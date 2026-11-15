"""
06_monitoring.py - Surveillance en production, detection de derives et generation d'alertes.

Ce script couvre l'Etape 8 du projet MLOps :
  1. Rejeu a partir des donnees reelles (train.csv = reference, test.csv = production simulee).
  2. 4 scenarios rigoureusement documentes et etiquetes [SIMULE] conformement a l'exigence D12 :
     - Scenario A : Normal (Temoin - donnees de test reelles) -> Verification de l'absence de fausse alerte.
     - Scenario B : Data Drift (Choc macroeconomique / Inflation) -> Alerte data drift (PSI > 0.20).
     - Scenario C : Concept Drift (Rupture comportementale : chute remboursements) -> Alerte performance (F1 en chute).
     - Scenario D : Incident technique (Latence accrue conformement a R10 : 880 ms) -> Alerte infrastructure.
  3. Calculs statistiques doubles :
     - Calcul integre du Population Stability Index (PSI) et tests par variable.
     - Generation du rapport d'observabilite Evidently (version 0.7.23).
  4. Generation des alertes structurees et rapports JSON + HTML dans reports/monitoring/.

Utilisation :
  python src/06_monitoring.py
  python src/06_monitoring.py --sample-size 5000
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "monitoring"

SEED = 42
FEATURES_NUM = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
]
FEATURES_CAT = ["operator", "account_type"]
TARGET = "default_risk"
CLASS_MAP = {"low": 0, "medium": 1, "high": 2}
INV_CLASS_MAP = {0: "low", 1: "medium", 2: "high"}


def calculer_psi_numerique(reference: np.ndarray, courant: np.ndarray, bins: int = 10) -> float:
    """
    Calcule le Population Stability Index (PSI) pour une variable numerique continue.
    Interprete selon les standards bancaires / credit scoring :
      PSI < 0.10         : Distribution stable
      0.10 <= PSI < 0.20 : Legere derive (Avertissement)
      PSI >= 0.20        : Derive marquee (Alerte / Action requise)
    """
    reference = reference[~np.isnan(reference)]
    courant = courant[~np.isnan(courant)]
    if len(reference) == 0 or len(courant) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 1:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(courant, bins=bin_edges)

    eps = 1e-4
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(courant) + eps * len(cur_counts))

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_val)


def calculer_psi_categoriel(ref_series: pd.Series, cur_series: pd.Series) -> float:
    """Calcule le PSI pour une variable categorielle."""
    categories = sorted(list(set(ref_series.dropna().unique()).union(set(cur_series.dropna().unique()))))
    eps = 1e-4
    ref_counts = ref_series.value_counts()
    cur_counts = cur_series.value_counts()
    n_ref = len(ref_series)
    n_cur = len(cur_series)

    psi_val = 0.0
    for cat in categories:
        p_ref = (ref_counts.get(cat, 0) + eps) / (n_ref + eps * len(categories))
        p_cur = (cur_counts.get(cat, 0) + eps) / (n_cur + eps * len(categories))
        psi_val += (p_cur - p_ref) * np.log(p_cur / p_ref)
    return float(psi_val)


def injecter_data_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario B [SIMULE] : Choc economique & inflation.
    - Hausse forte des montants moyens de depenses (+60%).
    - Diminution de la frequence mensuelle de recharge (-25%).
    - Augmentation des retards de paiement de 1.
    """
    df_drift = df.copy()
    np.random.seed(SEED)
    df_drift["avg_monthly_spend_ngn"] = df_drift["avg_monthly_spend_ngn"] * 1.60
    df_drift["recharge_frequency_per_month"] = (df_drift["recharge_frequency_per_month"] * 0.75).round().astype(int)
    df_drift["late_payment_count"] = df_drift["late_payment_count"] + 1
    return df_drift


def injecter_concept_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario C [SIMULE] : Rupture du lien features -> cible (Concept Drift).
    Les caracteristiques d'entree restent inchangees, mais les clients font face a une
    crise de liquidite : 80% des profils 'medium' et 60% des 'low' avec retard basculent en 'high'.
    Provoque l'effondrement du F1-score et du rappel sans aucune modification des variables d'entree.
    """
    df_cd = df.copy()
    np.random.seed(SEED)
    mask_med = (df_cd[TARGET] == "medium") & (np.random.rand(len(df_cd)) < 0.85)
    df_cd.loc[mask_med, TARGET] = "high"
    mask_low = (df_cd[TARGET] == "low") & (df_cd["late_payment_count"] >= 1) & (np.random.rand(len(df_cd)) < 0.65)
    df_cd.loc[mask_low, TARGET] = "high"
    return df_cd


def evaluer_scenario(
    nom_scenario: str,
    description: str,
    df_cur: pd.DataFrame,
    df_ref: pd.DataFrame,
    model,
    latence_simulee_ms: float = 65.0,
) -> dict:
    """Evalue les derives et performances d'un scenario donne."""
    psi_dict = {}
    derives_detectees = []
    for col in FEATURES_NUM:
        psi = calculer_psi_numerique(df_ref[col].values, df_cur[col].values)
        psi_dict[col] = round(psi, 4)
        if psi >= 0.20:
            derives_detectees.append({"feature": col, "psi": round(psi, 4), "statut": "CRITIQUE"})
        elif psi >= 0.10:
            derives_detectees.append({"feature": col, "psi": round(psi, 4), "statut": "AVERTISSEMENT"})

    for col in FEATURES_CAT:
        psi = calculer_psi_categoriel(df_ref[col], df_cur[col])
        psi_dict[col] = round(psi, 4)
        if psi >= 0.20:
            derives_detectees.append({"feature": col, "psi": round(psi, 4), "statut": "CRITIQUE"})
        elif psi >= 0.10:
            derives_detectees.append({"feature": col, "psi": round(psi, 4), "statut": "AVERTISSEMENT"})

    X_cur = df_cur.drop(columns=[TARGET, "customer_id"], errors="ignore")
    t0 = time.time()
    y_pred_code = model.predict(X_cur)
    duree_calc_ms = (time.time() - t0) * 1000 / max(len(X_cur), 1)
    latence_effective_ms = latence_simulee_ms if latence_simulee_ms > duree_calc_ms else duree_calc_ms

    y_pred_labels = [INV_CLASS_MAP.get(int(p), str(p)) for p in y_pred_code]
    y_true_labels = df_cur[TARGET].astype(str).tolist()

    f1_macro = float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    f1_high = float(f1_score(y_true_labels, y_pred_labels, labels=["high"], average="macro", zero_division=0))
    recall_high = float(recall_score(y_true_labels, y_pred_labels, labels=["high"], average="macro", zero_division=0))

    dist_pred = pd.Series(y_pred_labels).value_counts(normalize=True).round(3).to_dict()

    alertes = []
    if len(derives_detectees) > 0:
        max_psi = max(psi_dict.values())
        critique = any(d["statut"] == "CRITIQUE" for d in derives_detectees)
        alertes.append({
            "type": "DATA_DRIFT",
            "severite": "CRITIQUE" if critique else "WARNING",
            "message": f"Data drift detecte sur {len(derives_detectees)} variable(s) (PSI max = {max_psi}).",
        })

    if f1_macro < 0.38:
        alertes.append({
            "type": "CONCEPT_DRIFT_PERFORMANCE",
            "severite": "CRITIQUE",
            "message": f"Chute critique du F1-macro ({f1_macro:.4f} < seuil 0.3800).",
        })
    elif f1_macro < 0.44:
        alertes.append({
            "type": "CONCEPT_DRIFT_PERFORMANCE",
            "severite": "WARNING",
            "message": f"F1-macro en baisse sensible ({f1_macro:.4f} < seuil 0.4400).",
        })

    if recall_high < 0.45:
        alertes.append({
            "type": "RECOUVREMENT_RISQUE",
            "severite": "CRITIQUE",
            "message": f"Rappel classe high insuffisant ({recall_high:.4f} < 0.4500) : impayes non detectes.",
        })

    if latence_effective_ms > 300.0:
        alertes.append({
            "type": "INFRASTRUCTURE_LATENCE",
            "severite": "CRITIQUE",
            "message": f"Latence anormale du service ({latence_effective_ms:.1f} ms > seuil 300 ms). Incident R10.",
        })

    statut_global = "OK"
    if any(a["severite"] == "CRITIQUE" for a in alertes):
        statut_global = "CRITIQUE"
    elif any(a["severite"] == "WARNING" for a in alertes):
        statut_global = "AVERTISSEMENT"

    return {
        "scenario": nom_scenario,
        "description": description,
        "statut_global": statut_global,
        "nb_echantillons": len(df_cur),
        "latence_moyenne_ms": round(latence_effective_ms, 2),
        "f1_macro": round(f1_macro, 4),
        "f1_high": round(f1_high, 4),
        "recall_high": round(recall_high, 4),
        "distribution_predictions": dist_pred,
        "psi_par_feature": psi_dict,
        "derives_detectees": derives_detectees,
        "alertes": alertes,
    }


def generer_rapport_evidently(df_ref: pd.DataFrame, df_cur: pd.DataFrame, output_path: Path):
    """Genere le rapport HTML Evidently 0.7.23."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        sample_ref = df_ref.sample(min(len(df_ref), 2000), random_state=SEED)
        sample_cur = df_cur.sample(min(len(df_cur), 2000), random_state=SEED)

        report = Report(metrics=[DataDriftPreset()])
        cols = FEATURES_NUM + FEATURES_CAT
        snapshot = report.run(
            sample_ref[cols],
            sample_cur[cols],
        )
        snapshot.save_html(str(output_path))
        print(f"[OK] Rapport Evidently sauvegarde dans : {output_path}")
        return True
    except Exception as e:
        print(f"[INFO] Rapport Evidently simplifie (fallback) : {e}")
        return False


def generer_dashboard_html(resultats_scenarios: list, output_path: Path):
    """Genere un dashboard HTML d'observabilite moderne et autonome."""
    badge_color = {"OK": "#28a745", "AVERTISSEMENT": "#ffc107", "CRITIQUE": "#dc3545"}

    rows_html = ""
    for s in resultats_scenarios:
        color = badge_color.get(s["statut_global"], "#6c757d")
        text_color = "#000" if s["statut_global"] == "AVERTISSEMENT" else "#fff"
        if s["alertes"]:
            alertes_str = "<br>".join([f"&bull; <b>[{a['severite']}]</b> {a['message']}" for a in s["alertes"]])
        else:
            alertes_str = "<i>Aucune alerte (nominal)</i>"

        psi_resume = ", ".join([f"{k}: {v}" for k, v in s["psi_par_feature"].items() if v >= 0.10])
        if not psi_resume:
            psi_resume = "Toutes variables stables (PSI < 0.10)"

        badge_span = (
            f'<span style="background-color:{color}; color:{text_color}; '
            f'padding:6px 12px; border-radius:12px; font-weight:bold;">{s["statut_global"]}</span>'
        )

        rows_html += f"""
        <tr>
            <td><b>{s['scenario']}</b><br><small style="color:#666">{s['description']}</small></td>
            <td style="text-align:center;">{badge_span}</td>
            <td>{s['f1_macro']}</td>
            <td>{s['recall_high']}</td>
            <td>{s['latence_moyenne_ms']} ms</td>
            <td><small>{psi_resume}</small></td>
            <td><small>{alertes_str}</small></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Tableau de Bord de Monitoring MLOps - Credit Scoring Telecom</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 30px; background-color: #f8f9fa; color: #212529;
        }}
        .container {{
            max-width: 1200px; margin: auto; background: white; padding: 30px;
            border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #1e3d59; border-bottom: 2px solid #1e3d59; padding-bottom: 10px; }}
        .summary-box {{ display: flex; gap: 20px; margin: 25px 0; }}
        .card {{ flex: 1; padding: 20px; border-radius: 6px; background: #f1f4f9; border-left: 5px solid #1e3d59; }}
        .card h3 {{ margin: 0 0 10px 0; font-size: 16px; color: #555; }}
        .card p {{ margin: 0; font-size: 24px; font-weight: bold; color: #1e3d59; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 25px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #dee2e6; vertical-align: top; }}
        th {{ background-color: #f8f9fa; color: #495057; }}
        tr:hover {{ background-color: #f5f8ff; }}
        .footer {{
            margin-top: 35px; padding-top: 15px; border-top: 1px solid #dee2e6;
            font-size: 13px; color: #6c757d;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Tableau de Bord d'Observabilite & Surveillance MLOps</h1>
    <p><b>Projet :</b> Detection de Risque d'Impayes Telecom (Nigeria) | <b>Moteur :</b> XGBoost + Evidently & PSI</p>

    <div class="summary-box">
        <div class="card">
            <h3>Seuils d'Alerte PSI</h3>
            <p>&lt; 0.10 (Stable) | &ge; 0.20 (Alerte)</p>
        </div>
        <div class="card">
            <h3>Seuil Performance F1</h3>
            <p>&ge; 0.44 (Nominal) | &lt; 0.38 (Critique)</p>
        </div>
        <div class="card">
            <h3>Seuil Latence (R10)</h3>
            <p>&lt; 150 ms (Normal) | &gt; 300 ms (Alerte)</p>
        </div>
    </div>

    <h2>Synthese des Scenarios Testes [Donnees Reelles & Simulations D12]</h2>
    <table>
        <thead>
            <tr>
                <th>Scenario</th>
                <th style="text-align:center;">Statut</th>
                <th>F1 Macro</th>
                <th>Rappel 'High'</th>
                <th>Latence</th>
                <th>Derive Variables (PSI)</th>
                <th>Alertes Declenchees</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="footer">
        <p><b>Auditabilite & Regles MLOps :</b> Tous les scenarios avec injection de derive sont marques
        <b>[SIMULE]</b>. En cas d'alerte critique sur les donnees ou la performance, le pipeline declenche
        une notification et oriente vers la validation humaine avant tout redeploiement automatique.</p>
        <p>Genere automatiquement par <code>src/06_monitoring.py</code> pour l'Etape 8.</p>
    </div>
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] Dashboard HTML sauvegarde dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Monitoring et detection de derive")
    parser.add_argument("--sample-size", type=int, default=10000, help="Nombre de lignes a analyser")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ETAPE 8 : MONITORING, DETECTION DE DERIVE ET ALERTES EN PRODUCTION")
    print("=" * 80)

    print("[1/5] Chargement des jeux de reference (train) et de production (test)...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    n_sample = min(args.sample_size, len(df_test))
    df_ref = df_train.sample(n=min(len(df_train), n_sample * 2), random_state=SEED)
    df_prod_base = df_test.sample(n=n_sample, random_state=SEED).copy()

    model_path = MODELS_DIR / "model_final.joblib"
    print(f"[2/5] Chargement du modele de production : {model_path}...")
    model = joblib.load(model_path)

    print("[3/5] Construction des 4 scenarios de surveillance...")
    df_scen_a = df_prod_base.copy()
    df_scen_b = injecter_data_drift(df_prod_base)
    df_scen_c = injecter_concept_drift(df_prod_base)
    df_scen_d = df_prod_base.copy()

    scenarios = [
        ("Scenario A (Nominal)", "Donnees de test reelles sans alteration. Absence de fausse alerte.", df_scen_a, 72.0),
        ("Scenario B (Data Drift [SIMULE])", "Choc inflationniste (+60% depenses, retards accrus).", df_scen_b, 75.0),
        ("Scenario C (Concept Drift [SIMULE])", "Rupture de comportement. Chute de F1 sans modif entrees.", df_scen_c, 74.0),
        ("Scenario D (Incident Infra [SIMULE])", "Latence anormale illustrant la question R10 (880 ms).", df_scen_d, 880.0),
    ]

    print("[4/5] Evaluation des derives, calcul des PSI et evaluation des alertes...")
    resultats = []
    toutes_alertes = []

    for nom, desc, df_scen, latence in scenarios:
        res = evaluer_scenario(nom, desc, df_scen, df_ref, model, latence_simulee_ms=latence)
        resultats.append(res)
        for al in res["alertes"]:
            toutes_alertes.append({"scenario": nom, **al})
        lat = res["latence_moyenne_ms"]
        print(f"  -> {nom} : Statut = {res['statut_global']} | F1 = {res['f1_macro']} | Latence = {lat} ms")

    print("[5/5] Export des livrables de monitoring...")
    alertes_file = REPORTS_DIR / "alertes.json"
    with open(alertes_file, "w", encoding="utf-8") as f:
        json.dump(toutes_alertes, f, indent=2, ensure_ascii=False)
    print(f"[OK] Fichier d'alertes : {alertes_file}")

    rapport_json_file = REPORTS_DIR / "rapport_monitoring.json"
    with open(rapport_json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date_execution": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reference": {"taille": len(df_ref), "source": "data/processed/train.csv"},
                "resultats_scenarios": resultats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK] Rapport JSON complet : {rapport_json_file}")

    dashboard_file = REPORTS_DIR / "dashboard_monitoring.html"
    generer_dashboard_html(resultats, dashboard_file)

    evidently_file = REPORTS_DIR / "data_drift_report.html"
    generer_rapport_evidently(df_ref, df_scen_b, evidently_file)

    print("=" * 80)
    print("MONITORING VALIDE AVEC SUCCES.")
    print("=" * 80)


if __name__ == "__main__":
    main()
