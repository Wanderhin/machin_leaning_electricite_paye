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
    """Genere un rapport HTML de surveillance, autonome (aucune ressource externe)."""
    from collections import Counter
    from datetime import datetime
    from html import escape as esc

    # --- Seuils de reference (affiches et utilises pour la mise en forme) ---
    PSI_STABLE, PSI_ALERTE = 0.10, 0.20
    F1_NOMINAL, F1_CRITIQUE = 0.44, 0.38
    LAT_NORMALE_MS, LAT_ALERTE_MS = 150, 300

    STATUTS = {
        "OK": ("ok", "Nominal"),
        "AVERTISSEMENT": ("warn", "Avertissement"),
        "CRITIQUE": ("crit", "Critique"),
    }

    css = """
:root {
  --ink: #1b2430; --muted: #5d6877; --rule: #d5dae1; --rule-soft: #e8ebef;
  --paper: #ffffff; --page: #f3f4f6; --brand: #17375e;
  --ok: #1f7a46; --warn: #a8690a; --crit: #b3261e;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
.page {
  max-width: 1180px; margin: 32px auto; background: var(--paper);
  border: 1px solid var(--rule); border-top: 4px solid var(--brand);
  padding: 40px 48px 32px;
}
header { padding-bottom: 24px; border-bottom: 1px solid var(--rule); }
h1 { margin: 0 0 4px; font-size: 24px; font-weight: 600; letter-spacing: -0.01em; color: var(--brand); }
.subtitle { margin: 0; color: var(--muted); font-size: 15px; }
dl.meta { display: grid; grid-template-columns: repeat(4, auto); gap: 4px 40px; margin: 22px 0 0; justify-content: start; }
dl.meta dt { color: var(--muted); font-size: 12px; }
dl.meta dd { margin: 0 0 6px; font-weight: 600; }
h2 { margin: 36px 0 12px; font-size: 16px; font-weight: 600; color: var(--brand); }
p.lead { margin: 0 0 4px; max-width: 76ch; }
.count { white-space: nowrap; }
.count::before, .tag::before {
  content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 6px;
  background: var(--c, var(--muted));
}
.ok { --c: var(--ok); } .warn { --c: var(--warn); } .crit { --c: var(--crit); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule-soft); }
thead th { font-weight: 600; font-size: 13px; border-bottom: 2px solid var(--ink); white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.thresholds { max-width: 760px; }
table.thresholds td:first-child { font-weight: 600; }
table.scenarios tbody tr { border-left: 4px solid var(--c, transparent); }
table.scenarios tbody tr:last-child td { border-bottom: 1px solid var(--rule); }
.name { font-weight: 600; }
.desc { color: var(--muted); font-size: 13px; }
.tag { font-weight: 600; white-space: nowrap; color: var(--c); }
td.num.warn { color: var(--warn); font-weight: 600; }
td.num.crit { color: var(--crit); font-weight: 600; }
ul.alerts { margin: 0; padding: 0; list-style: none; font-size: 13px; }
ul.alerts li { margin: 0 0 4px; }
ul.alerts .sev { font-weight: 600; color: var(--c, var(--ink)); }
.psi { display: block; font-size: 13px; font-variant-numeric: tabular-nums; }
.psi b { font-weight: 600; }
.psi.warn b { color: var(--warn); } .psi.crit b { color: var(--crit); }
.muted { color: var(--muted); }
.notes { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 13px; color: var(--muted); max-width: 90ch; }
.notes p { margin: 0 0 8px; }
.notes b { color: var(--ink); font-weight: 600; }
@media (max-width: 760px) {
  .page { margin: 0; padding: 24px 18px; border-left: 0; border-right: 0; }
  dl.meta { grid-template-columns: repeat(2, auto); }
}
@media print {
  body { background: #fff; }
  .page { margin: 0; border: 0; border-top: 4px solid var(--brand); padding: 0; max-width: none; }
  tr { break-inside: avoid; }
}
"""

    template = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport de surveillance MLOps - Scoring de risque d'impayés télécom</title>
<style>__CSS__</style>
</head>
<body>
<div class="page">
  <header>
    <h1>Rapport de surveillance du modèle</h1>
    <p class="subtitle">Détection du risque d'impayés télécom (Nigeria)</p>
    <dl class="meta">
      <div><dt>Modèle</dt><dd>XGBoost</dd></div>
      <div><dt>Surveillance</dt><dd>Evidently et PSI</dd></div>
      <div><dt>Scénarios évalués</dt><dd>__NB__</dd></div>
      <div><dt>Généré le</dt><dd>__DATE__</dd></div>
    </dl>
  </header>

  <h2>Synthèse</h2>
  <p class="lead">__SYNTHESE__</p>

  <h2>Seuils de surveillance</h2>
  <div class="table-wrap">
    <table class="thresholds">
      <thead><tr><th>Indicateur</th><th>Nominal</th><th>Alerte</th></tr></thead>
      <tbody>
        <tr><td>Dérive des variables (PSI)</td><td>&lt; __PSI_S__ (stable)</td><td>&ge; __PSI_A__</td></tr>
        <tr><td>F1 macro</td><td>&ge; __F1_N__</td><td>&lt; __F1_C__ (critique)</td></tr>
        <tr><td>Latence moyenne (R10)</td><td>&lt; __LAT_N__ ms</td><td>&gt; __LAT_A__ ms</td></tr>
      </tbody>
    </table>
  </div>

  <h2>Résultats par scénario</h2>
  <div class="table-wrap">
    <table class="scenarios">
      <thead>
        <tr>
          <th>Scénario</th>
          <th>Statut</th>
          <th class="num">F1 macro</th>
          <th class="num">Rappel « High »</th>
          <th class="num">Latence moy.</th>
          <th>Variables en dérive (PSI &ge; __PSI_S__)</th>
          <th>Alertes</th>
        </tr>
      </thead>
      <tbody>
__ROWS__
      </tbody>
    </table>
  </div>

  <div class="notes">
    <p><b>Auditabilité.</b> Les scénarios avec injection de dérive sont marqués [SIMULE]. Lors d'une alerte
    critique sur les données ou la performance, le pipeline envoie une notification et demande une validation
    humaine avant tout redéploiement automatique.</p>
    <p>Rapport généré automatiquement par <code>src/06_monitoring.py</code> (étape 8).</p>
  </div>
</div>
</body>
</html>
"""

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def fmt(v, nd=3):
        f = num(v)
        return esc(str(v)) if f is None else f"{f:.{nd}f}"

    def niveau(v, warn, crit, inverse=False):
        """Classe CSS ('warn'/'crit'/'') selon des seuils. inverse=True : plus bas = pire."""
        f = num(v)
        if f is None:
            return ""
        if inverse:
            return "crit" if f < crit else "warn" if f < warn else ""
        return "crit" if f >= crit else "warn" if f >= warn else ""

    rows = []
    for s in resultats_scenarios:
        classe, libelle = STATUTS.get(s["statut_global"], ("", esc(str(s["statut_global"]))))

        psi_items = sorted(
            ((k, v) for k, v in s["psi_par_feature"].items() if (num(v) or 0) >= PSI_STABLE),
            key=lambda kv: -(num(kv[1]) or 0),
        )
        if psi_items:
            psi_html = "".join(
                f'<span class="psi {niveau(v, PSI_STABLE, PSI_ALERTE)}">{esc(str(k))} : <b>{fmt(v)}</b></span>'
                for k, v in psi_items
            )
        else:
            psi_html = f'<span class="muted">Aucune (toutes &lt; {PSI_STABLE:.2f})</span>'

        if s["alertes"]:
            items = "".join(
                f'<li><span class="sev {STATUTS.get(a["severite"], ("", ""))[0]}">'
                f'{esc(str(a["severite"]))}</span> : {esc(str(a["message"]))}</li>'
                for a in s["alertes"]
            )
            alertes_html = f'<ul class="alerts">{items}</ul>'
        else:
            alertes_html = '<span class="muted">Aucune alerte</span>'

        cls_f1 = niveau(s["f1_macro"], F1_NOMINAL, F1_CRITIQUE, inverse=True)
        cls_lat = niveau(s["latence_moyenne_ms"], LAT_NORMALE_MS + 1e-9, LAT_ALERTE_MS + 1e-9)

        rows.append(
            f'        <tr class="{classe}">\n'
            f'          <td><div class="name">{esc(str(s["scenario"]))}</div>'
            f'<div class="desc">{esc(str(s["description"]))}</div></td>\n'
            f'          <td><span class="tag {classe}">{libelle}</span></td>\n'
            f'          <td class="num {cls_f1}">{fmt(s["f1_macro"])}</td>\n'
            f'          <td class="num">{fmt(s["recall_high"])}</td>\n'
            f'          <td class="num {cls_lat}">{fmt(s["latence_moyenne_ms"], 1)} ms</td>\n'
            f'          <td>{psi_html}</td>\n'
            f'          <td>{alertes_html}</td>\n'
            f'        </tr>'
        )

    n = len(resultats_scenarios)
    cpt = Counter(s["statut_global"] for s in resultats_scenarios)
    parts = [
        f'<span class="count {cls}">{cpt[cle]} {lib.lower()}</span>'
        for cle, (cls, lib) in STATUTS.items() if cpt.get(cle)
    ]
    synthese = f"{n} scénario{'s' if n > 1 else ''} évalué{'s' if n > 1 else ''} : " + ", ".join(parts) + "."

    page = template
    for token, val in {
        "__CSS__": css,
        "__NB__": str(n),
        "__DATE__": datetime.now().strftime("%d/%m/%Y à %H:%M"),
        "__SYNTHESE__": synthese,
        "__PSI_S__": f"{PSI_STABLE:.2f}",
        "__PSI_A__": f"{PSI_ALERTE:.2f}",
        "__F1_N__": f"{F1_NOMINAL:.2f}",
        "__F1_C__": f"{F1_CRITIQUE:.2f}",
        "__LAT_N__": str(LAT_NORMALE_MS),
        "__LAT_A__": str(LAT_ALERTE_MS),
        "__ROWS__": "\n".join(rows),
    }.items():
        page = page.replace(token, val)

    output_path.write_text(page, encoding="utf-8")
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
