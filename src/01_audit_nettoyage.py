"""
ETAPE 1 - Audit et nettoyage des donnees (telecom_credit_scoring_nigeria.csv)

Ce script NE MODIFIE PAS votre fichier d'origine : il lit le CSV, affiche un rapport
d'audit, puis ecrit une version nettoyee dans data/processed/telecom_clean.csv.

Utilisation (depuis la racine du projet, environnement virtuel active) :
    pip install pandas numpy scikit-learn
    python src/01_audit_nettoyage.py
    # ou, si votre fichier est ailleurs :
    python src/01_audit_nettoyage.py --input data/telecom_credit_scoring_nigeria.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

TARGET = "default_risk"
ORDRE_CIBLE = ["low", "medium", "high"]
COLS_ID = ["customer_id", "phone_number"]
COLS_NUM = ["credit_score", "tenure_months", "avg_monthly_spend_ngn", "recharge_frequency_per_month",
            "payment_consistency_score", "late_payment_count", "max_loan_amount_ngn"]
COLS_CAT = ["operator", "credit_rating", "account_type"]
# Colonnes "suspectes" : elles ressemblent a un RESULTAT de scoring, pas a un comportement du client
SUSPECTES = ["credit_score", "credit_rating", "loan_eligibility", "max_loan_amount_ngn"]
OPERATEURS = {"mtn": "MTN", "airtel": "Airtel", "glo": "Glo", "9mobile": "9mobile"}

LIGNES = []


def log(msg=""):
    print(msg)
    LIGNES.append(str(msg))


def titre(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def main(chemin_in, chemin_out, chemin_rapport):
    df = pd.read_csv(chemin_in)
    n0 = len(df)

    titre("1. DIMENSIONS ET TYPES")
    log(f"Lignes : {df.shape[0]:,}   Colonnes : {df.shape[1]}")
    log(df.dtypes.to_string())
    attendues = set(COLS_ID + COLS_NUM + COLS_CAT + [TARGET, "assessment_date", "loan_eligibility"])
    if attendues - set(df.columns):
        log(f"\n!! Colonnes attendues absentes : {sorted(attendues - set(df.columns))}")
    if set(df.columns) - attendues:
        log(f"\n!! Colonnes inattendues (a examiner) : {sorted(set(df.columns) - attendues)}")

    titre("2. VALEURS MANQUANTES")
    na = df.isna().sum()
    log(pd.DataFrame({"manquantes": na, "pourcentage": (na / len(df) * 100).round(2)}).to_string())

    titre("3. DOUBLONS")
    log(f"Lignes entierement dupliquees : {df.duplicated().sum():,}")
    if "customer_id" in df:
        log(f"customer_id dupliques          : {df['customer_id'].duplicated().sum():,}")

    titre("4. VARIABLE CIBLE : " + TARGET)
    cible = df[TARGET].astype(str).str.strip().str.lower()
    rep = cible.value_counts(dropna=False)
    log(pd.DataFrame({"effectif": rep, "pourcentage": (rep / len(df) * 100).round(2)}).to_string())
    inattendues = set(cible.unique()) - set(ORDRE_CIBLE)
    if inattendues:
        log(f"!! Valeurs inattendues dans la cible : {inattendues}")
    part_min = rep.min() / len(df)
    log(f"\nClasse la plus rare = {part_min:.1%} des lignes. "
        + ("Desequilibre FORT : il faudra suivre le F1 macro et le rappel de 'high', pas l'accuracy."
           if part_min < 0.15 else "Desequilibre modere/faible."))

    titre("5. COLONNES CONSTANTES OU INUTILES")
    constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    log(f"Colonnes constantes : {constantes or 'aucune'}")
    if "assessment_date" in df:
        log(f"assessment_date : {df['assessment_date'].nunique()} valeur(s) distincte(s)")

    titre("6. VALEURS INVALIDES (hors plage logique)")
    regles = {
        "payment_consistency_score": lambda s: (s < 0) | (s > 1),
        "tenure_months": lambda s: s < 0,
        "avg_monthly_spend_ngn": lambda s: s < 0,
        "recharge_frequency_per_month": lambda s: s < 0,
        "late_payment_count": lambda s: s < 0,
        "credit_score": lambda s: s < 0,
        "max_loan_amount_ngn": lambda s: s < 0,
    }
    for col, regle in regles.items():
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce")
            log(f"{col:32s} invalides : {int(regle(s).sum()):,}")

    titre("7. STATISTIQUES DES VARIABLES NUMERIQUES")
    num = [c for c in COLS_NUM if c in df]
    log(df[num].describe().T[["min", "25%", "50%", "75%", "max", "mean", "std"]].round(2).to_string())

    titre("8. VARIABLES CATEGORIELLES")
    for c in COLS_CAT + ["loan_eligibility"]:
        if c in df:
            log(f"\n-- {c}")
            log(df[c].astype(str).value_counts(dropna=False).to_string())

    titre("9. TEST DE FUITE DE DONNEES (data leakage)")
    log("Principe : on entraine un petit arbre (profondeur 4) avec UNE SEULE colonne a la fois.")
    log("Si une colonne seule predit presque parfaitement la cible, elle contient la reponse")
    log("(ou en est derivee) -> a EXCLURE du modele final.\n")
    ech = df.sample(min(100_000, len(df)), random_state=42).copy()
    ech["_y"] = ech[TARGET].astype(str).str.strip().str.lower().map({k: i for i, k in enumerate(ORDRE_CIBLE)})
    ech = ech.dropna(subset=["_y"])
    resultats = []
    for col in [c for c in ech.columns if c not in COLS_ID + [TARGET, "_y", "assessment_date"]]:
        x = ech[[col]].copy()
        if not pd.api.types.is_numeric_dtype(x[col]) or pd.api.types.is_bool_dtype(x[col]):
            x = pd.get_dummies(x.astype(str), dtype=float)      # texte / booleen -> colonnes 0/1
        x = x.apply(pd.to_numeric, errors="coerce").fillna(-1)
        xtr, xte, ytr, yte = train_test_split(x, ech["_y"], test_size=0.3, random_state=0, stratify=ech["_y"])
        arbre = DecisionTreeClassifier(max_depth=4, random_state=0, class_weight="balanced").fit(xtr, ytr)
        resultats.append((col, f1_score(yte, arbre.predict(xte), average="macro")))
    res = pd.DataFrame(resultats, columns=["colonne", "F1_macro_seule"]).sort_values("F1_macro_seule", ascending=False)
    res["verdict"] = np.where(res["F1_macro_seule"] > 0.85, "FUITE PROBABLE",
                              np.where(res["F1_macro_seule"] > 0.65, "tres informative (a examiner)", "normal"))
    log(res.round(3).to_string(index=False))
    log("\nColonnes que je soupconnais a priori : " + ", ".join(SUSPECTES))

    titre("10. NETTOYAGE")
    d = df.copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):   # colonnes texte
            d[c] = d[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan})
    d[TARGET] = d[TARGET].str.lower()
    avant = len(d)
    d = d[d[TARGET].isin(ORDRE_CIBLE)]
    log(f"Lignes supprimees (cible manquante/invalide) : {avant - len(d):,}")

    for c in ["operator"]:
        if c in d:
            d[c] = d[c].str.lower().map(OPERATEURS).fillna(d[c])
    for c in ["credit_rating", "account_type"]:
        if c in d:
            d[c] = d[c].str.lower()
    if "loan_eligibility" in d:
        d["loan_eligibility"] = d["loan_eligibility"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})

    for c in COLS_NUM:
        if c in d:
            avant_na = d[c].isna().sum()
            d[c] = pd.to_numeric(d[c], errors="coerce")
            if d[c].isna().sum() > avant_na:
                log(f"{c}: {d[c].isna().sum() - avant_na} valeur(s) non numerique(s) -> NaN")
    for col, regle in regles.items():
        if col in d:
            masque = regle(d[col])
            if masque.sum():
                log(f"{col}: {int(masque.sum())} valeur(s) invalide(s) -> NaN (imputation plus tard, apres le decoupage)")
                d.loc[masque, col] = np.nan

    n_dup = d.duplicated().sum()
    d = d.drop_duplicates()
    log(f"Lignes dupliquees supprimees : {n_dup:,}")
    if "customer_id" in d:
        n_id = d["customer_id"].duplicated().sum()
        d = d.drop_duplicates(subset="customer_id", keep="first")
        log(f"customer_id dupliques supprimes (premiere occurrence gardee) : {n_id:,}")

    a_retirer = [c for c in ["phone_number"] if c in d]            # donnee personnelle, aucune valeur predictive
    a_retirer += [c for c in d.columns if d[c].nunique(dropna=False) <= 1 and c not in a_retirer]
    d = d.drop(columns=a_retirer)
    log(f"Colonnes retirees : {a_retirer}")

    Path(chemin_out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(chemin_out, index=False)
    log(f"\nLignes : {n0:,} -> {len(d):,}   Colonnes : {df.shape[1]} -> {d.shape[1]}")
    log(f"Fichier propre ecrit : {chemin_out}")
    Path(chemin_rapport).parent.mkdir(parents=True, exist_ok=True)
    Path(chemin_rapport).write_text("\n".join(LIGNES), encoding="utf-8")
    log(f"Rapport enregistre   : {chemin_rapport}")
    log("\nAUCUNE imputation ni normalisation ici : elles se font APRES le decoupage train/test (etape 2).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/telecom_credit_scoring_nigeria.csv")
    ap.add_argument("--output", default="../data/processed/telecom_clean.csv")
    ap.add_argument("--report", default="../reports/01_audit.txt")
    a = ap.parse_args()
    main(a.input, a.output, a.report)
