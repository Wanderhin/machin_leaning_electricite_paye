"""
ETAPE 2 - Decoupage train / validation / test (stratifie) + verification du pretraitement

Lit      : data/processed/telecom_clean.csv
Ecrit    : data/processed/train.csv, val.csv, test.csv  et  reports/02_decoupage.json

Les chemins sont calcules a partir de l'emplacement de ce fichier : on peut donc le lancer
depuis la racine du projet OU depuis src/, sans risque d'ecrire hors du projet.

    python src/02_decoupage.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from features import CLASSES, FEATURE_SETS, ID_COL, SEED, TARGET, build_preprocessor, separer_X_y

ROOT = Path(__file__).resolve().parents[1]
ENTREE = ROOT / "data" / "processed" / "telecom_clean.csv"
SORTIE = ROOT / "data" / "processed"
RAPPORT = ROOT / "reports" / "02_decoupage.json"

PART_TEST = 0.15
PART_VAL = 0.15


def repartition(df):
    return (df[TARGET].value_counts(normalize=True).reindex(CLASSES) * 100).round(2).to_dict()


def main():
    df = pd.read_csv(ENTREE)
    print(f"Donnees propres : {len(df):,} lignes, {df.shape[1]} colonnes")

    # 1) On met le TEST de cote en premier, puis on separe train / validation.
    #    stratify = on conserve les proportions low/medium/high dans chaque morceau.
    train_val, test = train_test_split(df, test_size=PART_TEST, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(train_val, test_size=PART_VAL / (1 - PART_TEST),
                                  stratify=train_val[TARGET], random_state=SEED)

    # 2) Controles d'integrite
    print("\n--- Tailles et proportions de la cible (%) ---")
    resume = {}
    for nom, part in [("train", train), ("val", val), ("test", test)]:
        resume[nom] = {"lignes": len(part), "cible_pct": repartition(part)}
        print(f"{nom:6s} {len(part):>8,}  {resume[nom]['cible_pct']}")

    ids_tr, ids_va, ids_te = set(train[ID_COL]), set(val[ID_COL]), set(test[ID_COL])
    chevauchement = len(ids_tr & ids_va) + len(ids_tr & ids_te) + len(ids_va & ids_te)
    print(f"\nclients presents dans plusieurs morceaux : {chevauchement}  (doit etre 0)")
    assert chevauchement == 0, "Fuite : un meme client est dans deux morceaux !"
    assert len(train) + len(val) + len(test) == len(df)

    # 3) Ecriture des fichiers
    SORTIE.mkdir(parents=True, exist_ok=True)
    for nom, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(SORTIE / f"{nom}.csv", index=False)

    # 4) Test du pretraitement : on l'AJUSTE sur le train uniquement, puis on l'applique au val
    print("\n--- Verification du pretraitement (fit sur train, transform sur val) ---")
    X_tr, _ = separer_X_y(train)
    X_va, _ = separer_X_y(val)
    for jeu in FEATURE_SETS:
        pre = build_preprocessor(jeu)
        A = pre.fit_transform(X_tr)
        B = pre.transform(X_va)
        assert A.shape[1] == B.shape[1] and not A.isna().any().any() and not B.isna().any().any()
        print(f"{jeu:12s} {A.shape[1]:>2} colonnes : {list(A.columns)}")

    RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    RAPPORT.write_text(json.dumps({"seed": SEED, "source": ENTREE.name, "decoupage": resume},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFichiers ecrits dans {SORTIE} et rapport dans {RAPPORT}")


if __name__ == "__main__":
    main()
