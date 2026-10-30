"""
ETAPE 3 - Entrainement XGBoost + suivi des experiences avec MLflow

Lit      : data/processed/train.csv et val.csv   (le TEST n'est PAS utilise ici)
Ecrit    : models/*.joblib, mlflow.db + mlartifacts/ (suivi MLflow), reports/03_entrainement.csv

Trois "runs" sont enregistres dans MLflow :
  1. dummy_reference   : un modele qui repond toujours la classe majoritaire (le plancher a battre)
  2. xgb_comportement  : XGBoost sur le jeu A (7 variables de comportement)
  3. xgb_complet       : XGBoost sur le jeu B (jeu A + colonnes de scoring)

    python src/03_entrainement.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db        # pour visualiser (depuis la racine)

Isolation d'un CANDIDAT (utilisee par le flow CT, etape 7) : par defaut le script ecrit dans models/ et reports/
(comportement inchange). Pour ne PAS toucher au modele valide :
    python src/03_entrainement.py --modeles-dir models/candidats/20261030_1200 --reports-dir models/candidats/20261030_1200/reports
"""
import argparse
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from features import CLASSES, SEED, build_preprocessor, separer_X_y

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELES = ROOT / "models"                      # dossier par defaut (modele valide)
REPORTS = ROOT / "reports"                     # dossier par defaut
EXPERIENCE = "default_risk_telecom"

PARAMS_XGB = dict(
    objective="multi:softprob",
    n_estimators=1000,            # maximum ; l'arret anticipe choisit le bon nombre
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",           # rapide sur 340 000 lignes
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=-1,
)


def md5(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def metriques(y, pred, proba):
    """Metriques sur la validation. Le rappel de 'high' = part des mauvais payeurs reperes."""
    m = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro"),
        "auc_ovr_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "log_loss": log_loss(y, proba),
    }
    for i, c in enumerate(CLASSES):
        m[f"recall_{c}"] = recall_score(y, pred, labels=[i], average=None, zero_division=0)[0]
    return {k: float(v) for k, v in m.items()}


def preparer_mlflow():
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIENCE) is None:
        mlflow.create_experiment(EXPERIENCE, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIENCE)


def run_dummy(X_tr, y_tr, X_va, y_va, empreintes):
    with mlflow.start_run(run_name="dummy_reference"):
        mlflow.set_tags({"modele": "dummy", "jeu_variables": "aucun"})
        mlflow.log_params({"strategy": "most_frequent", **empreintes})
        d = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        m = metriques(y_va, d.predict(X_va), d.predict_proba(X_va))
        mlflow.log_metrics(m)
    return {"run": "dummy_reference", **m}


def run_xgb(jeu, X_tr, y_tr, X_va, y_va, empreintes, modeles):
    nom = f"xgb_{jeu}"
    with mlflow.start_run(run_name=nom):
        mlflow.set_tags({"modele": "xgboost", "jeu_variables": jeu})
        mlflow.log_params({**PARAMS_XGB, "jeu_variables": jeu, "poids": "balanced", **empreintes})

        # 1) Le pretraitement est ajuste sur le TRAIN uniquement
        pre = build_preprocessor(jeu, scale=False)
        A_tr = pre.fit_transform(X_tr)
        A_va = pre.transform(X_va)

        # 2) Poids "balanced" : les classes rares (medium, high) comptent davantage
        poids = compute_sample_weight("balanced", y_tr)

        # 3) Entrainement avec arret anticipe surveille sur la VALIDATION
        modele = XGBClassifier(**PARAMS_XGB)
        modele.fit(A_tr, y_tr, sample_weight=poids, eval_set=[(A_va, y_va)], verbose=False)
        mlflow.log_param("meilleure_iteration", int(modele.best_iteration))

        # 4) Un seul objet deployable : pretraitement + modele
        pipeline = Pipeline([("prep", pre), ("model", modele)])
        proba = pipeline.predict_proba(X_va)
        m = metriques(y_va, proba.argmax(axis=1), proba)
        mlflow.log_metrics(m)

        importances = modele.get_booster().get_score(importance_type="gain")
        mlflow.log_dict({k: float(v) for k, v in sorted(importances.items(), key=lambda t: -t[1])},
                        "importance_variables_gain.json")

        modeles.mkdir(parents=True, exist_ok=True)
        chemin = modeles / f"{nom}.joblib"
        joblib.dump(pipeline, chemin)
        mlflow.log_artifact(str(chemin))
        mlflow.sklearn.log_model(pipeline, "model",
                                 serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        print(f"{nom:18s} F1 macro = {m['f1_macro']:.4f}   AUC = {m['auc_ovr_macro']:.4f}   "
              f"rappel high = {m['recall_high']:.3f}   (arbres retenus : {modele.best_iteration + 1})")
    return {"run": nom, **m}


def main(modeles=MODELES, reports=REPORTS):
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    X_tr, y_tr = separer_X_y(train)
    X_va, y_va = separer_X_y(val)
    empreintes = {"train_md5": md5(DATA / "train.csv"), "val_md5": md5(DATA / "val.csv"),
                  "n_train": len(train), "n_val": len(val)}

    preparer_mlflow()
    resultats = [run_dummy(X_tr, y_tr, X_va, y_va, empreintes)]
    for jeu in ("comportement", "complet"):
        resultats.append(run_xgb(jeu, X_tr, y_tr, X_va, y_va, empreintes, modeles))

    tableau = pd.DataFrame(resultats).set_index("run").round(4)
    rapport = reports / "03_entrainement.csv"
    rapport.parent.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(rapport)
    print("\n=== Comparaison sur la VALIDATION ===")
    print(tableau[["accuracy", "f1_macro", "auc_ovr_macro", "recall_low", "recall_medium", "recall_high"]]
          .to_string())
    print(f"\nModeles ecrits dans : {modeles}")
    print(f"Tableau enregistre : {rapport}")
    print("Pour visualiser :  mlflow ui --backend-store-uri sqlite:///mlflow.db   (puis http://127.0.0.1:5000)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etape 3 : entrainement XGBoost + MLflow")
    ap.add_argument("--modeles-dir", type=Path, default=MODELES,
                    help="dossier de sortie des .joblib (defaut : models/ ; CT : un dossier candidat)")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS,
                    help="dossier de sortie du tableau CSV (defaut : reports/)")
    a = ap.parse_args()
    main(a.modeles_dir.resolve(), a.reports_dir.resolve())