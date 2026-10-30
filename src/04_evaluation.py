"""
ETAPE 4 - Evaluation finale sur le TEST + enregistrement dans le registre de modeles MLflow

Ordre des operations (important pour que le test reste honnete) :
  1. On charge le modele retenu a l'etape 3 (par defaut : jeu "comportement").
  2. Sur la VALIDATION uniquement, on cherche des "poids de decision" (un multiplicateur par classe
     applique aux probabilites avant l'argmax) qui maximisent le F1 macro. On ne les garde que si
     le gain sur la validation est >= 0.01 ; sinon on garde l'argmax simple.
  3. Le TEST est utilise UNE SEULE FOIS, avec la regle de decision deja figee a l'etape 2.
  4. Resultats, matrice de confusion, poids et modele sont enregistres (reports/, models/, MLflow).
  5. Le modele est ajoute au registre MLflow avec l'alias "staging".

Lancement (depuis la racine du projet) :
    python src/04_evaluation.py
    python src/04_evaluation.py --jeu complet            # pour evaluer l'autre jeu de variables
    python src/04_evaluation.py --promouvoir 1           # passe la version 1 du registre en "production"

Isolation d'un CANDIDAT (utilisee par le flow CT, etape 7). Par defaut, le script lit et ECRASE models/ et reports/
et pose l'alias "staging" (comportement inchange). Pour evaluer un candidat SANS toucher au modele valide :
    python src/04_evaluation.py --modeles-dir models/candidats/X --reports-dir models/candidats/X/reports --alias challenger
Le script lit alors models/candidats/X/xgb_<jeu>.joblib (produit par 03 avec le meme --modeles-dir) et ecrit
model_final.joblib, decision_poids.json et les rapports dans ces dossiers, pas dans models/ ni reports/.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from features import CLASSES, SEED, separer_X_y

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELES = ROOT / "models"                      # dossier par defaut (modele valide, lu par l'API)
REPORTS = ROOT / "reports"                     # dossier par defaut
EXPERIENCE = "default_risk_telecom"
NOM_MODELE = "default_risk_xgb"
GAIN_MIN = 0.01          # gain minimal de F1 macro (validation) pour adopter les poids de decision


def md5(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def f1_macro_rapide(y, pred):
    """F1 macro calcule via la matrice de confusion (tres rapide, sert a la recherche de poids)."""
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean())


def chercher_poids(y_val, proba_val):
    """Recherche en grille des multiplicateurs (low fixe a 1) sur la VALIDATION."""
    y = np.asarray(y_val)
    base = f1_macro_rapide(y, proba_val.argmax(axis=1))
    meilleur, meilleurs_poids = base, (1.0, 1.0, 1.0)
    for w_med in np.arange(0.5, 6.01, 0.25):
        for w_high in np.arange(0.5, 3.01, 0.25):
            w = np.array([1.0, w_med, w_high])
            f = f1_macro_rapide(y, (proba_val * w).argmax(axis=1))
            if f > meilleur:
                meilleur, meilleurs_poids = f, (1.0, float(w_med), float(w_high))
    return base, meilleur, meilleurs_poids


def predire(proba, poids):
    return (proba * np.asarray(poids)).argmax(axis=1)


def metriques_test(y, pred, proba, prefixe):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    m = {
        f"{prefixe}accuracy": accuracy_score(y, pred),
        f"{prefixe}f1_macro": float(f.mean()),
        f"{prefixe}precision_macro": float(p.mean()),
        f"{prefixe}recall_macro": float(r.mean()),
        f"{prefixe}auc_ovr_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        f"{prefixe}log_loss": log_loss(y, proba),
    }
    for i, c in enumerate(CLASSES):
        m[f"{prefixe}precision_{c}"] = float(p[i])
        m[f"{prefixe}recall_{c}"] = float(r[i])
        m[f"{prefixe}f1_{c}"] = float(f[i])
    return {k: float(v) for k, v in m.items()}


def dessiner_matrice(y, pred, chemin, titre):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=CLASSES, ax=axes[0],
                                            cmap="Blues", colorbar=False)
    axes[0].set_title("Effectifs")
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=CLASSES, ax=axes[1], normalize="true",
                                            values_format=".2f", cmap="Blues", colorbar=False)
    axes[1].set_title("Normalisee par classe reelle (= rappel sur la diagonale)")
    for ax in axes:
        ax.set_xlabel("Classe predite")
        ax.set_ylabel("Classe reelle")
    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def promouvoir(version):
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    client.set_registered_model_alias(NOM_MODELE, "production", str(version))
    client.set_model_version_tag(NOM_MODELE, str(version), "statut", "production")
    for alias in ("staging", "challenger"):          # la version promue n'est plus ni staging ni challenger
        try:
            if client.get_model_version_by_alias(NOM_MODELE, alias).version == str(version):
                client.delete_registered_model_alias(NOM_MODELE, alias)
        except Exception:
            pass
    print(f"Version {version} de '{NOM_MODELE}' promue en PRODUCTION (alias 'production').")
    print("En vrai MLOps, cette promotion est declenchee par la CI/CD apres une porte de qualite (etape 7).")


def main(jeu, modeles=MODELES, reports=REPORTS, alias="staging"):
    chemin_modele = modeles / f"xgb_{jeu}.joblib"
    pipeline = joblib.load(chemin_modele)
    val, test = pd.read_csv(DATA / "val.csv"), pd.read_csv(DATA / "test.csv")
    X_va, y_va = separer_X_y(val)
    X_te, y_te = separer_X_y(test)
    y_va, y_te = y_va.to_numpy(), y_te.to_numpy()

    # --- 1) Poids de decision : cherches sur la VALIDATION seulement
    proba_va = pipeline.predict_proba(X_va)
    base_val, best_val, poids_best = chercher_poids(y_va, proba_va)
    gain = best_val - base_val
    adopte = gain >= GAIN_MIN
    poids = poids_best if adopte else (1.0, 1.0, 1.0)
    print(f"Validation : F1 macro argmax simple = {base_val:.4f} | avec poids {poids_best} = {best_val:.4f} "
          f"(gain {gain:+.4f})")
    print("=> poids de decision " + ("ADOPTES" if adopte else f"NON adoptes (gain < {GAIN_MIN}) : argmax simple"))

    # --- 2) Evaluation unique sur le TEST (regle de decision deja figee)
    proba_te = pipeline.predict_proba(X_te)
    pred_final = predire(proba_te, poids)
    m = metriques_test(y_te, pred_final, proba_te, "test_")
    m_simple = metriques_test(y_te, proba_te.argmax(axis=1), proba_te, "test_argmax_")

    reports.mkdir(parents=True, exist_ok=True)
    modeles.mkdir(parents=True, exist_ok=True)
    png = reports / "04_matrice_confusion_test.png"
    dessiner_matrice(y_te, pred_final, png, f"Matrice de confusion (TEST) - xgb_{jeu} - poids {poids}")
    rapport = classification_report(y_te, pred_final, target_names=CLASSES, digits=4, output_dict=True)
    (reports / "04_evaluation_test.json").write_text(json.dumps(
        {"jeu_variables": jeu, "poids_decision": poids, "poids_adoptes": bool(adopte),
         "validation": {"f1_macro_argmax": base_val, "f1_macro_poids": best_val},
         "metriques_test": m, "metriques_test_argmax_simple": m_simple,
         "classification_report": rapport,
         "matrice_confusion": confusion_matrix(y_te, pred_final).tolist()}, indent=2), encoding="utf-8")

    # Fichiers de reference pour l'API (etape 5) : modele final + poids de decision
    shutil.copy(chemin_modele, modeles / "model_final.joblib")
    (modeles / "decision_poids.json").write_text(json.dumps(
        {"classes": CLASSES, "poids": list(poids), "jeu_variables": jeu}, indent=2), encoding="utf-8")

    # --- 3) MLflow : run d'evaluation + enregistrement dans le registre
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIENCE)
    empreintes = {"train_md5": md5(DATA / "train.csv"), "val_md5": md5(DATA / "val.csv"),
                  "test_md5": md5(DATA / "test.csv")}
    with mlflow.start_run(run_name=f"evaluation_test_{jeu}"):
        mlflow.set_tags({"etape": "evaluation_test", "jeu_variables": jeu, "alias_vise": alias})
        mlflow.log_params({"jeu_variables": jeu, "poids_decision": str(poids), "poids_adoptes": bool(adopte),
                           "seed": SEED, **empreintes})
        mlflow.log_metrics({**m, **m_simple, "val_f1_macro_argmax": base_val, "val_f1_macro_poids": best_val})
        mlflow.log_artifact(str(png))
        mlflow.log_artifact(str(reports / "04_evaluation_test.json"))
        mlflow.log_artifact(str(modeles / "decision_poids.json"))
        mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=NOM_MODELE,
                                 serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                                 input_example=X_te.head(3))

    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{NOM_MODELE}'"))
    for cle, val_tag in {"statut": alias, "jeu_variables": jeu, "poids_decision": str(poids),
                         "f1_macro_test": f"{m['test_f1_macro']:.4f}",
                         "auc_test": f"{m['test_auc_ovr_macro']:.4f}", "train_md5": empreintes["train_md5"],
                         "seed": str(SEED)}.items():
        client.set_model_version_tag(NOM_MODELE, str(version), cle, val_tag)
    client.set_registered_model_alias(NOM_MODELE, alias, str(version))

    # --- 4) Resume
    print(f"\n=== TEST (une seule evaluation) - xgb_{jeu} - poids {poids} ===")
    print(f"accuracy {m['test_accuracy']:.4f} | F1 macro {m['test_f1_macro']:.4f} | "
          f"AUC {m['test_auc_ovr_macro']:.4f} | log-loss {m['test_log_loss']:.4f}")
    print(f"{'classe':8s} {'precision':>10s} {'rappel':>8s} {'F1':>8s}")
    for c in CLASSES:
        print(f"{c:8s} {m[f'test_precision_{c}']:10.3f} {m[f'test_recall_{c}']:8.3f} {m[f'test_f1_{c}']:8.3f}")
    print(f"\n(pour comparaison, argmax simple sur le test : F1 macro {m_simple['test_argmax_f1_macro']:.4f})")
    print(f"\nMatrice de confusion : {png}")
    print(f"Registre MLflow : '{NOM_MODELE}' version {version}, alias '{alias}'")
    print(f"Fichiers pour l'API : {modeles / 'model_final.joblib'} et {modeles / 'decision_poids.json'}")
    print(f"Pour passer en production plus tard :  python src/04_evaluation.py --promouvoir {version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeu", default="comportement", choices=["comportement", "complet"])
    ap.add_argument("--modeles-dir", type=Path, default=MODELES,
                    help="dossier des modeles lus et ecrits (defaut : models/ ; CT : un dossier candidat)")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS,
                    help="dossier des rapports ecrits (defaut : reports/)")
    ap.add_argument("--alias", default="staging", choices=["staging", "challenger"],
                    help="alias pose sur la nouvelle version du registre (defaut : staging ; CT : challenger)")
    ap.add_argument("--promouvoir", type=int, default=None, help="numero de version a passer en production")
    a = ap.parse_args()
    if a.promouvoir is not None:
        promouvoir(a.promouvoir)
    else:
        main(a.jeu, a.modeles_dir.resolve(), a.reports_dir.resolve(), a.alias)