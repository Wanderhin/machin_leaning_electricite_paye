"""
ETAPE 5 - API de serving (FastAPI)

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --app-dir src --reload
Documentation interactive (OpenAPI / Swagger) : http://127.0.0.1:8000/docs

Configuration par variables d'environnement (utile pour Docker, etape 6) :
    MODEL_DIR       dossier contenant model_final.joblib et decision_poids.json   (defaut : <racine>/models)
    MODEL_VERSION   identifiant de version a afficher                              (defaut : hash du modele)
    PREDICTION_LOG  fichier journal des predictions, "" pour desactiver           (defaut : <racine>/logs/predictions.jsonl)
    EVAL_REPORT     rapport d'evaluation JSON de l'etape 4 (affiche par /model-info)

Le modele charge est un Pipeline sklearn complet (pretraitement + XGBoost) : l'API lui envoie des donnees
BRUTES, exactement comme a l'entrainement. Securite : joblib/pickle peut executer du code au chargement,
on ne charge donc que des fichiers produits par notre propre pipeline (jamais un fichier recu d'un tiers).
"""
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from .schemas import PLAGES_ENTRAINEMENT, ClientFeatures, Health, ModelInfo, Prediction

ROOT = Path(__file__).resolve().parents[2]
JEU_ATTENDU = "comportement"
VARIABLES = ["tenure_months", "avg_monthly_spend_ngn", "recharge_frequency_per_month",
             "payment_consistency_score", "late_payment_count", "operator", "account_type"]

log = logging.getLogger("api")
_verrou_journal = threading.Lock()


def md5_fichier(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele UNE fois au demarrage (pas a chaque requete)."""
    dossier = Path(os.getenv("MODEL_DIR", str(ROOT / "models")))
    chemin_modele, chemin_poids = dossier / "model_final.joblib", dossier / "decision_poids.json"
    cfg = json.loads(chemin_poids.read_text(encoding="utf-8"))
    if cfg.get("jeu_variables") != JEU_ATTENDU:
        raise RuntimeError(f"Cette API attend le jeu '{JEU_ATTENDU}', mais le modele a ete entraine avec "
                           f"'{cfg.get('jeu_variables')}'. Adapter schemas.py avant de deployer.")
    app.state.pipeline = joblib.load(chemin_modele)
    app.state.classes = list(cfg["classes"])
    app.state.poids = np.array(cfg["poids"], dtype=float)
    app.state.version = os.getenv("MODEL_VERSION") or md5_fichier(chemin_modele)[:12]
    app.state.charge_le = datetime.now(timezone.utc).isoformat(timespec="seconds")
    app.state.journal = os.getenv("PREDICTION_LOG", str(ROOT / "logs" / "predictions.jsonl"))
    rapport = Path(os.getenv("EVAL_REPORT", str(ROOT / "reports" / "04_evaluation_test.json")))
    app.state.metriques_test = None
    if rapport.exists():
        brut = json.loads(rapport.read_text(encoding="utf-8")).get("metriques_test", {})
        app.state.metriques_test = {k: float(v) for k, v in brut.items()}
    log.info("Modele charge : version %s, poids %s", app.state.version, app.state.poids.tolist())
    yield


app = FastAPI(
    title="API de prediction du risque d'impaye (telecom Nigeria)",
    description=("Predit le niveau de risque d'impaye (`low`, `medium`, `high`) d'un client a partir de son "
                 "comportement. Modele XGBoost versionne (DVC + MLflow). Les scores renvoyes ne sont pas des "
                 "probabilites calibrees."),
    version="1.0.0",
    lifespan=lifespan,
)


def avertissements(c: ClientFeatures):
    out = []
    for champ, (mini, maxi) in PLAGES_ENTRAINEMENT.items():
        v = getattr(c, champ)
        if v < mini or v > maxi:
            out.append(f"{champ}={v} hors de la plage d'entrainement [{mini}, {maxi}] : prediction moins fiable")
    return out


def ecrire_journal(app, c: ClientFeatures, classe, scores, latence_ms, avert):
    """Une ligne JSON par prediction : alimentera le monitoring (etape 8). L'ID client est hashe."""
    if not app.state.journal:
        return
    ligne = {
        "horodatage": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "client_hash": hashlib.sha256(c.customer_id.encode()).hexdigest()[:16] if c.customer_id else None,
        "entrees": c.model_dump(exclude={"customer_id"}),
        "classe_predite": classe,
        "scores": scores,
        "version_modele": app.state.version,
        "latence_ms": round(latence_ms, 3),
        "avertissements": len(avert),
    }
    try:
        chemin = Path(app.state.journal)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with _verrou_journal, open(chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    except OSError:                       # un probleme de journal ne doit jamais casser la prediction
        log.exception("Ecriture du journal impossible")


@app.get("/health", response_model=Health, summary="Etat du service")
def health(request: Request):
    return Health(status="ok", modele_charge=True, version_modele=request.app.state.version)


@app.get("/model-info", response_model=ModelInfo, summary="Informations sur le modele en service")
def model_info(request: Request):
    s = request.app.state
    return ModelInfo(version_modele=s.version, jeu_variables=JEU_ATTENDU, variables_attendues=VARIABLES,
                     classes=s.classes, poids_decision=s.poids.tolist(), charge_le=s.charge_le,
                     metriques_test=s.metriques_test)


@app.post("/predict", response_model=Prediction, summary="Predire le risque d'impaye d'un client")
def predict(client: ClientFeatures, request: Request):
    s = request.app.state
    t0 = time.perf_counter()
    try:
        X = pd.DataFrame([client.model_dump(exclude={"customer_id"})])     # noms de colonnes exacts
        proba = s.pipeline.predict_proba(X)[0]
    except Exception:
        log.exception("Echec de la prediction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prediction")
    classe = s.classes[int((proba * s.poids).argmax())]                    # meme regle qu'a l'etape 4
    scores = {c: round(float(p), 4) for c, p in zip(s.classes, proba)}
    latence = (time.perf_counter() - t0) * 1000
    avert = avertissements(client)
    ecrire_journal(request.app, client, classe, scores, latence, avert)
    return Prediction(classe_predite=classe, scores=scores, version_modele=s.version,
                      avertissements=avert, latence_ms=round(latence, 3))
