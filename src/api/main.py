"""
ETAPE 5 - API de serving (FastAPI) de niveau production

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --app-dir src --reload
Documentation interactive (OpenAPI / Swagger) : http://127.0.0.1:8000/docs
Metriques d'observabilite Prometheus :          http://127.0.0.1:8000/metrics

Configuration par variables d'environnement (utilise par Docker) :
    MODEL_DIR       dossier contenant model_final.joblib et decision_poids.json   (defaut : <racine>/models)
    MODEL_VERSION   identifiant de version a afficher                              (defaut : hash du modele)
    PREDICTION_LOG  fichier journal des predictions, "" pour desactiver           (defaut : <racine>/logs/predictions.jsonl)
    EVAL_REPORT     rapport d'evaluation JSON de l'etape 4 (affiche par /model-info)

Le modele charge est un Pipeline sklearn complet (pretraitement + XGBoost).
Securite : joblib/pickle peut executer du code au chargement, on ne charge donc
que des fichiers certifies par notre propre pipeline de build.
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

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response
import joblib
import numpy as np
import pandas as pd

from .schemas import (
    PLAGES_ENTRAINEMENT,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClientFeatures,
    Health,
    ModelInfo,
    Prediction,
)

ROOT = Path(__file__).resolve().parents[2]
JEU_ATTENDU = "comportement"
VARIABLES = [
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
    "operator",
    "account_type",
]

log = logging.getLogger("api")
_verrou_journal = threading.Lock()

# --- Metriques d'observabilite Prometheus (Couche 2 du monitoring MLOps) ---
PREDICTIONS_COUNTER = Counter(
    "model_predictions_total",
    "Nombre total de predictions realisees",
    ["classe_predite", "version_modele"],
)
LATENCY_HISTOGRAM = Histogram(
    "model_prediction_duration_seconds",
    "Duree d'inference du modele en secondes",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
WARNINGS_COUNTER = Counter(
    "model_data_drift_warnings_total",
    "Nombre de valeurs hors plage d'entrainement detectees (signal precoce de drift)",
    ["champ"],
)

SIGNIFICATIONS_METIER = {
    "low": {
        "description": "Risque d'impaye faible (client solvable et regulier).",
        "recommandation": "Accorder le credit telecom ou l'avance de forfait sans restriction.",
    },
    "medium": {
        "description": "Risque d'impaye modere (profil necessitant une surveillance).",
        "recommandation": "Accorder un credit plafonne et envoyer des rappels de paiement preventifs.",
    },
    "high": {
        "description": "Risque d'impaye eleve (forte probabilite de defaut de paiement).",
        "recommandation": "Refuser le credit en postpaye ou exiger un rechargement prepaye integral.",
    },
}


def extraire_motifs(c: ClientFeatures) -> list[str]:
    """Extrait les facteurs cles d'explicabilite (XAI) pour la decision."""
    motifs = []
    if c.late_payment_count >= 3:
        motifs.append(f"Nombre eleve d'antecedents de retards ({c.late_payment_count} retards)")
    elif c.late_payment_count == 0:
        motifs.append("Aucun retard de paiement passe (historique propre)")

    if c.payment_consistency_score < 0.40:
        motifs.append(f"Faible regularite des reglements (score: {c.payment_consistency_score:.2f}/1.0)")
    elif c.payment_consistency_score > 0.70:
        motifs.append(f"Bonne regularite des reglements (score: {c.payment_consistency_score:.2f}/1.0)")

    if c.tenure_months < 6:
        motifs.append(f"Client recent dans le reseau ({c.tenure_months} mois d'anciennete)")
    elif c.tenure_months >= 24:
        motifs.append(f"Client fidele ({c.tenure_months} mois d'anciennete)")

    if c.account_type == "postpaid" and c.late_payment_count >= 2:
        motifs.append("Compte postpaye avec retards repetes (risque d'accumulation de dette)")

    return motifs


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
        raise RuntimeError(
            f"Cette API attend le jeu '{JEU_ATTENDU}', mais le modele a ete entraine avec "
            f"'{cfg.get('jeu_variables')}'. Adapter schemas.py avant de deployer."
        )
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
    log.info("Modele charge : version %s, classes %s", app.state.version, app.state.classes)
    yield


app = FastAPI(
    title="API de prediction du risque d'impaye (telecom Nigeria)",
    description=(
        "Predit le niveau de risque d'impaye (`low`, `medium`, `high`) d'un client a partir de son "
        "comportement de consommation telecom. Modele XGBoost industrialise avec observabilite "
        "Prometheus et journalisation des flux."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def avertissements(c: ClientFeatures):
    out = []
    for champ, (mini, maxi) in PLAGES_ENTRAINEMENT.items():
        v = getattr(c, champ)
        if v < mini or v > maxi:
            out.append(f"{champ}={v} hors de la plage d'entrainement [{mini}, {maxi}] : prediction moins fiable")
            WARNINGS_COUNTER.labels(champ=champ).inc()
    return out


def ecrire_journal(app, c: ClientFeatures, classe, scores, latence_ms, avert):
    """Journalise chaque prediction au format JSONL (avec hash SHA-256 de l'ID client)."""
    if not app.state.journal:
        return
    ligne = {
        "horodatage": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "client_hash": hashlib.sha256(c.customer_id.encode()).hexdigest()[:16] if c.customer_id else None,
        "entrees": c.model_dump(include=set(VARIABLES)),
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
    except OSError:
        log.exception("Ecriture du journal impossible")


@app.get("/health", response_model=Health, summary="Etat de sante du service (Healthcheck)")
def health(request: Request):
    return Health(status="ok", modele_charge=True, version_modele=request.app.state.version)


@app.get("/model-info", response_model=ModelInfo, summary="Metadonnees et gouvernance du modele actif")
def model_info(request: Request):
    s = request.app.state
    return ModelInfo(
        version_modele=s.version,
        jeu_variables=JEU_ATTENDU,
        variables_attendues=VARIABLES,
        classes=s.classes,
        poids_decision=s.poids.tolist(),
        charge_le=s.charge_le,
        metriques_test=s.metriques_test,
    )


@app.get("/metrics", summary="Endpoint d'observabilite Prometheus (Couche 2)")
def metrics():
    """Expose les compteurs et histogrammes au format text/plain de Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction, summary="Predire le risque d'impaye d'un client unique")
def predict(client: ClientFeatures, request: Request):
    s = request.app.state
    t0 = time.perf_counter()
    try:
        donnees = {k: getattr(client, k) for k in VARIABLES}
        X = pd.DataFrame([donnees])
        proba = s.pipeline.predict_proba(X)[0]
    except Exception:
        log.exception("Echec de la prediction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prediction")

    classe = s.classes[int((proba * s.poids).argmax())]
    scores = {c: round(float(p), 4) for c, p in zip(s.classes, proba)}
    latence_sec = time.perf_counter() - t0
    latence_ms = latence_sec * 1000

    PREDICTIONS_COUNTER.labels(classe_predite=classe, version_modele=s.version).inc()
    LATENCY_HISTOGRAM.observe(latence_sec)

    avert = avertissements(client)
    ecrire_journal(request.app, client, classe, scores, latence_ms, avert)

    info_metier = SIGNIFICATIONS_METIER.get(classe, {
        "description": f"Risque evalue a {classe}",
        "recommandation": "Se conformer aux regles de gestion du risque de l'operateur.",
    })
    motifs = extraire_motifs(client)

    return Prediction(
        classe_predite=classe,
        description_risque=info_metier["description"],
        decision_recommandee=info_metier["recommandation"],
        motifs_principaux=motifs,
        scores=scores,
        version_modele=s.version,
        avertissements=avert,
        latence_ms=round(latence_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Scoring de masse / lot de clients")
def predict_batch(batch: BatchPredictionRequest, request: Request):
    s = request.app.state
    if not batch.clients:
        return BatchPredictionResponse(predictions=[], total_clients=0, latence_totale_ms=0.0)

    t0 = time.perf_counter()
    try:
        rows = [{k: getattr(c, k) for k in VARIABLES} for c in batch.clients]
        X = pd.DataFrame(rows)
        probas = s.pipeline.predict_proba(X)
    except Exception:
        log.exception("Echec du scoring par lot")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prediction par lot")

    classes_idx = (probas * s.poids).argmax(axis=1)
    predictions = []

    for i, c in enumerate(batch.clients):
        classe = s.classes[int(classes_idx[i])]
        scores = {cl: round(float(p), 4) for cl, p in zip(s.classes, probas[i])}
        avert = avertissements(c)
        PREDICTIONS_COUNTER.labels(classe_predite=classe, version_modele=s.version).inc()
        info_metier = SIGNIFICATIONS_METIER.get(classe, {
            "description": f"Risque evalue a {classe}",
            "recommandation": "Se conformer aux regles de gestion du risque de l'operateur.",
        })
        motifs = extraire_motifs(c)
        pred = Prediction(
            classe_predite=classe,
            description_risque=info_metier["description"],
            decision_recommandee=info_metier["recommandation"],
            motifs_principaux=motifs,
            scores=scores,
            version_modele=s.version,
            avertissements=avert,
            latence_ms=0.0,
        )
        predictions.append(pred)
        ecrire_journal(request.app, c, classe, scores, 0.0, avert)

    latence_totale_ms = (time.perf_counter() - t0) * 1000
    LATENCY_HISTOGRAM.observe(latence_totale_ms / 1000.0)

    return BatchPredictionResponse(
        predictions=predictions,
        total_clients=len(predictions),
        latence_totale_ms=round(latence_totale_ms, 3),
    )
