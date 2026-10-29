"""
Tests de l'API. Ils construisent un PETIT modele temporaire (donnees aleatoires) : ils n'ont donc besoin
ni du vrai modele ni des donnees DVC, et pourront tourner tels quels dans la CI (etape 7).

    pytest -q
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.api.main import app
from src.features import CLASSES, build_preprocessor

CLIENT_OK = {
    "customer_id": "CUST1234567", "tenure_months": 36, "avg_monthly_spend_ngn": 18500.0,
    "recharge_frequency_per_month": 12, "payment_consistency_score": 0.35, "late_payment_count": 4,
    "operator": "MTN", "account_type": "prepaid",
}


def fabriquer_modele(dossier, poids=(1, 1, 1), jeu="comportement"):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 121, n), "avg_monthly_spend_ngn": rng.uniform(500, 50000, n),
        "recharge_frequency_per_month": rng.integers(1, 31, n), "payment_consistency_score": rng.uniform(0, 1, n),
        "late_payment_count": rng.integers(0, 11, n), "operator": rng.choice(["MTN", "Airtel", "Glo", "9mobile"], n),
        "account_type": rng.choice(["prepaid", "postpaid"], n),
    })
    y = rng.integers(0, 3, n)
    modele = XGBClassifier(n_estimators=5, max_depth=2, objective="multi:softprob", random_state=0)
    pipe = Pipeline([("prep", build_preprocessor("comportement")), ("model", modele)]).fit(df, y)
    joblib.dump(pipe, dossier / "model_final.joblib")
    (dossier / "decision_poids.json").write_text(json.dumps(
        {"classes": CLASSES, "poids": list(poids), "jeu_variables": jeu}), encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    fabriquer_modele(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("EVAL_REPORT", str(tmp_path / "absent.json"))
    with TestClient(app) as c:
        c.journal = tmp_path / "predictions.jsonl"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["modele_charge"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    d = r.json()
    assert d["jeu_variables"] == "comportement" and d["classes"] == CLASSES and len(d["variables_attendues"]) == 7


def test_predict_valide(client):
    r = client.post("/predict", json=CLIENT_OK)
    assert r.status_code == 200
    d = r.json()
    assert d["classe_predite"] in CLASSES
    assert set(d["scores"]) == set(CLASSES) and abs(sum(d["scores"].values()) - 1) < 1e-2
    assert d["avertissements"] == [] and d["version_modele"] and d["latence_ms"] >= 0


def test_operateur_et_compte_insensibles_a_la_casse(client):
    r = client.post("/predict", json={**CLIENT_OK, "operator": "mtn", "account_type": "PostPaid"})
    assert r.status_code == 200


@pytest.mark.parametrize("champ,valeur", [
    ("payment_consistency_score", 1.5), ("payment_consistency_score", -0.1), ("late_payment_count", -1),
    ("avg_monthly_spend_ngn", -5), ("operator", "Vodafone"), ("account_type", "hybride"),
    ("tenure_months", "beaucoup"),
])
def test_valeurs_impossibles_refusees(client, champ, valeur):
    assert client.post("/predict", json={**CLIENT_OK, champ: valeur}).status_code == 422


def test_champ_manquant_refuse(client):
    mauvais = {k: v for k, v in CLIENT_OK.items() if k != "operator"}
    assert client.post("/predict", json=mauvais).status_code == 422


def test_hors_plage_entrainement_accepte_avec_avertissement(client):
    r = client.post("/predict", json={**CLIENT_OK, "avg_monthly_spend_ngn": 80000})
    assert r.status_code == 200
    assert any("avg_monthly_spend_ngn" in a for a in r.json()["avertissements"])


def test_journal_ecrit_et_id_client_hashe(client):
    client.post("/predict", json=CLIENT_OK)
    lignes = client.journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    assert "CUST1234567" not in lignes[0]                       # jamais l'ID en clair
    d = json.loads(lignes[0])
    assert d["client_hash"] and d["entrees"]["operator"] == "MTN" and d["classe_predite"] in CLASSES


def test_poids_de_decision_appliques(tmp_path, monkeypatch):
    """Avec un poids enorme sur 'medium', la classe retenue doit etre 'medium' : les poids sont bien appliques."""
    fabriquer_modele(tmp_path, poids=(1, 1e6, 1))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("PREDICTION_LOG", "")
    with TestClient(app) as c:
        assert c.post("/predict", json=CLIENT_OK).json()["classe_predite"] == "medium"


def test_refus_de_demarrer_si_mauvais_jeu_de_variables(tmp_path, monkeypatch):
    fabriquer_modele(tmp_path, jeu="complet")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass
