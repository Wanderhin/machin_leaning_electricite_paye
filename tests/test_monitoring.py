"""
Tests unitaires des calculs statistiques de monitoring et de derive (etape 8).
Verifie le bon calcul du PSI, la detection de derive et l'attribution des alertes.

    pytest tests/test_monitoring.py -v
"""
import importlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor

# Import dynamique car le nom du script commence par un chiffre (06_monitoring)
monitoring_mod = importlib.import_module("src.06_monitoring")
calculer_psi_numerique = monitoring_mod.calculer_psi_numerique
calculer_psi_categoriel = monitoring_mod.calculer_psi_categoriel
injecter_data_drift = monitoring_mod.injecter_data_drift
injecter_concept_drift = monitoring_mod.injecter_concept_drift
evaluer_scenario = monitoring_mod.evaluer_scenario


def test_psi_numerique_distribution_identique():
    rng = np.random.default_rng(42)
    ref = rng.normal(100, 15, 2000)
    cur = rng.normal(100, 15, 2000)
    psi = calculer_psi_numerique(ref, cur)
    assert psi < 0.10, f"Le PSI d'une distribution identique devrait etre < 0.10, obtenu: {psi}"


def test_psi_numerique_distribution_derivee():
    rng = np.random.default_rng(42)
    ref = rng.normal(100, 15, 2000)
    cur = rng.normal(160, 25, 2000)
    psi = calculer_psi_numerique(ref, cur)
    assert psi >= 0.20, f"Le PSI d'une distribution tres derivee devrait etre >= 0.20, obtenu: {psi}"


def test_psi_categoriel_distribution():
    ref = pd.Series(["MTN"] * 500 + ["Airtel"] * 300 + ["Glo"] * 200)
    cur = pd.Series(["MTN"] * 500 + ["Airtel"] * 300 + ["Glo"] * 200)
    psi = calculer_psi_categoriel(ref, cur)
    assert psi < 0.05

    cur_derive = pd.Series(["MTN"] * 100 + ["Airtel"] * 100 + ["Glo"] * 800)
    psi_derive = calculer_psi_categoriel(ref, cur_derive)
    assert psi_derive > 0.20


def test_injection_data_drift():
    df = pd.DataFrame({
        "avg_monthly_spend_ngn": [1000.0, 2000.0],
        "recharge_frequency_per_month": [10, 20],
        "late_payment_count": [0, 1],
    })
    drifted = injecter_data_drift(df)
    assert drifted["avg_monthly_spend_ngn"].iloc[0] == 1600.0
    assert drifted["late_payment_count"].iloc[0] == 1


def test_injection_concept_drift():
    df = pd.DataFrame({
        "default_risk": ["medium"] * 50 + ["low"] * 50,
        "late_payment_count": [3] * 100,
    })
    cd = injecter_concept_drift(df)
    assert cd["default_risk"].value_counts().to_dict() != df["default_risk"].value_counts().to_dict()


def test_evaluer_scenario_nominal():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "tenure_months": rng.integers(1, 120, n),
        "avg_monthly_spend_ngn": rng.uniform(1000, 40000, n),
        "recharge_frequency_per_month": rng.integers(1, 30, n),
        "payment_consistency_score": rng.uniform(0, 1, n),
        "late_payment_count": rng.integers(0, 5, n),
        "operator": rng.choice(["MTN", "Airtel", "Glo", "9mobile"], n),
        "account_type": rng.choice(["prepaid", "postpaid"], n),
        "default_risk": rng.choice(["low", "medium", "high"], n),
    })
    y = df["default_risk"].map({"low": 0, "medium": 1, "high": 2})
    X = df.drop(columns=["default_risk", "customer_id"])
    pipe = Pipeline([
        ("prep", build_preprocessor("comportement")),
        ("model", XGBClassifier(n_estimators=3, max_depth=2, random_state=0)),
    ]).fit(X, y)

    res = evaluer_scenario("Test Nominal", "Sans derive", df, df, pipe, latence_simulee_ms=45.0)
    assert res["scenario"] == "Test Nominal"
    assert "psi_par_feature" in res
    assert all(psi < 0.10 for psi in res["psi_par_feature"].values())
