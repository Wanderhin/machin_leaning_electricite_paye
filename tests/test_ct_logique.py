"""
Tests de la logique du flow CT (pipeline/ct_logique.py). Ils n'ont besoin ni de Prefect, ni de XGBoost, ni des donnees
reelles : de petits tableaux construits ici suffisent. Ils pourront tourner tels quels dans la CI.

    pytest -q
"""
import json
from datetime import datetime

import pandas as pd
import pytest

from pipeline.ct_logique import (ALIAS_CT, COLONNES_REQUISES, ID_COL, TARGET, commandes, dossier_candidat,
                                 extraire_version_registre, nom_candidat, reference_depuis_rapport, tout_ok,
                                 valider_donnees, verdict_ct, verifier_isolation, verifier_raison)

REF = {nom: {"low": 50.0, "medium": 30.0, "high": 20.0} for nom in ("train", "val", "test")}
PETIT = {"train": 10, "val": 10, "test": 10}


def morceau(debut, n=10):
    """n lignes (50 % low, 30 % medium, 20 % high) avec toutes les colonnes requises et des identifiants uniques."""
    df = pd.DataFrame({c: 0 for c in COLONNES_REQUISES}, index=range(n))
    df[ID_COL] = [f"C{debut + i}" for i in range(n)]
    df[TARGET] = ["low"] * (n // 2) + ["medium"] * (n * 3 // 10) + ["high"] * (n - n // 2 - n * 3 // 10)
    return df


@pytest.fixture
def splits():
    return {"train": morceau(0), "val": morceau(100), "test": morceau(200)}


def echecs(controles):
    return {c["nom"] for c in controles if not c["ok"]}


# ---------------------------------------------------------------- validation des donnees
def test_donnees_valides(splits):
    controles = valider_donnees(splits, REF, min_lignes=PETIT)
    assert tout_ok(controles) and echecs(controles) == set()


def test_colonne_manquante_detectee(splits):
    splits["val"] = splits["val"].drop(columns=["tenure_months"])
    assert "colonnes_val" in echecs(valider_donnees(splits, REF, min_lignes=PETIT))


def test_effectif_trop_faible_detecte(splits):
    assert "effectif_train" in echecs(valider_donnees(splits, REF, min_lignes={"train": 1000, "val": 10, "test": 10}))


def test_cible_invalide_detectee(splits):
    splits["test"].loc[0, TARGET] = "inconnu"
    assert "cible_valide_test" in echecs(valider_donnees(splits, REF, min_lignes=PETIT))


def test_cible_manquante_detectee(splits):
    splits["train"].loc[0, TARGET] = None
    assert "cible_valide_train" in echecs(valider_donnees(splits, REF, min_lignes=PETIT))


def test_proportions_hors_tolerance_detectees(splits):
    splits["train"][TARGET] = "high"             # 100 % high au lieu de 20 %
    assert "proportions_train" in echecs(valider_donnees(splits, REF, min_lignes=PETIT))


def test_fuite_entre_morceaux_detectee(splits):
    splits["test"].loc[0, ID_COL] = splits["train"].loc[0, ID_COL]
    assert "clients_disjoints_train_test" in echecs(valider_donnees(splits, REF, min_lignes=PETIT))


def test_reference_absente_refusee(splits):
    ref = {k: v for k, v in REF.items() if k != "val"}
    assert "proportions_val" in echecs(valider_donnees(splits, ref, min_lignes=PETIT))


def test_reference_depuis_rapport(tmp_path):
    rapport = tmp_path / "02_decoupage.json"
    cible = {"low": 54.6, "medium": 23.1, "high": 22.3}
    rapport.write_text(json.dumps({"decoupage": {"train": {"lignes": 5, "cible_pct": cible}}}))
    assert reference_depuis_rapport(rapport)["train"]["medium"] == 23.1
    rapport.write_text("{}")
    with pytest.raises(ValueError):
        reference_depuis_rapport(rapport)


# ---------------------------------------------------------------- isolation du candidat
def test_isolation_acceptee_dans_candidats(tmp_path):
    verifier_isolation(dossier_candidat(tmp_path, "essai"), tmp_path)


@pytest.mark.parametrize("sous_chemin", ["models", "reports", "models/candidats", "models/autre", "data/processed", ".."])
def test_isolation_refuse_tout_ailleurs(tmp_path, sous_chemin):
    with pytest.raises(ValueError):
        verifier_isolation(tmp_path / sous_chemin, tmp_path)


def test_isolation_refuse_l_evasion_par_points_points(tmp_path):
    with pytest.raises(ValueError):
        verifier_isolation(tmp_path / "models" / "candidats" / ".." / ".." / "models", tmp_path)


def test_nom_candidat_horodate_et_raison_controlee():
    assert nom_candidat("derive_donnees", datetime(2026, 10, 30, 14, 5, 9)) == "20261030_140509_derive_donnees"
    with pytest.raises(ValueError):
        nom_candidat("n_importe_quoi")
    with pytest.raises(ValueError):
        verifier_raison("../../models")


# ---------------------------------------------------------------- commandes
def test_commandes_isolent_le_candidat_et_visent_challenger(tmp_path):
    dossier = dossier_candidat(tmp_path, "x")
    c = commandes(tmp_path, dossier, "python")
    for script in ("03_entrainement", "04_evaluation"):
        assert c[script][c[script].index("--modeles-dir") + 1] == str(dossier)
        assert c[script][c[script].index("--reports-dir") + 1] == str(dossier / "reports")
    assert c["04_evaluation"][c["04_evaluation"].index("--alias") + 1] == ALIAS_CT == "challenger"
    porte = c["05_porte_qualite"]
    assert porte[porte.index("--reference") + 1] == str(tmp_path / "reports" / "04_evaluation_test.json")
    assert porte[porte.index("--candidat") + 1] == str(dossier / "reports" / "04_evaluation_test.json")
    assert "--f1-min" not in porte                # seuils : defauts de 05_porte_qualite.py, pas de doublon ici


def test_commandes_transmettent_les_seuils_donnes(tmp_path):
    porte = commandes(tmp_path, dossier_candidat(tmp_path, "x"), "python", f1_min=0.5, tolerance=0.02)["05_porte_qualite"]
    assert porte[porte.index("--f1-min") + 1] == "0.5" and porte[porte.index("--tolerance") + 1] == "0.02"
    assert "--rappel-high-min" not in porte


def test_commandes_ne_touchent_jamais_aux_dossiers_du_modele_valide(tmp_path):
    cmds = commandes(tmp_path, dossier_candidat(tmp_path, "x"), "python")
    for script in ("03_entrainement", "04_evaluation"):
        valeurs = cmds[script]
        assert str(tmp_path / "models") not in valeurs and str(tmp_path / "reports") not in valeurs


# ---------------------------------------------------------------- verdict
def test_verdict_refuse():
    assert verdict_ct({}, 1)[0] == "REFUSE"


def test_verdict_equivalent_quand_aucun_ecart():
    assert verdict_ct({"ecart_f1_vs_reference": 0.0}, 0)[0] == "ACCEPTE_EQUIVALENT"
    assert verdict_ct({"ecart_f1_vs_reference": 0.0003}, 0)[0] == "ACCEPTE_EQUIVALENT"


def test_verdict_meilleur_avec_ecart():
    verdict, message = verdict_ct({"ecart_f1_vs_reference": 0.02}, 0)
    assert verdict == "ACCEPTE_AVEC_ECART" and "meilleur" in message


def test_verdict_legerement_moins_bon_deconseille():
    verdict, message = verdict_ct({"ecart_f1_vs_reference": -0.005}, 0)
    assert verdict == "ACCEPTE_AVEC_ECART" and "deconseillee" in message


def test_verdict_sans_reference():
    assert verdict_ct({"ecart_f1_vs_reference": None}, 0)[0] == "ACCEPTE_AVEC_ECART"


def test_verdict_refuse_un_code_inattendu():
    with pytest.raises(ValueError):
        verdict_ct({}, 2)


def test_extraire_version_registre():
    sortie = "Registre MLflow : 'default_risk_xgb' version 7, alias 'challenger'\nautre ligne"
    assert extraire_version_registre(sortie) == 7
    assert extraire_version_registre("rien") is None and extraire_version_registre(None) is None
