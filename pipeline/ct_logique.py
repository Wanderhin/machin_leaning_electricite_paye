"""
ETAPE 7 (iv) - Logique PURE du flow CT (reentrainement continu).

Ce module n'utilise PAS Prefect : il contient les decisions (donnees valides ? candidat bien isole ? quel verdict ?),
ce qui le rend testable partout (pytest, CI) sans installer d'orchestrateur. Le flow Prefect (flow_ct.py) ne fait
qu'enchainer ces fonctions et lancer les scripts 03, 04 et 05.
"""
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import (CLASSES, COMPORTEMENT_CAT, COMPORTEMENT_NUM, ID_COL, SCORING_NUM,  # noqa: E402
                      SCORING_ORD, TARGET)

RAISONS = ("manuel", "planning", "derive_donnees", "degradation_performance")
ALIAS_CT = "challenger"
EPS_EQUIVALENT = 0.0005       # meme seuil que la note « equivalent » de 05_porte_qualite.py
TOLERANCE_PTS = 5.0           # ecart max (en points de %) entre la repartition des classes et celle du decoupage enregistre
MIN_LIGNES = {"train": 10_000, "val": 2_000, "test": 2_000}      # garde-fou contre un fichier tronque
COLONNES_REQUISES = [ID_COL, TARGET] + COMPORTEMENT_NUM + COMPORTEMENT_CAT + SCORING_NUM + SCORING_ORD


# ------------------------------------------------------------------ 1) validation des donnees
def _controle(nom, ok, detail):
    return {"nom": nom, "ok": bool(ok), "detail": detail}


def reference_depuis_rapport(chemin):
    """Lit reports/02_decoupage.json : repartition (%) des classes de chaque morceau au moment du decoupage."""
    try:
        brut = json.loads(Path(chemin).read_text(encoding="utf-8"))["decoupage"]
        return {nom: {c: float(v["cible_pct"][c]) for c in CLASSES} for nom, v in brut.items()}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Rapport de decoupage illisible ({chemin}) : {type(e).__name__}: {e}") from e


def valider_donnees(splits, reference_pct, tolerance_pts=TOLERANCE_PTS, min_lignes=None):
    """
    splits : {"train": DataFrame, "val": DataFrame, "test": DataFrame}
    reference_pct : {"train": {"low": .., "medium": .., "high": ..}, ...}  (voir reference_depuis_rapport)
    Retourne la liste des controles ; le CT ne s'entraine que si TOUS sont OK.
    """
    min_lignes = MIN_LIGNES if min_lignes is None else min_lignes
    controles, completes = [], []
    for nom, df in splits.items():
        manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
        controles.append(_controle(f"colonnes_{nom}", not manquantes,
                                   "toutes presentes" if not manquantes else f"absentes : {manquantes}"))
        if manquantes:
            continue                       # les controles suivants supposent ces colonnes
        completes.append(nom)
        mini = min_lignes.get(nom, 0)
        controles.append(_controle(f"effectif_{nom}", len(df) >= mini, f"{len(df):,} lignes (minimum {mini:,})"))
        valide = bool(df[TARGET].isin(CLASSES).all())
        controles.append(_controle(f"cible_valide_{nom}", valide,
                                   "valeurs dans low/medium/high" if valide else "valeurs manquantes ou inattendues"))
        if not valide:
            continue
        ref = reference_pct.get(nom)
        if ref is None:
            controles.append(_controle(f"proportions_{nom}", False, "aucune reference dans le rapport de decoupage"))
            continue
        ecarts = {c: abs(float((df[TARGET] == c).mean() * 100) - ref[c]) for c in CLASSES}
        pire = max(ecarts, key=ecarts.get)
        controles.append(_controle(f"proportions_{nom}", ecarts[pire] <= tolerance_pts,
                                   f"ecart max {ecarts[pire]:.2f} point(s) sur '{pire}' (tolerance {tolerance_pts})"))
    for i, a in enumerate(completes):
        for b in completes[i + 1:]:
            commun = int(splits[a][ID_COL].isin(splits[b][ID_COL]).sum())
            detail = "aucun client commun" if commun == 0 else f"{commun} client(s) dans les deux morceaux (fuite)"
            controles.append(_controle(f"clients_disjoints_{a}_{b}", commun == 0, detail))
    return controles


def tout_ok(controles):
    return all(c["ok"] for c in controles)


# ------------------------------------------------------------------ 2) candidat isole
def verifier_raison(raison):
    if raison not in RAISONS:
        raise ValueError(f"Raison de declenchement inconnue : {raison!r} (attendu : {', '.join(RAISONS)})")


def nom_candidat(raison, maintenant=None):
    verifier_raison(raison)
    return f"{(maintenant or datetime.now()):%Y%m%d_%H%M%S}_{raison}"


def dossier_candidat(racine, nom):
    return Path(racine) / "models" / "candidats" / nom


def verifier_isolation(dossier, racine):
    """Garde-fou : le CT n'ecrit QUE dans models/candidats/<nom>/, jamais dans models/ ni reports/ (modele valide)."""
    base = (Path(racine) / "models" / "candidats").resolve()
    if base not in Path(dossier).resolve().parents:
        raise ValueError(f"Isolation violee : {dossier} n'est pas un sous-dossier de {base}. "
                         "Le modele valide (models/, reports/) ne doit jamais etre ecrase par le CT.")


def commandes(racine, dossier, python, f1_min=None, rappel_high_min=None, tolerance=None):
    """Commandes des scripts 03, 04, 05. Les seuils de la porte ne sont transmis que s'ils sont donnes
    (sinon : defauts de 05)."""
    racine, dossier = Path(racine), Path(dossier)
    reports, src = dossier / "reports", racine / "src"
    porte = [python, str(src / "05_porte_qualite.py"),
             "--candidat", str(reports / "04_evaluation_test.json"),
             "--reference", str(racine / "reports" / "04_evaluation_test.json")]
    for option, valeur in (("--f1-min", f1_min), ("--rappel-high-min", rappel_high_min), ("--tolerance", tolerance)):
        if valeur is not None:
            porte += [option, str(valeur)]
    isolation = ["--modeles-dir", str(dossier), "--reports-dir", str(reports)]
    return {
        "03_entrainement": [python, str(src / "03_entrainement.py"), *isolation],
        "04_evaluation": [python, str(src / "04_evaluation.py"), *isolation, "--alias", ALIAS_CT],
        "05_porte_qualite": porte,
    }


def extraire_version_registre(sortie):
    """Numero de version MLflow lu dans la sortie de 04 (« version 2, alias 'challenger' »), ou None."""
    m = re.search(r"version (\d+), alias '([a-z]+)'", sortie or "")
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ 3) verdict
def verdict_ct(porte, code_retour):
    """
    porte : contenu de 05_porte_qualite.json ; code_retour : code de 05 (0 accepte, 1 refuse).
    Retourne (verdict, message). Le CT ne promeut JAMAIS : meme ACCEPTE exige une decision humaine.
    """
    if code_retour == 1:
        return "REFUSE", "Candidat refuse par la porte de qualite : ne pas le promouvoir."
    if code_retour != 0:
        raise ValueError(f"Code retour inattendu de la porte de qualite : {code_retour}")
    ecart = porte.get("ecart_f1_vs_reference")
    if ecart is not None and abs(ecart) < EPS_EQUIVALENT:
        return "ACCEPTE_EQUIVALENT", "Candidat equivalent a la production : rien a promouvoir."
    if ecart is None:
        return "ACCEPTE_AVEC_ECART", "Candidat accepte, mais sans reference de comparaison : examen humain requis."
    if ecart < 0:
        return "ACCEPTE_AVEC_ECART", (f"Candidat accepte mais legerement moins bon que la production "
                                      f"({ecart:+.4f} de F1 macro, dans la tolerance) : promotion deconseillee.")
    return "ACCEPTE_AVEC_ECART", (f"Candidat accepte et meilleur que la production ({ecart:+.4f} de F1 macro) : "
                                  "examen humain avant toute promotion.")


# ------------------------------------------------------------------ 4) tracabilite
def md5_fichier(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def ecrire_resume(dossier, resume):
    chemin = Path(dossier) / "ct_resume.json"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(resume, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return chemin
