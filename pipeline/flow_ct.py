"""
ETAPE 7 (iv) - Flow Prefect : reentrainement continu (CT) sur les VRAIES donnees

Ce que fait le flow (dans cet ordre) :
  1. valide les donnees reelles (colonnes, cible, effectifs, proportions des classes, clients disjoints) ;
  2. entraine un CANDIDAT dans models/candidats/<horodatage>_<raison>/ (script 03) ;
  3. l'evalue sur le test et l'enregistre dans le registre MLflow avec l'alias "challenger" (script 04) ;
  4. applique la porte de qualite contre le modele en production (script 05) ;
  5. conclut par un verdict et ecrit ct_resume.json (raison, empreintes MD5, controles, verdict) : auditabilite.

Il ne PROMEUT JAMAIS rien : la mise en production reste une decision humaine (python src/04_evaluation.py --promouvoir N).
Il n'ecrit jamais dans models/ ni reports/ : le modele valide n'est pas touche (garde-fou dans ct_logique.py).
Donnees : les memes decoupages train/val/test que la production (le flow ne rejoue pas l'etape 02 : cela reecrirait des
sorties DVC). Avec les memes donnees, le candidat est identique a la production : verdict attendu ACCEPTE_EQUIVALENT.

Utilisation (depuis la racine du projet, environnement virtuel active) :
    python pipeline/flow_ct.py                                   # declenchement manuel
    python pipeline/flow_ct.py --raison derive_donnees           # declenche par une alerte de derive (etape 8)
    python pipeline/flow_ct.py --planifier "0 3 * * 1"           # planning : lundi 3 h (processus actif, Ctrl+C = arret)
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.ct_logique import (RAISONS, commandes, dossier_candidat, ecrire_resume,  # noqa: E402
                                 extraire_version_registre, md5_fichier, nom_candidat, reference_depuis_rapport,
                                 tout_ok, valider_donnees, verdict_ct, verifier_isolation, verifier_raison)


@task(name="valider_donnees")
def valider_donnees_task(racine: str) -> list:
    log = get_run_logger()
    donnees = Path(racine) / "data" / "processed"
    splits = {nom: pd.read_csv(donnees / f"{nom}.csv") for nom in ("train", "val", "test")}
    controles = valider_donnees(splits, reference_depuis_rapport(Path(racine) / "reports" / "02_decoupage.json"))
    for c in controles:
        (log.info if c["ok"] else log.error)("[%s] %s : %s", "OK" if c["ok"] else "ECHEC", c["nom"], c["detail"])
    return controles


@task(name="lancer_script")
def lancer_script(nom: str, commande: list, racine: str) -> dict:
    """Lance un script du projet. Le code retour est renvoye tel quel : c'est au flow de l'interpreter."""
    log = get_run_logger()
    log.info("Lancement de %s", nom)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(commande, cwd=racine, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sortie = (r.stdout or "") + (r.stderr or "")
    for ligne in sortie.strip().splitlines()[-12:]:
        log.info("  | %s", ligne)
    log.info("%s : code retour %s", nom, r.returncode)
    return {"script": nom, "code": r.returncode, "sortie": sortie}


def _echec(resume, dossier, verdict, message):
    resume["verdict"], resume["message"] = verdict, message
    ecrire_resume(dossier, resume)
    raise RuntimeError(message)


@flow(name="ct_impayes")
def flow_ct(raison: str = "manuel", f1_min: float | None = None, rappel_high_min: float | None = None,
            tolerance: float | None = None) -> dict:
    log = get_run_logger()
    verifier_raison(raison)
    racine = ROOT
    reference = racine / "reports" / "04_evaluation_test.json"
    for requis in (reference, racine / "reports" / "02_decoupage.json", racine / "models" / "model_final.joblib"):
        if not requis.exists():
            raise FileNotFoundError(f"Fichier indispensable absent : {requis} (modele valide ou rapport de decoupage)")

    nom = nom_candidat(raison)
    dossier = dossier_candidat(racine, nom)
    verifier_isolation(dossier, racine)
    dossier.mkdir(parents=True, exist_ok=False)
    resume = {
        "flow": "ct_impayes", "raison": raison, "candidat": nom, "dossier": str(dossier),
        "debut_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele_production_md5": md5_fichier(racine / "models" / "model_final.joblib"),
        "empreintes_donnees_md5": {n: md5_fichier(racine / "data" / "processed" / f"{n}.csv")
                                   for n in ("train", "val", "test")},
    }
    log.info("Candidat : %s (declenche par : %s)", nom, raison)

    # 1) donnees
    controles = valider_donnees_task(str(racine))
    resume["validation_donnees"] = controles
    if not tout_ok(controles):
        _echec(resume, dossier, "DONNEES_INVALIDES",
               "Validation des donnees echouee : aucun entrainement lance (voir les controles en echec).")

    # 2) entrainement puis 3) evaluation, dans le dossier isole
    cmds = commandes(racine, dossier, sys.executable, f1_min, rappel_high_min, tolerance)
    for script in ("03_entrainement", "04_evaluation"):
        res = lancer_script(script, cmds[script], str(racine))
        if res["code"] != 0:
            fin = "\n".join(res["sortie"].strip().splitlines()[-8:])
            _echec(resume, dossier, f"ECHEC_{script.upper()}",
                   f"{script} a echoue (code {res['code']}). Fin de la sortie :\n{fin}")
        if script == "04_evaluation":
            resume["version_registre_mlflow"] = extraire_version_registre(res["sortie"])

    # 4) porte de qualite : 0 = accepte, 1 = refuse (resultat metier, pas une erreur), autre = erreur d'entree
    res = lancer_script("05_porte_qualite", cmds["05_porte_qualite"], str(racine))
    if res["code"] not in (0, 1):
        _echec(resume, dossier, "ECHEC_05_PORTE_QUALITE",
               f"La porte de qualite a echoue (code {res['code']}) : {res['sortie'][-300:]}")
    porte = json.loads((dossier / "reports" / "05_porte_qualite.json").read_text(encoding="utf-8"))

    # 5) verdict
    verdict, message = verdict_ct(porte, res["code"])
    resume.update({"verdict": verdict, "message": message, "porte_de_qualite": porte,
                   "fin_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    chemin = ecrire_resume(dossier, resume)
    (log.warning if verdict == "REFUSE" else log.info)("VERDICT : %s - %s", verdict, message)
    log.info("Resume ecrit : %s", chemin)
    if verdict == "ACCEPTE_AVEC_ECART" and resume.get("version_registre_mlflow"):
        log.info("Pour promouvoir APRES validation humaine : python src/04_evaluation.py --promouvoir %s",
                 resume["version_registre_mlflow"])
    return resume


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etape 7 (iv) : flow Prefect de reentrainement continu (CT)")
    ap.add_argument("--raison", default="manuel", choices=RAISONS, help="ce qui declenche le reentrainement")
    ap.add_argument("--planifier", metavar="CRON", default=None,
                    help='planifie le flow (ex. "0 3 * * 1") ; le processus reste actif, Ctrl+C pour l\'arreter')
    ap.add_argument("--f1-min", type=float, default=None, help="seuil de F1 macro de la porte (defaut : celui de 05)")
    ap.add_argument("--rappel-high-min", type=float, default=None, help="seuil de rappel de 'high' (defaut : celui de 05)")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="tolerance de regression vs production (defaut : celle de 05)")
    a = ap.parse_args()
    if a.planifier:
        flow_ct.serve(name="ct-impayes-planifie", cron=a.planifier, parameters={"raison": "planning"})
    else:
        r = flow_ct(raison=a.raison, f1_min=a.f1_min, rappel_high_min=a.rappel_high_min, tolerance=a.tolerance)
        print(f"\nVERDICT : {r['verdict']}\n{r['message']}\nCandidat : {r['dossier']}")
        print(f"Resume   : {Path(r['dossier']) / 'ct_resume.json'}")
