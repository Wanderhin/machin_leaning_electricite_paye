"""
ETAPE 7 (ii) - Porte de qualite : un candidat est-il assez bon pour etre SOUMIS a une validation humaine ?

Lit      : le rapport d'evaluation d'un candidat (reports/04_evaluation_test.json produit par 04_evaluation.py)
           et, en option, celui du modele en production (la "reference").
Ecrit    : 05_porte_qualite.json (verdict + detail de chaque controle), a cote du rapport du candidat
Code retour : 0 = ACCEPTE, 1 = REFUSE, 2 = fichier illisible ou metriques absentes

Les trois controles (les seuils sont a justifier dans le rapport) :
  1. F1 macro du candidat        >= --f1-min            (defaut 0.45 ; plancher "toujours low" = 0.236)
  2. Rappel de 'high' du candidat >= --rappel-high-min  (defaut 0.55)  : part des mauvais payeurs reperes
  3. (si --reference) F1 macro   >= F1 macro de la reference - --tolerance (defaut 0.01) : pas de regression

IMPORTANT : reussir la porte ne deploie RIEN. Elle autorise seulement la suite (canari, validation humaine,
promotion manuelle : python src/04_evaluation.py --promouvoir N).

Exemples (depuis la racine du projet) :
    python src/05_porte_qualite.py --candidat models/candidats/essai1/reports/04_evaluation_test.json ^
           --reference reports/04_evaluation_test.json
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

CLE_F1 = "test_f1_macro"
CLE_RAPPEL_HIGH = "test_recall_high"


def erreur(message):
    """Erreur d'entree (pas un refus du candidat) : message sur stderr et code retour 2."""
    print(message, file=sys.stderr)
    sys.exit(2)


def lire_metriques(chemin):
    """Retourne (F1 macro, rappel de high) lus dans un rapport de 04_evaluation.py, ou quitte avec le code 2."""
    chemin = Path(chemin)
    try:
        brut = json.loads(chemin.read_text(encoding="utf-8"))
        m = brut["metriques_test"]
        f1, rappel = float(m[CLE_F1]), float(m[CLE_RAPPEL_HIGH])
    except (OSError, ValueError, KeyError, TypeError) as e:
        erreur(f"Rapport illisible ou metriques absentes dans {chemin} : {type(e).__name__}: {e}")
    if math.isnan(f1) or math.isnan(rappel):
        erreur(f"Metriques non numeriques (NaN) dans {chemin}")
    return f1, rappel


def evaluer(f1, rappel, f1_min, rappel_min, ref_f1=None, tolerance=0.01):
    """Applique les controles. Retourne la liste des controles (chacun avec son resultat 'ok')."""
    controles = [
        {"nom": "f1_macro_minimum", "valeur": f1, "regle": f">= {f1_min}", "ok": f1 >= f1_min},
        {"nom": "rappel_high_minimum", "valeur": rappel, "regle": f">= {rappel_min}", "ok": rappel >= rappel_min},
    ]
    if ref_f1 is not None:
        seuil = ref_f1 - tolerance
        controles.append({"nom": "pas_de_regression_vs_reference", "valeur": f1,
                          "regle": f">= {seuil:.4f} (reference {ref_f1:.4f} - tolerance {tolerance})",
                          "ok": f1 >= seuil})
    return controles


def main(a):
    f1, rappel = lire_metriques(a.candidat)
    ref_f1 = lire_metriques(a.reference)[0] if a.reference else None
    controles = evaluer(f1, rappel, a.f1_min, a.rappel_high_min, ref_f1, a.tolerance)
    accepte = all(c["ok"] for c in controles)

    notes = []
    if ref_f1 is not None and abs(f1 - ref_f1) < 0.0005:
        notes.append("Candidat equivalent a la reference (meme F1 macro a 3 decimales) : "
                     "la promotion n'apporterait aucun gain.")

    resultat = {
        "verdict": "ACCEPTE" if accepte else "REFUSE",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidat": str(a.candidat),
        "reference": str(a.reference) if a.reference else None,
        "metriques_candidat": {CLE_F1: f1, CLE_RAPPEL_HIGH: rappel},
        "f1_macro_reference": ref_f1,
        "ecart_f1_vs_reference": None if ref_f1 is None else f1 - ref_f1,
        "seuils": {"f1_min": a.f1_min, "rappel_high_min": a.rappel_high_min, "tolerance": a.tolerance},
        "controles": controles,
        "notes": notes,
        "avertissement": "Reussir la porte ne deploie rien : validation humaine puis promotion manuelle.",
    }

    sortie = Path(a.sortie) if a.sortie else Path(a.candidat).parent / "05_porte_qualite.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== PORTE DE QUALITE ===")
    for c in controles:
        print(f"[{'OK' if c['ok'] else 'ECHEC'}] {c['nom']:34s} valeur {c['valeur']:.4f}   regle {c['regle']}")
    for n in notes:
        print("NOTE :", n)
    print(f"\nVERDICT : {resultat['verdict']}   (detail : {sortie})")
    if accepte:
        print("Le candidat peut etre soumis a validation humaine. Rien n'est deploye.")
    return 0 if accepte else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etape 7 : porte de qualite d'un candidat")
    ap.add_argument("--candidat", required=True, type=Path, help="04_evaluation_test.json du candidat")
    ap.add_argument("--reference", type=Path, default=None, help="04_evaluation_test.json du modele en production")
    ap.add_argument("--f1-min", type=float, default=0.45)
    ap.add_argument("--rappel-high-min", type=float, default=0.55)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--sortie", type=Path, default=None, help="defaut : 05_porte_qualite.json a cote du candidat")
    sys.exit(main(ap.parse_args()))
