"""
simuler_trafic.py - Envoie des requêtes en continu vers l'API FastAPI
afin de visualiser les courbes et les graphiques s'animer en direct sur Grafana.

Usage :
    python scripts/simuler_trafic.py
    python scripts/simuler_trafic.py --n-requetes 100 --delai 0.3
"""
import argparse
import random
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"

OPERATEURS = ["MTN", "Airtel", "Glo", "9mobile"]
TYPES_COMPTE = ["prepaid", "postpaid"]


def generer_client():
    """Génère un profil client aléatoire (nominal ou avec dérive)."""
    # 15% de chances de générer un profil atypique (valeurs élevées)
    choc = random.random() < 0.15

    tenure = random.randint(1, 100)
    depenses = random.uniform(25000, 75000) if choc else random.uniform(500, 25000)
    recharges = random.randint(1, 10) if choc else random.randint(3, 30)
    regularite = random.uniform(0.1, 0.45) if choc else random.uniform(0.4, 0.98)
    retards = random.randint(3, 8) if choc else random.randint(0, 2)

    return {
        "customer_id": f"SIM_{random.randint(100000, 999999)}",
        "tenure_months": tenure,
        "avg_monthly_spend_ngn": round(depenses, 2),
        "recharge_frequency_per_month": recharges,
        "payment_consistency_score": round(regularite, 2),
        "late_payment_count": retards,
        "operator": random.choice(OPERATEURS),
        "account_type": random.choice(TYPES_COMPTE),
    }


def main():
    parser = argparse.ArgumentParser(description="Simulateur de trafic pour Grafana")
    parser.add_argument("--n-requetes", type=int, default=50, help="Nombre de requêtes (defaut: 50)")
    parser.add_argument("--delai", type=float, default=0.5, help="Délai entre requêtes en secondes (defaut: 0.5s)")
    args = parser.parse_args()

    print("=" * 70)
    print("SIMULATION DE TRAFIC EN COURS VERS L'API FASTAPI")
    print(f"Cible : {API_URL}")
    print(f"Nombre de requêtes : {args.n_requetes} (délai : {args.delai}s)")
    print("Ouvrez Grafana sur http://localhost:3000 pour voir les graphiques s'animer !")
    print("=" * 70)

    succes = 0
    erreurs = 0

    for i in range(1, args.n_requetes + 1):
        payload = generer_client()
        try:
            t0 = time.time()
            resp = requests.post(API_URL, json=payload, timeout=5)
            duree_ms = (time.time() - t0) * 1000

            if resp.status_code == 200:
                res = resp.json()
                classe = res.get("classe_predite")
                avert = len(res.get("avertissements", []))
                warn_str = f" [! {avert} alerte(s)]" if avert > 0 else ""
                print(f"[{i:03d}/{args.n_requetes}] Risque: {classe:<6} | Latence: {duree_ms:5.1f} ms{warn_str}")
                succes += 1
            else:
                print(f"[{i:03d}/{args.n_requetes}] Erreur HTTP {resp.status_code}")
                erreurs += 1
        except Exception as e:
            print(f"[{i:03d}/{args.n_requetes}] Connexion impossible : {e}")
            print("Astuce : assurez-vous que l'API est bien lancée sur le port 8000 !")
            break

        time.sleep(args.delai)

    print("=" * 70)
    print(f"Fin de la simulation. Succès : {succes} | Erreurs : {erreurs}")
    print("Consultez le dashboard sur Grafana : http://localhost:3000")
    print("=" * 70)


if __name__ == "__main__":
    main()
