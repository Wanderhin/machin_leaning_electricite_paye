"""
features.py - Definition UNIQUE des colonnes et du pretraitement.

Ce module sera importe par l'entrainement (etape 3), l'API (etape 5) et le monitoring (etape 8).
Un seul endroit pour definir le pretraitement = pas de decalage entre l'entrainement et la
production (le fameux "training-serving skew").
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
TARGET = "default_risk"
ID_COL = "customer_id"
CLASSES = ["low", "medium", "high"]                    # ordre du risque : 0, 1, 2
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

# --- Jeu A : "comportement" = ce qu'on observe du client AVANT toute decision de credit
COMPORTEMENT_NUM = ["tenure_months", "avg_monthly_spend_ngn", "recharge_frequency_per_month",
                    "payment_consistency_score", "late_payment_count"]
COMPORTEMENT_CAT = ["operator", "account_type"]

# --- Jeu B : "complet" = jeu A + colonnes qui ressemblent a un resultat de scoring
SCORING_NUM = ["credit_score", "max_loan_amount_ngn", "loan_eligibility"]
SCORING_ORD = ["credit_rating"]
RATING_ORDER = ["poor", "fair", "good", "excellent"]   # ordre naturel -> encodage ordinal

FEATURE_SETS = ("comportement", "complet")


def colonnes(feature_set):
    """Retourne (colonnes numeriques, colonnes categorielles, colonnes ordinales)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set doit etre dans {FEATURE_SETS}")
    num, cat, ordi = list(COMPORTEMENT_NUM), list(COMPORTEMENT_CAT), []
    if feature_set == "complet":
        num += SCORING_NUM
        ordi = list(SCORING_ORD)
    return num, cat, ordi


def build_preprocessor(feature_set="comportement", scale=False):
    """
    Construit le pretraitement (imputation + encodage, + standardisation si scale=True).
    A utiliser DANS un Pipeline sklearn avec le modele, et a ajuster (fit) sur le TRAIN seulement.
    scale=False pour les arbres/forets/boosting ; scale=True pour la regression logistique.
    """
    num, cat, ordi = colonnes(feature_set)

    etapes_num = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        etapes_num.append(("scaler", StandardScaler()))

    blocs = [
        ("num", Pipeline(etapes_num), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ]
    if ordi:
        blocs.append(("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[RATING_ORDER],
                                       handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ordi))

    pre = ColumnTransformer(blocs, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")      # garde les noms de colonnes (utile pour l'explicabilite)
    return pre


def separer_X_y(df):
    """Separe un DataFrame en X (sans id ni cible) et y (cible codee 0/1/2)."""
    y = df[TARGET].map(CLASS_TO_INT)
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y