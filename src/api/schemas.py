"""
Schemas Pydantic de l'API : ce que l'API accepte (entree) et ce qu'elle renvoie (sortie).

Deux niveaux de controle sur les entrees :
  - ERREUR 422 (valeur impossible) : negatif, score de coherence hors de [0, 1], operateur inconnu...
  - AVERTISSEMENT (valeur possible mais jamais vue a l'entrainement) : on repond quand meme, mais on
    previent. C'est un signal precoce de derive des donnees (utile pour le monitoring, etape 8).
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

OPERATEURS = {"mtn": "MTN", "airtel": "Airtel", "glo": "Glo", "9mobile": "9mobile"}

# Plages observees a l'entrainement (voir reports/01_audit.txt)
PLAGES_ENTRAINEMENT = {
    "tenure_months": (1, 120),
    "avg_monthly_spend_ngn": (500, 50000),
    "recharge_frequency_per_month": (1, 30),
    "late_payment_count": (0, 10),
}


class ClientFeatures(BaseModel):
    """Caracteristiques de comportement d'UN client (jeu de variables « comportement »)."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{
            "customer_id": "CUST1234567",
            "tenure_months": 36,
            "avg_monthly_spend_ngn": 18500.0,
            "recharge_frequency_per_month": 12,
            "payment_consistency_score": 0.35,
            "late_payment_count": 4,
            "operator": "MTN",
            "account_type": "prepaid",
        }]}
    )

    customer_id: Optional[str] = Field(
        None, max_length=64,
        description="Identifiant client (optionnel). Il n'est JAMAIS journalise en clair : seul un hash l'est.")
    tenure_months: int = Field(..., ge=0, le=1200, description="Anciennete en mois (entrainement : 1 a 120)")
    avg_monthly_spend_ngn: float = Field(..., ge=0, allow_inf_nan=False,
                                         description="Depense mensuelle moyenne en NGN (entrainement : 500 a 50 000)")
    recharge_frequency_per_month: int = Field(..., ge=0, le=1000,
                                              description="Nombre de recharges par mois (entrainement : 1 a 30)")
    payment_consistency_score: float = Field(..., ge=0, le=1, allow_inf_nan=False,
                                             description="Regularite des paiements, entre 0 et 1")
    late_payment_count: int = Field(..., ge=0, le=1000, description="Nombre de retards de paiement (entrainement : 0 a 10)")
    operator: Literal["MTN", "Airtel", "Glo", "9mobile"] = Field(..., description="Operateur (casse ignoree)")
    account_type: Literal["prepaid", "postpaid"] = Field(..., description="Type de compte (casse ignoree)")

    @field_validator("operator", mode="before")
    @classmethod
    def _operateur(cls, v):
        return OPERATEURS.get(v.strip().lower(), v) if isinstance(v, str) else v

    @field_validator("account_type", mode="before")
    @classmethod
    def _compte(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class Prediction(BaseModel):
    classe_predite: Literal["low", "medium", "high"] = Field(
        ..., description="Niveau de risque d'impaye retenu apres application des poids de decision")
    scores: dict[str, float] = Field(
        ..., description="Scores du modele par classe (somme = 1). ATTENTION : ce ne sont PAS des probabilites "
                         "calibrees (le modele est entraine avec des poids de classes) : a utiliser pour classer, "
                         "pas comme taux d'impaye reel.")
    version_modele: str = Field(..., description="Identifiant du modele (hash du fichier = hash DVC)")
    avertissements: list[str] = Field(default_factory=list, description="Valeurs hors de la plage d'entrainement")
    latence_ms: float = Field(..., description="Temps de calcul de la prediction")


class ModelInfo(BaseModel):
    version_modele: str
    jeu_variables: str
    variables_attendues: list[str]
    classes: list[str]
    poids_decision: list[float]
    charge_le: str
    metriques_test: Optional[dict[str, float]] = None


class Health(BaseModel):
    status: Literal["ok"]
    modele_charge: bool
    version_modele: str
