# Image de l'API de prediction des impayes (etape 6).
# Construire :  docker build -t default-risk-api:0.1 .
# Lancer     :  docker run --rm -p 8000:8000 default-risk-api:0.1
# Docs OpenAPI : http://127.0.0.1:8000/docs

# Meme version majeure de Python que le poste de developpement (3.12) :
# le .joblib depend des versions exactes de Python et des bibliotheques.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src

# libgomp1 : bibliotheque OpenMP requise par XGBoost sous Linux (absente de l'image slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependances d'abord : cette couche reste en cache tant que requirements-api.txt ne change pas.
COPY requirements-api.txt .
# Sous Linux, "xgboost" tire ~300 Mo de bibliotheques NVIDIA (GPU) inutiles ici.
# "xgboost-cpu" est la meme version, meme module Python, sans GPU : le .joblib se charge a l'identique.
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements-api.txt > /tmp/requirements-docker.txt \
    && pip install -r /tmp/requirements-docker.txt

# Utilisateur non-root (bonne pratique de securite).
RUN useradd --create-home --uid 1000 appuser

# Code de l'API et artefacts de modele (pas de donnees, pas de secrets : voir .dockerignore).
COPY src/ ./src/
COPY models/model_final.joblib models/decision_poids.json ./models/
COPY reports/04_evaluation_test.json ./reports/

RUN mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser

ENV MODEL_DIR=/app/models \
    PREDICTION_LOG=/app/logs/predictions.jsonl \
    EVAL_REPORT=/app/reports/04_evaluation_test.json

EXPOSE 8000

# Sonde de sante : l'image slim n'a pas curl, on utilise Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]