"""
Configuration MLflow avec Model Registry activé
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_PATH = PROJECT_ROOT / 'mlruns'
MLFLOW_DB_PATH = PROJECT_ROOT / '.mlflow' / 'mlflow.db'

# Créer les répertoires s'ils n'existent pas
MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MLRUNS_PATH.mkdir(parents=True, exist_ok=True)

# URIs pour le serveur MLflow
# Format SQLAlchemy pour la base de données backend (nécessaire pour Model Registry)
BACKEND_STORE_URI = f"sqlite:///{MLFLOW_DB_PATH}".replace("\\", "/")

# Default artifact root
ARTIFACT_ROOT = str(MLRUNS_PATH).replace("\\", "/")

# URI de tracking pour utiliser le serveur local
TRACKING_URI = "http://localhost:5000"

print(f" Backend Store URI: {BACKEND_STORE_URI}")
print(f" Artifact Root: {ARTIFACT_ROOT}")
print(f" Tracking URI: {TRACKING_URI}")
