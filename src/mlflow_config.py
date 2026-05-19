"""
Configuration MLflow avec Model Registry activé
"""
from pathlib import Path
import os
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLRUNS_PATH = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / ".mlflow" / "mlflow.db"

# Création dossiers
MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MLRUNS_PATH.mkdir(parents=True, exist_ok=True)

# Backend DB (Model Registry)
BACKEND_STORE_URI = f"sqlite:///{MLFLOW_DB_PATH}".replace("\\", "/")

# Artifacts
ARTIFACT_ROOT = str(MLRUNS_PATH).replace("\\", "/")

# MLflow server
TRACKING_URI = "http://127.0.0.1:5000"

# 🔥 IMPORTANT : appliquer automatiquement
mlflow.set_tracking_uri(TRACKING_URI)

print(f" Backend Store URI: {BACKEND_STORE_URI}")
print(f" Artifact Root: {ARTIFACT_ROOT}")
print(f" Tracking URI: {TRACKING_URI}")