import subprocess
from mlflow_config import BACKEND_STORE_URI, ARTIFACT_ROOT

def start_mlflow_server():
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", BACKEND_STORE_URI,
        "--default-artifact-root", ARTIFACT_ROOT,
        "--host", "127.0.0.1",
        "--port", "5000"
    ]

    print("Démarrage du serveur MLflow avec Model Registry...")
    print(f"Backend: {BACKEND_STORE_URI}")
    print(f"Artifacts: {ARTIFACT_ROOT}")
    print("URL: http://127.0.0.1:5000\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✓ Serveur MLflow arrêté")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    start_mlflow_server()