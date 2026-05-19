from pathlib import Path
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow_config import TRACKING_URI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_PATH = PROJECT_ROOT / 'mlruns'


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name('Movie_Success_Classification')
    if experiment is None:
        raise RuntimeError('Experiment Movie_Success_Classification introuvable')

    runs = client.search_runs([experiment.experiment_id], order_by=['metrics.accuracy DESC'], max_results=1)
    if not runs:
        raise RuntimeError('Aucun run de modèle trouvé dans MLflow')

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f'runs:/{run_id}/model'
    model_name = 'mon_modele_production'

    print(f'Enregistrement du modèle depuis {model_uri} dans le registry MLflow sous {model_name}')
    mv = mlflow.register_model(model_uri, model_name)

    # Attendre la création de la version pour la transition
    time.sleep(3)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='Production',
        archive_existing_versions=True
    )
    print(f'Modèle enregistré et promu en Production : {model_name} v{mv.version}')


if __name__ == '__main__':
    main()
