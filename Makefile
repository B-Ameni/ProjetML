.PHONY: setup mlflow-server train train-regression experiments register serve test drift pipeline

setup:
	pip install -r requirements.txt

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000

train:
	python src/train.py

train-regression:
	python src/regression_experiments.py

experiments:
	python src/classification_experiments.py

register:
	python src/register_best_model.py

serve:
	mlflow models serve -m models:/mon_modele_production/Production --port 1234 --no-conda

test:
	python tests/test_api.py

drift:
	python src/simulate_drift.py

pipeline: train register
	@echo 'Pipeline: entraînement + enregistrement du modèle complété'
