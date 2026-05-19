import argparse
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow_config import TRACKING_URI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'movies_credits_merged.csv'
MLRUNS_PATH = PROJECT_ROOT / 'mlruns'
FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    X = df[FEATURES].fillna(0)
    y = df['is_success']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }


def main(retrain: bool = False):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('Movie_Success_Classification')

    X_train, X_test, y_train, y_test = load_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    run_name = 'production_retrain' if retrain else 'production_train'

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({'model_type': 'RandomForest', 'n_estimators': 100})
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model')
        print('Metrics:', metrics)
        print(f"Modèle entraîné et loggé dans MLflow avec le nom de run '{run_name}'")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a production model and log it to MLflow')
    parser.add_argument('--retrain', action='store_true', help='Forcer un réentraînement du modèle')
    args = parser.parse_args()
    main(retrain=args.retrain)
