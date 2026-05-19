import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'movies_credits_merged.csv'
MLRUNS_PATH = PROJECT_ROOT / 'mlruns'
FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
SEUIL_DRIFT = 0.30
SEUIL_WARN = 0.15


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    X = df[FEATURES].fillna(0)
    y = df['is_success']
    return X, y


def simulate_production_drift(X_test):
    X_prod = X_test.copy()
    num_cols = X_prod.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols[:2]:
        X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, size=len(X_prod))
    return X_prod


def log_drift_report(X_ref, X_curr):
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment('monitoring_drift')

    with mlflow.start_run(run_name='drift_check_v1'):
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_ref, current_data=X_curr)
        report.save_html('drift_report.html')
        mlflow.log_artifact('drift_report.html')

        score_report = Report(metrics=[DatasetDriftMetric()])
        score_report.run(reference_data=X_ref, current_data=X_curr)
        result = score_report.as_dict()

        drift_share = result['metrics'][0]['result']['drift_share']
        dataset_drift = result['metrics'][0]['result']['dataset_drift']
        n_drifted = result['metrics'][0]['result']['number_of_drifted_columns']
        n_total = result['metrics'][0]['result']['number_of_columns']

        mlflow.log_metric('drift_share', drift_share)
        mlflow.log_metric('drifted_columns', n_drifted)
        mlflow.log_metric('total_columns', n_total)
        mlflow.log_metric('dataset_drifted', int(dataset_drift))

        print(f'Drift share : {drift_share:.2%} | Colonnes driftées : {n_drifted}/{n_total}')

        return drift_share, n_drifted, n_total


def log_ks_results(X_ref, X_curr):
    results = []
    for col in X_ref.select_dtypes(include='number').columns:
        stat, pvalue = stats.ks_2samp(X_ref[col], X_curr[col])
        results.append({
            'feature': col,
            'ks_stat': round(stat, 4),
            'p_value': round(pvalue, 4),
            'drifted': pvalue < 0.05
        })
        mlflow.log_metric(f'ks_pvalue_{col}', float(pvalue))

    df_drift = pd.DataFrame(results)
    df_drift.to_csv('ks_drift_results.csv', index=False)
    mlflow.log_artifact('ks_drift_results.csv')
    print(df_drift.to_string(index=False))


def main():
    X, y = load_data()
    X_train = X.sample(frac=0.8, random_state=42)
    X_test = X.drop(X_train.index)
    X_prod = simulate_production_drift(X_test)

    print(f'Moyenne feature 0 - Ref: {X_train.iloc[:,0].mean():.3f} | Prod: {X_prod.iloc[:,0].mean():.3f}')

    drift_share, n_drifted, n_total = log_drift_report(X_train, X_prod)

    with mlflow.start_run(run_name='drift_ks_test'):
        log_ks_results(X_train, X_prod)

        if drift_share > SEUIL_DRIFT:
            print(f'CRITIQUE : drift {drift_share:.2%} > seuil {SEUIL_DRIFT:.0%}')
            subprocess.run([sys.executable, str(PROJECT_ROOT / 'src' / 'train.py'),'--retrain'], check=True)           
            mlflow.log_metric('retrain_triggered', 1)
        elif drift_share > SEUIL_WARN:
            print(f'AVERTISSEMENT : drift {drift_share:.2%} — surveillance renforcée')
            mlflow.log_metric('retrain_triggered', 0)
        else:
            print(f'OK : drift {drift_share:.2%} — modèle stable')
            mlflow.log_metric('retrain_triggered', 0)


if __name__ == '__main__':
    main()
