import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths setup
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_credits_merged.csv'

# Configure MLflow
mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
mlflow.set_experiment("Movie_Success_Classification")

FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']


def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    X = df[FEATURES].fillna(0)
    y = df['is_success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, prec, rec, f1


def log_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig_path = f"cm_{model_name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)


def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, prec, rec, f1 = evaluate_model(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "cm_tn": int(tn),
            "cm_fp": int(fp),
            "cm_fn": int(fn),
            "cm_tp": int(tp)
        })
        mlflow.sklearn.log_model(model, "model")
        log_confusion_matrix(y_test, y_pred, model_name)
        print(f"[{model_name}] Accuracy: {acc:.4f} | F1: {f1:.4f}")


def tune_algorithm(model_name, X_train, X_test, y_train, y_test, method):
    if model_name == 'Random Forest':
        estimator = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'Logistic Regression':
        estimator = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }
    else:
        print(f"Tuning non supporté pour {model_name}. Entraînement direct lancé.")
        return None

    if method.lower() == 'gridsearch':
        search = GridSearchCV(estimator, param_grid, cv=3, scoring='f1', n_jobs=-1)
    else:
        search = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=8, cv=3, scoring='f1', random_state=42, n_jobs=-1)

    search.fit(X_train, y_train)
    print(f"Best params for {model_name}: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def model_from_name(name, params):
    if name == 'KNN':
        return KNeighborsClassifier(n_neighbors=int(params.get('k', 5)), weights='uniform')
    if name == 'SVM':
        return SVC(C=float(params.get('C', 1.0)), kernel='rbf', probability=True)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=int(params.get('n_estimators', 100)), max_depth=None if params.get('max_depth', 10) is None else int(params.get('max_depth', 10)), random_state=42)
    if name == 'Logistic Regression':
        return LogisticRegression(C=float(params.get('C', 1.0)), max_iter=1000, solver='liblinear', random_state=42)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Run classification experiments for Box-Office prediction')
    parser.add_argument('--algorithms', nargs='*', default=['KNN', 'SVM', 'Random Forest', 'Logistic Regression'])
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--tune-method', type=str, default=None, choices=['gridsearch', 'randomsearch', 'optuna'])
    return parser.parse_args()


def main():
    args = parse_args()
    print('Loading data...')
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f'Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples. Target positive rate: {y_train.mean():.2%}')
    print(f'Training algorithms: {args.algorithms}')

    if args.tune_method:
        for algo in args.algorithms:
            if algo in ['Random Forest', 'Logistic Regression']:
                result = tune_algorithm(algo, X_train, X_test, y_train, y_test, args.tune_method)
                if result:
                    estimator, best_params = result
                    run_experiment(estimator, f"{algo} ({args.tune_method})", best_params, X_train, X_test, y_train, y_test)
            else:
                print(f'Auto-tuning non supporté pour {algo}. Entraînement standard lancé.')

    for algo in args.algorithms:
        if args.tune_method and algo in ['Random Forest', 'Logistic Regression']:
            continue
        params = {
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'n_estimators': args.n_estimators,
            'k': args.k
        }
        model = model_from_name(algo, params)
        if model is None:
            print(f'Algorithme inconnu: {algo}')
            continue
        if algo == 'KNN':
            params = {'n_neighbors': args.k}
        elif algo == 'Random Forest':
            params = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth, 'random_state': 42}
        elif algo == 'Logistic Regression':
            params = {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
        elif algo == 'SVM':
            params = {'C': 1.0, 'kernel': 'rbf'}
        run_experiment(model, algo, params, X_train, X_test, y_train, y_test)

    print('Classification experiments finished successfully. You can view the results with `mlflow ui`.')

if __name__ == '__main__':
    main()
