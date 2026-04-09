import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Paths setup
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_clean.csv'

# Configure MLflow
mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
mlflow.set_experiment("Movie_Rating_Regression")

def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    
    # We define vote_average as the continuous target
    # We exclude rows missing the target or features
    df = df.dropna(subset=['vote_average'])
    
    # Filter valid votes and reasonable counts
    df = df[df['vote_count'] > 10].copy()
    
    # Features (numeric only) - exclude revenue to predict rating based purely on metadata if we want,
    # but since it's just experimentation, using all available numeric columns is fine.
    features = ['budget', 'popularity', 'runtime', 'revenue', 'vote_count']
    X = df[features].fillna(0)
    y = df['vote_average']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def log_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    
    fig_path = f"actual_vs_predicted_{model_name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)

def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae, mse, r2 = evaluate_model(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })
        
        # Log Model
        mlflow.sklearn.log_model(model, "model")
        
        # Log Visualization
        log_actual_vs_predicted(y_test, y_pred, model_name)
        
        print(f"[{model_name}] MAE: {mae:.4f} | R2: {r2:.4f}")

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples. Target Mean Rating: {y_train.mean():.2f}")
    
    # 1. Linear Regression
    lr_params = {"fit_intercept": True}
    run_experiment(
        LinearRegression(**lr_params),
        "Linear Regression", lr_params,
        X_train, X_test, y_train, y_test
    )
    
    # 2. Support Vector Regression
    svr_params = {"C": 1.0, "kernel": "rbf"}
    run_experiment(
        SVR(**svr_params),
        "SVR", svr_params,
        X_train, X_test, y_train, y_test
    )
    
    # 3. Random Forest Regression
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    run_experiment(
        RandomForestRegressor(**rf_params),
        "Random Forest Regression", rf_params,
        X_train, X_test, y_train, y_test
    )
    
    print("Regression experiments finished successfully. You can view the results with `mlflow ui`.")

if __name__ == "__main__":
    main()
