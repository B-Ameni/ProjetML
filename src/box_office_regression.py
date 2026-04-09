import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Paths setup
project_root = Path(__file__).parent.parent
movies_path = project_root / 'data' / 'movies_clean.csv'
credits_path = project_root / 'data' / 'credits_clean.csv'

# Configure MLflow
mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
mlflow.set_experiment("Box_Office_Revenue_Prediction")

def load_and_preprocess_data():
    print("Loading datasets...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    # Merge on movie id
    print("Merging datasets...")
    df = pd.merge(movies, credits, left_on='id', right_on='movie_id', how='inner')
    
    # Filter valid target
    df = df[df['revenue'] > 0].copy()
    
    # Feature Engineering
    # 1. Dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(2000)
    df['release_month'] = df['release_date'].dt.month.fillna(6)
    
    # 2. Text cleanups for TF-IDF (Replacing | with spaces)
    df['genres'] = df['genres'].fillna('').str.replace('|', ' ')
    df['cast'] = df['cast'].fillna('').str.replace('|', ' ')
    df['crew'] = df['crew'].fillna('').str.replace('|', ' ')
    
    # Target
    y = df['revenue']
    
    # Features
    X = df[['budget', 'runtime', 'release_year', 'release_month', 'genres', 'cast', 'crew']].copy()
    
    # Fill numeric NAs
    X['budget'] = X['budget'].fillna(X['budget'].median())
    X['runtime'] = X['runtime'].fillna(X['runtime'].median())
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline(model):
    # Column Transformer for mixed data types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['budget', 'runtime', 'release_year', 'release_month']),
            ('genres_tf', TfidfVectorizer(max_features=50, stop_words='english'), 'genres'),
            ('cast_tf', TfidfVectorizer(max_features=100, stop_words='english'), 'cast'),
            ('crew_tf', TfidfVectorizer(max_features=100, stop_words='english'), 'crew')
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def log_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title(f'Actual vs Predicted Box Office - {model_name}')
    plt.tight_layout()
    
    fig_path = f"box_office_vs_predicted_{model_name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)

def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test):
    pipeline = build_pipeline(model)
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        
        print(f"Training {model_name} pipeline...")
        pipeline.fit(X_train, y_train)
        
        # Predict
        print(f"Evaluating {model_name}...")
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })
        
        # Log Model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Log Visualization
        log_actual_vs_predicted(y_test, y_pred, model_name)
        
        print(f"[{model_name}] R2: {r2:.4f} | RMSE: {rmse:,.2f}")

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Data ready: {X_train.shape[0]} training examples.")
    
    # Linear Regression
    lr_params = {"model__fit_intercept": True}
    run_experiment(
        LinearRegression(),
        "Linear Regression", lr_params,
        X_train, X_test, y_train, y_test
    )
    
    # Random Forest Regression
    # We use fewer estimators for speed, but deep enough to capture non-linear relations
    rf_params = {"model__n_estimators": 50, "model__max_depth": 15, "model__random_state": 42}
    run_experiment(
        RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42),
        "Random Forest", rf_params,
        X_train, X_test, y_train, y_test
    )
    
    print("Box Office Regression tracking completed. Models are saved in MLflow.")

if __name__ == "__main__":
    main()
