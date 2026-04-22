import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_credits_merged.csv'

# Configure MLflow
mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
mlflow.set_experiment("Movie_Success_Classification")

FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']

def load_data():
    df = pd.read_csv(data_path)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    
    # We keep the raw dataframe to retrieve titles and unscaled features later
    X = df[FEATURES].fillna(0)
    y = df['is_success']
    
    # Stratified split to ensure same distribution, though classification_experiments.py didn't use stratify,
    # let's replicate the existing behavior for consistency:
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, indices_test, scaler

def run_analysis():
    print("Loading data...")
    df, X_train, X_test, y_train, y_test, indices_test, scaler = load_data()
    print("Data loaded.\n")

    # --- 1. Feature Importances ---
    print("=== 1. Feature Importances ===")
    rf_main = RandomForestClassifier(random_state=42)
    
    with mlflow.start_run(run_name="RF_Feature_Importance"):
        rf_main.fit(X_train, y_train)
        importances = rf_main.feature_importances_
        
        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=FEATURES)
        plt.title('Random Forest Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        fig_path = "feature_importances.png"
        plt.savefig(fig_path)
        plt.close()
        
        mlflow.log_artifact(fig_path)
        
        for name, imp in zip(FEATURES, importances):
            print(f"- {name}: {imp:.4f}")
            mlflow.log_metric(f"importance_{name}", imp)
            
        print("Feature importances plotted and saved.\n")


    # --- 2. Stability of Predictions ---
    print("=== 2. Stability of Predictions ===")
    random_states = [1, 23, 42, 99, 2024]
    accuracies = []
    
    for rs in random_states:
        with mlflow.start_run(run_name=f"RF_Stability_rs_{rs}"):
            rf_stab = RandomForestClassifier(random_state=rs)
            rf_stab.fit(X_train, y_train)
            preds = rf_stab.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)
            mlflow.log_param("random_state", rs)
            mlflow.log_metric("accuracy", acc)
            print(f"Random State {rs}: Accuracy = {acc:.4f}")
            
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}, Std Dev: {np.std(accuracies):.4f}\n")


    # --- 3. Error Analysis ---
    print("=== 3. Error Analysis ===")
    y_pred_main = rf_main.predict(X_test)
    
    # Find misclassified indices
    y_test_array = np.array(y_test)
    misclassified_mask = y_test_array != y_pred_main
    misclassified_relative_indices = np.where(misclassified_mask)[0]
    
    print("Sample Misclassified Examples:")
    # We take 3 examples
    sample_errors = misclassified_relative_indices[:3]
    for i, rel_idx in enumerate(sample_errors):
        original_idx = indices_test[rel_idx]
        movie_row = df.loc[original_idx]
        true_label = y_test_array[rel_idx]
        pred_label = y_pred_main[rel_idx]
        print(f"\nExample {i+1}: {movie_row['title']}")
        print(f"  True Label: {true_label} (Success) | Predicted Label: {pred_label}")
        for feature in FEATURES:
            print(f"  - {feature}: {movie_row[feature]}")


    # --- 4. Bias and Variance ---
    print("\n=== 4. Bias and Variance ===")
    configs = [
        {"n_estimators": 10, "max_depth": 2},     # Underfitting case
        {"n_estimators": 100, "max_depth": 10},   # Balanced case
        {"n_estimators": 200, "max_depth": None}  # Overfitting case
    ]
    
    print(f"{'n_estimators':<12} | {'max_depth':<9} | {'Train Acc':<9} | {'Test Acc':<8} | {'Gap (Variance)'}")
    print("-" * 65)
    
    for conf in configs:
        n_est = conf['n_estimators']
        mx_dep = conf['max_depth']
        with mlflow.start_run(run_name=f"RF_BiasVariance_n{n_est}_d{mx_dep}"):
            rf_bv = RandomForestClassifier(n_estimators=n_est, max_depth=mx_dep, random_state=42)
            rf_bv.fit(X_train, y_train)
            
            train_preds = rf_bv.predict(X_train)
            test_preds = rf_bv.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            gap = train_acc - test_acc
            
            mlflow.log_params(conf)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            
            print(f"{n_est:<12} | {str(mx_dep):<9} | {train_acc:.4f}    | {test_acc:.4f}   | {gap:.4f}")
            

    # --- 5. Comparison with Decision Tree ---
    print("\n=== 5. Comparison with Decision Tree ===")
    with mlflow.start_run(run_name="Decision_Tree_Baseline"):
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_preds)
        mlflow.log_metric("accuracy", dt_acc)
        
    print(f"Decision Tree Test Accuracy: {dt_acc:.4f}")
    rf_best = accuracy_score(y_test, y_pred_main)
    print(f"Default Random Forest Test Accuracy: {rf_best:.4f}")

if __name__ == '__main__':
    run_analysis()
