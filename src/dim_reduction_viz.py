import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths setup
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_clean.csv'

# Configure MLflow
mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
mlflow.set_experiment("Movie_Dimensionality_Reduction")

def load_data():
    df = pd.read_csv(data_path)
    # We define success as revenue > budget to color our plots
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    
    # Using Subsampling for t-SNE since it's slow on large datasets
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
        
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    
    # Features (numeric only)
    features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    X = df[features].fillna(0)
    y = df['is_success']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def apply_pca(X_scaled, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title(f'PCA - 2 Components (Explained Variance: {sum(pca.explained_variance_ratio_):.2%})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='Is Success (1=Yes, 0=No)')
    
    fig_path = "pca_2d.png"
    plt.savefig(fig_path)
    plt.close()
    
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    
    # Also log variance ratio
    mlflow.log_metrics({
        "pca_variance_pc1": pca.explained_variance_ratio_[0],
        "pca_variance_pc2": pca.explained_variance_ratio_[1],
        "pca_total_variance_2d": sum(pca.explained_variance_ratio_)
    })

def apply_tsne(X_scaled, y):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title('t-SNE components representation')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.colorbar(scatter, label='Is Success (1=Yes, 0=No)')
    
    fig_path = "tsne_2d.png"
    plt.savefig(fig_path)
    plt.close()
    
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)

def main():
    print("Loading and preparing data...")
    X_scaled, y = load_data()
    print(f"Data ready: {X_scaled.shape[0]} samples.")
    
    with mlflow.start_run(run_name="PCA_and_tSNE_Visualization"):
        mlflow.log_param("num_samples", X_scaled.shape[0])
        
        print("Applying PCA...")
        apply_pca(X_scaled, y)
        
        print("Applying t-SNE...")
        apply_tsne(X_scaled, y)
        
    print("Dimensionality reduction completed. View plots in MLflow UI!")

if __name__ == "__main__":
    main()
