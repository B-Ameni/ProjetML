#!/usr/bin/env python3
"""
MLflow Results Exporter & Analyzer
Extracts, analyzes, and exports experiment results from MLflow tracking.
"""

import json
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient
import sys

# Configuration
PROJECT_ROOT = Path(__file__).parent
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Movie_Success_Classification"

def main():
    """Extract and analyze MLflow results."""
    
    print("="*70)
    print("MLflow Results Analysis & Export")
    print("="*70)
    
    # Initialize MLflow client
    mlflow_uri = MLRUNS_PATH.as_uri()
    client = MlflowClient(tracking_uri=mlflow_uri)
    
    print(f"\n✓ Tracking URI: {mlflow_uri}")
    
    # Find experiment
    experiments = client.search_experiments()
    exp_id = None
    
    for exp in experiments:
        if exp.name == EXPERIMENT_NAME:
            exp_id = exp.experiment_id
            break
    
    if exp_id is None:
        print(f"✗ Experiment '{EXPERIMENT_NAME}' not found")
        print(f"Available experiments: {[e.name for e in experiments]}")
        sys.exit(1)
    
    print(f"✓ Found experiment: {EXPERIMENT_NAME} (ID: {exp_id})")
    
    # Fetch runs
    runs = client.search_runs(experiment_ids=[exp_id])
    print(f"✓ Found {len(runs)} runs")
    
    # Extract and organize data
    results = []
    
    for run in runs:
        if run.state != "FINISHED":
            continue
        
        run_data = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'status': run.info.status,
            'timestamp': run.info.start_time,
        }
        
        # Metrics
        for metric_key, metric_value in run.data.metrics.items():
            run_data[metric_key] = metric_value
        
        # Parameters
        for param_key, param_value in run.data.params.items():
            run_data[f'param_{param_key}'] = param_value
        
        results.append(run_data)
    
    if not results:
        print("✗ No completed runs found")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*70)
    print("RÉSULTATS DES EXPÉRIENCES")
    print("="*70)
    
    if 'accuracy' in df.columns:
        display_df = df[['run_name', 'accuracy', 'precision', 'recall', 'f1_score']].copy()
        display_df = display_df.sort_values('f1_score', ascending=False)
        print(display_df.to_string(index=False))
    else:
        print(df[['run_name', 'status']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*70)
    print("STATISTIQUES - Meilleurs Modèles")
    print("="*70)
    
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    for metric in available_metrics:
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx, 'run_name']
        best_score = df.loc[best_idx, metric]
        
        symbol = "⭐" if metric == 'f1_score' else ""
        print(f"  Best {metric:12} → {best_model:30} ({best_score:.4f}) {symbol}")
    
    # Export to CSV
    csv_path = PROJECT_ROOT / "run_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results exported to: {csv_path}")
    
    # Export to JSON
    json_path = PROJECT_ROOT / "run_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results exported to: {json_path}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    
    best_f1_idx = df['f1_score'].idxmax()
    best_model_name = df.loc[best_f1_idx, 'run_name']
    best_f1_score = df.loc[best_f1_idx, 'f1_score']
    
    print(f"\n✓ MEILLEUR MODÈLE: {best_model_name}")
    print(f"  F1-Score: {best_f1_score:.4f}")
    print(f"  Run ID: {df.loc[best_f1_idx, 'run_id']}")
    
    print("\n💡 Actions recommandées:")
    print(f"   1. Deployer le modèle {best_model_name}")
    print(f"   2. Register dans MLflow Model Registry")
    print(f"   3. Setup monitoring et alertes")
    print(f"   4. Planifier réentraînement mensuel")
    
    return df

if __name__ == "__main__":
    df = main()
    print("\n✓ Analysis complete!")
