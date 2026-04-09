from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import subprocess
import os
import json
import shutil
from pathlib import Path

app = FastAPI(title="Box-Office Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to local tracking
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"file:///{project_root}/mlruns".replace("\\", "/"))

# Config file to track deployed model and active dataset
deployed_model_file = os.path.join(project_root, ".deployed_model.json")
dataset_config_file = os.path.join(project_root, ".dataset_config.json")


def get_current_dataset_path():
    if os.path.exists(dataset_config_file):
        try:
            with open(dataset_config_file, 'r') as f:
                config = json.load(f)
                path = config.get('current_dataset')
                if path and os.path.exists(path):
                    return path
        except Exception:
            pass
    return os.path.join(project_root, 'data', 'movies_credits_merged.csv')


def set_current_dataset_path(path):
    with open(dataset_config_file, 'w') as f:
        json.dump({'current_dataset': path}, f)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API Backend Box-Office opérationnelle. Endpoints: /models, /results, /train"}

@app.get("/models")
def get_models():
    experiment = mlflow.get_experiment_by_name("Movie_Success_Classification")
    if not experiment:
        return {"models": []}
    
    # Load deployed model ID from config file
    deployed_run_id = None
    if os.path.exists(deployed_model_file):
        try:
            with open(deployed_model_file, 'r') as f:
                config = json.load(f)
                deployed_run_id = config.get('deployed_run_id')
        except:
            pass
    
    # If no deployed model set, use the most recent one
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if not runs.empty and not deployed_run_id:
        deployed_run_id = runs.iloc[0]['run_id']
        # Save it
        with open(deployed_model_file, 'w') as f:
            json.dump({'deployed_run_id': deployed_run_id}, f)
    
    if runs.empty:
        return {"models": []}
        
    models_list = []
    for idx, row in runs.iterrows():
        run_id = row.get('run_id', 'unknown')
        # Mark as "Deployed" if it's the one we saved, else "Archived"
        status = "Déployé" if run_id == deployed_run_id else "Archivé"
        
        # safely get row values
        run_name = row.get('tags.mlflow.runName', 'Model')
        acc = row.get('metrics.accuracy', 0.0)
        
        # handle missing safely
        if pd.isna(acc):
            acc = 0.0
        if pd.isna(run_name):
            run_name = 'Model'
            
        models_list.append({
            "id": f"exp-{str(run_id)[:6]}",
            "run_id": run_id,
            "name": f"Box_Office_{run_name}",
            "model": run_name,
            "accuracy": round(float(acc), 4),
            "status": status,
            "version": f"v1.{len(runs)-idx}",
            "dataset": os.path.basename(get_current_dataset_path())
        })
    
    return {"models": models_list}

@app.post("/rollback/{run_id}")
def rollback_model(run_id: str):
    """Set a specific model version as deployed"""
    try:
        # Save the deployed run ID
        with open(deployed_model_file, 'w') as f:
            json.dump({'deployed_run_id': run_id}, f)
        return {"status": "success", "message": f"Modèle {run_id} restauré et déployé."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/results")
def get_results():
    experiment = mlflow.get_experiment_by_name("Movie_Success_Classification")
    if not experiment:
        return {"models": [], "confusion_matrix": {}}
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        return {"models": [], "confusion_matrix": {}}

    models = []
    deployed_run_id = None
    if os.path.exists(deployed_model_file):
        try:
            with open(deployed_model_file, 'r') as f:
                deployed_run_id = json.load(f).get('deployed_run_id')
        except Exception:
            pass

    for _, row in runs.iterrows():
        run_id = row.get('run_id')
        model_name = row.get('tags.mlflow.runName', 'Unknown')
        if pd.isna(model_name):
            model_name = 'Unknown'

        metrics = {
            'accuracy': float(row.get('metrics.accuracy', 0.0) if not pd.isna(row.get('metrics.accuracy', 0.0)) else 0.0),
            'precision': float(row.get('metrics.precision', 0.0) if not pd.isna(row.get('metrics.precision', 0.0)) else 0.0),
            'recall': float(row.get('metrics.recall', 0.0) if not pd.isna(row.get('metrics.recall', 0.0)) else 0.0),
            'f1_score': float(row.get('metrics.f1_score', 0.0) if not pd.isna(row.get('metrics.f1_score', 0.0)) else 0.0)
        }
        models.append({
            'run_id': run_id,
            'model': model_name,
            'status': 'Déployé' if run_id == deployed_run_id else 'Archivé',
            'dataset': os.path.basename(get_current_dataset_path()),
            'metrics': metrics,
            'version': f"v1.{run_id[:6]}"
        })

    deployed_cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    deployed_run = runs[runs['run_id'] == deployed_run_id] if deployed_run_id else pd.DataFrame()
    if not deployed_run.empty:
        row = deployed_run.iloc[0]
        deployed_cm = {
            'tp': int(row.get('metrics.cm_tp', 0) if not pd.isna(row.get('metrics.cm_tp', 0)) else 0),
            'tn': int(row.get('metrics.cm_tn', 0) if not pd.isna(row.get('metrics.cm_tn', 0)) else 0),
            'fp': int(row.get('metrics.cm_fp', 0) if not pd.isna(row.get('metrics.cm_fp', 0)) else 0),
            'fn': int(row.get('metrics.cm_fn', 0) if not pd.isna(row.get('metrics.cm_fn', 0)) else 0)
        }
    elif not runs.empty:
        row = runs.iloc[0]
        deployed_cm = {
            'tp': int(row.get('metrics.cm_tp', 0) if not pd.isna(row.get('metrics.cm_tp', 0)) else 0),
            'tn': int(row.get('metrics.cm_tn', 0) if not pd.isna(row.get('metrics.cm_tn', 0)) else 0),
            'fp': int(row.get('metrics.cm_fp', 0) if not pd.isna(row.get('metrics.cm_fp', 0)) else 0),
            'fn': int(row.get('metrics.cm_fn', 0) if not pd.isna(row.get('metrics.cm_fn', 0)) else 0)
        }

    return {
        'models': models,
        'confusion_matrix': deployed_cm
    }

@app.get("/dataset")
def get_dataset(columns: str = Query(None), outcome: str = Query(None)):
    df = pd.read_csv(get_current_dataset_path())
    if 'is_success' not in df.columns:
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['is_success'] = ((df['revenue'] > df['budget']) & (df['budget'] > 0) & (df['revenue'] > 0)).astype(int)
    
    if outcome in ['hit', 'flop'] and 'is_success' in df.columns:
        df = df[df['is_success'] == (1 if outcome == 'hit' else 0)]

    selected_columns = df.columns.tolist()
    if columns:
        requested = [col for col in columns.split(',') if col in df.columns]
        if requested:
            selected_columns = requested
            df = df[requested]

    info = {
        "name": os.path.basename(get_current_dataset_path()),
        "size": f"{os.path.getsize(get_current_dataset_path()) / 1024 / 1024:.1f} MB",
        "columns": len(selected_columns),
        "lines": len(df),
        "status": "Prêt",
        "available_columns": df.columns.tolist(),
        "counts": {
            "hit": int(df[df.get('is_success') == 1].shape[0]) if 'is_success' in df.columns else None,
            "flop": int(df[df.get('is_success') == 0].shape[0]) if 'is_success' in df.columns else None
        }
    }
    # Clean NaN and infinite values before JSON serialization
    preview_df = df.head(20).copy()
    preview_df = preview_df.fillna(0)  # Replace NaN with 0
    preview_df = preview_df.replace([np.inf, -np.inf], 0)  # Replace infinities with 0
    preview = preview_df.to_dict('records')
    return {"info": info, "preview": preview}


@app.post('/dataset/upload')
async def upload_dataset(file: UploadFile = File(...)):
    target_path = os.path.join(project_root, 'data', file.filename)
    contents = await file.read()
    with open(target_path, 'wb') as f:
        f.write(contents)
    set_current_dataset_path(target_path)
    return {
        'status': 'success',
        'message': f'Dataset téléversé et activé : {file.filename}',
        'dataset': os.path.basename(target_path)
    }


@app.post('/dataset/clean')
def clean_dataset():
    path = get_current_dataset_path()
    df = pd.read_csv(path)
    if 'budget' in df.columns and 'revenue' in df.columns:
        df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df = df.dropna().reset_index(drop=True)
    cleaned_path = os.path.join(project_root, 'data', f'cleaned_{os.path.basename(path)}')
    df.to_csv(cleaned_path, index=False)
    set_current_dataset_path(cleaned_path)
    return {
        'status': 'success',
        'message': f'Dataset nettoyé et sauvegardé sous {os.path.basename(cleaned_path)}',
        'dataset': os.path.basename(cleaned_path)
    }


@app.post("/predict")
def predict(data: dict):
    experiment = mlflow.get_experiment_by_name("Movie_Success_Classification")
    if not experiment:
        return {"error": "No experiment found"}

    deployed_run_id = None
    if os.path.exists(deployed_model_file):
        try:
            with open(deployed_model_file, 'r') as f:
                deployed_run_id = json.load(f).get('deployed_run_id')
        except Exception:
            pass

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
    if runs.empty:
        return {"error": "No runs found"}

    if deployed_run_id and any(runs['run_id'] == deployed_run_id):
        best_run_id = deployed_run_id
    else:
        best_run_id = runs.iloc[0]['run_id']

    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    
    from sklearn.preprocessing import StandardScaler
    features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    input_data = {
        'budget': float(data.get('budget', 0)),
        'popularity': float(data.get('popularity', 0) or 0),
        'runtime': float(data.get('runtime', 0)),
        'vote_average': float(data.get('vote_average', 0) or 0),
        'vote_count': float(data.get('vote_count', 0) or 0)
    }
    input_df = pd.DataFrame([input_data])

    df = pd.read_csv(get_current_dataset_path())
    df = df[(df.get('budget', 0) > 0) & (df.get('revenue', 0) > 0)].copy()
    scaler = StandardScaler()
    scaler.fit(df[features].fillna(0))
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    return {"prediction": "Hit" if prediction == 1 else "Flop", "run_id": best_run_id}


class TrainRequest(BaseModel):
    learningRate: float = 0.01
    maxDepth: int = 10
    nEstimators: int = 100
    k: int = 5
    algorithms: list = None
    trainType: str = 'scratch'
    tuneMethod: str = None


def training_job(params=None):
    script_path = os.path.join(project_root, "src", "classification_experiments.py")
    python_exec = os.path.join(project_root, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exec):
        python_exec = os.path.join(project_root, "venv", "Scripts", "python")
    if not os.path.exists(python_exec):
        python_exec = "python"

    cmd = [python_exec, script_path]
    if params is not None:
        algorithms = params.get('algorithms')
        if algorithms:
            algo_map = {
                'svm': 'SVM',
                'knn': 'KNN',
                'rf': 'Random Forest',
                'lr': 'Logistic Regression'
            }
            mapped_algos = [algo_map.get(a.lower(), a) for a in algorithms]
            cmd.extend(['--algorithms', *mapped_algos])
            print(f"Training with algorithms: {mapped_algos}")

        if params.get('learningRate') is not None:
            cmd.extend(['--learning-rate', str(params['learningRate'])])
        if params.get('maxDepth') is not None:
            cmd.extend(['--max-depth', str(params['maxDepth'])])
        if params.get('nEstimators') is not None:
            cmd.extend(['--n-estimators', str(params['nEstimators'])])
        if params.get('k') is not None:
            cmd.extend(['--k', str(params['k'])])
        if params.get('tuneMethod'):
            tune_value = params['tuneMethod'].lower()
            if tune_value in ['gridsearch', 'randomsearch']:
                cmd.extend(['--tune-method', tune_value])
            else:
                cmd.extend(['--tune-method', 'gridsearch'])

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Training script error: {result.stderr}")
        else:
            print(f"Training completed: {result.stdout}")
    except Exception as e:
        print(f"Error running training job: {e}")

@app.post("/train")
def run_train(request: TrainRequest, background_tasks: BackgroundTasks):
    if request.trainType == 'pretrained':
        experiment = mlflow.get_experiment_by_name("Movie_Success_Classification")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
            if not runs.empty:
                deployed_id = runs.iloc[0]['run_id']
                with open(deployed_model_file, 'w') as f:
                    json.dump({'deployed_run_id': deployed_id}, f)
                return {"status": "success", "message": "Modèle pré-entraîné sélectionné et déployé."}
        return {"status": "error", "message": "Aucun modèle pré-entraîné disponible."}

    background_tasks.add_task(training_job, params=request.dict())
    return {"status": "success", "message": "Entraînement démarré en arrière-plan (MLflow Tracking actif)."}

@app.post("/tune")
def tune_model(request: TrainRequest, background_tasks: BackgroundTasks):
    if not request.algorithms:
        return {"status": "error", "message": "Veuillez sélectionner au moins un algorithme pour le tuning."}
    request.trainType = 'scratch'
    background_tasks.add_task(training_job, params=request.dict())
    return {"status": "success", "message": f"Tuning automatique ({request.tuneMethod}) démarré en arrière-plan."}


@app.get('/export-model/{run_id}')
def export_model(run_id: str):
    experiment = mlflow.get_experiment_by_name("Movie_Success_Classification")
    if not experiment:
        return {"status": "error", "message": "Aucun modèle trouvé pour l'export."}
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        out_dir = os.path.join(project_root, 'data', 'exports', run_id)
        os.makedirs(out_dir, exist_ok=True)
        mlflow.sklearn.save_model(model, out_dir)
        archive_path = os.path.join(project_root, 'data', 'exports', f'model_{run_id}')
        shutil.make_archive(archive_path, 'zip', out_dir)
        return FileResponse(f"{archive_path}.zip", media_type='application/zip', filename=f'model_{run_id}.zip')
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
