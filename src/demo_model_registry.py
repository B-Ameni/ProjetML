"""
Script de démonstration du Model Registry MLflow
Ce script montre comment :
1. Entraîner un modèle et le logger dans MLflow
2. Le retrouver dans le Model Registry
3. Gérer les versions et les stages (Staging, Production)
"""

import time
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlflow_config import TRACKING_URI

def demo_model_registry():
    """
    Démonstration complète du Model Registry
    """
    print("=" * 70)
    print("🎬 DÉMONSTRATION: MLflow Model Registry")
    print("=" * 70)
    print()
    
    # Configuration
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    model_name = "demo_box_office_model"
    
    # 1. Créer une expérience
    print("1️⃣  Création de l'expérience 'Demo_Model_Registry'...")
    exp_name = "Demo_Model_Registry"
    try:
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            exp_id = exp.experiment_id
            print(f"   ✓ Expérience existante: {exp_id}")
    except:
        exp_id = client.create_experiment(exp_name)
        print(f"   ✓ Nouvelle expérience créée: {exp_id}")
    print()
    
    # 2. Générer et entraîner un modèle
    print("2️⃣  Entraînement du modèle...")
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    with mlflow.start_run(experiment_id=exp_id, run_name="demo_run_1"):
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metrics({"accuracy": accuracy})
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"   ✓ Modèle entraîné avec accuracy: {accuracy:.4f}")
        print(f"   ✓ Run ID: {run_id}")
    print()
    
    # 3. Enregistrer le modèle dans le Model Registry
    print("3️⃣  Enregistrement dans le Model Registry...")
    model_uri = f"runs:/{run_id}/model"
    
    # Vérifier si le modèle existe déjà
    try:
        existing_model = client.get_registered_model(model_name)
        print(f"   ⚠️  Le modèle '{model_name}' existe déjà")
        print(f"   Versions existantes: {[v.version for v in existing_model.latest_versions]}")
    except:
        pass
    
    # Enregistrer le modèle
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version
    print(f"   ✓ Modèle enregistré: {model_name} v{version}")
    print(f"   ✓ URI: {model_uri}")
    print()
    
    # 4. Transitionner vers Staging
    print("4️⃣  Transition du modèle vers Staging...")
    time.sleep(2)  # Attendre la synchronisation
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"   ✓ {model_name} v{version} est maintenant en Staging")
    print()
    
    # 5. Afficher les modèles enregistrés
    print("5️⃣  Affichage des modèles enregistrés:")
    registered_models = client.list_registered_models()
    for rm in registered_models:
        print(f"   📦 {rm.name}")
        for version in rm.latest_versions:
            print(f"      v{version.version}: {version.current_stage}")
    print()
    
    # 6. Transition vers Production
    print("6️⃣  Promotion en Production...")
    time.sleep(2)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"   ✓ {model_name} v{version} est maintenant en Production")
    print()
    
    # 7. Charger le modèle depuis le registry
    print("7️⃣  Chargement du modèle depuis le Model Registry...")
    model_uri_production = f"models:/{model_name}/Production"
    loaded_model = mlflow.sklearn.load_model(model_uri_production)
    test_pred = loaded_model.predict(X_test[:5])
    print(f"   ✓ Modèle chargé avec succès")
    print(f"   ✓ Prédictions de test: {test_pred}")
    print()
    
    print("=" * 70)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 70)
    print()
    print("🎯 Prochaines étapes:")
    print("   1. Accédez à l'interface Web: http://localhost:5000")
    print("   2. Allez dans 'Models' pour voir votre modèle enregistré")
    print("   3. Utilisez 'serve' pour servir le modèle en production")
    print()

if __name__ == "__main__":
    demo_model_registry()
