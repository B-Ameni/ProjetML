# MLflow Model Registry - Guide Complet

## 🎯 Objectif
Activer le **Model Registry** de MLflow pour enregistrer, gérer et servir vos modèles ML en production.

## 📋 Prérequis
- MLflow installé
- Base de données SQLite (créée automatiquement)
- Serveur MLflow en cours d'exécution

---

## 🚀 Démarrage Rapide

### Étape 1: Démarrer le serveur MLflow avec Model Registry

Le serveur MLflow doit être lancé **avant** d'exécuter vos scripts d'entraînement.

**Option A - Via Makefile:**
```bash
make mlflow-server
```

**Option B - Directement:**
```bash
python src/start_mlflow_server.py
```

Vous devriez voir:
```
🚀 Démarrage du serveur MLflow avec Model Registry...
   Backend: sqlite:////path/to/.mlflow/mlflow.db
   Artifacts: /path/to/mlruns
   URL: http://localhost:5000
```

✅ Le serveur est prêt quand vous voyez: `[2026-05-19 XX:XX:XX] WARNING in werkzeug`

### Étape 2: Entraîner un modèle

Ouvrez un **nouveau terminal** et lancez:

```bash
# Entraîner le modèle de classification
python src/train.py

# Ou: Exécuter toutes les expériences
python src/classification_experiments.py

# Ou: Exécuter les expériences de régression
python src/regression_experiments.py
```

Vous verrez:
```
Metrics: {'accuracy': 0.85, 'precision': 0.87, 'recall': 0.82, 'f1_score': 0.84}
Modèle entraîné et loggé dans MLflow avec le nom de run 'production_train'
```

### Étape 3: Enregistrer le modèle dans le Model Registry

```bash
# Enregistrer le meilleur modèle en Production
python src/register_best_model.py
```

Vous verrez:
```
Enregistrement du modèle depuis runs:/xxx/model dans le registry MLflow sous mon_modele_production
Modèle enregistré et promu en Production : mon_modele_production v1
```

### Étape 4: Visualiser dans l'interface Web

Ouvrez: **http://localhost:5000**

Vous verrez:
- **Experiments**: Les expériences d'entraînement
- **Models**: Votre modèle enregistré avec ses versions

---

## 📊 Anatomie du Model Registry

### Structure des Modèles Enregistrés

```
mon_modele_production/
├── Version 1
│   ├── Stage: Production
│   ├── Accuracy: 0.85
│   └── Run: xxx
├── Version 2
│   ├── Stage: Staging
│   ├── Accuracy: 0.84
│   └── Run: yyy
└── Version 3
    ├── Stage: None (Archived)
    ├── Accuracy: 0.82
    └── Run: zzz
```

### Stages (Étapes du Cycle de Vie)

- **None**: Nouveau modèle (par défaut)
- **Staging**: Modèle en test/validation
- **Production**: Modèle en production (actif)
- **Archived**: Modèle archivé (ancien)

---

## 🔧 Workflows Courants

### A. Entraîner → Enregistrer → Servir

```bash
# Terminal 1: Démarrer le serveur MLflow
make mlflow-server

# Terminal 2: Entraîner le modèle
python src/train.py

# Terminal 3: Enregistrer le modèle
python src/register_best_model.py

# Terminal 3: Servir le modèle
make serve
```

### B. Mettre à Jour un Modèle (Bump Version)

```bash
# 1. Entraîner un nouveau modèle
python src/train.py

# 2. Enregistrer (crée automatiquement une nouvelle version)
python src/register_best_model.py

# Voir les versions dans l'interface Web
```

### C. Comparer les Versions

Via l'interface Web (http://localhost:5000):
1. Allez dans **Models** → **mon_modele_production**
2. Cliquez sur **Versions**
3. Comparez les métriques (Accuracy, F1, etc.)

### D. Promouvoir en Production

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="mon_modele_production",
    version=2,  # Nouvelle version
    stage="Production",
    archive_existing_versions=True  # Archive l'ancienne production
)
```

---

## 🧪 Démonstration Complète

Un script de démonstration est fourni pour tester toutes les fonctionnalités:

```bash
python src/demo_model_registry.py
```

Ce script:
1. ✅ Crée une expérience
2. ✅ Entraîne un modèle
3. ✅ L'enregistre dans le Model Registry
4. ✅ Le transition vers Staging
5. ✅ Le promote en Production
6. ✅ Le charge depuis le registry
7. ✅ Fait des prédictions

---

## 📁 Configuration MLflow

Les paramètres sont définis dans [src/mlflow_config.py](src/mlflow_config.py):

```python
BACKEND_STORE_URI = "sqlite:////path/to/.mlflow/mlflow.db"  # ✅ Model Registry
ARTIFACT_ROOT = "/path/to/mlruns"                           # Stockage des artefacts
TRACKING_URI = "http://localhost:5000"                       # Serveur MLflow
```

---

## 🔍 Dépannage

### Q: Erreur "Model Registry requires a backend store"

**R:** Assurez-vous que le serveur MLflow est lancé avec:
```bash
make mlflow-server
```

(Pas `mlflow ui` qui n'active pas le Model Registry)

### Q: Le modèle n'apparaît pas dans Models

**R:** 
1. ✅ Vérifiez que le serveur MLflow tourne
2. ✅ Exécutez `python src/register_best_model.py`
3. ✅ Rafraîchissez http://localhost:5000

### Q: Erreur de connexion au serveur

**R:** Vérifiez que le serveur tourne:
```bash
curl http://localhost:5000
# Devrait retourner une réponse HTML
```

### Q: Le modèle reste en stage "None"

**R:** Lancez explicitement la transition:
```bash
python src/register_best_model.py
# Cela promote automatiquement en Production
```

---

## 📝 Prochaines Étapes

1. **Intégrer avec l'API Backend**: Utiliser le modèle depuis [backend/main.py](../backend/main.py)
2. **Automatiser les Retraînements**: Via des pipelines MLflow
3. **Versioning des Données**: Tracker les versions de datasets
4. **Monitoring en Production**: Superviser la performance du modèle
5. **A/B Testing**: Comparer deux versions en production

---

## 🎓 Ressources

- Documentation MLflow: https://mlflow.org/docs/latest/model-registry.html
- API MLflowClient: https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient
- Workflow complet: https://mlflow.org/docs/latest/model-registry.html#workflow

---

## ✅ Vérification Finale

Pour vérifier que tout fonctionne:

```bash
# 1. Vérifier que le serveur tourne
curl http://localhost:5000

# 2. Entraîner un modèle
python src/train.py

# 3. Enregistrer le modèle
python src/register_best_model.py

# 4. Ouvrir l'interface Web
# http://localhost:5000 → Models → mon_modele_production
```

Si vous voyez votre modèle dans l'interface Web avec le stage "Production", c'est bon! 🎉
