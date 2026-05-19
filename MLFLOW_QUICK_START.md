# 🚀 ACTIVATION MLflow Model Registry - RÉSUMÉ RAPIDE

## ✨ Ce qui a changé

Vous pouvez maintenant **enregistrer et gérer vos modèles** dans le Model Registry MLflow!

### Fichiers créés/modifiés:
- ✅ `src/mlflow_config.py` - Configuration MLflow avec Backend Store (SQLite)
- ✅ `src/start_mlflow_server.py` - Démarrage du serveur MLflow avec Model Registry
- ✅ `src/demo_model_registry.py` - Démonstration complète
- ✅ `MLFLOW_REGISTRY_GUIDE.md` - Guide détaillé en français
- ✅ Mis à jour: `src/train.py`, `src/register_best_model.py`, `src/classification_experiments.py`, `src/regression_experiments.py`, `backend/main.py`, `Makefile`

---

## 🎯 INSTRUCTIONS D'UTILISATION (3 étapes)

### 1️⃣ Démarrer le serveur MLflow
```bash
make mlflow-server
```
ou
```bash
python src/start_mlflow_server.py
```

**Attendez de voir** (cela signifie que le serveur est prêt):
```
[2026-05-19 XX:XX:XX] WARNING in werkzeug
```

### 2️⃣ Entraîner un modèle (dans un nouveau terminal)
```bash
python src/train.py
```

### 3️⃣ Enregistrer le modèle dans le Model Registry
```bash
python src/register_best_model.py
```

Vous devriez voir:
```
Enregistrement du modèle depuis runs:/xxx/model dans le registry MLflow sous mon_modele_production
Modèle enregistré et promu en Production : mon_modele_production v1
```

---

## 🌐 Interface Web

Une fois le serveur lancé, ouvrez dans votre navigateur:

👉 **http://localhost:5000**

Vous verrez:
- **Experiments** - Vos entraînements
- **Models** - Vos modèles enregistrés ⭐

---

## 📊 Voir votre modèle enregistré

1. Allez à http://localhost:5000
2. Cliquez sur **Models** dans la barre de navigation
3. Vous verrez **mon_modele_production** avec:
   - Version 1
   - Stage: **Production** (en vert)
   - Métriques: accuracy, precision, recall, f1_score

---

## 🧪 Tester avec la démonstration

```bash
python src/demo_model_registry.py
```

Cela va:
1. Créer une expérience
2. Entraîner un modèle
3. L'enregistrer dans le Model Registry
4. Le passer en Staging, puis Production
5. Le charger et faire des prédictions

---

## 🔑 Points clés

| Avant | Après |
|-------|-------|
| ❌ Tracking URI local (file://) | ✅ Serveur MLflow (http://localhost:5000) |
| ❌ Pas de Model Registry | ✅ Model Registry SQLite activé |
| ❌ Modèles stockés en fichiers | ✅ Modèles avec versioning et stages |
| ❌ Pas de gestion des versions | ✅ Versions: v1, v2, v3... |
| ❌ Pas de stages de cycle de vie | ✅ Stages: None, Staging, Production, Archived |

---

## 📁 Structure

```
projet/
├── src/
│   ├── mlflow_config.py           ← Configuration
│   ├── start_mlflow_server.py     ← Démarrer le serveur
│   ├── train.py                   ← Entraîner (utilise mlflow_config)
│   ├── register_best_model.py     ← Enregistrer dans le registry
│   ├── demo_model_registry.py     ← Démonstration
│   └── check_mlflow_setup.py      ← Vérifier configuration
├── backend/
│   └── main.py                    ← API (utilise mlflow_config)
├── .mlflow/
│   └── mlflow.db                  ← Base de données SQLite (créée auto)
├── mlruns/                        ← Artefacts stockés ici
├── Makefile                       ← make mlflow-server, make train, etc
└── MLFLOW_REGISTRY_GUIDE.md       ← Guide complet
```

---

## ⚡ Makefile - Commandes rapides

```bash
make mlflow-server        # Démarrer le serveur MLflow
make train                # Entraîner le modèle
make experiments          # Lancer tous les expériences
make train-regression     # Entraîner les régressions
make register             # Enregistrer le modèle en Production
make serve                # Servir le modèle
make pipeline             # train + register
```

---

## 🆘 Dépannage rapide

**Q: Erreur "Model Registry requires a backend store"**
- ✅ Solution: Utilisez `python src/start_mlflow_server.py` (pas `mlflow ui`)

**Q: Le modèle n'apparaît pas dans Models**
- ✅ Solution: Exécutez `python src/register_best_model.py`

**Q: Connexion refusée à http://localhost:5000**
- ✅ Solution: Le serveur n'est pas lancé, exécutez `make mlflow-server`

---

## 🎓 Workflow complet (exemple)

```bash
# Terminal 1 - Serveur MLflow
$ make mlflow-server
🚀 Démarrage du serveur MLflow...
✓ Backend: SQLite
✓ URL: http://localhost:5000

# Terminal 2 - Entraînement
$ python src/train.py
Metrics: {'accuracy': 0.85, ...}
Modèle loggé dans MLflow

$ python src/register_best_model.py
✓ Modèle enregistré: mon_modele_production v1
✓ Stage: Production

# Terminal 3 - Consulter
$ curl http://localhost:5000/api/2.0/mlflow/registered-models/list
# Affiche tous les modèles enregistrés
```

---

## ✅ Validation finale

Pour vérifier que tout marche:

```bash
# 1. Démarrer le serveur
make mlflow-server

# 2. Dans un nouveau terminal, entraîner
python src/train.py

# 3. Enregistrer
python src/register_best_model.py

# 4. Ouvrir http://localhost:5000
# 5. Aller dans Models
# 6. Vous devriez voir "mon_modele_production" avec stage "Production" ✅
```

---

## 📚 Documentation complète

Consultez [MLFLOW_REGISTRY_GUIDE.md](MLFLOW_REGISTRY_GUIDE.md) pour:
- Workflows avancés
- Comparaison de versions
- API MLflow
- A/B Testing
- Monitoring en production

---

**Besoin d'aide?** Consultez la section Dépannage du guide complet! 🎉
