Movie Box-Office Prediction Using Machine Learning

 Classification Binaire avec MLflow  
Technologies utilisées : React + FastAPI + MLflow + scikit-learn

Ce projet prédit si un film est rentable ou non (revenue > budget) en fonction de ses caractéristiques:
- Budget
- Popularité
- Durée
- Note moyenne
- Nombre de votes

Modèles implémentés:
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest (donne le meilleur resultat ) 
- Logistic Regression
- AdaBoost
- XGBoost

Installation
- Python 3.10
- Node.js 16
- Git

python -m venv venv
source venv/Scripts/activate  
pip install -r requirements.txt

Dépendances :
- fastapi
- uvicorn
- mlflow
- scikit-learn
- pandas
- numpy

Entraîner les Modèles : 
python src/classification_experiments.py

Frontend :
cd frontend
npm install
npm start


Backend :
cd backend
python main.py


http://localhost:3000 : pour frontend
http://localhost:8000 : pour backend
http://localhost:5000 : pour mlflow

CI/CD local et détection de drift :
- `make setup` : installe les dépendances Python et lance MLflow UI.
- `make train` : entraîne un modèle Random Forest sur `data/movies_credits_merged.csv` et logge le modèle dans MLflow.
- `make register` : enregistre le meilleur modèle MLflow dans le registry sous `mon_modele_production` et le met en stage `Production`.
- `make serve` : sert le modèle enregistré sur `http://127.0.0.1:1234`.
- `make test` : vérifie que le serveur MLflow répond bien sur `/ping`.
- `make drift` : simule un drift de production, génère un rapport Evidently, exécute un KS-test et déclenche un ré-entraînement si le drift dépasse le seuil.

Hook Git :
- Le hook `.git/hooks/pre-commit` vérifie avant chaque commit que le dernier run MLflow du modèle `Movie_Success_Classification` a une `accuracy > 0.80`.

Remarque : le hook Git doit être rendu exécutable si vous utilisez un shell Unix ou Git Bash.

