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
http://localhost:3000 : pour backend
http://localhost:5000 : pour mlflow

