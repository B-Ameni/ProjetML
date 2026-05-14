import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_credits_merged.csv'
FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']

def run():
    df = pd.read_csv(data_path)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df['is_success'] = (df['revenue'] > df['budget']).astype(int)
    X = df[FEATURES].fillna(0)
    y = df['is_success']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    random_states = [1, 23, 42, 99, 2024]
    print(f"{'RS':<5} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
    print("-" * 60)
    
    for rs in random_states:
        rf = RandomForestClassifier(random_state=rs, n_estimators=100)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print(f"{rs:<5} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

if __name__ == '__main__':
    run()
