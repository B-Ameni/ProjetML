import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'movies_credits_merged.csv'
FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']

def run():
    df = pd.read_csv(data_path)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    X = df[FEATURES].fillna(0)
    y = df['revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    random_states = [1, 23, 42, 99, 2024]
    print(f"{'RS':<5} | {'MAE':<12} | {'MSE':<18} | {'RMSE':<12} | {'R2':<6}")
    print("-" * 65)
    
    for rs in random_states:
        rf = RandomForestRegressor(random_state=rs, n_estimators=50) # smaller n_estimators for speed
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        print(f"{rs:<5} | {mae:,.0f} | {mse:,.0f} | {rmse:,.0f} | {r2:.4f}")

if __name__ == '__main__':
    run()
