"""
Hyperparameter Grid Search for 70%+ accuracy
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json
import os
from datetime import datetime

MODELS_DIR = "models"

def load_data():
    df = pd.read_csv("data/training_data.csv")
    exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
               'home_score', 'visitor_score', 'home_won', 'season']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0).select_dtypes(include=[np.number])
    return X.values, df['home_won'].values, list(X.columns)

def main():
    print("=" * 50)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 50)
    
    X, y, features = load_data()
    print(f"Samples: {len(X)}, Features: {len(features)}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define param grid
    param_grid = {
        'max_iter': [400, 600, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.02, 0.03, 0.05],
        'max_leaf_nodes': [20, 31, 50],
        'l2_regularization': [1.0, 3.0, 5.0],
        'min_samples_leaf': [30, 50, 80]
    }
    
    # Smaller grid for speed
    param_grid = {
        'max_iter': [500, 700],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.02, 0.04],
        'max_leaf_nodes': [25, 40],
        'l2_regularization': [2.0, 4.0],
        'min_samples_leaf': [40, 60]
    }
    
    base_model = HistGradientBoostingClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nRunning GridSearchCV...")
    grid = GridSearchCV(
        base_model, param_grid, 
        cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    grid.fit(X_scaled, y)
    
    print(f"\nBest CV Score: {grid.best_score_:.4f} ({grid.best_score_*100:.1f}%)")
    print(f"Best Params: {grid.best_params_}")
    
    # Test on holdout
    split = int(len(X_scaled) * 0.8)
    X_tr, X_te = X_scaled[:split], X_scaled[split:]
    y_tr, y_te = y[:split], y[split:]
    
    best_model = grid.best_estimator_
    best_model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, best_model.predict(X_te))
    
    print(f"\nTest Accuracy: {acc*100:.1f}%")
    
    # Save best params
    with open(f"{MODELS_DIR}/best_hyperparameters.json", 'w') as f:
        json.dump({
            'best_params': grid.best_params_,
            'cv_score': float(grid.best_score_),
            'test_accuracy': float(acc)
        }, f, indent=2)
    
    print(f"\nBest params saved to {MODELS_DIR}/best_hyperparameters.json")
    return grid.best_params_, acc

if __name__ == "__main__":
    main()
