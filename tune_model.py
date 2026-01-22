"""
Hyperparameter Tuning Script
Optimizes RegularizedEnsembleModel using RandomizedSearchCV
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from model_engine_regularized import RegularizedEnsembleModel
import config
import joblib
import os

def load_training_data():
    """Load latest training data"""
    path = os.path.join(config.DATA_DIR, 'training_data.csv')
    if not os.path.exists(path):
        print("Training data not found. Please run train_model.py first to generate it.")
        return None
    return pd.read_csv(path)

def tune_rf(X, y):
    print("\nTuning Random Forest...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, max_features='sqrt')
    
    # Use TimeSeriesSplit to respect data order
    cv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=cv, 
        scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X, y)
    print(f"Best RF Params: {search.best_params_}")
    print(f"Best Score: {-search.best_score_:.4f}")
    return search.best_params_

def tune_gb(X, y):
    print("\nTuning Gradient Boosting...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    cv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        gb, param_dist, n_iter=10, cv=cv, 
        scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X, y)
    print(f"Best GB Params: {search.best_params_}")
    print(f"Best Score: {-search.best_score_:.4f}")
    return search.best_params_

def main():
    print("Optimization Phase: Hyperparameter Tuning")
    
    df = load_training_data()
    if df is None: return
    
    # Initialize basic model to prepare data
    model = RegularizedEnsembleModel()
    X, y_home, y_visitor, y_winner = model.prepare_training_data(df)
    
    # Scale features
    print("Scaling features...")
    X_scaled = model.scaler.fit_transform(X)
    
    # We tune on Home Score primarily as a proxy for general regression performance
    # 1. Tune Random Forest
    rf_params = tune_rf(X_scaled, y_home)
    
    # 2. Tune Gradient Boosting
    gb_params = tune_gb(X_scaled, y_home)
    
    # Construct final params dict
    final_params = {
        'rf': rf_params,
        'gb': gb_params,
        # Keep others default for now
        'et': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 5},
        'ridge': {'alpha': 10.0},
        'logistic': {'C': 0.1}
    }
    
    # Save parameters
    import json
    param_path = os.path.join(config.MODELS_DIR, 'best_hyperparameters.json')
    with open(param_path, 'w') as f:
        json.dump(final_params, f, indent=2)
        
    print(f"\nOptimization Complete. Parameters saved to {param_path}")
    print("Run `train_model.py` (updated to load these params) to verify improvements.")

if __name__ == "__main__":
    main()
