"""
Hyperparameter Tuning Script for Elite Ensemble (XGBoost/CatBoost)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score
import config
import os
import joblib
import json

def load_training_data():
    """Load latest training data"""
    path = os.path.join(config.DATA_DIR, 'training_data.csv')
    if not os.path.exists(path):
        print("Training data not found. Please run train_model.py first to generate it.")
        return None
    return pd.read_csv(path)

def tune_xgboost_regressor(X, y, name="HomeScore"):
    print(f"\nTuning XGBoost Regressor for {name}...")
    
    # Define parameter grid
    param_dist = {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0.1, 1, 10, 50]
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_jobs=-1,
        random_state=42
    )
    
    cv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        xgb_model, param_dist, n_iter=20, cv=cv, 
        scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X, y)
    print(f"Best XGB Params: {search.best_params_}")
    print(f"Best MAE: {-search.best_score_:.4f}")
    return search.best_params_

def tune_catboost_regressor(X, y, name="HomeScore"):
    print(f"\nTuning CatBoost Regressor for {name}...")
    
    # Native CatBoost tuning
    grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'random_strength': [1, 5]
    }
    
    cb_model = CatBoostRegressor(
        iterations=500,
        loss_function='MAE',
        verbose=0,
        allow_writing_files=False,
        random_state=42
    )
    
    # Simple randomized search using CatBoost's method
    # Note: randomized_search in catboost takes X, y as pool or separate
    print("Running CatBoost randomized_search...")
    
    randomized_search_result = cb_model.randomized_search(
        grid,
        X=X,
        y=y,
        n_iter=10,
        partition_random_seed=42,
        calc_cv_statistics=True,
        search_by_train_test_split=True, # Use train-test split for speed
        verbose=False,
        plot=False
    )
    
    best_params = randomized_search_result['params']
    print(f"Best CatBoost Params: {best_params}")
    return best_params

def tune_xgboost_classifier(X, y):
    print("\nTuning XGBoost Classifier for Winner...")
    
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    cv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        xgb_model, param_dist, n_iter=15, cv=cv, 
        scoring='accuracy', n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X, y)
    print(f"Best XGB Class Params: {search.best_params_}")
    print(f"Best Accuracy: {search.best_score_:.4f}")
    return search.best_params_

def tune_catboost_classifier(X, y):
    print("\nTuning CatBoost Classifier for Winner...")
    
    grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 9],
        'random_strength': [1, 5]
    }
    
    cb_model = CatBoostClassifier(
        iterations=500,
        loss_function='Logloss',
        verbose=0,
        allow_writing_files=False,
        random_state=42
    )
    
    print("Running CatBoost randomized_search...")
    
    randomized_search_result = cb_model.randomized_search(
        grid,
        X=X,
        y=y,
        n_iter=10,
        partition_random_seed=42,
        calc_cv_statistics=True,
        search_by_train_test_split=True,
        verbose=False,
        plot=False
    )
    
    best_params = randomized_search_result['params']
    print(f"Best CatBoost Class Params: {best_params}")
    return best_params

def main():
    print("ELITE MODEL HYPERPARAMETER TUNING (XGBoost & CatBoost)")
    
    df = load_training_data()
    if df is None: return
    
    # Prepare data (copy logic from EliteEnsembleModel)
    exclude_cols = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
                   'home_score', 'visitor_score', 'home_won', 'season']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    X.fillna(0, inplace=True)
    
    y_home = df['home_score']
    y_winner = df['home_won']
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols) # Keep as DF for CatBoost potentially? No, array is fine.
    
    # 1. Tune Score Regressors (Proxy: Home Score)
    xgb_reg_params = tune_xgboost_regressor(X_scaled, y_home)
    cb_reg_params = tune_catboost_regressor(X_scaled, y_home)
    
    # 2. Tune Winner Classifiers
    xgb_class_params = tune_xgboost_classifier(X_scaled, y_winner)
    cb_class_params = tune_catboost_classifier(X_scaled, y_winner)
    
    final_params = {
        'xgb_reg': xgb_reg_params,
        'cb_reg': cb_reg_params,
        'xgb_class': xgb_class_params,
        'cb_class': cb_class_params
    }
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    param_path = os.path.join(config.MODELS_DIR, 'elite_hyperparameters.json')
    with open(param_path, 'w') as f:
        json.dump(final_params, f, indent=2)
        
    print(f"\nOptimization Complete. Parameters saved to {param_path}")

if __name__ == "__main__":
    main()
