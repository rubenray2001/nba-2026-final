"""
Final optimized training with tuned hyperparameters
Uses best params from grid search + multiple diverse models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
import joblib
import os
import json
from datetime import datetime

MODELS_DIR = "models"

def load_data():
    df = pd.read_csv("data/training_data.csv")
    exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
               'home_score', 'visitor_score', 'home_won', 'season']
    feature_cols = [c for c in df.columns if c not in exclude]
    # Smart NaN filling
    fill_defaults = {}
    for col in feature_cols:
        if 'elo' in col.lower(): fill_defaults[col] = 1500
        elif 'win_pct' in col.lower() or 'prob' in col.lower(): fill_defaults[col] = 0.5
        elif 'efg_pct' in col.lower(): fill_defaults[col] = 0.54
        elif 'points_scored' in col.lower() or 'points_allowed' in col.lower(): fill_defaults[col] = 110
        elif 'pace' in col.lower(): fill_defaults[col] = 98
        elif 'rest_days' in col.lower(): fill_defaults[col] = 2
        elif 'vegas_total' in col.lower(): fill_defaults[col] = 220.0
        elif 'vegas_implied' in col.lower(): fill_defaults[col] = 0.5
        else: fill_defaults[col] = 0
    X = df[feature_cols].fillna(fill_defaults).select_dtypes(include=[np.number])
    return X, df['home_won'], df['home_score'], df['visitor_score'], list(X.columns)

def train():
    print("=" * 50)
    print("FINAL ENSEMBLE TRAINING")
    print("=" * 50)
    
    X, y, y_home, y_vis, features = load_data()
    print(f"Samples: {len(X)}, Features: {len(features)}")
    
    # Split
    split = int(len(X) * 0.8)
    X_tr_raw, X_te_raw = X.values[:split], X.values[split:]
    y_tr, y_te = y.values[:split], y.values[split:]
    y_h_tr, y_h_te = y_home.values[:split], y_home.values[split:]
    y_v_tr, y_v_te = y_vis.values[:split], y_vis.values[split:]
    
    # Scale (Fit on Train ONLY to avoid leakage)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
    
    # Best params from grid search + diverse variants
    print("\n[1/3] Training classifier ensemble...")
    clf = VotingClassifier([
        # Best from grid search
        ('hgb_best', HistGradientBoostingClassifier(
            max_iter=500, max_depth=5, learning_rate=0.04,
            max_leaf_nodes=25, l2_regularization=4.0,
            min_samples_leaf=60, random_state=42
        )),
        # Deeper variant
        ('hgb_deep', HistGradientBoostingClassifier(
            max_iter=600, max_depth=7, learning_rate=0.03,
            max_leaf_nodes=40, l2_regularization=3.0,
            min_samples_leaf=40, random_state=43
        )),
        # Shallower variant
        ('hgb_shallow', HistGradientBoostingClassifier(
            max_iter=800, max_depth=4, learning_rate=0.02,
            max_leaf_nodes=15, l2_regularization=5.0,
            min_samples_leaf=80, random_state=44
        )),
        # GradientBoosting (different algo)
        ('gb', GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            min_samples_leaf=20, subsample=0.8, random_state=42
        )),
        # RandomForest (bagging, decorrelated)
        ('rf', RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=8,
            max_features='sqrt', n_jobs=-1, random_state=42
        )),
        # ExtraTrees (more randomization)
        ('et', ExtraTreesClassifier(
            n_estimators=500, max_depth=15, min_samples_leaf=5,
            max_features='sqrt', n_jobs=-1, random_state=42
        ))
    ], voting='soft', n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba)
    
    print(f"\n*** TEST ACCURACY: {acc*100:.1f}% ***")
    print(f"Log Loss: {ll:.4f}")
    
    # Score predictors
    print("\n[2/3] Training score models...")
    home_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=6, l2_regularization=3.0, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42))
    ], n_jobs=-1)
    home_ens.fit(X_tr, y_h_tr)
    
    vis_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=6, l2_regularization=3.0, random_state=43)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=43)),
        ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=43))
    ], n_jobs=-1)
    vis_ens.fit(X_tr, y_v_tr)
    
    h_mae = mean_absolute_error(y_h_te, home_ens.predict(X_te))
    v_mae = mean_absolute_error(y_v_te, vis_ens.predict(X_te))
    print(f"Home MAE: {h_mae:.2f}, Visitor MAE: {v_mae:.2f}")
    
    # Save
    print("\n[3/3] Saving...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, f"{MODELS_DIR}/winner_ensemble.pkl")
    joblib.dump(home_ens, f"{MODELS_DIR}/home_score_ensemble.pkl")
    joblib.dump(vis_ens, f"{MODELS_DIR}/visitor_score_ensemble.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    
    meta = {
        'feature_names': features,
        'training_info': {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_tr),
            'test_samples': len(X_te),
            'features': len(features),
            'metrics': {
                'winner_test_accuracy': float(acc),
                'winner_test_logloss': float(ll),
                'home_score_test_mae': float(h_mae),
                'visitor_score_test_mae': float(v_mae)
            }
        }
    }
    with open(f"{MODELS_DIR}/model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"FINAL RESULT: {acc*100:.1f}%")
    if acc >= 0.70: print("🎉 70%+ ACHIEVED!")
    elif acc >= 0.67: print("📈 VERY CLOSE!")
    elif acc >= 0.65: print("📈 GOOD PROGRESS!")
    print("=" * 50)
    
    return acc

if __name__ == "__main__":
    train()
