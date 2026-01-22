"""
Final Training on FULL 2000-2026 Dataset
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier, VotingClassifier, 
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    VotingRegressor, HistGradientBoostingRegressor, 
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
import joblib
import json
import os
from datetime import datetime

MODELS_DIR = "models"

def train():
    print("=" * 60)
    print("TRAINING ON FULL 2000-2026 DATASET")
    print("=" * 60)
    
    # Load FULL dataset
    df = pd.read_csv("data/training_data.csv")
    print(f"FULL Dataset: {len(df)} games")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    
    exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
               'home_score', 'visitor_score', 'home_won', 'season']
    X = df[[c for c in df.columns if c not in exclude]].fillna(0).select_dtypes(include=[np.number])
    y = df['home_won']
    y_home = df['home_score']
    y_vis = df['visitor_score']
    features = list(X.columns)
    
    print(f"Features: {len(features)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # 80/20 temporal split
    split = int(len(X_scaled) * 0.8)
    X_tr, X_te = X_scaled[:split], X_scaled[split:]
    y_tr, y_te = y.values[:split], y.values[split:]
    y_h_tr, y_h_te = y_home.values[:split], y_home.values[split:]
    y_v_tr, y_v_te = y_vis.values[:split], y_vis.values[split:]
    
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
    
    # 6-model ensemble
    print("\n[1/3] Training classifier ensemble...")
    clf = VotingClassifier([
        ('hgb1', HistGradientBoostingClassifier(
            max_iter=800, max_depth=5, learning_rate=0.025,
            l2_regularization=4.0, min_samples_leaf=60, random_state=42
        )),
        ('hgb2', HistGradientBoostingClassifier(
            max_iter=600, max_depth=6, learning_rate=0.035,
            l2_regularization=3.0, min_samples_leaf=40, random_state=43
        )),
        ('hgb3', HistGradientBoostingClassifier(
            max_iter=500, max_depth=7, learning_rate=0.04,
            l2_regularization=2.5, min_samples_leaf=30, random_state=44
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            min_samples_leaf=30, subsample=0.8, random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=600, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=600, max_depth=14, min_samples_leaf=8,
            n_jobs=-1, random_state=42
        ))
    ], voting='soft', n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    proba = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba)
    
    print(f"\n*** ACCURACY: {acc*100:.1f}% ***")
    print(f"Log Loss: {ll:.4f}")
    
    # Score models
    print("\n[2/3] Training score models...")
    home_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=7, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42))
    ], n_jobs=-1)
    home_ens.fit(X_tr, y_h_tr)
    
    vis_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=7, random_state=43)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=43)),
        ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=43))
    ], n_jobs=-1)
    vis_ens.fit(X_tr, y_v_tr)
    
    h_mae = mean_absolute_error(y_h_te, home_ens.predict(X_te))
    v_mae = mean_absolute_error(y_v_te, vis_ens.predict(X_te))
    print(f"Home MAE: {h_mae:.2f}, Visitor MAE: {v_mae:.2f}")
    
    # Save
    print("\n[3/3] Saving models...")
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
            'data_range': '2000-2026',
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
    
    print("\n" + "=" * 60)
    print(f"FINAL: {acc*100:.1f}% on FULL 2000-2026 DATA")
    print("=" * 60)
    
    return acc

if __name__ == "__main__":
    train()
