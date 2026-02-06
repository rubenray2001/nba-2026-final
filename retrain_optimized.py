"""
Optimized Model Retraining Script
Addresses overfitting by using proper regularization and temporal validation
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib

from config import MODELS_DIR, DATA_DIR
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from data_manager import DataManager

def train_optimized_model():
    """Train model with optimized regularization to reduce overfitting"""
    
    print("=" * 70)
    print("OPTIMIZED MODEL TRAINING - Reducing Overfitting")
    print("=" * 70)
    
    # Load feature-engineered data
    feature_eng = FeatureEngineer()
    data_mgr = DataManager()
    
    # Only use 2023-2026 seasons (most recent, most relevant)
    seasons = list(range(2023, 2027))
    print(f"\nUsing seasons: {seasons}")
    
    # Fetch data and build features
    print("Fetching training data...")
    all_data = data_mgr.get_complete_training_data(seasons)
    
    print("Building feature dataset...")
    df = feature_eng.build_training_dataset(all_data, seasons)
    print(f"Total samples: {len(df)}")
    
    # Sort by date for proper temporal split
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target variables
    target_cols = ['home_score', 'visitor_score', 'home_won']
    feature_cols = [c for c in df.columns if c not in target_cols + ['date', 'season', 'game_id', 'home_team_id', 'visitor_team_id']]
    
    print(f"Features: {len(feature_cols)}")
    
    # Temporal split: Use last 20% as test (most recent games)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training samples: {len(train_df)} (games before {train_df['date'].max()})")
    print(f"Test samples: {len(test_df)} (games after {test_df['date'].min()})")
    
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
    
    X_train = train_df[feature_cols].fillna(fill_defaults)
    X_test = test_df[feature_cols].fillna(fill_defaults)
    
    y_home_train = train_df['home_score']
    y_home_test = test_df['home_score']
    y_visitor_train = train_df['visitor_score']
    y_visitor_test = test_df['visitor_score']
    y_winner_train = train_df['home_won'].astype(int)
    y_winner_test = test_df['home_won'].astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection - keep top 30 features (more aggressive than before)
    print("\n" + "=" * 60)
    print("FEATURE SELECTION - Top 30 features")
    print("=" * 60)
    
    # Use classifier + winner target for feature selection (not regressor + score)
    # This selects features relevant to predicting wins, not just home scores
    selector_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    selector = SelectFromModel(selector_model, threshold='median', max_features=30)
    selector.fit(X_train_scaled, y_winner_train)
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features")
    print(f"Top features: {selected_features[:10]}")
    
    # STRONGLY REGULARIZED MODELS
    print("\n" + "=" * 60)
    print("TRAINING REGULARIZED ENSEMBLE")
    print("=" * 60)
    
    # Score prediction ensemble - HEAVILY REGULARIZED
    home_score_models = [
        ('rf', RandomForestRegressor(
            n_estimators=50,  # Reduced from 200
            max_depth=5,  # Reduced from 15
            min_samples_split=20,  # Increased from 5
            min_samples_leaf=10,  # Added
            max_features='sqrt',  # Limit features per tree
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=50,  # Reduced from 200
            max_depth=3,  # Reduced from 5
            min_samples_split=20,
            min_samples_leaf=10,
            learning_rate=0.03,  # Reduced from 0.05
            subsample=0.7,
            random_state=42
        )),
        ('hgb', HistGradientBoostingRegressor(
            max_iter=50,
            max_depth=3,
            min_samples_leaf=20,
            learning_rate=0.03,
            l2_regularization=1.0,  # Added L2 regularization
            random_state=42
        )),
        ('ridge', Ridge(alpha=10.0))  # Increased regularization
    ]
    
    home_ensemble = VotingRegressor(home_score_models)
    visitor_ensemble = VotingRegressor(home_score_models)
    
    print("Training home score model...")
    home_ensemble.fit(X_train_selected, y_home_train)
    
    print("Training visitor score model...")
    visitor_ensemble.fit(X_train_selected, y_visitor_train)
    
    # Winner prediction ensemble - HEAVILY REGULARIZED
    winner_models = [
        ('rf', RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            min_samples_split=30,
            min_samples_leaf=15,
            learning_rate=0.02,
            subsample=0.7,
            random_state=42
        )),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=2,
            min_samples_leaf=30,
            learning_rate=0.02,
            l2_regularization=2.0,
            random_state=42
        )),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))  # Low C = high regularization
    ]
    
    winner_ensemble = VotingClassifier(winner_models, voting='soft')
    
    print("Training winner prediction model...")
    winner_ensemble.fit(X_train_selected, y_winner_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Training performance
    home_pred_train = home_ensemble.predict(X_train_selected)
    visitor_pred_train = visitor_ensemble.predict(X_train_selected)
    winner_pred_train = winner_ensemble.predict(X_train_selected)
    
    # Test performance
    home_pred_test = home_ensemble.predict(X_test_selected)
    visitor_pred_test = visitor_ensemble.predict(X_test_selected)
    winner_pred_test = winner_ensemble.predict(X_test_selected)
    
    train_acc = accuracy_score(y_winner_train, winner_pred_train)
    test_acc = accuracy_score(y_winner_test, winner_pred_test)
    overfit_gap = train_acc - test_acc
    
    home_mae_train = mean_absolute_error(y_home_train, home_pred_train)
    home_mae_test = mean_absolute_error(y_home_test, home_pred_test)
    visitor_mae_train = mean_absolute_error(y_visitor_train, visitor_pred_train)
    visitor_mae_test = mean_absolute_error(y_visitor_test, visitor_pred_test)
    
    print(f"\nWinner Prediction:")
    print(f"  Train Accuracy: {train_acc:.1%}")
    print(f"  Test Accuracy:  {test_acc:.1%}")
    print(f"  Overfit Gap:    {overfit_gap:.1%} {'[OK]' if overfit_gap < 0.10 else '[HIGH]'}")
    
    print(f"\nScore Prediction (MAE):")
    print(f"  Home - Train: {home_mae_train:.1f}, Test: {home_mae_test:.1f}")
    print(f"  Visitor - Train: {visitor_mae_train:.1f}, Test: {visitor_mae_test:.1f}")
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(home_ensemble, os.path.join(MODELS_DIR, 'home_score_ensemble.pkl'))
    joblib.dump(visitor_ensemble, os.path.join(MODELS_DIR, 'visitor_score_ensemble.pkl'))
    joblib.dump(winner_ensemble, os.path.join(MODELS_DIR, 'winner_ensemble.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'selector.pkl'))
    
    # Save metadata
    metadata = {
        'feature_names': feature_cols,
        'selected_features': selected_features,
        'training_info': {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(feature_cols),
            'selected_features': len(selected_features),
            'seasons': seasons,
            'metrics': {
                'home_score_test_mae': float(home_mae_test),
                'visitor_score_test_mae': float(visitor_mae_test),
                'home_score_test_r2': float(r2_score(y_home_test, home_pred_test)),
                'visitor_score_test_r2': float(r2_score(y_visitor_test, visitor_pred_test)),
                'winner_train_accuracy': float(train_acc),
                'winner_test_accuracy': float(test_acc),
                'overfit_gap': float(overfit_gap)
            }
        },
        'model_type': 'optimized_regularized_ensemble'
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update training history
    history_path = os.path.join(MODELS_DIR, 'training_history.json')
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        history = []
    
    history.append({
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'total_samples': len(df),
        'features': len(selected_features),
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc),
        'home_score_mae': float(home_mae_test),
        'visitor_score_mae': float(visitor_mae_test),
        'overfit_gap': float(overfit_gap),
        'model_type': 'optimized_regularized'
    })
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[SAVED] Models saved to {MODELS_DIR}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n>>> Test Accuracy: {test_acc:.1%}")
    print(f">>> Score MAE: {(home_mae_test + visitor_mae_test) / 2:.1f} pts")
    print(f">>> Overfit Gap: {overfit_gap:.1%}")
    
    if overfit_gap < 0.10:
        print("\n[SUCCESS] Model is well-regularized!")
    else:
        print("\n[WARNING] Consider further regularization if accuracy drops.")
    
    return test_acc, overfit_gap

if __name__ == "__main__":
    train_optimized_model()
