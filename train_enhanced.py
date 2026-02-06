"""
Enhanced Model Training with Vegas Integration
Key improvements:
1. Uses Vegas odds as features (strongest single predictor)
2. Proper feature selection to reduce noise
3. Better regularization 
4. Calibrated probability outputs
5. Focus on high-confidence picks
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, brier_score_loss, log_loss
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, DATA_DIR, TRAINING_SEASONS
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from data_manager import DataManager

def compute_calibration_stats(y_true, y_prob, n_bins=10):
    """Compute calibration curve statistics"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted = y_prob[mask].mean()
            actual_rate = y_true[mask].mean()
            count = mask.sum()
            calibration_data.append({
                'bin': i,
                'mean_predicted': mean_predicted,
                'actual_rate': actual_rate,
                'count': count,
                'gap': abs(mean_predicted - actual_rate)
            })
    
    return pd.DataFrame(calibration_data)


def train_enhanced_model():
    """Train model with Vegas integration and better calibration"""
    
    print("=" * 70)
    print("ENHANCED MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    feature_eng = FeatureEngineer()
    data_mgr = DataManager()
    
    # Use recent seasons only (2022-2026)
    seasons = [2022, 2023, 2024, 2025, 2026]
    print(f"\nUsing seasons: {seasons}")
    
    # Fetch data
    print("Fetching training data...")
    all_data = data_mgr.get_complete_training_data(seasons)
    
    print("Building feature dataset...")
    df = feature_eng.build_training_dataset(all_data, seasons)
    print(f"Total samples: {len(df)}")
    
    # Sort by date for temporal split
    df = df.sort_values('date').reset_index(drop=True)
    
    # Identify feature columns
    exclude_cols = ['game_id', 'date', 'season', 'home_team_id', 'visitor_team_id',
                   'home_score', 'visitor_score', 'home_won']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Feature importance analysis - reduce redundancy
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Identify highly correlated features
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
    
    X_all = df[feature_cols].fillna(fill_defaults)
    corr_matrix = X_all.corr().abs()
    
    # Find pairs with correlation > 0.9
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, corr_matrix.columns[i]) 
                       for i, col in enumerate(upper_tri.columns) 
                       for j, val in enumerate(upper_tri[col]) 
                       if val > 0.9]
    
    print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.9)")
    
    # Remove redundant features (keep first in each pair)
    features_to_drop = set()
    for f1, f2 in high_corr_pairs:
        if f1 not in features_to_drop:
            features_to_drop.add(f2)
    
    reduced_features = [f for f in feature_cols if f not in features_to_drop]
    print(f"Reduced from {len(feature_cols)} to {len(reduced_features)} features")
    
    # Temporal split (last 20% for test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTraining: {len(train_df)} games (up to {train_df['date'].max().date()})")
    print(f"Testing: {len(test_df)} games (from {test_df['date'].min().date()})")
    
    X_train = train_df[reduced_features].fillna(fill_defaults)
    X_test = test_df[reduced_features].fillna(fill_defaults)
    
    y_home_train = train_df['home_score']
    y_home_test = test_df['home_score']
    y_visitor_train = train_df['visitor_score']
    y_visitor_test = test_df['visitor_score']
    y_winner_train = train_df['home_won'].astype(int)
    y_winner_test = test_df['home_won'].astype(int)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature Selection - Top 40 most informative
    print("\n" + "=" * 60)
    print("FEATURE SELECTION (Top 40)")
    print("=" * 60)
    
    selector = SelectKBest(score_func=mutual_info_classif, k=min(40, len(reduced_features)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_winner_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(reduced_features, selected_mask) if s]
    print(f"Selected {len(selected_features)} features")
    
    # Show top features by importance
    feature_scores = pd.DataFrame({
        'feature': reduced_features,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    print("\nTop 10 predictive features:")
    for _, row in feature_scores.head(10).iterrows():
        print(f"  {row['feature']}: {row['score']:.4f}")
    
    # ========================================
    # TRAIN MODELS
    # ========================================
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    # Winner prediction - use calibrated classifier for better probabilities
    base_winner_models = [
        ('hgb1', HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=42
        )),
        ('hgb2', HistGradientBoostingClassifier(
            max_iter=150, max_depth=3, learning_rate=0.03,
            min_samples_leaf=30, l2_regularization=2.0,
            random_state=43
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=15,
            random_state=44, n_jobs=-1
        )),
        ('lr', LogisticRegression(C=0.5, max_iter=500, random_state=45))
    ]
    
    winner_ensemble = VotingClassifier(base_winner_models, voting='soft')
    
    print("Training winner classifier...")
    winner_ensemble.fit(X_train_selected, y_winner_train)
    
    # Calibrate for better probability estimates
    print("Calibrating probabilities...")
    calibrated_winner = CalibratedClassifierCV(winner_ensemble, cv=3, method='isotonic')
    calibrated_winner.fit(X_train_selected, y_winner_train)
    
    # Score prediction models
    score_models = [
        ('hgb', HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10,
            random_state=43, n_jobs=-1
        )),
        ('ridge', Ridge(alpha=5.0))
    ]
    
    home_score_ensemble = VotingRegressor(score_models)
    visitor_score_ensemble = VotingRegressor(score_models)
    
    print("Training home score model...")
    home_score_ensemble.fit(X_train_selected, y_home_train)
    
    print("Training visitor score model...")
    visitor_score_ensemble.fit(X_train_selected, y_visitor_train)
    
    # ========================================
    # EVALUATION
    # ========================================
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Predictions
    winner_pred_train = calibrated_winner.predict(X_train_selected)
    winner_pred_test = calibrated_winner.predict(X_test_selected)
    winner_prob_train = calibrated_winner.predict_proba(X_train_selected)[:, 1]
    winner_prob_test = calibrated_winner.predict_proba(X_test_selected)[:, 1]
    
    home_pred_train = home_score_ensemble.predict(X_train_selected)
    home_pred_test = home_score_ensemble.predict(X_test_selected)
    visitor_pred_train = visitor_score_ensemble.predict(X_train_selected)
    visitor_pred_test = visitor_score_ensemble.predict(X_test_selected)
    
    # Accuracy
    train_acc = accuracy_score(y_winner_train, winner_pred_train)
    test_acc = accuracy_score(y_winner_test, winner_pred_test)
    
    # Brier score (calibration quality)
    train_brier = brier_score_loss(y_winner_train, winner_prob_train)
    test_brier = brier_score_loss(y_winner_test, winner_prob_test)
    
    # Log loss
    train_logloss = log_loss(y_winner_train, winner_prob_train)
    test_logloss = log_loss(y_winner_test, winner_prob_test)
    
    # Score MAE
    home_mae_train = mean_absolute_error(y_home_train, home_pred_train)
    home_mae_test = mean_absolute_error(y_home_test, home_pred_test)
    visitor_mae_train = mean_absolute_error(y_visitor_train, visitor_pred_train)
    visitor_mae_test = mean_absolute_error(y_visitor_test, visitor_pred_test)
    
    print("\n--- WINNER PREDICTION ---")
    print(f"Train Accuracy: {train_acc:.1%}")
    print(f"Test Accuracy:  {test_acc:.1%}")
    print(f"Overfit Gap:    {(train_acc - test_acc):.1%}")
    print(f"Train Brier:    {train_brier:.4f}")
    print(f"Test Brier:     {test_brier:.4f} (lower is better, 0.25 = random)")
    print(f"Test Log Loss:  {test_logloss:.4f}")
    
    print("\n--- SCORE PREDICTION ---")
    print(f"Home MAE  - Train: {home_mae_train:.1f}, Test: {home_mae_test:.1f}")
    print(f"Visitor MAE - Train: {visitor_mae_train:.1f}, Test: {visitor_mae_test:.1f}")
    
    # Calibration analysis
    print("\n--- CALIBRATION ANALYSIS ---")
    cal_stats = compute_calibration_stats(y_winner_test.values, winner_prob_test)
    print("Bin | Predicted | Actual | Count | Gap")
    print("-" * 50)
    for _, row in cal_stats.iterrows():
        print(f"{row['bin']:3.0f} | {row['mean_predicted']:.1%}     | {row['actual_rate']:.1%}  | {row['count']:5.0f} | {row['gap']:.1%}")
    
    avg_calibration_gap = cal_stats['gap'].mean()
    print(f"\nAverage Calibration Gap: {avg_calibration_gap:.1%}")
    
    # High confidence performance
    print("\n--- HIGH CONFIDENCE PICKS ---")
    high_conf_mask = (winner_prob_test >= 0.60) | (winner_prob_test <= 0.40)
    if high_conf_mask.sum() > 0:
        high_conf_actual = y_winner_test[high_conf_mask]
        high_conf_pred = winner_pred_test[high_conf_mask]
        high_conf_acc = accuracy_score(high_conf_actual, high_conf_pred)
        print(f"Games with >60% confidence: {high_conf_mask.sum()} ({high_conf_mask.sum()/len(y_winner_test):.1%})")
        print(f"Accuracy on high-conf picks: {high_conf_acc:.1%}")
    
    very_high_conf_mask = (winner_prob_test >= 0.65) | (winner_prob_test <= 0.35)
    if very_high_conf_mask.sum() > 0:
        very_high_actual = y_winner_test[very_high_conf_mask]
        very_high_pred = winner_pred_test[very_high_conf_mask]
        very_high_acc = accuracy_score(very_high_actual, very_high_pred)
        print(f"Games with >65% confidence: {very_high_conf_mask.sum()} ({very_high_conf_mask.sum()/len(y_winner_test):.1%})")
        print(f"Accuracy on very-high-conf: {very_high_acc:.1%}")
    
    # ========================================
    # SAVE MODELS
    # ========================================
    
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the calibrated winner model and individual score models
    joblib.dump(home_score_ensemble, os.path.join(MODELS_DIR, 'home_score_ensemble.pkl'))
    joblib.dump(visitor_score_ensemble, os.path.join(MODELS_DIR, 'visitor_score_ensemble.pkl'))
    joblib.dump(calibrated_winner, os.path.join(MODELS_DIR, 'winner_ensemble.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'selector.pkl'))
    
    # Save metadata
    metadata = {
        'feature_names': reduced_features,  # Features before selection
        'selected_features': selected_features,
        'training_info': {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(reduced_features),
            'selected_features': len(selected_features),
            'seasons': seasons,
            'metrics': {
                'home_score_test_mae': float(home_mae_test),
                'visitor_score_test_mae': float(visitor_mae_test),
                'home_score_test_r2': float(r2_score(y_home_test, home_pred_test)),
                'visitor_score_test_r2': float(r2_score(y_visitor_test, visitor_pred_test)),
                'winner_train_accuracy': float(train_acc),
                'winner_test_accuracy': float(test_acc),
                'overfit_gap': float(train_acc - test_acc),
                'brier_score': float(test_brier),
                'log_loss': float(test_logloss),
                'calibration_gap': float(avg_calibration_gap)
            }
        },
        'model_type': 'enhanced_calibrated_ensemble'
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
        'overfit_gap': float(train_acc - test_acc),
        'brier_score': float(test_brier),
        'model_type': 'enhanced_calibrated'
    })
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModels saved to {MODELS_DIR}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTest Accuracy: {test_acc:.1%}")
    print(f"Brier Score:   {test_brier:.4f} (calibration quality)")
    print(f"Score MAE:     {(home_mae_test + visitor_mae_test)/2:.1f} pts")
    print(f"Overfit Gap:   {(train_acc - test_acc):.1%}")
    
    if test_acc >= 0.65:
        print("\n[GOOD] Model performing above baseline!")
    elif test_acc >= 0.60:
        print("\n[OK] Model performing reasonably well")
    else:
        print("\n[NOTE] NBA prediction is inherently difficult - 60%+ is competitive")
    
    return test_acc, test_brier


if __name__ == "__main__":
    train_enhanced_model()
