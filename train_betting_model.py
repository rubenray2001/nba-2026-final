"""
Betting-Focused Model Training
Trains models to predict AGAINST VEGAS rather than raw outcomes.
Focus: Moneyline, Spread (Regression), and Over/Under (Regression)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.calibration import CalibratedClassifierCV


def load_prebuilt_training_data():
    """Load pre-built training data from train_model.py output (fast path)"""
    path = "data/training_data.csv"
    if not os.path.exists(path):
        return None
    
    print(f"Loading pre-built training data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    
    # Check required columns exist
    required = ['home_score', 'visitor_score', 'home_won', 'date']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Pre-built data missing columns: {missing}")
        return None
    
    # Calculate actual outcomes
    df['actual_spread'] = df['home_score'] - df['visitor_score']
    df['actual_total'] = df['home_score'] + df['visitor_score']
    
    # Filter out games with zero scores
    df = df[(df['home_score'] > 0) & (df['visitor_score'] > 0)].copy()
    
    print(f"Loaded {len(df)} games with complete features")
    return df


def add_vegas_proxy_features(df):
    """
    Create Vegas proxy features for training.
    """
    # Estimate spread based on recent performance
    if 'home_point_diff_last10' in df.columns and 'visitor_point_diff_last10' in df.columns:
        df['est_vegas_spread'] = -(df['home_point_diff_last10'] - df['visitor_point_diff_last10']) / 2 - 2.5
    else:
        df['est_vegas_spread'] = -3.0  # Default home favorite
    
    # Estimate total based on scoring - IMPROVED with pace factors
    if 'home_points_scored_last10' in df.columns:
        h_off = df.get('home_points_scored_last10', 112)
        v_off = df.get('visitor_points_scored_last10', 112)
        h_def = df.get('home_points_allowed_last10', 112)
        v_def = df.get('visitor_points_allowed_last10', 112)
        
        # Estimate: average of offensive and defensive matchups
        est_home = (h_off + v_def) / 2
        est_visitor = (v_off + h_def) / 2
        df['est_vegas_total'] = est_home + est_visitor
        
        # Pace-related features (PREVIOUSLY EXCLUDED - NOW INCLUDED)
        df['combined_offense'] = h_off + v_off
        df['combined_defense'] = h_def + v_def
        df['pace_indicator'] = df['combined_offense'] - 224  # vs league avg
        df['defense_indicator'] = 224 - df['combined_defense']  # vs league avg
        
        # Scoring volatility
        if 'home_points_std_last10' in df.columns:
            df['scoring_volatility'] = df.get('home_points_std_last10', 10) + df.get('visitor_points_std_last10', 10)
        
    else:
        # Defaults
        df['est_vegas_total'] = 224.0
        df['combined_offense'] = 224.0
        df['combined_defense'] = 224.0
        df['pace_indicator'] = 0.0
        df['defense_indicator'] = 0.0
        df['scoring_volatility'] = 20.0
    
    return df


def train_betting_models(df):
    """Train specialized models for each betting type"""
    
    print("\n" + "="*60)
    print("TRAINING BETTING-FOCUSED MODELS (HYBRID)")
    print("="*60)
    
    # Feature columns (exclude targets and ID columns only)
    exclude_cols = [
        'actual_home_score', 'actual_visitor_score', 'actual_spread', 
        'actual_total', 'home_won', 'game_id', 'date',
        'home_covered', 'went_over', 'favorite_won', 'upset',
        'est_vegas_spread', 'est_vegas_total', 'est_home_win_prob',
        # CRITICAL: Exclude raw scores to prevent data leakage
        'home_score', 'visitor_score', 'season',
        'home_team_id', 'visitor_team_id'
        # NOTE: combined_offense, pace_indicator etc are kept!
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
    print(f"\nUsing {len(feature_cols)} features including Pace/Volatility")
    
    # Temporal split
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Fill defaults for Standard Gradient Boosting (Moneyline)
    fill_defaults = {}
    for col in feature_cols:
        if 'elo' in col.lower(): fill_defaults[col] = 1500
        elif 'win_pct' in col.lower(): fill_defaults[col] = 0.5
        elif 'points' in col.lower(): fill_defaults[col] = 110
        else: fill_defaults[col] = 0
    
    # Scaled data for Moneyline (GradientBoostingClassifier needs it)
    X_train = train_df[feature_cols].fillna(fill_defaults)
    X_test = test_df[feature_cols].fillna(fill_defaults)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Raw data for Spread/Totals (HistGradientBoosting handles NaNs)
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    
    models = {}
    results = {}
    
    # ========== MODEL 1: MONEYLINE (Classifier) - UNTOUCHED / ORIGINAL LOGIC ==========
    # Using standard GradientBoostingClassifier + CalibratedClassifierCV
    # This is exactly what was there before
    print("\n--- Training MONEYLINE Model (Original) ---")
    y_train_ml = train_df['home_won']
    y_test_ml = test_df['home_won']
    
    ml_model = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, 
            min_samples_leaf=20, subsample=0.8, random_state=42
        ),
        cv=3, method='isotonic'
    )
    ml_model.fit(X_train_scaled, y_train_ml)
    
    ml_prob = ml_model.predict_proba(X_test_scaled)[:, 1]
    ml_pred = ml_model.predict(X_test_scaled)
    ml_acc = accuracy_score(y_test_ml, ml_pred)
    
    # High confidence metrics (to match original output format)
    high_conf_mask = (ml_prob > 0.65) | (ml_prob < 0.35)
    hc_acc = accuracy_score(y_test_ml[high_conf_mask], ml_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0
    
    print(f"  Moneyline Accuracy: {ml_acc:.1%}")
    print(f"  High Conf Accuracy: {hc_acc:.1%}")
    
    models['moneyline'] = ml_model
    results['moneyline'] = {'accuracy': ml_acc, 'high_conf_accuracy': hc_acc}
    
    # ========== MODEL 2: SPREAD (Regressor) - NEW / FAST ==========
    # Switches to HistGradientBoosting for speed + regression power
    print("\n--- Training SPREAD Model (Reg) ---")
    y_train_spread = train_df['actual_spread']
    y_test_spread = test_df['actual_spread']
    
    spread_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        loss='absolute_error', l2_regularization=1.0, random_state=42
    )
    spread_model.fit(X_train_raw, y_train_spread)
    
    spread_pred = spread_model.predict(X_test_raw)
    spread_mae = mean_absolute_error(y_test_spread, spread_pred)
    print(f"  Spread MAE: {spread_mae:.2f} points")
    
    # Calculate cover accuracy against proxy line for reporting
    pred_cover = (spread_pred + test_df['est_vegas_spread']) > 0
    actual_cover = (y_test_spread + test_df['est_vegas_spread']) > 0
    cover_acc = accuracy_score(actual_cover, pred_cover)
    print(f"  Estimated Cover Accuracy: {cover_acc:.1%}")
    
    models['spread'] = spread_model
    results['spread'] = {'mae': spread_mae, 'accuracy': cover_acc}
    
    # ========== MODEL 3: TOTALS (Regressor) - NEW / FAST ==========
    print("\n--- Training TOTALS Model (Reg) ---")
    y_train_total = train_df['actual_total']
    y_test_total = test_df['actual_total']
    
    total_model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.03,
        loss='absolute_error', l2_regularization=1.0, random_state=42
    )
    total_model.fit(X_train_raw, y_train_total)
    
    total_pred = total_model.predict(X_test_raw)
    total_mae = mean_absolute_error(y_test_total, total_pred)
    print(f"  Total MAE: {total_mae:.2f} points")
    
    # O/U Accuracy against proxy line
    pred_over = total_pred > test_df['est_vegas_total']
    actual_over = y_test_total > test_df['est_vegas_total']
    ou_acc = accuracy_score(actual_over, pred_over)
    print(f"  Estimated O/U Accuracy: {ou_acc:.1%}")
    
    models['totals'] = total_model
    results['totals'] = {'mae': total_mae, 'accuracy': ou_acc}
    
    return models, scaler, feature_cols, results


def save_betting_models(models, scaler, feature_cols, results):
    """Save all betting models"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(models['moneyline'], 'models/betting_moneyline.joblib')
    joblib.dump(models['spread'], 'models/betting_spread.joblib')
    joblib.dump(models['totals'], 'models/betting_totals.joblib')
    joblib.dump(scaler, 'models/betting_scaler.joblib')
    
    metadata = {
        'feature_names': feature_cols,
        'trained_at': datetime.now().isoformat(),
        'results': results,
        'model_type': 'hybrid_regression'
    }
    
    with open('models/betting_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n[OK] Betting models saved (Hybrid Mode)")


def main():
    print("="*60)
    print("BETTING MODEL TRAINING (HYBRID)")
    print("Focus: Moneyline (Original), Spread/Total (Faster Regression)")
    print("="*60)
    
    betting_df = load_prebuilt_training_data()
    if betting_df is None: return
    
    betting_df = add_vegas_proxy_features(betting_df)
    models, scaler, feature_cols, results = train_betting_models(betting_df)
    save_betting_models(models, scaler, feature_cols, results)
    print("\n[DONE] TRAINING COMPLETE")


if __name__ == "__main__":
    main()
