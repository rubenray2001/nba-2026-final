"""
Betting-Focused Model Training for College Basketball
Trains models to predict Moneyline, Spread, and O/U
"""
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
import config


def load_prebuilt_training_data(gender: str):
    """Load pre-built training data"""
    path = os.path.join(config.DATA_DIR, gender, "training_data.csv")
    if not os.path.exists(path):
        print(f"No training data found at {path}. Run train_model.py first.")
        return None
    
    df = pd.read_csv(path, low_memory=False)
    required = ['home_score', 'visitor_score', 'home_won', 'date']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return None
    
    df['actual_spread'] = df['home_score'] - df['visitor_score']
    df['actual_total'] = df['home_score'] + df['visitor_score']
    df = df[(df['home_score'] > 0) & (df['visitor_score'] > 0)].copy()
    
    print(f"Loaded {len(df)} {gender} games")
    return df


def add_vegas_proxy_features(df, gender):
    """Create Vegas proxy features for training"""
    avg_total = config.GENDER_CONFIG[gender]["avg_total"]
    
    if 'home_point_diff_last10' in df.columns and 'visitor_point_diff_last10' in df.columns:
        df['est_vegas_spread'] = -(df['home_point_diff_last10'] - df['visitor_point_diff_last10']) / 2 - 3.0
    else:
        df['est_vegas_spread'] = -3.0
    
    if 'home_points_scored_last10' in df.columns:
        h_off = df.get('home_points_scored_last10', avg_total / 2)
        v_off = df.get('visitor_points_scored_last10', avg_total / 2)
        h_def = df.get('home_points_allowed_last10', avg_total / 2)
        v_def = df.get('visitor_points_allowed_last10', avg_total / 2)
        
        est_home = (h_off + v_def) / 2
        est_visitor = (v_off + h_def) / 2
        df['est_vegas_total'] = est_home + est_visitor
        df['combined_offense'] = h_off + v_off
        df['combined_defense'] = h_def + v_def
        df['pace_indicator'] = df['combined_offense'] - avg_total
        df['defense_indicator'] = avg_total - df['combined_defense']
    else:
        df['est_vegas_total'] = avg_total
        df['combined_offense'] = avg_total
        df['combined_defense'] = avg_total
        df['pace_indicator'] = 0.0
        df['defense_indicator'] = 0.0
    
    return df


def train_betting_models(df, gender):
    """Train specialized betting models"""
    print(f"\n{'='*60}")
    print(f"TRAINING {gender.upper()} BETTING MODELS")
    print(f"{'='*60}")
    
    avg_score = config.GENDER_CONFIG[gender]["avg_score"]
    
    exclude_cols = [
        'actual_home_score', 'actual_visitor_score', 'actual_spread',
        'actual_total', 'home_won', 'game_id', 'date',
        'est_vegas_spread', 'est_vegas_total',
        'home_score', 'visitor_score', 'season',
        'home_team_id', 'visitor_team_id'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    print(f"\nUsing {len(feature_cols)} features")
    
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    fill_defaults = {}
    for col in feature_cols:
        if 'elo' in col.lower(): fill_defaults[col] = 1500
        elif 'win_pct' in col.lower(): fill_defaults[col] = 0.5
        elif 'points' in col.lower(): fill_defaults[col] = avg_score
        else: fill_defaults[col] = 0
    
    X_train = train_df[feature_cols].fillna(fill_defaults)
    X_test = test_df[feature_cols].fillna(fill_defaults)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    
    models = {}
    results = {}
    
    # Moneyline
    print("\n--- MONEYLINE ---")
    y_train_ml = train_df['home_won']
    y_test_ml = test_df['home_won']
    
    ml_model = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42
        ), cv=3, method='isotonic'
    )
    ml_model.fit(X_train_scaled, y_train_ml)
    
    ml_prob = ml_model.predict_proba(X_test_scaled)[:, 1]
    ml_pred = ml_model.predict(X_test_scaled)
    ml_acc = accuracy_score(y_test_ml, ml_pred)
    
    high_conf_mask = (ml_prob > 0.65) | (ml_prob < 0.35)
    hc_acc = accuracy_score(y_test_ml[high_conf_mask], ml_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0
    
    print(f"  Accuracy: {ml_acc:.1%} | High Conf: {hc_acc:.1%}")
    models['moneyline'] = ml_model
    results['moneyline'] = {'accuracy': float(ml_acc), 'high_conf_accuracy': float(hc_acc)}
    
    # Spread
    print("\n--- SPREAD ---")
    spread_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        loss='absolute_error', l2_regularization=1.0, random_state=42
    )
    spread_model.fit(X_train_raw, train_df['actual_spread'])
    spread_pred = spread_model.predict(X_test_raw)
    spread_mae = mean_absolute_error(test_df['actual_spread'], spread_pred)
    
    pred_cover = (spread_pred + test_df['est_vegas_spread']) > 0
    actual_cover = (test_df['actual_spread'] + test_df['est_vegas_spread']) > 0
    cover_acc = accuracy_score(actual_cover, pred_cover)
    
    print(f"  MAE: {spread_mae:.2f} | Cover: {cover_acc:.1%}")
    models['spread'] = spread_model
    results['spread'] = {'mae': float(spread_mae), 'confident_accuracy': float(cover_acc)}
    
    # Totals
    print("\n--- TOTALS ---")
    total_model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.03,
        loss='absolute_error', l2_regularization=1.0, random_state=42
    )
    total_model.fit(X_train_raw, train_df['actual_total'])
    total_pred = total_model.predict(X_test_raw)
    total_mae = mean_absolute_error(test_df['actual_total'], total_pred)
    
    pred_over = total_pred > test_df['est_vegas_total']
    actual_over = test_df['actual_total'] > test_df['est_vegas_total']
    ou_acc = accuracy_score(actual_over, pred_over)
    
    print(f"  MAE: {total_mae:.2f} | O/U: {ou_acc:.1%}")
    models['totals'] = total_model
    results['totals'] = {'mae': float(total_mae), 'confident_accuracy': float(ou_acc)}
    
    return models, scaler, feature_cols, results


def save_betting_models(models, scaler, feature_cols, results, gender):
    """Save betting models"""
    models_dir = os.path.join(config.MODELS_DIR, gender)
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(models['moneyline'], os.path.join(models_dir, 'betting_moneyline.joblib'))
    joblib.dump(models['spread'], os.path.join(models_dir, 'betting_spread.joblib'))
    joblib.dump(models['totals'], os.path.join(models_dir, 'betting_totals.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'betting_scaler.joblib'))
    
    metadata = {
        'feature_names': feature_cols,
        'trained_at': datetime.now().isoformat(),
        'results': results,
        'model_type': 'hybrid_regression',
        'gender': gender
    }
    with open(os.path.join(models_dir, 'betting_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] {gender} betting models saved")


def train_gender(gender):
    """Train betting models for a gender"""
    df = load_prebuilt_training_data(gender)
    if df is None:
        return
    
    df = add_vegas_proxy_features(df, gender)
    models, scaler, feature_cols, results = train_betting_models(df, gender)
    save_betting_models(models, scaler, feature_cols, results, gender)


def main():
    if len(sys.argv) > 1:
        gender = sys.argv[1].lower()
        if gender in ['mens', 'womens']:
            train_gender(gender)
        else:
            print(f"Unknown gender: {gender}")
    else:
        for gender in ['mens', 'womens']:
            train_gender(gender)
            print()
    
    print("\n[DONE] BETTING MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
