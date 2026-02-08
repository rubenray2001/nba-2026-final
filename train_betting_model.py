"""
Betting-Focused Model Training
Trains models to predict AGAINST VEGAS rather than raw outcomes.
Focus: Moneyline, Spread, and Over/Under accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

from data_manager import DataManager
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from odds_api_client import TheOddsAPIClient


def fetch_historical_odds(seasons=[2024, 2025]):
    """Fetch historical Vegas odds for training"""
    print("Fetching historical odds data...")
    odds_client = TheOddsAPIClient()
    
    # We'll use recent games where we have odds
    all_odds = []
    
    # Get odds for recent dates
    for days_ago in range(1, 60):  # Last 60 days
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        try:
            odds = odds_client.get_odds()
            if odds:
                for game in odds:
                    game['fetch_date'] = date
                    all_odds.append(game)
        except Exception:
            pass
    
    return all_odds


def prepare_betting_data(games_df, historical_data, feature_eng):
    """Prepare training data with betting outcomes"""
    
    print(f"Processing {len(games_df)} games for betting outcomes...")
    
    records = []
    
    for idx, game in games_df.iterrows():
        try:
            # Only use completed games
            if game.get('status') != 'Final':
                continue
            
            home_score = game.get('home_team_score', 0)
            visitor_score = game.get('visitor_team_score', 0)
            
            if not home_score or not visitor_score:
                continue
            
            # Build features
            features = feature_eng.build_features_for_game(
                game.to_dict() if hasattr(game, 'to_dict') else game,
                historical_data,
                game.get('season', 2024)
            )
            
            if features is None:
                continue
            
            # Calculate actual outcomes
            actual_spread = home_score - visitor_score  # Positive = home won by X
            actual_total = home_score + visitor_score
            home_won = 1 if home_score > visitor_score else 0
            
            # Add outcomes to features
            features['actual_home_score'] = home_score
            features['actual_visitor_score'] = visitor_score
            features['actual_spread'] = actual_spread
            features['actual_total'] = actual_total
            features['home_won'] = home_won
            features['game_id'] = game.get('id')
            features['date'] = game.get('date')
            
            records.append(features)
            
        except Exception as e:
            continue
    
    print(f"Prepared {len(records)} games with complete data")
    return pd.DataFrame(records)


def add_vegas_proxy_features(df):
    """
    Since we don't have historical Vegas lines, create proxy features
    that approximate what Vegas would set based on team stats
    """
    
    # Estimate spread based on team strength difference
    # Vegas typically uses power ratings
    if 'home_point_diff_last10' in df.columns and 'visitor_point_diff_last10' in df.columns:
        df['est_vegas_spread'] = -(df['home_point_diff_last10'] - df['visitor_point_diff_last10']) / 2 - 2.5
    else:
        df['est_vegas_spread'] = -3.0  # Default home favorite
    
    # Estimate total based on scoring - IMPROVED with pace factors
    if 'home_points_scored_last10' in df.columns and 'visitor_points_scored_last10' in df.columns:
        h_off = df.get('home_points_scored_last10', 112)
        v_off = df.get('visitor_points_scored_last10', 112)
        h_def = df.get('home_points_allowed_last10', 112)
        v_def = df.get('visitor_points_allowed_last10', 112)
        
        # Estimate: average of offensive and defensive matchups
        est_home = (h_off + v_def) / 2
        est_visitor = (v_off + h_def) / 2
        df['est_vegas_total'] = est_home + est_visitor
        
        # NEW: Add pace-related features for O/U model
        df['combined_offense'] = h_off + v_off
        df['combined_defense'] = h_def + v_def
        df['pace_indicator'] = df['combined_offense'] - 224  # vs league avg
        df['defense_indicator'] = 224 - df['combined_defense']  # vs league avg
        
        # Scoring variance (high variance = harder to predict)
        if 'home_points_std_last10' in df.columns:
            df['scoring_volatility'] = df.get('home_points_std_last10', 10) + df.get('visitor_points_std_last10', 10)
        
    else:
        df['est_vegas_total'] = 224.0  # League average
        df['combined_offense'] = 224.0
        df['combined_defense'] = 224.0
        df['pace_indicator'] = 0.0
        df['defense_indicator'] = 0.0
    
    # Win probability proxy based on ELO
    if 'elo_diff' in df.columns:
        # Convert ELO diff to probability
        df['est_home_win_prob'] = 1 / (1 + 10 ** (-df['elo_diff'] / 400))
    else:
        df['est_home_win_prob'] = 0.55  # Slight home advantage
    
    return df


def create_betting_targets(df):
    """Create targets for betting models"""
    
    # Target 1: Did home team cover the spread?
    # Spread convention: negative = home favored (e.g., -5.5 means home gives 5.5)
    # Home covers when: actual_margin + spread > 0
    # e.g., home -5.5, wins by 8: 8 + (-5.5) = 2.5 > 0 → covered ✓
    # e.g., home -5.5, wins by 3: 3 + (-5.5) = -2.5 < 0 → didn't cover ✓
    df['home_covered'] = ((df['actual_spread'] + df['est_vegas_spread']) > 0).astype(int)
    
    # Target 2: Did game go over estimated total?
    df['went_over'] = (df['actual_total'] > df['est_vegas_total']).astype(int)
    
    # Target 3: Did favorite win? (Moneyline)
    df['favorite_won'] = np.where(
        df['est_home_win_prob'] > 0.5,
        df['home_won'],  # Home was favorite
        1 - df['home_won']  # Visitor was favorite
    )
    
    # Target 4: Upset detection (underdog won)
    df['upset'] = np.where(
        df['est_home_win_prob'] > 0.5,
        1 - df['home_won'],  # Home favorite lost
        df['home_won']  # Visitor favorite lost
    )
    
    return df


def train_betting_models(df):
    """Train specialized models for each betting type"""
    
    print("\n" + "="*60)
    print("TRAINING BETTING-FOCUSED MODELS")
    print("="*60)
    
    # Feature columns (exclude targets and identifiers)
    exclude_cols = [
        'actual_home_score', 'actual_visitor_score', 'actual_spread', 
        'actual_total', 'home_won', 'game_id', 'date',
        'home_covered', 'went_over', 'favorite_won', 'upset',
        'est_vegas_spread', 'est_vegas_total', 'est_home_win_prob'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Temporal split (train on older, test on newer)
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"Training set: {len(train_df)} games")
    print(f"Test set: {len(test_df)} games")
    
    # Smart NaN filling: use feature-appropriate defaults
    fill_defaults = {}
    for col in feature_cols:
        if 'elo' in col.lower():
            fill_defaults[col] = 1500
        elif 'win_pct' in col.lower() or 'prob' in col.lower():
            fill_defaults[col] = 0.5
        elif 'points_scored' in col.lower() or 'points_allowed' in col.lower():
            fill_defaults[col] = 110
        elif 'total' in col.lower() and 'vegas' in col.lower():
            fill_defaults[col] = 220
        else:
            fill_defaults[col] = 0
    
    X_train = train_df[feature_cols].fillna(fill_defaults)
    X_test = test_df[feature_cols].fillna(fill_defaults)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # ========== MODEL 1: MONEYLINE (Winner Prediction) ==========
    print("\n--- Training MONEYLINE Model ---")
    y_train_ml = train_df['home_won']
    y_test_ml = test_df['home_won']
    
    ml_model = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42
        ),
        cv=3,
        method='isotonic'
    )
    ml_model.fit(X_train_scaled, y_train_ml)
    
    ml_pred = ml_model.predict(X_test_scaled)
    ml_prob = ml_model.predict_proba(X_test_scaled)[:, 1]
    ml_acc = accuracy_score(y_test_ml, ml_pred)
    
    # High confidence accuracy
    high_conf_mask = (ml_prob > 0.65) | (ml_prob < 0.35)
    if high_conf_mask.sum() > 0:
        hc_acc = accuracy_score(y_test_ml[high_conf_mask], ml_pred[high_conf_mask])
        hc_count = high_conf_mask.sum()
    else:
        hc_acc = 0
        hc_count = 0
    
    print(f"  Overall Accuracy: {ml_acc:.1%}")
    print(f"  High Conf (65%+): {hc_acc:.1%} ({hc_count} games)")
    
    models['moneyline'] = ml_model
    results['moneyline'] = {'accuracy': ml_acc, 'high_conf_accuracy': hc_acc}
    
    # ========== MODEL 2: SPREAD (Cover Prediction) ==========
    print("\n--- Training SPREAD Model ---")
    y_train_spread = train_df['home_covered']
    y_test_spread = test_df['home_covered']
    
    spread_model = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42
        ),
        cv=3,
        method='isotonic'
    )
    spread_model.fit(X_train_scaled, y_train_spread)
    
    spread_pred = spread_model.predict(X_test_scaled)
    spread_prob = spread_model.predict_proba(X_test_scaled)[:, 1]
    spread_acc = accuracy_score(y_test_spread, spread_pred)
    
    # Confident spread picks
    conf_spread_mask = (spread_prob > 0.58) | (spread_prob < 0.42)
    if conf_spread_mask.sum() > 0:
        conf_spread_acc = accuracy_score(y_test_spread[conf_spread_mask], spread_pred[conf_spread_mask])
        conf_spread_count = conf_spread_mask.sum()
    else:
        conf_spread_acc = 0
        conf_spread_count = 0
    
    print(f"  Overall Accuracy: {spread_acc:.1%}")
    print(f"  Confident (58%+): {conf_spread_acc:.1%} ({conf_spread_count} games)")
    
    models['spread'] = spread_model
    results['spread'] = {'accuracy': spread_acc, 'confident_accuracy': conf_spread_acc}
    
    # ========== MODEL 3: TOTALS (Over/Under) ==========
    print("\n--- Training TOTALS Model ---")
    y_train_total = train_df['went_over']
    y_test_total = test_df['went_over']
    
    total_model = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.02,
            min_samples_leaf=15,
            subsample=0.85,
            max_features='sqrt',
            random_state=42
        ),
        cv=3,
        method='isotonic'
    )
    total_model.fit(X_train_scaled, y_train_total)
    
    total_pred = total_model.predict(X_test_scaled)
    total_prob = total_model.predict_proba(X_test_scaled)[:, 1]
    total_acc = accuracy_score(y_test_total, total_pred)
    
    # Confident total picks
    conf_total_mask = (total_prob > 0.58) | (total_prob < 0.42)
    if conf_total_mask.sum() > 0:
        conf_total_acc = accuracy_score(y_test_total[conf_total_mask], total_pred[conf_total_mask])
        conf_total_count = conf_total_mask.sum()
    else:
        conf_total_acc = 0
        conf_total_count = 0
    
    print(f"  Overall Accuracy: {total_acc:.1%}")
    print(f"  Confident (58%+): {conf_total_acc:.1%} ({conf_total_count} games)")
    
    models['totals'] = total_model
    results['totals'] = {'accuracy': total_acc, 'confident_accuracy': conf_total_acc}
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("BETTING MODEL SUMMARY")
    print("="*60)
    print(f"  MONEYLINE:  {ml_acc:.1%} overall, {hc_acc:.1%} high-conf")
    print(f"  SPREAD:     {spread_acc:.1%} overall, {conf_spread_acc:.1%} confident")
    print(f"  TOTALS:     {total_acc:.1%} overall, {conf_total_acc:.1%} confident")
    print("="*60)
    
    return models, scaler, feature_cols, results


def save_betting_models(models, scaler, feature_cols, results):
    """Save all betting models"""
    
    os.makedirs('models', exist_ok=True)
    
    # Save individual models
    joblib.dump(models['moneyline'], 'models/betting_moneyline.joblib')
    joblib.dump(models['spread'], 'models/betting_spread.joblib')
    joblib.dump(models['totals'], 'models/betting_totals.joblib')
    joblib.dump(scaler, 'models/betting_scaler.joblib')
    
    # Save metadata
    metadata = {
        'feature_names': feature_cols,
        'trained_at': datetime.now().isoformat(),
        'results': results
    }
    
    with open('models/betting_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n[OK] Betting models saved to models/")


def main():
    print("="*60)
    print("BETTING MODEL TRAINING")
    print("Focus: Moneyline, Spread, Over/Under")
    print("="*60)
    
    # Initialize
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    
    # Get historical games
    print("\nFetching historical game data...")
    seasons = [2023, 2024, 2025]
    historical_data = data_mgr.get_complete_training_data(seasons)
    
    if historical_data['games'].empty:
        print("ERROR: No historical data available")
        return
    
    games_df = historical_data['games']
    print(f"Loaded {len(games_df)} total games")
    
    # Filter to completed games
    games_df = games_df[games_df['status'] == 'Final'].copy()
    print(f"Filtered to {len(games_df)} completed games")
    
    # Prepare betting data
    betting_df = prepare_betting_data(games_df, historical_data, feature_eng)
    
    if len(betting_df) < 100:
        print("ERROR: Not enough games for training")
        return
    
    # Add Vegas proxy features
    betting_df = add_vegas_proxy_features(betting_df)
    
    # Create betting targets
    betting_df = create_betting_targets(betting_df)
    
    # Train models
    models, scaler, feature_cols, results = train_betting_models(betting_df)
    
    # Save models
    save_betting_models(models, scaler, feature_cols, results)
    
    print("\n[DONE] TRAINING COMPLETE")
    print("Run the app to use the new betting models")


if __name__ == "__main__":
    main()
