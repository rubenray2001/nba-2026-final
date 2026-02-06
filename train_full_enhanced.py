"""
Full Enhanced Model Training
Includes: Vegas odds, H2H history, injuries, situational factors
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, brier_score_loss
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, DATA_DIR, TRAINING_SEASONS
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from data_manager import DataManager


def train_full_enhanced_model():
    """Train model with ALL enhanced features"""
    
    print("=" * 70)
    print("FULL ENHANCED MODEL TRAINING")
    print("Vegas + H2H + Injuries + Situational")
    print("=" * 70)
    
    # Initialize
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    enhanced_eng = FeatureEngineer()  # Same class now (EnhancedFeatureEngineer)
    
    # Use recent seasons
    seasons = [2022, 2023, 2024, 2025, 2026]
    print(f"\nUsing seasons: {seasons}")
    
    # ===========================================
    # FETCH ALL DATA
    # ===========================================
    print("\n" + "=" * 60)
    print("FETCHING DATA")
    print("=" * 60)
    
    # Get base training data
    print("Fetching game data...")
    all_data = data_mgr.get_complete_training_data(seasons)
    
    # Build base features
    print("\nBuilding base features...")
    df = feature_eng.build_training_dataset(all_data, seasons)
    print(f"Base dataset: {len(df)} games, {df.shape[1]} columns")
    
    # Get historical games for H2H calculation
    historical_games = all_data.get('games', pd.DataFrame())
    
    # Fetch Vegas odds for training data
    print("\nFetching Vegas odds...")
    try:
        unique_dates = df['date'].dt.strftime('%Y-%m-%d').unique().tolist()
        # Sample dates to avoid too many API calls (get odds for ~10% of games)
        sample_dates = unique_dates[::10][:100]  # Every 10th date, max 100
        odds_df = data_mgr.fetch_vegas_odds(dates=sample_dates)
        print(f"Fetched odds for {len(odds_df)} game-vendor combinations")
    except Exception as e:
        print(f"Could not fetch odds: {e}")
        odds_df = pd.DataFrame()
    
    # Fetch standings (for situational features)
    print("\nFetching standings...")
    try:
        standings_list = []
        for season in seasons[-2:]:  # Last 2 seasons for standings
            s = data_mgr.api_client.get_standings(season)
            if s:
                for standing in s:
                    standing['season'] = season
                standings_list.extend(s)
        standings_df = pd.DataFrame(standings_list)
        print(f"Fetched {len(standings_df)} standing records")
    except Exception as e:
        print(f"Could not fetch standings: {e}")
        standings_df = pd.DataFrame()
    
    # ===========================================
    # ENHANCE FEATURES
    # ===========================================
    print("\n" + "=" * 60)
    print("ENHANCING FEATURES")
    print("=" * 60)
    
    # Add enhanced features
    df_enhanced = enhanced_eng.enhance_features(
        df,
        odds_df=odds_df,
        injuries=None,  # No historical injuries available for training
        standings=standings_df,
        historical_games=historical_games
    )
    
    print(f"\nEnhanced dataset: {len(df_enhanced)} games, {df_enhanced.shape[1]} columns")
    
    # List new features added
    new_cols = [c for c in df_enhanced.columns if c not in df.columns]
    print(f"New features added: {len(new_cols)}")
    print(f"  Vegas: {[c for c in new_cols if 'vegas' in c.lower()]}")
    print(f"  H2H: {[c for c in new_cols if 'h2h' in c.lower()]}")
    print(f"  Injury: {[c for c in new_cols if 'injur' in c.lower() or 'star' in c.lower()]}")
    print(f"  Situational: {[c for c in new_cols if c not in ['vegas', 'h2h'] and c in new_cols][:10]}...")
    
    # ===========================================
    # PREPARE TRAINING DATA
    # ===========================================
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    
    # Sort by date
    df_enhanced = df_enhanced.sort_values('date').reset_index(drop=True)
    
    # Identify feature columns
    exclude_cols = ['game_id', 'date', 'season', 'home_team_id', 'visitor_team_id',
                   'home_score', 'visitor_score', 'home_won',
                   'home_date', 'visitor_date', 'home_team_id_y', 'visitor_team_id_y']
    
    feature_cols = [c for c in df_enhanced.columns 
                   if c not in exclude_cols 
                   and df_enhanced[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"Total features: {len(feature_cols)}")
    
    # Remove highly correlated features
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
    
    X_all = df_enhanced[feature_cols].fillna(fill_defaults)
    corr_matrix = X_all.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri.index[upper_tri[col] > 0.95].tolist()
        for hc in high_corr:
            if hc not in features_to_drop and col not in features_to_drop:
                features_to_drop.add(hc)
    
    reduced_features = [f for f in feature_cols if f not in features_to_drop]
    print(f"After removing >0.95 correlation: {len(reduced_features)} features")
    
    # Temporal split
    split_idx = int(len(df_enhanced) * 0.8)
    train_df = df_enhanced.iloc[:split_idx]
    test_df = df_enhanced.iloc[split_idx:]
    
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
    
    # Feature selection - Top 50
    print("\nSelecting top 50 features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=min(50, len(reduced_features)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_winner_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(reduced_features, selected_mask) if s]
    
    # Show top features
    feature_scores = pd.DataFrame({
        'feature': reduced_features,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print("\nTop 15 most predictive features:")
    for i, (_, row) in enumerate(feature_scores.head(15).iterrows()):
        marker = "*" if any(x in row['feature'].lower() for x in ['vegas', 'h2h', 'travel', 'motivation']) else " "
        print(f"  {marker} {row['feature']}: {row['score']:.4f}")
    
    # ===========================================
    # TRAIN MODELS
    # ===========================================
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    # Winner classifier
    base_models = [
        ('hgb1', HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=0.5,
            random_state=42
        )),
        ('hgb2', HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=43
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=10,
            random_state=44, n_jobs=-1
        )),
        ('lr', LogisticRegression(C=0.5, max_iter=500, random_state=45))
    ]
    
    winner_ensemble = VotingClassifier(base_models, voting='soft')
    
    print("Training winner classifier...")
    winner_ensemble.fit(X_train_selected, y_winner_train)
    
    print("Calibrating probabilities...")
    calibrated_winner = CalibratedClassifierCV(winner_ensemble, cv=3, method='isotonic')
    calibrated_winner.fit(X_train_selected, y_winner_train)
    
    # Score models
    score_models = [
        ('hgb', HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=0.5,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=8,
            random_state=43, n_jobs=-1
        )),
        ('ridge', Ridge(alpha=3.0))
    ]
    
    home_score_ensemble = VotingRegressor(score_models)
    visitor_score_ensemble = VotingRegressor(score_models)
    
    print("Training score models...")
    home_score_ensemble.fit(X_train_selected, y_home_train)
    visitor_score_ensemble.fit(X_train_selected, y_visitor_train)
    
    # ===========================================
    # EVALUATION
    # ===========================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Predictions
    winner_pred_train = calibrated_winner.predict(X_train_selected)
    winner_pred_test = calibrated_winner.predict(X_test_selected)
    winner_prob_test = calibrated_winner.predict_proba(X_test_selected)[:, 1]
    
    home_pred_test = home_score_ensemble.predict(X_test_selected)
    visitor_pred_test = visitor_score_ensemble.predict(X_test_selected)
    
    # Metrics
    train_acc = accuracy_score(y_winner_train, winner_pred_train)
    test_acc = accuracy_score(y_winner_test, winner_pred_test)
    test_brier = brier_score_loss(y_winner_test, winner_prob_test)
    
    home_mae = mean_absolute_error(y_home_test, home_pred_test)
    visitor_mae = mean_absolute_error(y_visitor_test, visitor_pred_test)
    
    print(f"\n--- WINNER PREDICTION ---")
    print(f"Train Accuracy: {train_acc:.1%}")
    print(f"Test Accuracy:  {test_acc:.1%}")
    print(f"Overfit Gap:    {(train_acc - test_acc):.1%}")
    print(f"Brier Score:    {test_brier:.4f}")
    
    print(f"\n--- SCORE PREDICTION ---")
    print(f"Home MAE:    {home_mae:.1f} pts")
    print(f"Visitor MAE: {visitor_mae:.1f} pts")
    
    # High confidence analysis
    print(f"\n--- HIGH CONFIDENCE PICKS ---")
    
    for threshold in [0.60, 0.65, 0.70]:
        mask = (winner_prob_test >= threshold) | (winner_prob_test <= (1 - threshold))
        if mask.sum() > 10:
            acc = accuracy_score(y_winner_test[mask], winner_pred_test[mask])
            print(f"  >{threshold:.0%} confidence: {mask.sum()} games ({mask.sum()/len(y_winner_test):.1%}), Accuracy: {acc:.1%}")
    
    # ===========================================
    # SAVE MODELS
    # ===========================================
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(home_score_ensemble, os.path.join(MODELS_DIR, 'home_score_ensemble.pkl'))
    joblib.dump(visitor_score_ensemble, os.path.join(MODELS_DIR, 'visitor_score_ensemble.pkl'))
    joblib.dump(calibrated_winner, os.path.join(MODELS_DIR, 'winner_ensemble.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'selector.pkl'))
    
    # Save metadata
    metadata = {
        'feature_names': reduced_features,
        'selected_features': selected_features,
        'enhanced_features': new_cols,
        'training_info': {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(train_df),
            'test_samples': len(test_df),
            'features': len(reduced_features),
            'selected_features': len(selected_features),
            'seasons': seasons,
            'enhancements': ['vegas_odds', 'h2h_history', 'situational'],
            'metrics': {
                'home_score_test_mae': float(home_mae),
                'visitor_score_test_mae': float(visitor_mae),
                'home_score_test_r2': float(r2_score(y_home_test, home_pred_test)),
                'visitor_score_test_r2': float(r2_score(y_visitor_test, visitor_pred_test)),
                'winner_train_accuracy': float(train_acc),
                'winner_test_accuracy': float(test_acc),
                'overfit_gap': float(train_acc - test_acc),
                'brier_score': float(test_brier)
            }
        },
        'model_type': 'full_enhanced_ensemble'
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update history
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
        'features': len(selected_features),
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc),
        'home_score_mae': float(home_mae),
        'visitor_score_mae': float(visitor_mae),
        'overfit_gap': float(train_acc - test_acc),
        'brier_score': float(test_brier),
        'model_type': 'full_enhanced',
        'enhancements': ['vegas', 'h2h', 'situational']
    })
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModels saved to {MODELS_DIR}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTest Accuracy: {test_acc:.1%}")
    print(f"Score MAE:     {(home_mae + visitor_mae)/2:.1f} pts")
    print(f"Brier Score:   {test_brier:.4f}")
    
    # Compare to baseline
    baseline_acc = 0.642  # From previous training
    improvement = (test_acc - baseline_acc) * 100
    if improvement > 0:
        print(f"\n[IMPROVED] +{improvement:.1f}% vs baseline!")
    else:
        print(f"\n[NOTE] {improvement:+.1f}% vs baseline")
    
    return test_acc


if __name__ == "__main__":
    train_full_enhanced_model()
