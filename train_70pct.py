"""
Enhanced Feature Engineering + Training
Adds H2H matchup features and MOV-weighted ELO
Target: 70%+ accuracy
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
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

def calculate_mov_elo(games_df):
    """
    Calculate Margin-of-Victory weighted ELO ratings
    MOV multiplier makes ELO more predictive
    """
    df = games_df.sort_values('date').copy()
    
    K = 20
    MEAN = 1500
    HOME_ADV = 100
    WIDTH = 400
    
    elo_ratings = {}
    home_elos, visitor_elos = [], []
    
    for _, row in df.iterrows():
        hid, vid = row['home_team_id'], row['visitor_team_id']
        
        h_elo = elo_ratings.get(hid, MEAN)
        v_elo = elo_ratings.get(vid, MEAN)
        
        home_elos.append(h_elo)
        visitor_elos.append(v_elo)
        
        h_elo_adj = h_elo + HOME_ADV
        prob_home = 1 / (10 ** ((v_elo - h_elo_adj) / WIDTH) + 1)
        
        h_score = row['home_team_score']
        v_score = row['visitor_team_score']
        margin = abs(h_score - v_score)
        
        actual = 1.0 if h_score > v_score else 0.0
        
        # MOV multiplier (FiveThirtyEight-style)
        elo_diff = abs(h_elo_adj - v_elo)
        mov_mult = np.log(margin + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
        
        shift = K * mov_mult * (actual - prob_home)
        elo_ratings[hid] = h_elo + shift
        elo_ratings[vid] = v_elo - shift
    
    df['home_elo_mov'] = home_elos
    df['visitor_elo_mov'] = visitor_elos
    df['elo_mov_diff'] = df['home_elo_mov'] - df['visitor_elo_mov']
    
    return df[['id', 'home_elo_mov', 'visitor_elo_mov', 'elo_mov_diff']]

def calculate_h2h_features(games_df):
    """
    Calculate head-to-head historical features between matchups
    """
    df = games_df.sort_values('date').copy()
    
    # Create matchup key (sorted team ids for consistency)
    df['matchup'] = df.apply(lambda r: tuple(sorted([r['home_team_id'], r['visitor_team_id']])), axis=1)
    
    h2h_wins = []
    h2h_margins = []
    h2h_games_played = []
    
    # Track H2H history
    matchup_history = {}  # matchup -> list of (home_team_id, home_won, margin)
    
    for _, row in df.iterrows():
        matchup = row['matchup']
        home_id = row['home_team_id']
        
        # Get historical H2H
        history = matchup_history.get(matchup, [])
        
        if len(history) == 0:
            h2h_wins.append(0.5)
            h2h_margins.append(0)
            h2h_games_played.append(0)
        else:
            # Count wins for current home team in this matchup
            wins = sum(1 for h in history if (h[0] == home_id and h[1]) or (h[0] != home_id and not h[1]))
            h2h_wins.append(wins / len(history))
            
            # Average margin from home team perspective
            margins = [h[2] if h[0] == home_id else -h[2] for h in history]
            h2h_margins.append(np.mean(margins))
            h2h_games_played.append(len(history))
        
        # Update history
        home_won = row['home_team_score'] > row['visitor_team_score']
        margin = row['home_team_score'] - row['visitor_team_score']
        
        if matchup not in matchup_history:
            matchup_history[matchup] = []
        matchup_history[matchup].append((home_id, home_won, margin))
    
    df['h2h_win_pct'] = h2h_wins
    df['h2h_avg_margin'] = h2h_margins
    df['h2h_games'] = h2h_games_played
    
    return df[['id', 'h2h_win_pct', 'h2h_avg_margin', 'h2h_games']]

def enhance_training_data():
    """Add new features to training data"""
    print("Loading historical games...")
    games = pd.read_csv("data/games_historical.csv")
    print(f"Games: {len(games)}")
    
    print("Calculating MOV-weighted ELO...")
    elo_df = calculate_mov_elo(games)
    
    print("Calculating H2H features...")
    h2h_df = calculate_h2h_features(games)
    
    print("Loading existing training data...")
    train_df = pd.read_csv("data/training_data.csv")
    
    # Merge new features
    print("Merging new features...")
    train_df = train_df.merge(elo_df, left_on='game_id', right_on='id', how='left')
    train_df = train_df.merge(h2h_df, left_on='game_id', right_on='id', how='left')
    
    # Fill NaN
    for col in ['home_elo_mov', 'visitor_elo_mov', 'elo_mov_diff', 'h2h_win_pct', 'h2h_avg_margin', 'h2h_games']:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0 if 'margin' in col or 'games' in col else 0.5 if 'pct' in col else 1500)
    
    # Drop duplicate id columns
    train_df = train_df.drop(columns=['id_x', 'id_y'], errors='ignore')
    
    print(f"Enhanced features: {len(train_df.columns)}")
    return train_df

def train():
    print("=" * 50)
    print("ENHANCED TRAINING - TARGET 70%+")
    print("=" * 50)
    
    # Get enhanced data
    df = enhance_training_data()
    
    exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
               'home_score', 'visitor_score', 'home_won', 'season']
    
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0).select_dtypes(include=[np.number])
    y = df['home_won']
    y_home = df['home_score']
    y_vis = df['visitor_score']
    features = list(X.columns)
    
    print(f"\nSamples: {len(X)}, Features: {len(features)}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Split
    split = int(len(X_scaled) * 0.8)
    X_tr, X_te = X_scaled[:split], X_scaled[split:]
    y_tr, y_te = y.values[:split], y.values[split:]
    y_h_tr, y_h_te = y_home.values[:split], y_home.values[split:]
    y_v_tr, y_v_te = y_vis.values[:split], y_vis.values[split:]
    
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")
    
    # Train classifier
    print("\n[1/3] Training classifier...")
    clf = VotingClassifier([
        ('hgb1', HistGradientBoostingClassifier(
            max_iter=800, max_depth=6, learning_rate=0.025,
            max_leaf_nodes=31, l2_regularization=3.0, random_state=42
        )),
        ('hgb2', HistGradientBoostingClassifier(
            max_iter=600, max_depth=8, learning_rate=0.04,
            max_leaf_nodes=50, l2_regularization=2.0, random_state=43
        )),
        ('hgb3', HistGradientBoostingClassifier(
            max_iter=500, max_depth=10, learning_rate=0.03,
            max_leaf_nodes=64, l2_regularization=1.5, random_state=44
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            min_samples_leaf=5, random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=500, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        ))
    ], voting='soft', n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba)
    
    print(f"\n*** ACCURACY: {acc*100:.1f}% ***")
    print(f"Log Loss: {ll:.4f}")
    
    # Score models
    print("\n[2/3] Training score models...")
    home_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=8, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42))
    ], n_jobs=-1)
    home_ens.fit(X_tr, y_h_tr)
    
    vis_ens = VotingRegressor([
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=8, random_state=43)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=43)),
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
                'home_score_test_mae': float(h_mae),
                'visitor_score_test_mae': float(v_mae)
            }
        }
    }
    with open(f"{MODELS_DIR}/model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"RESULT: {acc*100:.1f}%")
    if acc >= 0.70: print("🎉 70%+ ACHIEVED!")
    elif acc >= 0.65: print("📈 GOOD PROGRESS!")
    print("=" * 50)
    
    return acc

if __name__ == "__main__":
    train()
