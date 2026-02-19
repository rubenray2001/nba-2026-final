"""
Feature Engineering Pipeline for College Basketball
Includes: ELO Rating, Rolling Stats, Schedule Fatigue
Adapted from NBA model with college-specific defaults
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import json
import config


class FeatureEngineer:
    """Feature engineering for college basketball predictions"""
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.gender_config = config.GENDER_CONFIG[gender]
        self.rolling_windows = config.ROLLING_WINDOWS
        self.avg_score = self.gender_config["avg_score"]
        self.avg_total = self.gender_config["avg_total"]
        self.avg_pace = self.gender_config["avg_pace"]
        
        # ELO Constants
        self.ELO_K = 20
        self.ELO_MEAN = 1500
        self.ELO_HOME_ADV = 100  # Home court is very important in college
        self.ELO_WIDTH = 400
    
    def _calculate_elo(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ELO ratings for every game with season regression.
        Between seasons, ratings regress 1/3 toward 1500.
        """
        df = games_df.sort_values('date').copy()
        elo_ratings = {}
        current_season = None
        SEASON_REGRESSION = 1/3
        
        home_elos = []
        visitor_elos = []
        
        for _, row in df.iterrows():
            hid = row['home_team_id']
            vid = row['visitor_team_id']
            game_season = row.get('season', None)
            
            if game_season is not None and current_season is not None and game_season != current_season:
                for team_id in list(elo_ratings.keys()):
                    old_elo = elo_ratings[team_id]
                    elo_ratings[team_id] = old_elo + SEASON_REGRESSION * (self.ELO_MEAN - old_elo)
            current_season = game_season
            
            h_elo = elo_ratings.get(hid, self.ELO_MEAN)
            v_elo = elo_ratings.get(vid, self.ELO_MEAN)
            
            home_elos.append(h_elo)
            visitor_elos.append(v_elo)
            
            h_elo_adj = h_elo + self.ELO_HOME_ADV
            prob_home_win = 1 / (10 ** ((v_elo - h_elo_adj) / self.ELO_WIDTH) + 1)
            
            h_score = row['home_team_score']
            v_score = row['visitor_team_score']
            actual_home_win = 1.0 if h_score > v_score else 0.0
            mov_mult = np.log(abs(h_score - v_score) + 1) if (h_score > 0 and v_score > 0) else 1.0
            
            shift = self.ELO_K * mov_mult * (actual_home_win - prob_home_win)
            elo_ratings[hid] = h_elo + shift
            elo_ratings[vid] = v_elo - shift
        
        df['home_elo'] = home_elos
        df['visitor_elo'] = visitor_elos
        
        return df[['game_id', 'home_elo', 'visitor_elo']]
    
    def build_training_dataset(self, all_data: Dict, seasons: List[int]) -> pd.DataFrame:
        """Build feature-rich training dataset from historical games"""
        print("=" * 50)
        print("FEATURE ENGINEERING (ELO + Rolling Stats)")
        print("=" * 50)
        
        games_df = all_data['games'].copy()
        if 'id' in games_df.columns:
            games_df = games_df.rename(columns={'id': 'game_id'})
        
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        # Calculate ELO
        print("Calculating ELO Ratings...")
        elo_df = self._calculate_elo(games_df)
        games_df = pd.merge(games_df, elo_df, on='game_id', how='left')
        
        # Build team-centric metrics
        print("Computing rolling stats...")
        
        # Create records for each team from each game (home perspective + away perspective)
        records = []
        for _, row in games_df.iterrows():
            # Home team record
            records.append({
                'game_id': row['game_id'],
                'date': row['date'],
                'season': row.get('season', 0),
                'team_id': row['home_team_id'],
                'opponent_id': row['visitor_team_id'],
                'is_home': 1,
                'points_scored': row['home_team_score'],
                'points_allowed': row['visitor_team_score'],
                'won': 1 if row['home_team_score'] > row['visitor_team_score'] else 0,
            })
            # Away team record
            records.append({
                'game_id': row['game_id'],
                'date': row['date'],
                'season': row.get('season', 0),
                'team_id': row['visitor_team_id'],
                'opponent_id': row['home_team_id'],
                'is_home': 0,
                'points_scored': row['visitor_team_score'],
                'points_allowed': row['home_team_score'],
                'won': 1 if row['visitor_team_score'] > row['home_team_score'] else 0,
            })
        
        team_df = pd.DataFrame(records)
        team_df = team_df.sort_values(['team_id', 'date']).reset_index(drop=True)
        
        # Point differential
        team_df['point_diff'] = team_df['points_scored'] - team_df['points_allowed']
        
        # Rolling stats
        grouped = team_df.groupby('team_id')
        for window in self.rolling_windows:
            team_df[f'win_pct_last{window}'] = grouped['won'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            for m in ['points_scored', 'points_allowed', 'point_diff']:
                team_df[f'{m}_last{window}'] = grouped[m].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
        
        # Schedule metrics
        team_df['date'] = pd.to_datetime(team_df['date'])
        team_df['prev_game_date'] = grouped['date'].shift(1)
        team_df['rest_days'] = (team_df['date'] - team_df['prev_game_date']).dt.days.fillna(3).clip(upper=7)
        team_df['is_b2b'] = (team_df['rest_days'] == 1).astype(int)
        
        # Merge back to game-level features
        feature_cols = ['team_id', 'date', 'rest_days', 'is_b2b']
        for w in self.rolling_windows:
            feature_cols.append(f'win_pct_last{w}')
            feature_cols += [f'points_scored_last{w}', f'points_allowed_last{w}', f'point_diff_last{w}']
        
        features_subset = team_df[feature_cols].copy()
        
        # Merge home features
        game_features = pd.merge(
            games_df,
            features_subset.add_prefix('home_'),
            left_on=['home_team_id', 'date'],
            right_on=['home_team_id', 'home_date'],
            how='inner'
        )
        
        # Merge visitor features
        game_features = pd.merge(
            game_features,
            features_subset.add_prefix('visitor_'),
            left_on=['visitor_team_id', 'date'],
            right_on=['visitor_team_id', 'visitor_date'],
            how='inner'
        )
        
        # Derived features
        game_features['elo_diff'] = game_features['home_elo'] - game_features['visitor_elo']
        game_features['rest_advantage'] = game_features['home_rest_days'] - game_features['visitor_rest_days']
        game_features['momentum_diff_5'] = game_features['home_win_pct_last5'] - game_features['visitor_win_pct_last5']
        game_features['momentum_diff_10'] = game_features['home_win_pct_last10'] - game_features['visitor_win_pct_last10']
        
        if 'home_point_diff_last10' in game_features.columns:
            game_features['net_rating_diff'] = game_features['home_point_diff_last10'] - game_features['visitor_point_diff_last10']
        
        # Targets
        game_features['home_score'] = game_features['home_team_score']
        game_features['visitor_score'] = game_features['visitor_team_score']
        game_features['home_won'] = (game_features['home_team_score'] > game_features['visitor_team_score']).astype(int)
        
        # Final column selection
        final_cols = ['game_id', 'date', 'season', 'home_team_id', 'visitor_team_id']
        final_cols += ['home_score', 'visitor_score', 'home_won']
        final_cols += ['home_elo', 'visitor_elo', 'elo_diff']
        final_cols += ['rest_advantage', 'momentum_diff_5', 'momentum_diff_10']
        if 'net_rating_diff' in game_features.columns:
            final_cols.append('net_rating_diff')
        
        feat_cols_only = [c for c in features_subset.columns if c not in ['team_id', 'date']]
        for c in feat_cols_only:
            final_cols.append(f'home_{c}')
            final_cols.append(f'visitor_{c}')
        
        # Only keep columns that exist
        final_cols = [c for c in final_cols if c in game_features.columns]
        final_df = game_features[final_cols].copy()
        
        # Smart NaN filling
        fill_defaults = {}
        for col in final_df.columns:
            if 'elo' in col.lower():
                fill_defaults[col] = 1500
            elif 'win_pct' in col.lower():
                fill_defaults[col] = 0.5
            elif 'points_scored' in col.lower():
                fill_defaults[col] = self.avg_score
            elif 'points_allowed' in col.lower():
                fill_defaults[col] = self.avg_score
            elif 'rest_days' in col.lower():
                fill_defaults[col] = 2
            elif 'vegas_total' in col.lower():
                fill_defaults[col] = self.avg_total
            elif 'vegas_implied' in col.lower():
                fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower():
                fill_defaults[col] = 0.5
            else:
                fill_defaults[col] = 0
        final_df = final_df.fillna(fill_defaults)
        
        # Remove cold-start games
        initial_count = len(final_df)
        cold_start_mask = (
            (final_df['home_win_pct_last10'] > 0) &
            (final_df['visitor_win_pct_last10'] > 0) &
            (final_df['home_points_scored_last5'] > 0) &
            (final_df['visitor_points_scored_last5'] > 0)
        )
        final_df = final_df[cold_start_mask]
        removed = initial_count - len(final_df)
        print(f"Removed {removed} cold-start games ({removed/max(initial_count,1)*100:.1f}%)")
        
        print(f"Engineering Complete. Data shape: {final_df.shape}")
        return final_df
    
    def build_features_for_game(self, game: dict, historical_data: dict, current_season: int) -> dict:
        """Build features for a single upcoming game using historical data"""
        # Load feature names from model metadata
        models_dir = os.path.join(config.MODELS_DIR, self.gender)
        metadata_path = os.path.join(models_dir, 'model_metadata.json')
        
        expected_features = []
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                expected_features = metadata.get('feature_names', [])
            except Exception:
                pass
        
        home_id = game.get('home_team_id') or game.get('home_team', {}).get('id')
        visitor_id = game.get('visitor_team_id') or game.get('visitor_team', {}).get('id')
        
        if not home_id or not visitor_id:
            return None
        
        games_df = historical_data.get('games', pd.DataFrame())
        if games_df.empty:
            return None
        
        games_df['date'] = pd.to_datetime(games_df['date'])
        features = {}
        
        # ELO
        if 'home_elo' in games_df.columns:
            for prefix, team_id in [('home', home_id), ('visitor', visitor_id)]:
                t_games = games_df[(games_df['home_team_id'] == team_id) | (games_df['visitor_team_id'] == team_id)].sort_values('date', ascending=False)
                if not t_games.empty:
                    last_g = t_games.iloc[0]
                    features[f'{prefix}_elo'] = last_g['home_elo'] if last_g['home_team_id'] == team_id else last_g['visitor_elo']
                else:
                    features[f'{prefix}_elo'] = 1500
        else:
            features['home_elo'] = 1500
            features['visitor_elo'] = 1500
        
        features['elo_diff'] = features.get('home_elo', 1500) - features.get('visitor_elo', 1500)
        
        # Rolling stats for each team
        for prefix, team_id in [('home', home_id), ('visitor', visitor_id)]:
            team_games = games_df[(games_df['home_team_id'] == team_id) | (games_df['visitor_team_id'] == team_id)].copy()
            team_games = team_games.sort_values('date', ascending=False)
            
            # Normalize perspective
            team_games['points_scored'] = np.where(
                team_games['home_team_id'] == team_id,
                team_games['home_team_score'], team_games['visitor_team_score']
            )
            team_games['points_allowed'] = np.where(
                team_games['home_team_id'] == team_id,
                team_games['visitor_team_score'], team_games['home_team_score']
            )
            team_games['won'] = (team_games['points_scored'] > team_games['points_allowed']).astype(int)
            team_games['point_diff'] = team_games['points_scored'] - team_games['points_allowed']
            
            for window in [5, 10, 20]:
                recent = team_games.head(window)
                if len(recent) > 0:
                    features[f'{prefix}_win_pct_last{window}'] = recent['won'].mean()
                    features[f'{prefix}_points_scored_last{window}'] = recent['points_scored'].mean()
                    features[f'{prefix}_points_allowed_last{window}'] = recent['points_allowed'].mean()
                    features[f'{prefix}_point_diff_last{window}'] = recent['point_diff'].mean()
                else:
                    features[f'{prefix}_win_pct_last{window}'] = 0.5
                    features[f'{prefix}_points_scored_last{window}'] = self.avg_score
                    features[f'{prefix}_points_allowed_last{window}'] = self.avg_score
                    features[f'{prefix}_point_diff_last{window}'] = 0
            
            # Rest days
            if not team_games.empty:
                game_date_str = game.get('date') or game.get('status', '')
                try:
                    game_date = pd.to_datetime(game_date_str).tz_localize(None) if 'T' in str(game_date_str) else pd.to_datetime(game_date_str)
                except (ValueError, TypeError):
                    game_date = pd.Timestamp.now()
                
                last_game_date = pd.to_datetime(team_games.iloc[0]['date'])
                try:
                    last_game_date = last_game_date.tz_localize(None)
                except TypeError:
                    pass
                
                rest_days = max(0, min((game_date - last_game_date).days, 10))
                features[f'{prefix}_rest_days'] = rest_days
                features[f'{prefix}_is_b2b'] = 1 if rest_days <= 1 else 0
            else:
                features[f'{prefix}_rest_days'] = 3
                features[f'{prefix}_is_b2b'] = 0
            
            # Season record
            current_season_games = team_games[team_games['season'] == current_season] if 'season' in team_games.columns else team_games
            if not current_season_games.empty:
                wins = current_season_games['won'].sum()
                losses = len(current_season_games) - wins
                features[f'{prefix}_wins'] = int(wins)
                features[f'{prefix}_losses'] = int(losses)
            else:
                features[f'{prefix}_wins'] = 0
                features[f'{prefix}_losses'] = 0
        
        # Derived features
        features['rest_advantage'] = features.get('home_rest_days', 2) - features.get('visitor_rest_days', 2)
        features['momentum_diff_5'] = features.get('home_win_pct_last5', 0.5) - features.get('visitor_win_pct_last5', 0.5)
        features['momentum_diff_10'] = features.get('home_win_pct_last10', 0.5) - features.get('visitor_win_pct_last10', 0.5)
        features['net_rating_diff'] = features.get('home_point_diff_last10', 0) - features.get('visitor_point_diff_last10', 0)
        
        # Default Vegas features
        features['vegas_spread_home'] = 0.0
        features['vegas_total'] = self.avg_total
        features['vegas_implied_home_prob'] = 0.5
        features['vegas_has_odds'] = 0
        
        # Default H2H
        features['h2h_games'] = 0
        features['h2h_home_wins'] = 0
        features['h2h_home_win_pct'] = 0.5
        features['h2h_avg_margin'] = 0
        features['h2h_last3_home_wins'] = 0.5
        
        # Default injury features
        features['injury_impact_diff'] = 0.0
        features['home_injury_impact'] = 0.0
        features['visitor_injury_impact'] = 0.0
        
        # Fill missing expected features
        if expected_features:
            for feat in expected_features:
                if feat not in features:
                    if 'elo' in feat.lower(): features[feat] = 1500
                    elif 'win_pct' in feat.lower(): features[feat] = 0.5
                    elif 'points_scored' in feat.lower(): features[feat] = self.avg_score
                    elif 'points_allowed' in feat.lower(): features[feat] = self.avg_score
                    elif 'rest_days' in feat.lower(): features[feat] = 2
                    elif 'vegas_total' in feat.lower(): features[feat] = self.avg_total
                    elif 'vegas_implied' in feat.lower(): features[feat] = 0.5
                    elif 'h2h_home_win_pct' in feat.lower() or 'h2h_last3' in feat.lower(): features[feat] = 0.5
                    else: features[feat] = 0.0
        
        return features
