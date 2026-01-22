"""
Feature Engineering Pipeline (Enhanced)
Includes: ELO Rating, Four Factors, Schedule Fatigue, and Rolling Stats.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import config

class FeatureEngineer:
    """High-performance feature engineering using vectorized operations + ELO"""
    
    def __init__(self):
        self.rolling_windows = config.ROLLING_WINDOWS
        # ELO Constants
        self.ELO_K = 20
        self.ELO_MEAN = 1500
        self.ELO_HOME_ADV = 100
        self.ELO_WIDTH = 400

    def _create_team_metrics_df(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Transform games to Team-Metric format (Long)"""
        # 1. Prepare Home Perspective
        home_df = games_df.copy()
        home_df['team_id'] = home_df['home_team_id']
        home_df['opponent_id'] = home_df['visitor_team_id']
        home_df['is_home'] = 1
        home_df['points_scored'] = home_df['home_team_score']
        home_df['points_allowed'] = home_df['visitor_team_score']
        home_df['won'] = (home_df['home_team_score'] > home_df['visitor_team_score']).astype(int)
        
        # Extract Advanced Stats (Home)
        # Note: We must be careful to use the Correct Team's stats.
        # for 'home_df', 'fgm' is 'home_fgm'
        stat_cols = ['fgm', 'fga', 'fg3m', 'ftm', 'fta', 'oreb', 'dreb', 'tov', 'stl', 'blk', 'pf']
        for col in stat_cols:
            home_df[col] = home_df.get(f'home_{col}', 0)
            home_df[f'opp_{col}'] = home_df.get(f'visitor_{col}', 0)
            
        # 2. Prepare Visitor Perspective
        visitor_df = games_df.copy()
        visitor_df['team_id'] = visitor_df['visitor_team_id']
        visitor_df['opponent_id'] = visitor_df['home_team_id']
        visitor_df['is_home'] = 0
        visitor_df['points_scored'] = visitor_df['visitor_team_score']
        visitor_df['points_allowed'] = visitor_df['home_team_score']
        visitor_df['won'] = (visitor_df['visitor_team_score'] > visitor_df['home_team_score']).astype(int)
        
        # Extract Advanced Stats (Visitor)
        for col in stat_cols:
            visitor_df[col] = visitor_df.get(f'visitor_{col}', 0)
            visitor_df[f'opp_{col}'] = visitor_df.get(f'home_{col}', 0)
        
        # 3. Concatenate and Sort
        team_df = pd.concat([home_df, visitor_df], ignore_index=True)
        # Sort by team then date for rolling calc
        team_df = team_df.sort_values(['team_id', 'date']).reset_index(drop=True)
        
        return team_df

    def _calculate_four_factors(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Dean Oliver's Four Factors
        1. eFG% = (FGM + 0.5 * 3PM) / FGA
        2. TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        3. OREB% = OREB / (OREB + OppDREB)
        4. FTR = FTM / FGA
        """
        # Avoid division by zero
        
        # 1. Effective Field Goal Percentage (eFG%)
        team_df['efg_pct'] = (team_df['fgm'] + 0.5 * team_df['fg3m']) / team_df['fga'].replace(0, 1)
        
        # 2. Turnover Percentage (TOV%)
        poss_est = team_df['fga'] + 0.44 * team_df['fta'] + team_df['tov']
        team_df['tov_pct'] = team_df['tov'] / poss_est.replace(0, 1)
        
        # 3. Offensive Rebound Percentage (OREB%)
        total_rebs = team_df['oreb'] + team_df['opp_dreb']
        team_df['oreb_pct'] = team_df['oreb'] / total_rebs.replace(0, 1)
        
        # 4. Free Throw Rate (FTR)
        team_df['ftr'] = team_df['ftm'] / team_df['fga'].replace(0, 1)
        
        # --- DEFENSIVE FOUR FACTORS (Opponent's stats) ---
        # 1. Opp eFG%
        team_df['opp_efg_pct'] = (team_df['opp_fgm'] + 0.5 * team_df['opp_fg3m']) / team_df['opp_fga'].replace(0, 1)
        
        # 2. Opp TOV%
        opp_poss_est = team_df['opp_fga'] + 0.44 * team_df['opp_fta'] + team_df['opp_tov']
        team_df['opp_tov_pct'] = team_df['opp_tov'] / opp_poss_est.replace(0, 1)
        
        # 3. Opp OREB% (Note: This is Opp OREB / (Opp OREB + My DREB))
        opp_total_rebs = team_df['opp_oreb'] + team_df['dreb']
        team_df['opp_oreb_pct'] = team_df['opp_oreb'] / opp_total_rebs.replace(0, 1)
        
        # 4. Opp FTR
        team_df['opp_ftr'] = team_df['opp_ftm'] / team_df['opp_fga'].replace(0, 1)
        
        return team_df

    def _calculate_elo(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ELO ratings for every game.
        Returns a dataframe with 'elo_home' and 'elo_visitor' for each game_id.
        """
        # Sort by date
        df = games_df.sort_values('date').copy()
        
        # Initialize ELO dict
        elo_ratings = {} # team_id -> rating
        
        # Store results lists to build dataframe later (faster than appending rows)
        home_elos = []
        visitor_elos = []
        
        # Iterate through all games chronologically
        for _, row in df.iterrows():
            hid = row['home_team_id']
            vid = row['visitor_team_id']
            
            # Get current ELO (default 1500)
            h_elo = elo_ratings.get(hid, self.ELO_MEAN)
            v_elo = elo_ratings.get(vid, self.ELO_MEAN)
            
            # Store entering ELOs (Features for this game)
            home_elos.append(h_elo)
            visitor_elos.append(v_elo)
            
            # Calculate Expected Win Prob
            # Home Advantage adds to Home ELO
            h_elo_adj = h_elo + self.ELO_HOME_ADV
            prob_home_win = 1 / (10 ** ((v_elo - h_elo_adj) / self.ELO_WIDTH) + 1)
            
            # Actual Outcome
            h_score = row['home_team_score']
            v_score = row['visitor_team_score']
            
            if h_score > v_score:
                actual_home_win = 1.0
            else:
                actual_home_win = 0.0
            
            # Margin of Victory Multiplier (Optional but good)
            # mov_mult = np.log(abs(h_score - v_score) + 1) # Simple log margin
            mov_mult = 1.0 # Keep simple for now
            
            # Update Ratings
            shift = self.ELO_K * mov_mult * (actual_home_win - prob_home_win)
            
            elo_ratings[hid] = h_elo + shift
            elo_ratings[vid] = v_elo - shift
            
        # Add to dataframe
        df['home_elo'] = home_elos
        df['visitor_elo'] = visitor_elos
        
        return df[['game_id', 'home_elo', 'visitor_elo']]

    def _calculate_rolling_stats(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages of raw stats AND four factors + enhanced features"""
        grouped = team_df.groupby('team_id')
        
        # Calculate pace (possessions estimate) first
        team_df['pace'] = team_df['fga'] + 0.44 * team_df['fta'] - team_df['oreb'] + team_df['tov']
        
        # Calculate point differential for net rating
        team_df['point_diff'] = team_df['points_scored'] - team_df['points_allowed']
        
        # Metrics to roll
        metrics = [
            'points_scored', 'points_allowed', 'point_diff', 'pace',
            'efg_pct', 'tov_pct', 'oreb_pct', 'ftr',
            'opp_efg_pct', 'opp_tov_pct', 'opp_oreb_pct', 'opp_ftr'
        ]
        
        for window in self.rolling_windows:
            # Win Percentage
            team_df[f'win_pct_last{window}'] = grouped['won'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            
            for m in metrics:
                team_df[f'{m}_last{window}'] = grouped[m].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
            
            # Home/Away Win Percentage splits
            team_df[f'home_win_pct_last{window}'] = grouped.apply(
                lambda g: g[g['is_home'] == 1]['won'].shift(1).rolling(window, min_periods=1).mean()
            ).reset_index(level=0, drop=True).fillna(0.5)
            
            team_df[f'away_win_pct_last{window}'] = grouped.apply(
                lambda g: g[g['is_home'] == 0]['won'].shift(1).rolling(window, min_periods=1).mean()
            ).reset_index(level=0, drop=True).fillna(0.5)
        
        return team_df


    def _calculate_schedule_metrics(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Rest Days and Fatigue (3-in-4, etc.)"""
        team_df['date'] = pd.to_datetime(team_df['date'])
        grouped = team_df.groupby('team_id')['date']
        
        # 1. Rest Days
        team_df['prev_game_date'] = grouped.shift(1)
        team_df['rest_days'] = (team_df['date'] - team_df['prev_game_date']).dt.days.fillna(3).clip(upper=7)
        
        # 2. Games in Last X Days (Fatigue)
        # Use rolling count on date index? simpler: Rolling count of rows is just "Games played"
        # We need "Games in last 7 days".
        # This is hard to vectorise purely with 'rolling' on DataFrame index unless we set Date as index.
        
        # Fast approximation: 'Fatigue Score'
        # High if played yesterday, played 2 days ago, etc.
        # We can use the 'rest_days' sequence.
        
        # B2B (Back to Back) : Rest = 1
        team_df['is_b2b'] = (team_df['rest_days'] == 1).astype(int)
        
        # 3 in 4 Nights:
        # Check if 3rd prev game was <= 3 days ago
        team_df['date_3_games_ago'] = grouped.shift(2) # 0 is current, 1 is prev, 2 is 2nd prev (total 3 games)
        msg_gap = (team_df['date'] - team_df['date_3_games_ago']).dt.days
        team_df['is_3in4'] = (msg_gap <= 4).astype(float).fillna(0)
        
        # 4 in 5 Nights
        team_df['date_4_games_ago'] = grouped.shift(3)
        msg_gap_4 = (team_df['date'] - team_df['date_4_games_ago']).dt.days
        team_df['is_4in5'] = (msg_gap_4 <= 5).astype(float).fillna(0)

        return team_df

    def build_training_dataset(self, all_data: Dict, seasons: List[int]) -> pd.DataFrame:
        print("="*50)
        print("ADVANCED FEATURE ENGINEERING (ELO + FOUR FACTORS)")
        print("="*50)
        
        games_df = all_data['games'].copy()
        if 'id' in games_df.columns:
            games_df = games_df.rename(columns={'id': 'game_id'})
            
        # Ensure date is datetime for merging
        games_df['date'] = pd.to_datetime(games_df['date'])
            
        # 1. Calculate ELO (Requires chronological scan of ALL games first)
        print("Calculating ELO Ratings...")
        elo_df = self._calculate_elo(games_df)
        games_df = pd.merge(games_df, elo_df, on='game_id')
        
        # 2. Transform to Team-Centric
        team_df = self._create_team_metrics_df(games_df)
        
        # 3. Calculate Base Metrics (Four Factors)
        team_df = self._calculate_four_factors(team_df)
        
        # 4. Rolling Stats & Schedule
        team_df = self._calculate_schedule_metrics(team_df)
        team_df = self._calculate_rolling_stats(team_df)
        
        # 5. Merge back to Home/Visitor matchups
        print("Reconstructing matchups...")
        
        # ELO is already in games_df, so we just need the Team Performance Features
        # Features to keep from team_df
        feature_cols = [
            'team_id', 'date', # Keys
            'rest_days', 'is_b2b', 'is_3in4', 'is_4in5'
        ]
        
        # Add rolling columns
        for w in self.rolling_windows:
            feature_cols.append(f'win_pct_last{w}')
            feature_cols += [c for c in team_df.columns if f'_last{w}' in c and 'win_pct' not in c]
            
        features_subset = team_df[feature_cols].copy()
        
        # Merge Home
        # Ensure we are merging on same types. features_subset date is datetime.
        game_features = pd.merge(
            games_df, 
            features_subset.add_prefix('home_'), 
            left_on=['home_team_id', 'date'],
            right_on=['home_team_id', 'home_date'],
            how='inner'
        )
        
        # Merge Visitor
        game_features = pd.merge(
            game_features,
            features_subset.add_prefix('visitor_'),
            left_on=['visitor_team_id', 'date'],
            right_on=['visitor_team_id', 'visitor_date'],
            how='inner'
        )
        
        # 6. Create Derived Interaction Features
        print("Creating interaction features...")
        
        # ELO Difference (most predictive single feature)
        game_features['elo_diff'] = game_features['home_elo'] - game_features['visitor_elo']
        
        # Rest advantage
        game_features['rest_advantage'] = game_features['home_rest_days'] - game_features['visitor_rest_days']
        
        # Momentum (recent performance differential)
        game_features['momentum_diff_5'] = game_features['home_win_pct_last5'] - game_features['visitor_win_pct_last5']
        game_features['momentum_diff_10'] = game_features['home_win_pct_last10'] - game_features['visitor_win_pct_last10']
        
        # Net Rating differential
        if 'home_point_diff_last10' in game_features.columns:
            game_features['net_rating_diff'] = game_features['home_point_diff_last10'] - game_features['visitor_point_diff_last10']
        
        # 7. Final Selection
        # Targets
        game_features['home_score'] = game_features['home_team_score']
        game_features['visitor_score'] = game_features['visitor_team_score']
        game_features['home_won'] = (game_features['home_team_score'] > game_features['visitor_team_score']).astype(int)
        
        # Construct Final Column List
        final_cols = ['game_id', 'date', 'season', 'home_team_id', 'visitor_team_id']
        final_cols += ['home_score', 'visitor_score', 'home_won']
        
        # Add ELO (individual + diff)
        final_cols += ['home_elo', 'visitor_elo', 'elo_diff']
        
        # Add Derived Features
        final_cols += ['rest_advantage', 'momentum_diff_5', 'momentum_diff_10']
        if 'net_rating_diff' in game_features.columns:
            final_cols.append('net_rating_diff')
        
        # Add Engineered Features (excluding keys)
        feat_cols_only = [c for c in features_subset.columns if c not in ['team_id', 'date']]
        for c in feat_cols_only:
            final_cols.append(f'home_{c}')
            final_cols.append(f'visitor_{c}')
            
        final_df = game_features[game_features['season'].isin(seasons)][final_cols].copy()
        final_df = final_df.fillna(0)
        
        print(f"Engineering Complete. Data shape: {final_df.shape}")
        return final_df

    def build_features_for_game(self, game: dict, historical_data: dict, current_season: int) -> dict:
        """
        Build features for a single upcoming game using historical data.
        
        Args:
            game: Dict with game info (home_team_id, visitor_team_id, etc.)
            historical_data: Dict with 'games', 'standings', etc. from DataManager
            current_season: Current NBA season
            
        Returns:
            Dict of features for this game matching model's expected features
        """
        import json
        import os
        
        # Load feature names from model metadata
        metadata_path = os.path.join('models', 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            expected_features = metadata.get('feature_names', [])
        else:
            expected_features = []
        
        home_id = game.get('home_team_id') or game.get('home_team', {}).get('id')
        visitor_id = game.get('visitor_team_id') or game.get('visitor_team', {}).get('id')
        
        if not home_id or not visitor_id:
            return None
        
        # Get historical games
        games_df = historical_data.get('games', pd.DataFrame())
        if games_df.empty:
            return None
        
        # Build team performance stats from historical data
        features = {}
        
        # Get ELO ratings (use most recent from historical data)
        if 'home_elo' in games_df.columns:
            # Get latest ELO for each team
            home_games = games_df[games_df['home_team_id'] == home_id].sort_values('date', ascending=False)
            visitor_games = games_df[games_df['home_team_id'] == visitor_id].sort_values('date', ascending=False)
            
            if not home_games.empty:
                features['home_elo'] = home_games['home_elo'].iloc[0]
            else:
                # Check as visitor
                home_as_vis = games_df[games_df['visitor_team_id'] == home_id].sort_values('date', ascending=False)
                features['home_elo'] = home_as_vis['visitor_elo'].iloc[0] if not home_as_vis.empty else 1500
                
            if not visitor_games.empty:
                features['visitor_elo'] = visitor_games['home_elo'].iloc[0]
            else:
                vis_as_vis = games_df[games_df['visitor_team_id'] == visitor_id].sort_values('date', ascending=False)
                features['visitor_elo'] = vis_as_vis['visitor_elo'].iloc[0] if not vis_as_vis.empty else 1500
        else:
            features['home_elo'] = 1500
            features['visitor_elo'] = 1500
        
        features['elo_diff'] = features['home_elo'] - features['visitor_elo']
        
        # Calculate rolling stats for each team
        for prefix, team_id in [('home', home_id), ('visitor', visitor_id)]:
            # Get team's recent games (as home or visitor)
            team_home_games = games_df[games_df['home_team_id'] == team_id].copy()
            team_away_games = games_df[games_df['visitor_team_id'] == team_id].copy()
            
            # Combine and sort by date
            team_home_games['team_points'] = team_home_games.get('home_team_score', 0)
            team_home_games['opp_points'] = team_home_games.get('visitor_team_score', 0)
            team_home_games['won'] = (team_home_games['team_points'] > team_home_games['opp_points']).astype(int)
            
            team_away_games['team_points'] = team_away_games.get('visitor_team_score', 0)
            team_away_games['opp_points'] = team_away_games.get('home_team_score', 0)
            team_away_games['won'] = (team_away_games['team_points'] > team_away_games['opp_points']).astype(int)
            
            all_games = pd.concat([team_home_games, team_away_games]).sort_values('date', ascending=False)
            
            # Rolling windows
            for window in [5, 10, 20]:
                recent = all_games.head(window)
                if len(recent) > 0:
                    features[f'{prefix}_win_pct_last{window}'] = recent['won'].mean()
                    features[f'{prefix}_points_scored_last{window}'] = recent['team_points'].mean()
                    features[f'{prefix}_points_allowed_last{window}'] = recent['opp_points'].mean()
                    features[f'{prefix}_point_diff_last{window}'] = features[f'{prefix}_points_scored_last{window}'] - features[f'{prefix}_points_allowed_last{window}']
                else:
                    features[f'{prefix}_win_pct_last{window}'] = 0.5
                    features[f'{prefix}_points_scored_last{window}'] = 110
                    features[f'{prefix}_points_allowed_last{window}'] = 110
                    features[f'{prefix}_point_diff_last{window}'] = 0
            
            # Rest days (default 2)
            features[f'{prefix}_rest_days'] = 2
            features[f'{prefix}_is_b2b'] = 0
            features[f'{prefix}_is_3in4'] = 0
            features[f'{prefix}_is_4in5'] = 0
            
            # Default four factors
            for window in [5, 10, 20]:
                features[f'{prefix}_efg_pct_last{window}'] = 0.5
                features[f'{prefix}_tov_pct_last{window}'] = 0.12
                features[f'{prefix}_oreb_pct_last{window}'] = 0.25
                features[f'{prefix}_ftr_last{window}'] = 0.25
                features[f'{prefix}_opp_efg_pct_last{window}'] = 0.5
                features[f'{prefix}_opp_tov_pct_last{window}'] = 0.12
                features[f'{prefix}_opp_oreb_pct_last{window}'] = 0.25
                features[f'{prefix}_opp_ftr_last{window}'] = 0.25
                features[f'{prefix}_pace_last{window}'] = 100
        
        # Derived features
        features['rest_advantage'] = features['home_rest_days'] - features['visitor_rest_days']
        features['momentum_diff_5'] = features['home_win_pct_last5'] - features['visitor_win_pct_last5']
        features['momentum_diff_10'] = features['home_win_pct_last10'] - features['visitor_win_pct_last10']
        features['net_rating_diff'] = features['home_point_diff_last10'] - features['visitor_point_diff_last10']
        
        # Enhanced features (for new model)
        features['implied_home_prob'] = 1 / (1 + 10 ** (-features['elo_diff'] / 400))
        features['implied_away_prob'] = 1 - features['implied_home_prob']
        features['prob_edge'] = abs(features['implied_home_prob'] - 0.5)
        features['log_odds_home'] = np.log(features['implied_home_prob'] / (1 - features['implied_home_prob'] + 0.001))
        features['weighted_momentum'] = features['momentum_diff_5'] * 0.6 + features['momentum_diff_10'] * 0.4
        features['b2b_advantage'] = features['visitor_is_b2b'] - features['home_is_b2b']
        
        # Consistency diffs
        for window in [5, 10, 20]:
            features[f'consistency_diff_{window}'] = features[f'home_point_diff_last{window}'] - features[f'visitor_point_diff_last{window}']
        
        # Net ratings
        features['home_net_rating_5'] = features['home_points_scored_last5'] - features['home_points_allowed_last5']
        features['visitor_net_rating_5'] = features['visitor_points_scored_last5'] - features['visitor_points_allowed_last5']
        features['home_net_rating_10'] = features['home_points_scored_last10'] - features['home_points_allowed_last10']
        features['visitor_net_rating_10'] = features['visitor_points_scored_last10'] - features['visitor_points_allowed_last10']
        features['net_rating_diff_5'] = features['home_net_rating_5'] - features['visitor_net_rating_5']
        features['net_rating_diff_10'] = features['home_net_rating_10'] - features['visitor_net_rating_10']
        
        # Additional features
        features['home_court_strength'] = 0.05
        features['elo_momentum_interaction'] = features['elo_diff'] * features['momentum_diff_10']
        features['rest_performance_interaction'] = features['rest_advantage'] * features['net_rating_diff']
        features['min_elo'] = min(features['home_elo'], features['visitor_elo'])
        features['quality_game'] = 1.0 if features['min_elo'] > 1500 else 0.0
        features['underdog_at_home'] = 1.0 if features['visitor_elo'] > features['home_elo'] + 50 else 0.0
        
        # Fill any missing expected features with 0
        for feat in expected_features:
            if feat not in features:
                features[feat] = 0
        
        return features

if __name__ == "__main__":
    pass

