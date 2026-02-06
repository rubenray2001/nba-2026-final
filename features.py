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
        Calculate ELO ratings for every game with season regression.
        Between seasons, ratings regress 1/3 toward 1500 to account for
        roster turnover, draft picks, trades, and coaching changes.
        Returns a dataframe with 'elo_home' and 'elo_visitor' for each game_id.
        """
        # Sort by date
        df = games_df.sort_values('date').copy()
        
        # Initialize ELO dict
        elo_ratings = {} # team_id -> rating
        current_season = None  # Track season transitions
        
        # Season regression factor (regress 1/3 toward mean between seasons)
        SEASON_REGRESSION = 1/3
        
        # Store results lists to build dataframe later (faster than appending rows)
        home_elos = []
        visitor_elos = []
        
        # Iterate through all games chronologically
        for _, row in df.iterrows():
            hid = row['home_team_id']
            vid = row['visitor_team_id']
            game_season = row.get('season', None)
            
            # Apply season regression when transitioning to a new season
            if game_season is not None and current_season is not None and game_season != current_season:
                for team_id in list(elo_ratings.keys()):
                    old_elo = elo_ratings[team_id]
                    elo_ratings[team_id] = old_elo + SEASON_REGRESSION * (self.ELO_MEAN - old_elo)
            current_season = game_season
            
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
            
            # Margin of Victory Multiplier — log(MOV+1) improves ELO accuracy
            mov_mult = np.log(abs(h_score - v_score) + 1) if (h_score > 0 and v_score > 0) else 1.0
            
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
        # shift(2) = 2 games ago. With current game that's 3 games total.
        # "3 games in 4 nights" means the span from 2-games-ago to now is <= 3 days.
        # (Night1=game, Night2=game, Night3=off, Night4=game = 3 day span)
        team_df['date_3_games_ago'] = grouped.shift(2)
        msg_gap = (team_df['date'] - team_df['date_3_games_ago']).dt.days
        team_df['is_3in4'] = (msg_gap <= 3).astype(float).fillna(0)
        
        # 4 in 5 Nights:
        # shift(3) = 3 games ago. With current game that's 4 games total.
        # "4 games in 5 nights" means the span from 3-games-ago to now is <= 4 days.
        team_df['date_4_games_ago'] = grouped.shift(3)
        msg_gap_4 = (team_df['date'] - team_df['date_4_games_ago']).dt.days
        team_df['is_4in5'] = (msg_gap_4 <= 4).astype(float).fillna(0)

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
        
        # Smart NaN filling: use feature-appropriate defaults instead of blanket 0
        # Percentages -> 0.5 (neutral), ELO -> 1500 (average), scores -> league avg
        fill_defaults = {}
        for col in final_df.columns:
            if 'elo' in col.lower():
                fill_defaults[col] = 1500
            elif 'win_pct' in col.lower():
                fill_defaults[col] = 0.5
            elif 'efg_pct' in col.lower():
                fill_defaults[col] = 0.54
            elif 'tov_pct' in col.lower():
                fill_defaults[col] = 0.13
            elif 'oreb_pct' in col.lower():
                fill_defaults[col] = 0.25
            elif 'ftr' in col.lower() and 'last' in col.lower():
                fill_defaults[col] = 0.20
            elif 'points_scored' in col.lower():
                fill_defaults[col] = 110
            elif 'points_allowed' in col.lower():
                fill_defaults[col] = 110
            elif 'pace' in col.lower():
                fill_defaults[col] = 98
            elif 'rest_days' in col.lower():
                fill_defaults[col] = 2
            elif 'vegas_total' in col.lower():
                fill_defaults[col] = 220.0
            elif 'vegas_implied' in col.lower():
                fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower():
                fill_defaults[col] = 0.5
            else:
                fill_defaults[col] = 0
        final_df = final_df.fillna(fill_defaults)
        
        # FIX #2: Remove cold-start games where teams haven't played enough games
        # These games have zero/meaningless rolling stats and pollute training data
        initial_count = len(final_df)
        
        # Filter: both teams must have played at least 5 games (win_pct_last5 > 0 means at least 1 game in window)
        # More reliable: check if points_scored > 0 for both teams
        cold_start_mask = (
            (final_df['home_win_pct_last10'] > 0) & 
            (final_df['visitor_win_pct_last10'] > 0) &
            (final_df['home_points_scored_last5'] > 0) &
            (final_df['visitor_points_scored_last5'] > 0)
        )
        final_df = final_df[cold_start_mask]
        
        removed_count = initial_count - len(final_df)
        print(f"Removed {removed_count} cold-start games ({removed_count/initial_count*100:.1f}%)")
        
        print(f"Engineering Complete. Data shape: {final_df.shape}")
        return final_df

    def build_features_for_game(self, game: dict, historical_data: dict, current_season: int) -> dict:
        """
        Build features for a single upcoming game using historical data.
        """
        import json
        import os
        
        # Load feature names from model metadata
        metadata_path = os.path.join(config.MODELS_DIR, 'model_metadata.json')
        print(f"DEBUG: Loading metadata from {metadata_path}")
        
        expected_features = []
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                expected_features = metadata.get('feature_names', [])
                print(f"DEBUG: Successfully loaded {len(expected_features)} expected features")
            except Exception as e:
                print(f"DEBUG: Failed to parse metadata JSON: {e}")
        else:
            print(f"DEBUG: Metadata file NOT found at {metadata_path}")
        
        home_id = game.get('home_team_id') or game.get('home_team', {}).get('id')
        visitor_id = game.get('visitor_team_id') or game.get('visitor_team', {}).get('id')
        
        if not home_id or not visitor_id:
            return None
        
        # Get historical games
        games_df = historical_data.get('games', pd.DataFrame())
        if games_df.empty:
            return None
            
        # Ensure dates
        games_df['date'] = pd.to_datetime(games_df['date'])
        
        features = {}
        
        # 1. ELO Rating
        if 'home_elo' in games_df.columns:
            # Get latest ELO for each team
            h_games = games_df[(games_df['home_team_id'] == home_id) | (games_df['visitor_team_id'] == home_id)].sort_values('date', ascending=False)
            v_games = games_df[(games_df['home_team_id'] == visitor_id) | (games_df['visitor_team_id'] == visitor_id)].sort_values('date', ascending=False)
            
            if not h_games.empty:
                last_g = h_games.iloc[0]
                features['home_elo'] = last_g['home_elo'] if last_g['home_team_id'] == home_id else last_g['visitor_elo']
            else:
                features['home_elo'] = 1500
                
            if not v_games.empty:
                last_g = v_games.iloc[0]
                features['visitor_elo'] = last_g['home_elo'] if last_g['home_team_id'] == visitor_id else last_g['visitor_elo']
            else:
                features['visitor_elo'] = 1500
        else:
            features['home_elo'] = 1500
            features['visitor_elo'] = 1500
            
        features['elo_diff'] = features['home_elo'] - features['visitor_elo']
        
        # 2. Rolling Stats Calculation
        # We need to reconstruct the performance history for each team
        # including the advanced stats columns
        
        stat_map = ['fgm', 'fga', 'fg3m', 'ftm', 'fta', 'oreb', 'dreb', 'tov', 'pf']
        
        for prefix, team_id in [('home', home_id), ('visitor', visitor_id)]:
            # Get all games for this team
            team_games = games_df[(games_df['home_team_id'] == team_id) | (games_df['visitor_team_id'] == team_id)].copy()
            team_games = team_games.sort_values('date', ascending=False)
            
            # Normalize to team/opponent perspective
            # Create temporary columns for "my team" stats
            
            # Function to extract 'my' stat taking into account home/away
            def get_stat(row, stat):
                if row['home_team_id'] == team_id:
                    return row.get(f'home_{stat}', 0)
                else:
                    return row.get(f'visitor_{stat}', 0)
            
            def get_opp_stat(row, stat):
                if row['home_team_id'] == team_id:
                    return row.get(f'visitor_{stat}', 0)
                else:
                    return row.get(f'home_{stat}', 0)

            # Pre-calculate key metrics for the dataframe
            # We do this vectorized for speed if possible, but apply is safe
            team_games['points_scored'] = np.where(team_games['home_team_id'] == team_id, team_games['home_team_score'], team_games['visitor_team_score'])
            team_games['points_allowed'] = np.where(team_games['home_team_id'] == team_id, team_games['visitor_team_score'], team_games['home_team_score'])
            team_games['won'] = (team_games['points_scored'] > team_games['points_allowed']).astype(int)
            
            # Extract advanced calc components
            for s in stat_map:
                # Vectorized lookup
                # col = 'home_' + s if home else 'visitor_' + s
                team_games[s] = np.where(team_games['home_team_id'] == team_id, 
                                        team_games.get(f'home_{s}', 0), 
                                        team_games.get(f'visitor_{s}', 0))
                team_games[f'opp_{s}'] = np.where(team_games['home_team_id'] == team_id, 
                                        team_games.get(f'visitor_{s}', 0), 
                                        team_games.get(f'home_{s}', 0))
            
            # Calculate Derived Factors per game
            # eFG%
            team_games['efg_pct'] = (team_games['fgm'] + 0.5 * team_games['fg3m']) / team_games['fga'].replace(0, 1)
            team_games['opp_efg_pct'] = (team_games['opp_fgm'] + 0.5 * team_games['opp_fg3m']) / team_games['opp_fga'].replace(0, 1)
            
            # TOV%
            poss = team_games['fga'] + 0.44 * team_games['fta'] + team_games['tov']
            team_games['tov_pct'] = team_games['tov'] / poss.replace(0, 1)
            opp_poss = team_games['opp_fga'] + 0.44 * team_games['opp_fta'] + team_games['opp_tov']
            team_games['opp_tov_pct'] = team_games['opp_tov'] / opp_poss.replace(0, 1)
            
            # OREB%
            # approx: my_oreb / (my_oreb + opp_dreb)
            team_games['oreb_pct'] = team_games['oreb'] / (team_games['oreb'] + team_games['opp_dreb']).replace(0, 1)
            team_games['opp_oreb_pct'] = team_games['opp_oreb'] / (team_games['opp_oreb'] + team_games['dreb']).replace(0, 1)
            
            # FTR
            team_games['ftr'] = team_games['ftm'] / team_games['fga'].replace(0, 1)
            team_games['opp_ftr'] = team_games['opp_ftm'] / team_games['opp_fga'].replace(0, 1)
            
            # Pace
            team_games['pace'] = poss # Approx pace per game
            
            # Point Diff
            team_games['point_diff'] = team_games['points_scored'] - team_games['points_allowed']

            # Rolling Calculations
            for window in [5, 10, 20]:
                recent = team_games.head(window)
                if len(recent) > 0:
                    features[f'{prefix}_win_pct_last{window}'] = recent['won'].mean()
                    features[f'{prefix}_points_scored_last{window}'] = recent['points_scored'].mean()
                    features[f'{prefix}_points_allowed_last{window}'] = recent['points_allowed'].mean()
                    features[f'{prefix}_point_diff_last{window}'] = recent['point_diff'].mean()
                    
                    # Factors
                    features[f'{prefix}_efg_pct_last{window}'] = recent['efg_pct'].mean()
                    features[f'{prefix}_tov_pct_last{window}'] = recent['tov_pct'].mean()
                    features[f'{prefix}_oreb_pct_last{window}'] = recent['oreb_pct'].mean()
                    features[f'{prefix}_ftr_last{window}'] = recent['ftr'].mean()
                    features[f'{prefix}_opp_efg_pct_last{window}'] = recent['opp_efg_pct'].mean()
                    features[f'{prefix}_opp_tov_pct_last{window}'] = recent['opp_tov_pct'].mean()
                    features[f'{prefix}_opp_oreb_pct_last{window}'] = recent['opp_oreb_pct'].mean()
                    features[f'{prefix}_opp_ftr_last{window}'] = recent['opp_ftr'].mean()
                    
                    features[f'{prefix}_pace_last{window}'] = recent['pace'].mean()
                else:
                    # Fallbacks
                    features[f'{prefix}_win_pct_last{window}'] = 0.5
                    features[f'{prefix}_points_scored_last{window}'] = 110
                    features[f'{prefix}_points_allowed_last{window}'] = 110
                    features[f'{prefix}_point_diff_last{window}'] = 0
                    features[f'{prefix}_efg_pct_last{window}'] = 0.54
                    features[f'{prefix}_tov_pct_last{window}'] = 0.13
                    features[f'{prefix}_oreb_pct_last{window}'] = 0.25
                    features[f'{prefix}_ftr_last{window}'] = 0.20
                    features[f'{prefix}_opp_efg_pct_last{window}'] = 0.54
                    features[f'{prefix}_opp_tov_pct_last{window}'] = 0.13
                    features[f'{prefix}_opp_oreb_pct_last{window}'] = 0.25
                    features[f'{prefix}_opp_ftr_last{window}'] = 0.20
                    features[f'{prefix}_pace_last{window}'] = 98

            # Schedule metrics (Rest, B2B) - Calculate accurately
            if not team_games.empty:
                # Get the game date from the game dict
                game_date_str = game.get('date') or game.get('status', '')
                try:
                    if 'T' in str(game_date_str):
                        game_date = pd.to_datetime(game_date_str).tz_localize(None)
                    else:
                        game_date = pd.to_datetime(game_date_str)
                except (ValueError, TypeError):
                    game_date = pd.Timestamp.now()
                
                # Get team's last game date (team_games is sorted descending by date)
                last_game_date = pd.to_datetime(team_games.iloc[0]['date'])
                if hasattr(last_game_date, 'tz_localize'):
                    try:
                        last_game_date = last_game_date.tz_localize(None)
                    except TypeError:
                        pass
                
                # Calculate rest days
                rest_days = (game_date - last_game_date).days
                rest_days = max(0, min(rest_days, 10))  # Cap between 0 and 10
                
                features[f'{prefix}_rest_days'] = rest_days
                features[f'{prefix}_is_b2b'] = 1 if rest_days <= 1 else 0
                
                # Check 3 in 4 nights (3 games in 4 days)
                if len(team_games) >= 2:
                    third_game_date = pd.to_datetime(team_games.iloc[1]['date'])
                    days_for_3_games = (game_date - third_game_date).days
                    features[f'{prefix}_is_3in4'] = 1 if days_for_3_games <= 3 else 0
                else:
                    features[f'{prefix}_is_3in4'] = 0
                
                # Check 4 in 5 nights
                if len(team_games) >= 3:
                    fourth_game_date = pd.to_datetime(team_games.iloc[2]['date'])
                    days_for_4_games = (game_date - fourth_game_date).days
                    features[f'{prefix}_is_4in5'] = 1 if days_for_4_games <= 4 else 0
                else:
                    features[f'{prefix}_is_4in5'] = 0
            else:
                features[f'{prefix}_rest_days'] = 3
                features[f'{prefix}_is_b2b'] = 0
                features[f'{prefix}_is_3in4'] = 0
                features[f'{prefix}_is_4in5'] = 0

            # ---------------------------------------------------------
            # SEASON RECORD CALCULATION (Fix for 0-0 Record)
            # ---------------------------------------------------------
            # Filter games for CURRENT SEASON ONLY
            current_season_games = team_games[team_games['season'] == current_season]
            
            if not current_season_games.empty:
                wins = current_season_games['won'].sum()
                losses = len(current_season_games) - wins
                features[f'{prefix}_wins'] = int(wins)
                features[f'{prefix}_losses'] = int(losses)
                
                # Current Streak
                streak = 0
                if len(current_season_games) > 0:
                    last_results = current_season_games['won'].head(15).tolist() # Check last 15
                    if last_results:
                        current_status = last_results[0] # 1 for win, 0 for loss
                        for r in last_results:
                            if r == current_status:
                                streak += 1
                            else:
                                break
                        # If current status is 0 (loss), make streak negative
                        if current_status == 0:
                            streak = -streak
                
                features[f'{prefix}_streak'] = int(streak)
                
                # Conference Rank (Mock if missing)
                # In a real app we'd fetch this from standings API, but here we can approx or leave 0
                features[f'{prefix}_conference_rank'] = 0 
            else:
                features[f'{prefix}_wins'] = 0
                features[f'{prefix}_losses'] = 0
                features[f'{prefix}_streak'] = 0
                features[f'{prefix}_conference_rank'] = 0

        # 3. Derived Features
        features['rest_advantage'] = features['home_rest_days'] - features['visitor_rest_days']
        features['momentum_diff_5'] = features['home_win_pct_last5'] - features['visitor_win_pct_last5']
        features['momentum_diff_10'] = features['home_win_pct_last10'] - features['visitor_win_pct_last10']
        features['net_rating_diff'] = features['home_point_diff_last10'] - features['visitor_point_diff_last10']
        
        # 4. HEAD-TO-HEAD FEATURES
        h2h_mask = (
            ((games_df['home_team_id'] == home_id) & (games_df['visitor_team_id'] == visitor_id)) |
            ((games_df['home_team_id'] == visitor_id) & (games_df['visitor_team_id'] == home_id))
        )
        h2h_games = games_df[h2h_mask].sort_values('date', ascending=False)
        
        if len(h2h_games) > 0:
            # Calculate H2H record for home team
            h2h_home_wins = 0
            h2h_margins = []
            
            for _, g in h2h_games.head(10).iterrows():  # Last 10 H2H games
                if g['home_team_id'] == home_id:
                    if g['home_team_score'] > g['visitor_team_score']:
                        h2h_home_wins += 1
                    h2h_margins.append(g['home_team_score'] - g['visitor_team_score'])
                else:
                    if g['visitor_team_score'] > g['home_team_score']:
                        h2h_home_wins += 1
                    h2h_margins.append(g['visitor_team_score'] - g['home_team_score'])
            
            features['h2h_games'] = len(h2h_games.head(10))
            features['h2h_home_wins'] = h2h_home_wins
            features['h2h_home_win_pct'] = h2h_home_wins / len(h2h_games.head(10))
            features['h2h_avg_margin'] = np.mean(h2h_margins) if h2h_margins else 0
            features['h2h_last3_home_wins'] = sum(1 for i, (_, g) in enumerate(h2h_games.head(3).iterrows()) 
                                                   if (g['home_team_id'] == home_id and g['home_team_score'] > g['visitor_team_score'])
                                                   or (g['home_team_id'] != home_id and g['visitor_team_score'] > g['home_team_score'])) / max(1, len(h2h_games.head(3)))
        else:
            features['h2h_games'] = 0
            features['h2h_home_wins'] = 0
            features['h2h_home_win_pct'] = 0.5
            features['h2h_avg_margin'] = 0
            features['h2h_last3_home_wins'] = 0.5
        
        # 5. SITUATIONAL FEATURES
        # Conference rank difference
        features['conf_rank_diff'] = features.get('visitor_conference_rank', 8) - features.get('home_conference_rank', 8)
        
        # Playoff race (teams ranked 6-10)
        home_rank = features.get('home_conference_rank', 8)
        visitor_rank = features.get('visitor_conference_rank', 8)
        features['home_in_playoff_race'] = 1 if 6 <= home_rank <= 10 else 0
        features['visitor_in_playoff_race'] = 1 if 6 <= visitor_rank <= 10 else 0
        
        # Motivation score
        def get_motivation(rank):
            if rank <= 4: return 3  # Contender
            elif rank <= 10: return 2  # Playoff team
            else: return 1  # Lottery
        
        features['home_motivation'] = get_motivation(home_rank)
        features['visitor_motivation'] = get_motivation(visitor_rank)
        features['motivation_diff'] = features['home_motivation'] - features['visitor_motivation']
        
        # Season phase (based on current date)
        try:
            game_date = pd.to_datetime(game.get('date', pd.Timestamp.now()))
            month = game_date.month
            if month in [10, 11]:
                features['season_phase'] = 1  # Early
            elif month in [12, 1, 2]:
                features['season_phase'] = 2  # Mid
            elif month in [3, 4]:
                features['season_phase'] = 3  # Late/playoff push
            else:
                features['season_phase'] = 4  # Playoffs
            features['is_late_season'] = 1 if month in [3, 4] else 0
        except (ValueError, TypeError, KeyError):
            features['season_phase'] = 2
            features['is_late_season'] = 0
        
        # Default Vegas features (will be overridden if odds available)
        features['vegas_spread_home'] = 0.0
        features['vegas_total'] = 220.0
        features['vegas_implied_home_prob'] = 0.5
        features['vegas_has_odds'] = 0
        
        # Default injury features (including new PPG-weighted features)
        features['home_injuries_out'] = 0
        features['visitor_injuries_out'] = 0
        features['home_questionable'] = 0
        features['visitor_questionable'] = 0
        features['injury_diff'] = 0
        features['home_stars_out'] = 0
        features['visitor_stars_out'] = 0
        features['star_injury_diff'] = 0
        features['home_injury_impact'] = 0.0
        features['visitor_injury_impact'] = 0.0
        features['injury_impact_diff'] = 0.0
        
        # Default travel features
        features['visitor_travel_miles'] = 0
        features['visitor_tz_change'] = 0
        features['is_long_travel'] = 0
        
        # Fill missing with feature-appropriate defaults based on expected_features
        if expected_features:
            for feat in expected_features:
                if feat not in features:
                    # Use smart defaults based on feature name
                    if 'elo' in feat.lower():
                        features[feat] = 1500
                    elif 'win_pct' in feat.lower():
                        features[feat] = 0.5
                    elif 'efg_pct' in feat.lower():
                        features[feat] = 0.54
                    elif 'tov_pct' in feat.lower():
                        features[feat] = 0.13
                    elif 'oreb_pct' in feat.lower():
                        features[feat] = 0.25
                    elif 'ftr' in feat.lower() and 'last' in feat.lower():
                        features[feat] = 0.20
                    elif 'points_scored' in feat.lower():
                        features[feat] = 110
                    elif 'points_allowed' in feat.lower():
                        features[feat] = 110
                    elif 'pace' in feat.lower():
                        features[feat] = 98
                    elif 'rest_days' in feat.lower():
                        features[feat] = 2
                    elif 'vegas_total' in feat.lower():
                        features[feat] = 220.0
                    elif 'vegas_implied' in feat.lower():
                        features[feat] = 0.5
                    elif 'h2h_home_win_pct' in feat.lower() or 'h2h_last3' in feat.lower():
                        features[feat] = 0.5
                    else:
                        features[feat] = 0.0
                    
        return features

if __name__ == "__main__":
    pass

