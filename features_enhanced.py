"""
Enhanced Feature Engineering for College Basketball
Adds: Vegas odds, H2H history, situational factors
Adapted from NBA model
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import config
from features import FeatureEngineer


class EnhancedFeatureEngineer(FeatureEngineer):
    """Enhanced feature engineering with Vegas, H2H, and situational data"""
    
    def __init__(self, gender: str = "mens"):
        super().__init__(gender)
    
    def build_features_for_game(self, game_dict, historical_data, current_season, **kwargs):
        """Build complete feature set (Base + Enhanced)"""
        base_features = super().build_features_for_game(game_dict, historical_data, current_season)
        
        if base_features is None:
            base_features = {
                'home_elo': 1500, 'visitor_elo': 1500, 'elo_diff': 0,
                'home_rest_days': 2, 'visitor_rest_days': 2,
                'rest_advantage': 0, 'momentum_diff_5': 0, 'momentum_diff_10': 0,
                'net_rating_diff': 0,
                'home_is_b2b': 0, 'visitor_is_b2b': 0,
            }
            for w in [5, 10, 20]:
                for p in ['home', 'visitor']:
                    base_features[f'{p}_win_pct_last{w}'] = 0.5
                    base_features[f'{p}_points_scored_last{w}'] = self.avg_score
                    base_features[f'{p}_points_allowed_last{w}'] = self.avg_score
                    base_features[f'{p}_point_diff_last{w}'] = 0
        
        # Ensure required keys
        if 'home_team_id' not in base_features:
            base_features['home_team_id'] = game_dict.get('home_team_id')
        if 'visitor_team_id' not in base_features:
            base_features['visitor_team_id'] = game_dict.get('visitor_team_id')
        if 'date' not in base_features:
            base_features['date'] = game_dict.get('date')
        if 'game_id' not in base_features:
            base_features['game_id'] = game_dict.get('game_id') or game_dict.get('id')
        
        df = pd.DataFrame([base_features])
        
        enhanced_df = self.enhance_features(
            df,
            odds_df=kwargs.get('odds_df'),
            standings=kwargs.get('standings'),
            historical_games=historical_data.get('games') if historical_data else None,
        )
        
        if not enhanced_df.empty:
            return enhanced_df.iloc[0].to_dict()
        return base_features
    
    def add_vegas_features(self, games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Add Vegas odds as features"""
        if odds_df is None or odds_df.empty:
            games_df['vegas_spread_home'] = 0.0
            games_df['vegas_total'] = self.avg_total
            games_df['vegas_implied_home_prob'] = 0.5
            games_df['vegas_ml_home'] = -110
            games_df['vegas_ml_away'] = -110
            games_df['vegas_has_odds'] = 0
            return games_df
        
        vegas_features = []
        for idx, game in games_df.iterrows():
            raw_id = game.get('game_id') or game.get('id')
            if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)):
                game_odds = pd.DataFrame()
            else:
                game_id = str(int(raw_id)) if isinstance(raw_id, float) else str(raw_id)
                game_odds = odds_df[odds_df['game_id'].astype(str) == game_id]
            
            if game_odds.empty:
                vegas_features.append({
                    'vegas_spread_home': 0.0,
                    'vegas_total': self.avg_total,
                    'vegas_implied_home_prob': 0.5,
                    'vegas_ml_home': -110,
                    'vegas_ml_away': -110,
                    'vegas_has_odds': 0
                })
            else:
                spread_home = pd.to_numeric(game_odds['spread_home_value'], errors='coerce').median()
                total = pd.to_numeric(game_odds['total_value'], errors='coerce').median()
                ml_home = pd.to_numeric(game_odds['moneyline_home_odds'], errors='coerce').median()
                ml_away = pd.to_numeric(game_odds['moneyline_away_odds'], errors='coerce').median()
                
                if pd.notna(ml_home) and abs(ml_home) <= 5000:
                    if ml_home < 0:
                        implied_prob = abs(ml_home) / (abs(ml_home) + 100)
                    else:
                        implied_prob = 100 / (ml_home + 100)
                elif pd.notna(spread_home):
                    implied_prob = 0.5 + (-spread_home * 0.03)
                else:
                    implied_prob = 0.5
                
                implied_prob = max(0.05, min(0.95, implied_prob))
                clean_spread = spread_home if pd.notna(spread_home) and abs(spread_home) <= 40 else 0.0
                clean_total = total if pd.notna(total) and 90 <= total <= 220 else self.avg_total
                
                vegas_features.append({
                    'vegas_spread_home': clean_spread,
                    'vegas_total': clean_total,
                    'vegas_implied_home_prob': implied_prob,
                    'vegas_ml_home': ml_home if pd.notna(ml_home) and abs(ml_home) <= 5000 else -110,
                    'vegas_ml_away': ml_away if pd.notna(ml_away) and abs(ml_away) <= 5000 else -110,
                    'vegas_has_odds': 1
                })
        
        vegas_df = pd.DataFrame(vegas_features, index=games_df.index)
        return pd.concat([games_df, vegas_df], axis=1)
    
    def add_h2h_features(self, games_df: pd.DataFrame, historical_games: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head history features"""
        h2h_features = []
        
        for idx, game in games_df.iterrows():
            home_id = game['home_team_id']
            visitor_id = game['visitor_team_id']
            game_date = pd.to_datetime(game['date'])
            
            mask = (
                ((historical_games['home_team_id'] == home_id) & (historical_games['visitor_team_id'] == visitor_id)) |
                ((historical_games['home_team_id'] == visitor_id) & (historical_games['visitor_team_id'] == home_id))
            )
            matchups = historical_games[mask].copy()
            matchups['date'] = pd.to_datetime(matchups['date'])
            matchups = matchups[matchups['date'] < game_date].sort_values('date', ascending=False)
            
            if len(matchups) == 0:
                h2h_features.append({
                    'h2h_games': 0, 'h2h_home_wins': 0, 'h2h_home_win_pct': 0.5,
                    'h2h_avg_margin': 0.0, 'h2h_last3_home_wins': 0.5,
                })
            else:
                home_wins = 0
                margins = []
                for _, m in matchups.head(10).iterrows():
                    if m['home_team_id'] == home_id:
                        if m['home_team_score'] > m['visitor_team_score']:
                            home_wins += 1
                        margins.append(m['home_team_score'] - m['visitor_team_score'])
                    else:
                        if m['visitor_team_score'] > m['home_team_score']:
                            home_wins += 1
                        margins.append(m['visitor_team_score'] - m['home_team_score'])
                
                total = len(matchups.head(10))
                last3 = matchups.head(3)
                l3_wins = sum(1 for _, m in last3.iterrows()
                    if (m['home_team_id'] == home_id and m['home_team_score'] > m['visitor_team_score'])
                    or (m['home_team_id'] != home_id and m['visitor_team_score'] > m['home_team_score']))
                
                h2h_features.append({
                    'h2h_games': total,
                    'h2h_home_wins': home_wins,
                    'h2h_home_win_pct': home_wins / total if total > 0 else 0.5,
                    'h2h_avg_margin': np.mean(margins) if margins else 0.0,
                    'h2h_last3_home_wins': l3_wins / len(last3) if len(last3) > 0 else 0.5,
                })
        
        h2h_df = pd.DataFrame(h2h_features, index=games_df.index)
        return pd.concat([games_df, h2h_df], axis=1)
    
    def add_situational_features(self, games_df: pd.DataFrame,
                                  standings: pd.DataFrame = None) -> pd.DataFrame:
        """Add situational features: season phase, conference rankings"""
        situational = []
        
        for idx, game in games_df.iterrows():
            features = {}
            game_date = pd.to_datetime(game['date'])
            month = game_date.month
            
            # Season phase for college basketball
            if month in [11]:
                features['season_phase'] = 1  # Early (non-conference)
            elif month in [12, 1]:
                features['season_phase'] = 2  # Mid (conference starts)
            elif month in [2]:
                features['season_phase'] = 3  # Late (conference grind)
            elif month in [3]:
                features['season_phase'] = 4  # Tournament time
            else:
                features['season_phase'] = 2
            
            features['is_late_season'] = 1 if month in [2, 3] else 0
            features['is_march'] = 1 if month == 3 else 0
            
            situational.append(features)
        
        sit_df = pd.DataFrame(situational, index=games_df.index)
        return pd.concat([games_df, sit_df], axis=1)
    
    def build_training_dataset(self, all_data: Dict, seasons: List[int]) -> pd.DataFrame:
        """Override base to add enhanced features"""
        base_df = super().build_training_dataset(all_data, seasons)
        if base_df.empty:
            return base_df
        
        historical_games = all_data.get('games', pd.DataFrame())
        if not historical_games.empty:
            historical_games = historical_games.copy()
            historical_games['date'] = pd.to_datetime(historical_games['date'])
            
            print("\nAdding H2H features...")
            try:
                base_df = self.add_h2h_features(base_df, historical_games)
            except Exception as e:
                print(f"  H2H features failed: {e}")
                base_df['h2h_games'] = 0
                base_df['h2h_home_wins'] = 0
                base_df['h2h_home_win_pct'] = 0.5
                base_df['h2h_avg_margin'] = 0.0
                base_df['h2h_last3_home_wins'] = 0.5
        
        print("Adding situational features...")
        try:
            standings = all_data.get('standings', pd.DataFrame())
            base_df = self.add_situational_features(base_df, standings)
        except Exception as e:
            print(f"  Situational features failed: {e}")
            base_df['season_phase'] = 2
            base_df['is_late_season'] = 0
            base_df['is_march'] = 0
        
        # Default Vegas features
        base_df['vegas_spread_home'] = 0.0
        base_df['vegas_total'] = self.avg_total
        base_df['vegas_implied_home_prob'] = 0.5
        base_df['vegas_ml_home'] = -110
        base_df['vegas_ml_away'] = -110
        base_df['vegas_has_odds'] = 0
        
        # Default injury features
        base_df['injury_impact_diff'] = 0.0
        base_df['home_injury_impact'] = 0.0
        base_df['visitor_injury_impact'] = 0.0
        
        print(f"\nEnhanced training dataset: {base_df.shape[0]} games x {base_df.shape[1]} features")
        return base_df
    
    def enhance_features(self, games_df: pd.DataFrame,
                         odds_df: pd.DataFrame = None,
                         standings: pd.DataFrame = None,
                         historical_games: pd.DataFrame = None) -> pd.DataFrame:
        """Apply all feature enhancements"""
        result = games_df.copy()
        
        if odds_df is not None:
            result = self.add_vegas_features(result, odds_df)
        else:
            result['vegas_spread_home'] = 0.0
            result['vegas_total'] = self.avg_total
            result['vegas_implied_home_prob'] = 0.5
            result['vegas_ml_home'] = -110
            result['vegas_ml_away'] = -110
            result['vegas_has_odds'] = 0
        
        if historical_games is not None and not historical_games.empty:
            result = self.add_h2h_features(result, historical_games)
        else:
            result['h2h_games'] = 0
            result['h2h_home_wins'] = 0
            result['h2h_home_win_pct'] = 0.5
            result['h2h_avg_margin'] = 0.0
            result['h2h_last3_home_wins'] = 0.5
        
        result['injury_impact_diff'] = 0.0
        result['home_injury_impact'] = 0.0
        result['visitor_injury_impact'] = 0.0
        
        result = self.add_situational_features(result, standings)
        
        return result
