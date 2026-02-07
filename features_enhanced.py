"""
Enhanced Feature Engineering
Adds: Vegas odds, H2H history, injury impact, situational factors
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import config

# Team location data for travel calculations (approximate timezone/region)
from features import FeatureEngineer

TEAM_LOCATIONS = {
    # Eastern Conference
    1: {'city': 'Atlanta', 'tz': 'ET', 'lat': 33.75, 'lon': -84.39},
    2: {'city': 'Boston', 'tz': 'ET', 'lat': 42.36, 'lon': -71.06},
    3: {'city': 'Brooklyn', 'tz': 'ET', 'lat': 40.68, 'lon': -73.97},
    4: {'city': 'Charlotte', 'tz': 'ET', 'lat': 35.23, 'lon': -80.84},
    5: {'city': 'Chicago', 'tz': 'CT', 'lat': 41.88, 'lon': -87.63},
    6: {'city': 'Cleveland', 'tz': 'ET', 'lat': 41.50, 'lon': -81.69},
    7: {'city': 'Detroit', 'tz': 'ET', 'lat': 42.33, 'lon': -83.05},
    8: {'city': 'Indiana', 'tz': 'ET', 'lat': 39.76, 'lon': -86.16},
    9: {'city': 'Miami', 'tz': 'ET', 'lat': 25.76, 'lon': -80.19},
    10: {'city': 'Milwaukee', 'tz': 'CT', 'lat': 43.04, 'lon': -87.92},
    11: {'city': 'New York', 'tz': 'ET', 'lat': 40.75, 'lon': -73.99},
    12: {'city': 'Orlando', 'tz': 'ET', 'lat': 28.54, 'lon': -81.38},
    13: {'city': 'Philadelphia', 'tz': 'ET', 'lat': 39.95, 'lon': -75.17},
    14: {'city': 'Toronto', 'tz': 'ET', 'lat': 43.64, 'lon': -79.38},
    15: {'city': 'Washington', 'tz': 'ET', 'lat': 38.90, 'lon': -77.02},
    # Western Conference
    16: {'city': 'Dallas', 'tz': 'CT', 'lat': 32.79, 'lon': -96.81},
    17: {'city': 'Denver', 'tz': 'MT', 'lat': 39.74, 'lon': -104.99},
    18: {'city': 'Golden State', 'tz': 'PT', 'lat': 37.77, 'lon': -122.42},
    19: {'city': 'Houston', 'tz': 'CT', 'lat': 29.76, 'lon': -95.36},
    20: {'city': 'LA Clippers', 'tz': 'PT', 'lat': 34.04, 'lon': -118.27},
    21: {'city': 'LA Lakers', 'tz': 'PT', 'lat': 34.04, 'lon': -118.27},
    22: {'city': 'Memphis', 'tz': 'CT', 'lat': 35.14, 'lon': -90.05},
    23: {'city': 'Minnesota', 'tz': 'CT', 'lat': 44.98, 'lon': -93.26},
    24: {'city': 'New Orleans', 'tz': 'CT', 'lat': 29.95, 'lon': -90.08},
    25: {'city': 'Oklahoma City', 'tz': 'CT', 'lat': 35.46, 'lon': -97.52},
    26: {'city': 'Phoenix', 'tz': 'MT', 'lat': 33.45, 'lon': -112.07},
    27: {'city': 'Portland', 'tz': 'PT', 'lat': 45.53, 'lon': -122.67},
    28: {'city': 'Sacramento', 'tz': 'PT', 'lat': 38.58, 'lon': -121.49},
    29: {'city': 'San Antonio', 'tz': 'CT', 'lat': 29.43, 'lon': -98.49},
    30: {'city': 'Utah', 'tz': 'MT', 'lat': 40.77, 'lon': -111.89},
}



# Star player PPG lookup for accurate injury weighting
STAR_PLAYER_PPG = {
    # 30+ PPG stars
    'shai gilgeous-alexander': 31.5, 'luka doncic': 33.0, 'giannis antetokounmpo': 31.0,
    'joel embiid': 35.0, 'jayson tatum': 27.5, 'kevin durant': 27.0, 'donovan mitchell': 26.0,
    'anthony edwards': 27.0, 'devin booker': 27.5, 'damian lillard': 25.5,
    # 25+ PPG
    'lebron james': 24.0, 'stephen curry': 23.0, 'jaylen brown': 24.0, 'trae young': 24.0,
    'jalen brunson': 25.5, 'de\'aaron fox': 26.5, 'anthony davis': 25.0, 'kyrie irving': 24.0,
    'ja morant': 21.0, 'lamelo ball': 22.0, 'cade cunningham': 24.0, 'tyrese haliburton': 20.0,
    # 20+ PPG
    'darius garland': 21.5, 'evan mobley': 18.5, 'scottie barnes': 20.0, 'karl-anthony towns': 25.0,
    'zion williamson': 23.0, 'brandon ingram': 23.0, 'pascal siakam': 21.5, 'bam adebayo': 18.5,
    'jimmy butler': 19.0, 'paul george': 22.5, 'kawhi leonard': 24.0, 'james harden': 21.0,
    'tyrese maxey': 26.0, 'chet holmgren': 17.0, 'jalen williams': 21.0,
    # 15+ PPG role players
    'fred vanvleet': 15.5, 'nikola vucevic': 18.0, 'alperen sengun': 19.0, 'domantas sabonis': 20.0,
    'lauri markkanen': 23.0, 'dejounte murray': 18.0, 'victor wembanyama': 22.0,
    'aaron gordon': 15.0, 'alex caruso': 10.0, 'luguentz dort': 12.0, 'isaiah hartenstein': 12.0,
    'keegan murray': 15.5, 'jamal murray': 21.0, 'michael porter jr': 15.0,
    'khris middleton': 16.0, 'bobby portis': 14.0, 'bradley beal': 18.0, 'chris paul': 10.0,
    'max strus': 12.0, 'miles mcbride': 10.0, 'sam hauser': 9.0, 'jakob poeltl': 14.0,
}


# Star players by team (keeping for legacy compatibility if needed)
STAR_PLAYERS = {
    1: ['Trae Young', 'Dejounte Murray'],
    2: ['Jayson Tatum', 'Jaylen Brown', 'Kristaps Porzingis'],
    3: ['Mikal Bridges', 'Cameron Johnson'],
    4: ['LaMelo Ball', 'Brandon Miller'],
    5: ['DeMar DeRozan', 'Zach LaVine'],
    6: ['Donovan Mitchell', 'Darius Garland', 'Evan Mobley'],
    7: ['Cade Cunningham', 'Jaden Ivey'],
    8: ['Tyrese Haliburton', 'Pascal Siakam'],
    9: ['Jimmy Butler', 'Bam Adebayo', 'Tyler Herro'],
    10: ['Giannis Antetokounmpo', 'Damian Lillard', 'Khris Middleton'],
    11: ['Jalen Brunson', 'Julius Randle', 'OG Anunoby'],
    12: ['Paolo Banchero', 'Franz Wagner'],
    13: ['Joel Embiid', 'Tyrese Maxey'],
    14: ['Scottie Barnes', 'Pascal Siakam', 'RJ Barrett'],
    15: ['Bradley Beal', 'Kyle Kuzma'],
    16: ['Luka Doncic', 'Kyrie Irving'],
    17: ['Nikola Jokic', 'Jamal Murray', 'Michael Porter Jr'],
    18: ['Stephen Curry', 'Klay Thompson', 'Draymond Green'],
    19: ['Jalen Green', 'Alperen Sengun'],
    20: ['Kawhi Leonard', 'Paul George', 'James Harden'],
    21: ['LeBron James', 'Anthony Davis', 'Austin Reaves'],
    22: ['Ja Morant', 'Desmond Bane'],
    23: ['Anthony Edwards', 'Karl-Anthony Towns', 'Rudy Gobert'],
    24: ['Zion Williamson', 'Brandon Ingram', 'CJ McCollum'],
    25: ['Shai Gilgeous-Alexander', 'Chet Holmgren', 'Jalen Williams'],
    26: ['Kevin Durant', 'Devin Booker', 'Bradley Beal'],
    27: ['Anfernee Simons', 'Jerami Grant'],
    28: ['De\'Aaron Fox', 'Domantas Sabonis'],
    29: ['Victor Wembanyama', 'Devin Vassell'],
    30: ['Lauri Markkanen', 'Collin Sexton', 'Jordan Clarkson'],
}


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in miles between two points"""
    # Haversine formula approximation
    R = 3959  # Earth's radius in miles
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def get_travel_distance(from_team_id: int, to_team_id: int) -> float:
    """Get travel distance between two teams"""
    if from_team_id not in TEAM_LOCATIONS or to_team_id not in TEAM_LOCATIONS:
        return 0.0
    
    from_loc = TEAM_LOCATIONS[from_team_id]
    to_loc = TEAM_LOCATIONS[to_team_id]
    
    return calculate_distance(from_loc['lat'], from_loc['lon'], 
                              to_loc['lat'], to_loc['lon'])


def get_timezone_change(from_team_id: int, to_team_id: int) -> int:
    """Get timezone difference (hours) for travel"""
    tz_map = {'PT': 0, 'MT': 1, 'CT': 2, 'ET': 3}
    
    if from_team_id not in TEAM_LOCATIONS or to_team_id not in TEAM_LOCATIONS:
        return 0
    
    from_tz = TEAM_LOCATIONS[from_team_id].get('tz', 'ET')
    to_tz = TEAM_LOCATIONS[to_team_id].get('tz', 'ET')
    
    return abs(tz_map.get(to_tz, 0) - tz_map.get(from_tz, 0))


class EnhancedFeatureEngineer(FeatureEngineer):
    """Enhanced feature engineering with Vegas, H2H, injuries, and situational data"""
    
    def __init__(self):
        super().__init__()
        
    def build_features_for_game(self, game_dict, historical_data, current_season, **kwargs):
        """
        Build complete feature set (Base + Enhanced)
        kwargs can include:
        - injuries: List[dict]
        - odds_df: pd.DataFrame
        - player_stats: Dict[int, float]
        - standings: pd.DataFrame
        """
        # 1. Get Base Features (Rolling stats, ELO, schedule)
        base_features = super().build_features_for_game(game_dict, historical_data, current_season)
        
        if base_features is None:
            # If base features failed (e.g. no history), use defaults so we can still add Injury/Vegas
            base_features = {
                'home_elo': 1500, 'visitor_elo': 1500, 'elo_diff': 0,
                'home_rest_days': 2, 'visitor_rest_days': 2,
                'rest_advantage': 0, 'momentum_diff_5': 0, 'momentum_diff_10': 0,
                'net_rating_diff': 0,
                # Schedule flags
                'home_is_b2b': 0, 'visitor_is_b2b': 0,
                'home_is_3in4': 0, 'visitor_is_3in4': 0,
                'home_is_4in5': 0, 'visitor_is_4in5': 0,
                # Rolling stats defaults (avoid KeyError in enhancers)
                'home_win_pct_last5': 0.5, 'visitor_win_pct_last5': 0.5,
                'home_win_pct_last10': 0.5, 'visitor_win_pct_last10': 0.5,
                'home_win_pct_last20': 0.5, 'visitor_win_pct_last20': 0.5,
                # Add minimal others if needed by enhancers, usually 0 is fine
            }
            
        # 2. Enhance with new features (Injury, Vegas, Situational)
        # Fix: Ensure base_features has the keys required by enhance_features (team_ids, date)
        if 'home_team_id' not in base_features:
            base_features['home_team_id'] = game_dict.get('home_team_id')
        if 'visitor_team_id' not in base_features:
            base_features['visitor_team_id'] = game_dict.get('visitor_team_id')
        if 'date' not in base_features:
            base_features['date'] = game_dict.get('date')
        if 'game_id' not in base_features:
            base_features['game_id'] = game_dict.get('game_id') or game_dict.get('id')

        # Convert dict to single-row DataFrame explicitly
        df = pd.DataFrame([base_features])
        
        # Call enhance_features with the passed kwargs
        # We need to extract them from kwargs or pass None
        enhanced_df = self.enhance_features(
            df, 
            odds_df=kwargs.get('odds_df'),
            injuries=kwargs.get('injuries'),
            standings=kwargs.get('standings'),
            historical_games=historical_data.get('games') if historical_data else None,
            player_stats=kwargs.get('player_stats')
        )
        
        # Convert back to dict (first row)
        if not enhanced_df.empty:
            return enhanced_df.iloc[0].to_dict()
        return base_features
    
    def add_vegas_features(self, games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Add Vegas odds as features"""
        if odds_df.empty:
            # Add default values if no odds
            games_df['vegas_spread_home'] = 0.0
            games_df['vegas_total'] = 220.0
            games_df['vegas_implied_home_prob'] = 0.5
            games_df['vegas_ml_home'] = -110
            games_df['vegas_ml_away'] = -110
            games_df['vegas_has_odds'] = 0
            return games_df
        
        # Merge odds with games
        vegas_features = []
        
        for idx, game in games_df.iterrows():
            raw_id = game.get('game_id') or game.get('id')
            # Guard against None/NaN game IDs
            if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)):
                game_odds = pd.DataFrame()  # no match possible
            else:
                game_id = str(int(raw_id)) if isinstance(raw_id, float) else str(raw_id)
                # Ensure odds_df game_id is also string for comparison
                game_odds = odds_df[odds_df['game_id'].astype(str) == game_id]
            
            if game_odds.empty:
                vegas_features.append({
                    'vegas_spread_home': 0.0,
                    'vegas_total': 220.0,
                    'vegas_implied_home_prob': 0.5,
                    'vegas_ml_home': -110,
                    'vegas_ml_away': -110,
                    'vegas_has_odds': 0
                })
            else:
                # Get median values across vendors
                spread_home = pd.to_numeric(game_odds['spread_home_value'], errors='coerce').median()
                total = pd.to_numeric(game_odds['total_value'], errors='coerce').median()
                ml_home = pd.to_numeric(game_odds['moneyline_home_odds'], errors='coerce').median()
                ml_away = pd.to_numeric(game_odds['moneyline_away_odds'], errors='coerce').median()
                
                # Calculate implied probability from moneyline
                # Filter out garbage moneyline values (API sometimes returns -199900 etc.)
                if pd.notna(ml_home) and abs(ml_home) <= 5000:
                    if ml_home < 0:
                        implied_prob = abs(ml_home) / (abs(ml_home) + 100)
                    else:
                        implied_prob = 100 / (ml_home + 100)
                elif pd.notna(spread_home):
                    # Fallback: estimate from spread (each point ~ 3% shift)
                    implied_prob = 0.5 + (-spread_home * 0.03)
                else:
                    implied_prob = 0.5
                
                # Clamp to realistic range (no team is >95% or <5% to win)
                implied_prob = max(0.05, min(0.95, implied_prob))
                
                vegas_features.append({
                    'vegas_spread_home': spread_home if pd.notna(spread_home) else 0.0,
                    'vegas_total': total if pd.notna(total) else 220.0,
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
            
            # Find previous matchups between these teams (before this game)
            mask = (
                ((historical_games['home_team_id'] == home_id) & (historical_games['visitor_team_id'] == visitor_id)) |
                ((historical_games['home_team_id'] == visitor_id) & (historical_games['visitor_team_id'] == home_id))
            )
            matchups = historical_games[mask].copy()
            matchups['date'] = pd.to_datetime(matchups['date'])
            matchups = matchups[matchups['date'] < game_date].sort_values('date', ascending=False)
            
            if len(matchups) == 0:
                h2h_features.append({
                    'h2h_games': 0,
                    'h2h_home_wins': 0,
                    'h2h_home_win_pct': 0.5,
                    'h2h_avg_margin': 0.0,
                    'h2h_home_avg_score': 110.0,
                    'h2h_visitor_avg_score': 110.0,
                    'h2h_last3_home_wins': 0.5,
                })
            else:
                # Calculate H2H stats
                # "Home wins" = wins by the team that is home in THIS game
                home_wins = 0
                margins = []
                home_scores = []
                visitor_scores = []
                
                for _, m in matchups.iterrows():
                    if m['home_team_id'] == home_id:
                        # This team was home in the historical game
                        if m['home_team_score'] > m['visitor_team_score']:
                            home_wins += 1
                        margins.append(m['home_team_score'] - m['visitor_team_score'])
                        home_scores.append(m['home_team_score'])
                        visitor_scores.append(m['visitor_team_score'])
                    else:
                        # This team was away in the historical game
                        if m['visitor_team_score'] > m['home_team_score']:
                            home_wins += 1
                        margins.append(m['visitor_team_score'] - m['home_team_score'])
                        home_scores.append(m['visitor_team_score'])
                        visitor_scores.append(m['home_team_score'])
                
                total_games = len(matchups)
                last3 = matchups.head(3)
                
                # Last 3 H2H games
                last3_wins = 0
                for _, m in last3.iterrows():
                    if m['home_team_id'] == home_id:
                        if m['home_team_score'] > m['visitor_team_score']:
                            last3_wins += 1
                    else:
                        if m['visitor_team_score'] > m['home_team_score']:
                            last3_wins += 1
                
                h2h_features.append({
                    'h2h_games': total_games,
                    'h2h_home_wins': home_wins,
                    'h2h_home_win_pct': home_wins / total_games if total_games > 0 else 0.5,
                    'h2h_avg_margin': np.mean(margins) if margins else 0.0,
                    'h2h_home_avg_score': np.mean(home_scores) if home_scores else 110.0,
                    'h2h_visitor_avg_score': np.mean(visitor_scores) if visitor_scores else 110.0,
                    'h2h_last3_home_wins': last3_wins / len(last3) if len(last3) > 0 else 0.5,
                })
        
        h2h_df = pd.DataFrame(h2h_features, index=games_df.index)
        return pd.concat([games_df, h2h_df], axis=1)
    
    def add_injury_features(self, games_df: pd.DataFrame, injuries: List[dict], 
                             player_stats: Dict[int, float] = None) -> pd.DataFrame:
        """
        Add injury impact features with PPG weighting
        
        Args:
            games_df: DataFrame with games
            injuries: List of injury dicts from API
            player_stats: Optional dict mapping player_id -> ppg for weighting
        """
        injury_features = []
        
        # Build injury lookup by team with player stats
        injuries_by_team = {}
        for inj in injuries:
            player = inj.get('player', {})
            team_id = player.get('team_id')
            player_id = player.get('id')
            
            if team_id:
                if team_id not in injuries_by_team:
                    injuries_by_team[team_id] = []
                
                # Get player name for lookup
                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".lower().strip()

                # Get player PPG from stats or lookup or estimate
                ppg = 0.0
                if player_stats and player_id in player_stats:
                    ppg = player_stats[player_id]
                elif player_name in STAR_PLAYER_PPG:
                    ppg = STAR_PLAYER_PPG[player_name]
                else:
                    # Estimate PPG from position/role
                    position = player.get('position', '')
                    if position in ['G', 'PG', 'SG']:
                        ppg = 10.0  # Guard avg estimate
                    elif position in ['F', 'SF', 'PF']:
                        ppg = 8.0  # Forward avg estimate
                    else:
                        ppg = 6.0  # Center avg estimate
                
                injuries_by_team[team_id].append({
                    'name': f"{player.get('first_name', '')} {player.get('last_name', '')}",
                    'status': inj.get('status', '').lower(),
                    'ppg': ppg,
                    'player_id': player_id
                })
        
        for idx, game in games_df.iterrows():
            home_id = game['home_team_id']
            visitor_id = game['visitor_team_id']
            
            home_injuries = injuries_by_team.get(home_id, [])
            visitor_injuries = injuries_by_team.get(visitor_id, [])
            
            # Count injuries by status
            home_out = len([i for i in home_injuries if i['status'] in ['out', 'doubtful', 'out for season']])
            visitor_out = len([i for i in visitor_injuries if i['status'] in ['out', 'doubtful', 'out for season']])
            home_questionable = len([i for i in home_injuries if i['status'] in ['questionable', 'day-to-day']])
            visitor_questionable = len([i for i in visitor_injuries if i['status'] in ['questionable', 'day-to-day']])
            
            # PPG-weighted impact (OUT = 100%, Questionable = 50%)
            home_ppg_out = sum(i['ppg'] for i in home_injuries if i['status'] in ['out', 'doubtful', 'out for season'])
            home_ppg_quest = sum(i['ppg'] for i in home_injuries if i['status'] in ['questionable', 'day-to-day']) * 0.5
            home_injury_impact = home_ppg_out + home_ppg_quest
            
            visitor_ppg_out = sum(i['ppg'] for i in visitor_injuries if i['status'] in ['out', 'doubtful', 'out for season'])
            visitor_ppg_quest = sum(i['ppg'] for i in visitor_injuries if i['status'] in ['questionable', 'day-to-day']) * 0.5
            visitor_injury_impact = visitor_ppg_out + visitor_ppg_quest
            
            # Check star players (anyone > 15 PPG)
            home_stars_out = len([i for i in home_injuries if i['ppg'] >= 15 and i['status'] in ['out', 'doubtful', 'out for season']])
            visitor_stars_out = len([i for i in visitor_injuries if i['ppg'] >= 15 and i['status'] in ['out', 'doubtful', 'out for season']])
            
            injury_features.append({
                'home_injuries_out': home_out,
                'visitor_injuries_out': visitor_out,
                'home_questionable': home_questionable,
                'visitor_questionable': visitor_questionable,
                'injury_diff': visitor_out - home_out,  # Positive = home advantage
                'home_stars_out': home_stars_out,
                'visitor_stars_out': visitor_stars_out,
                'star_injury_diff': visitor_stars_out - home_stars_out,
                # NEW: PPG-weighted impact features
                'home_injury_impact': home_injury_impact,
                'visitor_injury_impact': visitor_injury_impact,
                'injury_impact_diff': visitor_injury_impact - home_injury_impact,  # Positive = home advantage
            })
        
        injury_df = pd.DataFrame(injury_features, index=games_df.index)
        return pd.concat([games_df, injury_df], axis=1)
    
    def add_situational_features(self, games_df: pd.DataFrame, 
                                  standings: pd.DataFrame = None,
                                  historical_games: pd.DataFrame = None) -> pd.DataFrame:
        """Add situational features: travel, playoff race, motivation"""
        situational_features = []
        
        for idx, game in games_df.iterrows():
            home_id = game['home_team_id']
            visitor_id = game['visitor_team_id']
            game_date = pd.to_datetime(game['date'])
            
            features = {}
            
            # 1. Travel distance for visitor
            # Find visitor's last game location
            if historical_games is not None and not historical_games.empty:
                visitor_games = historical_games[
                    ((historical_games['home_team_id'] == visitor_id) | 
                     (historical_games['visitor_team_id'] == visitor_id))
                ].copy()
                visitor_games['date'] = pd.to_datetime(visitor_games['date'])
                visitor_games = visitor_games[visitor_games['date'] < game_date].sort_values('date', ascending=False)
                
                if len(visitor_games) > 0:
                    last_game = visitor_games.iloc[0]
                    # Where was the visitor team last?
                    if last_game['home_team_id'] == visitor_id:
                        last_location = visitor_id  # They were home
                    else:
                        last_location = last_game['home_team_id']  # They were away at this team's arena
                    
                    travel_dist = get_travel_distance(last_location, home_id)
                    tz_change = get_timezone_change(last_location, home_id)
                else:
                    travel_dist = 0
                    tz_change = 0
            else:
                travel_dist = 0
                tz_change = 0
            
            features['visitor_travel_miles'] = travel_dist
            features['visitor_tz_change'] = tz_change
            features['is_long_travel'] = 1 if travel_dist > 1500 else 0  # Cross-country trip
            
            # 2. Playoff race / motivation
            if standings is not None and not standings.empty:
                home_standing = standings[standings['team_id'] == home_id]
                visitor_standing = standings[standings['team_id'] == visitor_id]
                
                if len(home_standing) > 0:
                    home_conf_rank = home_standing.iloc[0].get('conference_rank', 8)
                    home_games_behind = home_standing.iloc[0].get('games_behind', 0)
                else:
                    home_conf_rank = 8
                    home_games_behind = 0
                
                if len(visitor_standing) > 0:
                    visitor_conf_rank = visitor_standing.iloc[0].get('conference_rank', 8)
                    visitor_games_behind = visitor_standing.iloc[0].get('games_behind', 0)
                else:
                    visitor_conf_rank = 8
                    visitor_games_behind = 0
            else:
                home_conf_rank = 8
                visitor_conf_rank = 8
                home_games_behind = 0
                visitor_games_behind = 0
            
            features['home_conf_rank'] = home_conf_rank
            features['visitor_conf_rank'] = visitor_conf_rank
            features['conf_rank_diff'] = visitor_conf_rank - home_conf_rank  # Positive = home better
            
            # Playoff race (teams ranked 6-10 fighting for spots)
            features['home_in_playoff_race'] = 1 if 6 <= home_conf_rank <= 10 else 0
            features['visitor_in_playoff_race'] = 1 if 6 <= visitor_conf_rank <= 10 else 0
            
            # Motivation: contender (1-4), playoff team (5-10), lottery (11-15)
            def get_motivation(rank):
                if rank <= 4:
                    return 3  # Contender - high motivation
                elif rank <= 10:
                    return 2  # Playoff race - medium-high
                else:
                    return 1  # Lottery - lower motivation (tanking?)
            
            features['home_motivation'] = get_motivation(home_conf_rank)
            features['visitor_motivation'] = get_motivation(visitor_conf_rank)
            features['motivation_diff'] = features['home_motivation'] - features['visitor_motivation']
            
            # 3. Month/Season phase (early, mid, late, playoffs approaching)
            month = game_date.month
            if month in [10, 11]:
                season_phase = 1  # Early season
            elif month in [12, 1, 2]:
                season_phase = 2  # Mid season
            elif month in [3, 4]:
                season_phase = 3  # Late season / playoff push
            else:
                season_phase = 4  # Playoffs
            
            features['season_phase'] = season_phase
            features['is_late_season'] = 1 if month in [3, 4] else 0
            
            situational_features.append(features)
        
        sit_df = pd.DataFrame(situational_features, index=games_df.index)
        return pd.concat([games_df, sit_df], axis=1)
    
    def build_training_dataset(self, all_data: Dict, seasons: List[int]) -> pd.DataFrame:
        """
        Override base to add enhanced features (H2H, situational) to training data.
        
        Vegas and injury features get neutral defaults during training (no historical data),
        but are included so the model's feature_names list contains them. At prediction time,
        real values are used when available.
        """
        # 1. Get base training dataset (ELO, rolling stats, four factors, schedule)
        base_df = super().build_training_dataset(all_data, seasons)
        
        if base_df.empty:
            return base_df
        
        # 2. Add H2H features (computable from historical games)
        historical_games = all_data.get('games', pd.DataFrame())
        if not historical_games.empty:
            historical_games = historical_games.copy()
            historical_games['date'] = pd.to_datetime(historical_games['date'])
            
            print("\nAdding H2H features to training data...")
            try:
                base_df = self.add_h2h_features(base_df, historical_games)
                print(f"  H2H features added: h2h_games, h2h_home_win_pct, etc.")
            except Exception as e:
                print(f"  Warning: H2H features failed: {e}")
                # Add defaults
                base_df['h2h_games'] = 0
                base_df['h2h_home_wins'] = 0
                base_df['h2h_home_win_pct'] = 0.5
                base_df['h2h_avg_margin'] = 0.0
                base_df['h2h_home_avg_score'] = 110.0
                base_df['h2h_visitor_avg_score'] = 110.0
                base_df['h2h_last3_home_wins'] = 0.5
        
        # 3. Add situational features (travel, motivation, season phase)
        standings = all_data.get('standings', pd.DataFrame())
        print("Adding situational features to training data...")
        try:
            base_df = self.add_situational_features(base_df, standings, historical_games)
            print(f"  Situational features added: travel, motivation, season_phase, etc.")
        except Exception as e:
            print(f"  Warning: Situational features failed: {e}")
            base_df['visitor_travel_miles'] = 0
            base_df['visitor_tz_change'] = 0
            base_df['is_long_travel'] = 0
            base_df['home_conf_rank'] = 8
            base_df['visitor_conf_rank'] = 8
            base_df['conf_rank_diff'] = 0
            base_df['home_in_playoff_race'] = 0
            base_df['visitor_in_playoff_race'] = 0
            base_df['home_motivation'] = 2
            base_df['visitor_motivation'] = 2
            base_df['motivation_diff'] = 0
            base_df['season_phase'] = 2
            base_df['is_late_season'] = 0
        
        # 4. Add default Vegas features (no historical odds available for training)
        # Including these in training ensures they're in the model's feature_names.
        # At prediction time, real values replace these defaults.
        base_df['vegas_spread_home'] = 0.0
        base_df['vegas_total'] = 220.0
        base_df['vegas_implied_home_prob'] = 0.5
        base_df['vegas_ml_home'] = -110
        base_df['vegas_ml_away'] = -110
        base_df['vegas_has_odds'] = 0
        
        # 5. Add default injury features (no historical injury data for training)
        base_df['home_injuries_out'] = 0
        base_df['visitor_injuries_out'] = 0
        base_df['home_questionable'] = 0
        base_df['visitor_questionable'] = 0
        base_df['injury_diff'] = 0
        base_df['home_stars_out'] = 0
        base_df['visitor_stars_out'] = 0
        base_df['star_injury_diff'] = 0
        base_df['home_injury_impact'] = 0.0
        base_df['visitor_injury_impact'] = 0.0
        base_df['injury_impact_diff'] = 0.0
        
        print(f"\nEnhanced training dataset: {base_df.shape[0]} games x {base_df.shape[1]} features")
        return base_df
    
    def enhance_features(self, games_df: pd.DataFrame, 
                         odds_df: pd.DataFrame = None,
                         injuries: List[dict] = None,
                         standings: pd.DataFrame = None,
                         historical_games: pd.DataFrame = None,
                         player_stats: Dict[int, float] = None) -> pd.DataFrame:
        """Apply all feature enhancements"""
        result = games_df.copy()
        
        print("Adding Vegas features...")
        if odds_df is not None:
            result = self.add_vegas_features(result, odds_df)
        
        print("Adding H2H features...")
        if historical_games is not None:
            result = self.add_h2h_features(result, historical_games)
        
        print("Adding injury features...")
        if injuries is not None:
            result = self.add_injury_features(result, injuries, player_stats)
        else:
            # Add default injury features
            result['home_injuries_out'] = 0
            result['visitor_injuries_out'] = 0
            result['home_questionable'] = 0
            result['visitor_questionable'] = 0
            result['injury_diff'] = 0
            result['home_stars_out'] = 0
            result['visitor_stars_out'] = 0
            result['star_injury_diff'] = 0
            result['home_injury_impact'] = 0.0
            result['visitor_injury_impact'] = 0.0
            result['injury_impact_diff'] = 0.0
        
        print("Adding situational features...")
        result = self.add_situational_features(result, standings, historical_games)
        
        return result


# Quick test function
def test_features():
    """Test the enhanced features"""
    print("Testing enhanced features...")
    
    # Test travel distance
    dist = get_travel_distance(21, 2)  # Lakers to Boston
    print(f"LA Lakers to Boston: {dist:.0f} miles")
    
    dist = get_travel_distance(20, 21)  # Clippers to Lakers
    print(f"Clippers to Lakers: {dist:.0f} miles")
    
    tz = get_timezone_change(21, 2)  # Lakers to Boston
    print(f"Lakers to Boston timezone change: {tz} hours")


if __name__ == "__main__":
    test_features()
