"""
Data Manager - Orchestrates data collection and caching for college basketball
Supports both Men's and Women's
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json
import traceback
from api_client import ESPNCollegeBasketballClient
from odds_api_client import TheOddsAPIClient
import config

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class DataManager:
    """Manages data fetching, caching, and aggregation for college basketball"""
    
    def __init__(self, gender: str = "mens"):
        """
        Args:
            gender: 'mens' or 'womens'
        """
        self.gender = gender
        self.gender_config = config.GENDER_CONFIG[gender]
        self.client = ESPNCollegeBasketballClient(gender)
        self.odds_client = TheOddsAPIClient(sport=self.gender_config["odds_sport"])
        self.data_dir = os.path.join(config.DATA_DIR, gender)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_cache_path(self, cache_name: str) -> str:
        """Get path for cache file"""
        return os.path.join(self.data_dir, f"{cache_name}.csv")
    
    def _is_cache_valid(self, cache_path: str, expiry_hours: int = None,
                        required_cols: list = None) -> bool:
        """Check if cache file exists and is not expired"""
        if not os.path.exists(cache_path):
            return False
        if expiry_hours is None:
            expiry_hours = config.CACHE_EXPIRY_HOURS
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_modified > timedelta(hours=expiry_hours):
            return False
        if required_cols:
            try:
                header_df = pd.read_csv(cache_path, nrows=0)
                missing = [c for c in required_cols if c not in header_df.columns]
                if missing:
                    return False
            except Exception:
                return False
        return True
    
    def fetch_historical_games(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical games for specified seasons from ESPN
        
        Args:
            seasons: List of seasons to fetch (e.g., [2024, 2025])
            force_refresh: Force refresh from API
        """
        cache_path = self._get_cache_path("games_historical")
        
        required_cols = ['home_team_id', 'visitor_team_id', 'home_team_score', 'visitor_team_score']
        if not force_refresh and self._is_cache_valid(cache_path, expiry_hours=12, required_cols=required_cols):
            print(f"Loading {self.gender} games from cache: {cache_path}")
            return pd.read_csv(cache_path, low_memory=False)
        
        print(f"Fetching {self.gender} games for seasons: {seasons}")
        all_games = []
        
        for season in seasons:
            games = self.client.get_historical_games(season)
            all_games.extend(games)
        
        df = pd.DataFrame(all_games)
        
        if not df.empty:
            # Filter to completed games with valid scores
            df = df[df['status'] == 'Final'].copy()
            df = df[(df['home_team_score'] > 0) & (df['visitor_team_score'] > 0)].copy()
            
            df.to_csv(cache_path, index=False)
            print(f"Saved {len(df)} {self.gender} games to cache")
        
        return df
    
    def update_historical_with_recent(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch the last N days of games from ESPN and merge into the historical cache.
        This ensures yesterday's results are always reflected in features like
        L10 Win %, PPG, ELO, and rest days — without doing a full multi-season refetch.
        """
        cache_path = self._get_cache_path("games_historical")
        
        # Load existing cache
        if os.path.exists(cache_path):
            try:
                existing_df = pd.read_csv(cache_path, low_memory=False)
            except Exception:
                print("Warning: Could not read existing historical cache, skipping incremental update")
                return pd.DataFrame()
        else:
            print("No historical cache found — run full fetch first")
            return pd.DataFrame()
        
        # Fetch recent games day by day
        recent_games = []
        for d in range(days):
            date = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
            try:
                day_games = self.client.get_games(dates=[date])
                recent_games.extend(day_games)
            except Exception as e:
                print(f"Warning: Could not fetch games for {date}: {e}")
        
        if not recent_games:
            print("No recent games found to merge")
            return existing_df
        
        recent_df = pd.DataFrame(recent_games)
        
        # Keep only completed games with valid scores
        if 'status' in recent_df.columns:
            recent_df = recent_df[recent_df['status'] == 'Final'].copy()
        if 'home_team_score' in recent_df.columns and 'visitor_team_score' in recent_df.columns:
            recent_df = recent_df[
                (recent_df['home_team_score'] > 0) & (recent_df['visitor_team_score'] > 0)
            ].copy()
        
        if recent_df.empty:
            print("No completed recent games to merge")
            return existing_df
        
        # Determine game ID column
        id_col = 'id' if 'id' in recent_df.columns else 'game_id'
        existing_id_col = 'id' if 'id' in existing_df.columns else 'game_id'
        
        # Deduplicate: remove games already present, then append new ones
        existing_ids = set(existing_df[existing_id_col].astype(str).values)
        new_games = recent_df[~recent_df[id_col].astype(str).isin(existing_ids)]
        
        if new_games.empty:
            print("Historical cache already up-to-date")
            return existing_df
        
        # Align columns
        for col in existing_df.columns:
            if col not in new_games.columns:
                new_games[col] = None
        new_games = new_games[[c for c in existing_df.columns if c in new_games.columns]]
        
        merged_df = pd.concat([existing_df, new_games], ignore_index=True)
        merged_df.to_csv(cache_path, index=False)
        print(f"Merged {len(new_games)} new games into historical cache (total: {len(merged_df)})")
        
        return merged_df
    
    def fetch_todays_games(self, target_date: str = None) -> pd.DataFrame:
        """Fetch games for a specific date"""
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching {self.gender} games for {target_date}")
        games = self.client.get_games(dates=[target_date])
        df = pd.DataFrame(games)
        return df
    
    def fetch_vegas_odds(self, dates: List[str] = None, game_ids: List[int] = None,
                         games_df: pd.DataFrame = None) -> pd.DataFrame:
        """Fetch betting odds with team name matching. Falls back to ESPN for WNCAAB."""
        # Build team name mapping for matching
        team_mapping = {}
        if games_df is not None and not games_df.empty:
            for _, game in games_df.iterrows():
                game_id = game.get('id')
                team_mapping[game_id] = {
                    'home': game.get('home_team_name', ''),
                    'away': game.get('visitor_team_name', '')
                }
        
        try:
            print(f"   Fetching {self.gender} odds from The Odds API...")
            raw_odds = self.odds_client.get_odds()
            
            if raw_odds:
                converted = self.odds_client.convert_to_standard_format(raw_odds, team_mapping)
                df = pd.DataFrame(converted)
                
                if not df.empty:
                    print(f"   Got odds for {df['game_id'].nunique()} games")
                    return df
            
            print(f"   No odds from The Odds API, trying ESPN fallback...")
        except Exception as e:
            print(f"   Odds API error: {e}, trying ESPN fallback...")
        
        # Fallback: ESPN header API (works for WNCAAB via DraftKings)
        try:
            espn_league = self.gender_config.get("espn_league", 
                "womens-college-basketball" if self.gender == "womens" else "mens-college-basketball")
            espn_odds = self.odds_client.get_espn_odds(
                league=espn_league, team_name_mapping=team_mapping)
            
            if espn_odds:
                df = pd.DataFrame(espn_odds)
                return df
        except Exception as e:
            print(f"   ESPN fallback error: {e}")
        
        return pd.DataFrame()
    
    def fetch_standings(self, season: int = None) -> pd.DataFrame:
        """Fetch standings"""
        cache_path = self._get_cache_path(f"standings_{season or 'current'}")
        
        if self._is_cache_valid(cache_path, expiry_hours=24):
            return pd.read_csv(cache_path)
        
        print(f"Fetching {self.gender} standings...")
        standings = self.client.get_standings(season=season)
        df = pd.DataFrame(standings)
        
        if not df.empty:
            df.to_csv(cache_path, index=False)
        
        return df
    
    def fetch_rankings(self) -> List[Dict]:
        """Fetch AP Top 25 rankings"""
        return self.client.get_rankings()
    
    def get_complete_training_data(self, seasons: List[int]) -> Dict:
        """
        Build complete training dataset
        
        Returns dict with 'games', 'standings' DataFrames
        """
        print("=" * 50)
        print(f"BUILDING {self.gender.upper()} TRAINING DATASET")
        print("=" * 50)
        
        # Fetch games
        games_df = self.fetch_historical_games(seasons, force_refresh=False)
        
        if games_df.empty:
            print("No games found!")
            return {'games': pd.DataFrame(), 'standings': pd.DataFrame()}
        
        # Always merge in the last 7 days so yesterday's results are reflected
        try:
            games_df = self.update_historical_with_recent(days=7)
        except Exception as e:
            print(f"Warning: Incremental update failed ({e}), using cached data")
            traceback.print_exc()
        
        print(f"Found {len(games_df)} valid completed games")
        
        # Fetch standings for each season
        all_standings = []
        for season in seasons:
            standings = self.fetch_standings(season)
            if not standings.empty:
                all_standings.append(standings)
        
        standings_df = pd.concat(all_standings, ignore_index=True) if all_standings else pd.DataFrame()
        
        print(f"Fetched {len(standings_df)} standing records")
        
        return {
            'games': games_df,
            'standings': standings_df
        }
