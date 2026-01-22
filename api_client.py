"""
BallDontLie API Client - GOAT Tier
Handles all API interactions with balldontlie.io
"""
import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config


class BallDontLieClient:
    """API client for balldontlie.io with GOAT tier access"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.API_KEY
        self.base_url = config.API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.api_key
        })
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Rate limiting
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None, version: str = "v1") -> Dict:
        """Make API request with error handling, timeout, and retries"""
        self._rate_limit()
        
        url = f"{self.base_url}/{version}/{endpoint}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent locking (User requirement)
                response = self.session.get(url, params=params, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    return {"data": [], "meta": {}}
    
    def _paginate_request(self, endpoint: str, params: Dict = None, version: str = "v1", 
                         max_pages: int = None) -> List[Dict]:
        """Handle paginated API requests"""
        all_data = []
        cursor = None
        page_count = 0
        
        if params is None:
            params = {}
        
        while True:
            if cursor:
                params["cursor"] = cursor
            
            result = self._make_request(endpoint, params, version)
            data = result.get("data", [])
            all_data.extend(data)
            
            # Check for next page
            meta = result.get("meta", {})
            cursor = meta.get("next_cursor")
            
            page_count += 1
            if not cursor or (max_pages and page_count >= max_pages):
                break
            
            print(f"Fetching page {page_count + 1}...")
        
        return all_data
    
    # ==================== GAMES ====================
    
    def get_games(self, seasons: List[int] = None, dates: List[str] = None, 
                  team_ids: List[int] = None, per_page: int = 100) -> List[Dict]:
        """
        Get NBA games
        
        Args:
            seasons: List of seasons (e.g., [2023, 2024])
            dates: List of dates in YYYY-MM-DD format
            team_ids: List of team IDs
            per_page: Results per page (max 100)
        """
        params = {"per_page": per_page}
        
        # Build array parameters correctly for requests library
        if seasons:
            params["seasons[]"] = seasons
        if dates:
            params["dates[]"] = dates
        if team_ids:
            params["team_ids[]"] = team_ids
        
        return self._paginate_request("games", params)
    
    def get_game_by_id(self, game_id: int) -> Dict:
        """Get specific game by ID"""
        result = self._make_request(f"games/{game_id}")
        return result.get("data", {})
    
    # ==================== ADVANCED STATS ====================
    
    def get_advanced_stats(self, seasons: List[int] = None, game_ids: List[int] = None,
                          per_page: int = 100) -> List[Dict]:
        """
        Get advanced statistics (GOAT tier required)
        Includes: PIE, pace, offensive/defensive rating, eFG%, true shooting %, etc.
        """
        params = {"per_page": per_page}
        
        if seasons:
            params["seasons[]"] = seasons
        if game_ids:
            params["game_ids[]"] = game_ids
        
        return self._paginate_request("stats/advanced", params)
    
    # ==================== TEAM SEASON AVERAGES ====================
    
    def get_team_season_averages(self, season: int, category: str = "general", 
                                stat_type: str = "base", season_type: str = "regular") -> List[Dict]:
        """
        Get team season averages (GOAT tier required)
        
        Categories: general, advanced, four_factors, hustle
        Types: base, advanced, misc, scoring, opponent (varies by category)
        """
        endpoint = f"team_season_averages/{category}"
        params = {
            "season": season,
            "season_type": season_type,
            "per_page": 100
        }
        
        # Hustle category doesn't require type parameter
        if category != "hustle":
            params["type"] = stat_type
        
        result = self._make_request(endpoint, params, version="nba/v1")
        return result.get("data", [])
    
    # ==================== BETTING ODDS ====================
    
    def get_betting_odds(self, dates: List[str] = None, game_ids: List[int] = None) -> List[Dict]:
        """
        Get betting odds (GOAT tier required)
        Includes spreads, totals, moneylines
        """
        params = {}
        
        if dates:
            params["dates[]"] = dates
        if game_ids:
            params["game_ids[]"] = game_ids
        
        result = self._make_request("odds", params, version="v2")
        return result.get("data", [])
    
    # ==================== STANDINGS ====================
    
    def get_standings(self, season: int, conference: str = None) -> List[Dict]:
        """Get team standings for a season"""
        params = {"season": season}
        if conference:
            params["conference"] = conference
        
        result = self._make_request("standings", params)
        return result.get("data", [])
    
    # ==================== BOX SCORES ====================
    
    def get_box_scores(self, game_ids: List[int] = None, dates: List[str] = None) -> List[Dict]:
        """
        Get box scores for games
        Note: API v1 requires 'date' parameter. 
        If multiple dates are provided, multiple requests will be made.
        If game_ids are provided, results will be filtered client-side (dates must be known/provided).
        """
        if not dates:
            print("Warning: get_box_scores requires 'dates' parameter")
            return []
            
        all_data = []
        for date in dates:
            params = {"date": date, "per_page": 100}
            # v1 endpoint requires date
            data = self._paginate_request("box_scores", params, version="v1")
            
            # Filter by game_ids if provided
            if game_ids:
                game_ids_set = set(game_ids)
                filtered_data = [
                    game for game in data 
                    if game.get('id') in game_ids_set
                ]
                all_data.extend(filtered_data)
            else:
                all_data.extend(data)
                
        return all_data
    
    # ==================== PLAYERS ====================
    
    def get_active_players(self) -> List[Dict]:
        """Get all active players"""
        return self._paginate_request("active_players")
    
    def get_player_injuries(self, per_page: int = 100) -> List[Dict]:
        """
        Get ALL player injuries with pagination (GOAT tier)
        
        Returns all injury statuses: Out, Questionable, Probable, Doubtful, Out For Season
        """
        params = {"per_page": per_page}
        return self._paginate_request("player_injuries", params)
    
    # ==================== SEASON AVERAGES ====================
    
    def get_season_averages(self, season: int, player_ids: List[int] = None) -> List[Dict]:
        """Get player season averages"""
        params = {"season": season}
        
        if player_ids:
            params["player_ids[]"] = player_ids
        
        result = self._make_request("season_averages", params)
        return result.get("data", [])
    
    # ==================== TEAMS ====================
    
    def get_teams(self) -> List[Dict]:
        """Get all NBA teams"""
        result = self._make_request("teams")
        return result.get("data", [])
    
    def get_team_by_id(self, team_id: int) -> Dict:
        """Get specific team by ID"""
        result = self._make_request(f"teams/{team_id}")
        return result.get("data", {})


# Convenience instance
client = BallDontLieClient()
