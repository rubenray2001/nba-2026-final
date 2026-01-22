"""
The Odds API Client - Backup odds source for complete coverage
Provides odds when BallDontLie doesn't have them
"""
import requests
import time
from typing import List, Dict
import config


class TheOddsAPIClient:
    """Client for The Odds API - comprehensive sports betting odds"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.ODDS_API_KEY
        self.base_url = config.ODDS_API_BASE_URL
        self.sport = "basketball_nba"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting (1 second between requests)
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def get_odds(self, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
        """
        Get NBA odds from The Odds API
        
        Args:
            regions: Betting regions (us, uk, eu, au)
            markets: Comma-separated markets (h2h=moneyline, spreads, totals)
        
        Returns:
            List of odds dictionaries
        """
        self._rate_limit()
        
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Check remaining requests
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                print(f"   The Odds API - Requests remaining: {remaining}")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"The Odds API request failed: {e}")
            return []
    
    def convert_to_balldontlie_format(self, odds_data: List[Dict], team_name_mapping: Dict = None) -> List[Dict]:
        """
        Convert The Odds API format to BallDontLie format for compatibility
        
        Args:
            odds_data: Raw odds from The Odds API
            team_name_mapping: Optional mapping of team names to game IDs
        
        Returns:
            List of odds in BallDontLie-compatible format
        """
        converted_odds = []
        
        for game in odds_data:
            game_id = self._extract_game_id(game, team_name_mapping)
            
            # Process each bookmaker
            for bookmaker in game.get('bookmakers', []):
                vendor = bookmaker.get('key', 'unknown')
                
                # Initialize odds record
                odds_record = {
                    'game_id': game_id,
                    'vendor': vendor,
                    'spread_home_value': None,
                    'spread_home_odds': None,
                    'spread_away_value': None,
                    'spread_away_odds': None,
                    'moneyline_home_odds': None,
                    'moneyline_away_odds': None,
                    'total_value': None,
                    'total_over_odds': None,
                    'total_under_odds': None,
                    'updated_at': bookmaker.get('last_update')
                }
                
                # Determine home/away teams
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                
                # Extract markets
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    
                    if market_key == 'h2h':  # Moneyline
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                odds_record['moneyline_home_odds'] = outcome.get('price')
                            elif outcome.get('name') == away_team:
                                odds_record['moneyline_away_odds'] = outcome.get('price')
                    
                    elif market_key == 'spreads':  # Point spread
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                odds_record['spread_home_value'] = outcome.get('point')
                                odds_record['spread_home_odds'] = outcome.get('price')
                            elif outcome.get('name') == away_team:
                                odds_record['spread_away_value'] = outcome.get('point')
                                odds_record['spread_away_odds'] = outcome.get('price')
                    
                    elif market_key == 'totals':  # Over/Under
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over':
                                odds_record['total_value'] = outcome.get('point')
                                odds_record['total_over_odds'] = outcome.get('price')
                            elif outcome.get('name') == 'Under':
                                odds_record['total_under_odds'] = outcome.get('price')
                
                converted_odds.append(odds_record)
        
        return converted_odds
    
    def _extract_game_id(self, game: Dict, team_name_mapping: Dict = None) -> str:
        """
        Extract or generate game ID
        The Odds API uses different IDs, so we need to map to BallDontLie IDs
        """
        # If mapping provided, try to find matching game
        if team_name_mapping:
            home_team = game.get('home_team', '').lower()
            away_team = game.get('away_team', '').lower()
            
            # Try to find matching game in mapping using normalized names
            for game_id, teams in team_name_mapping.items():
                bdl_home = teams.get('home', '').lower()
                bdl_away = teams.get('away', '').lower()
                
                # Check for exact match or key word match (city or nickname)
                home_match = self._teams_match(home_team, bdl_home)
                away_match = self._teams_match(away_team, bdl_away)
                
                # Require BOTH teams to match to avoid false positives
                if home_match and away_match:
                    return game_id
        
        # Fallback: use The Odds API game ID
        return game.get('id', 'unknown')
    
    def _teams_match(self, odds_api_team: str, bdl_team: str) -> bool:
        """
        Check if two team names refer to the same team
        Handles variations like 'Los Angeles Lakers' vs 'LA Lakers'
        """
        if not odds_api_team or not bdl_team:
            return False
        
        # Direct match
        if odds_api_team == bdl_team:
            return True
        
        # Extract key identifiers (last word is usually nickname)
        odds_words = odds_api_team.split()
        bdl_words = bdl_team.split()
        
        # Match on nickname (last word) - most reliable
        if odds_words and bdl_words:
            odds_nickname = odds_words[-1]
            bdl_nickname = bdl_words[-1]
            
            if odds_nickname == bdl_nickname:
                return True
        
        # Handle special cases like "LA Clippers" vs "Los Angeles Clippers"
        # Check if one contains the other's nickname
        for word in odds_words:
            if word in ['clippers', 'lakers', 'warriors', 'celtics', 'heat', 'nets',
                       'knicks', 'bulls', 'cavaliers', 'mavericks', 'nuggets', 
                       'pistons', 'rockets', 'pacers', 'grizzlies', 'bucks',
                       'timberwolves', 'pelicans', 'thunder', 'magic', '76ers',
                       'suns', 'blazers', 'kings', 'spurs', 'raptors', 'jazz',
                       'wizards', 'hawks', 'hornets']:
                if word in bdl_team:
                    return True
        
        return False
    
    def check_usage(self) -> Dict:
        """Check API usage stats"""
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h"
        }
        
        try:
            response = self.session.get(url, params=params)
            
            return {
                'remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                'used': response.headers.get('x-requests-used', 'Unknown'),
                'status': response.status_code
            }
        except Exception as e:
            return {'error': str(e)}


# Convenience instance
odds_api_client = TheOddsAPIClient()
