"""
The Odds API Client - Primary odds source for college basketball
Supports both Men's (NCAAB) and Women's (WNCAAB)
"""
import requests
import time
from typing import List, Dict
import config


class TheOddsAPIClient:
    """Client for The Odds API - college basketball odds"""
    
    def __init__(self, sport: str = None, api_key: str = None):
        """
        Args:
            sport: Sport key (e.g., 'basketball_ncaab' or 'basketball_wncaab')
            api_key: API key for The Odds API
        """
        self.api_key = api_key or config.ODDS_API_KEY
        self.base_url = config.ODDS_API_BASE_URL
        self.sport = sport or "basketball_ncaab"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def get_odds(self, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
        """
        Get college basketball odds
        
        Args:
            regions: Betting regions (us, uk, eu, au)
            markets: Comma-separated markets (h2h=moneyline, spreads, totals)
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
            
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                print(f"   The Odds API ({self.sport}) - Requests remaining: {remaining}")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"The Odds API request failed: {e}")
            return []
    
    def get_events(self) -> List[Dict]:
        """Get list of upcoming events (games) for the sport"""
        self._rate_limit()
        events_url = f"{self.base_url}/sports/{self.sport}/events"
        
        try:
            response = self.session.get(events_url, params={"apiKey": self.api_key})
            response.raise_for_status()
            events = response.json()
            
            if 'x-requests-remaining' in response.headers:
                print(f"   The Odds API (Events) - Requests remaining: {response.headers['x-requests-remaining']}")
            
            return events
        except requests.exceptions.RequestException as e:
            print(f"The Odds API events request failed: {e}")
            return []
    
    def get_player_props(self, markets: str = None, max_events: int = 8) -> List[Dict]:
        """
        Get player props from all available bookmakers (Bovada, DraftKings, FanDuel, etc.)
        
        Args:
            markets: Comma-separated player prop markets
            max_events: Max number of events to query (each costs 1 API call)
        """
        if markets is None:
            markets = ",".join([
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_threes",
                "player_points_rebounds_assists",
                "player_blocks",
                "player_steals",
                "player_turnovers",
                "player_double_double"
            ])
        
        events = self.get_events()
        
        if not events:
            return []
        
        all_props = []
        for event in events[:max_events]:
            event_id = event.get('id')
            if not event_id:
                continue
            
            self._rate_limit()
            event_odds_url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "us,eu,uk,au",  # expanded regions for more props
                "markets": markets,
                "oddsFormat": "american"
            }
            
            try:
                response = self.session.get(event_odds_url, params=params)
                response.raise_for_status()
                event_data = response.json()
                
                if 'x-requests-remaining' in response.headers:
                    remaining = response.headers['x-requests-remaining']
                    print(f"   The Odds API (Props: {event.get('home_team', 'game')}) - Remaining: {remaining}")
                
                props = self._parse_event_props(event_data)
                all_props.extend(props)
            except requests.exceptions.RequestException as e:
                print(f"Props request failed for event {event_id}: {e}")
                continue
        
        return all_props
    
    def _parse_event_props(self, event_data: Dict) -> List[Dict]:
        """Parse player props from a single event response"""
        props = []
        market_names = {
            'player_points': 'Points', 'player_rebounds': 'Rebounds',
            'player_assists': 'Assists', 'player_threes': '3-Pointers',
            'player_points_rebounds_assists': 'PRA',
            'player_double_double': 'Double-Double',
            'player_blocks': 'Blocks', 'player_steals': 'Steals',
            'player_turnovers': 'Turnovers',
        }
        
        # Friendly bookmaker names
        book_display_names = {
            'bovada': 'Bovada', 
            'draftkings': 'DraftKings', 
            'fanduel': 'FanDuel',
            'betmgm': 'BetMGM', 
            'caesars': 'Caesars',
            'williamhill_us': 'Caesars',  # WH US rebranded to Caesars
            'betrivers': 'BetRivers',
            'fanatics': 'Fanatics',
            'superbook': 'SuperBook',
            'barstool': 'ESPN BET', # Rebranded
            'espnbet': 'ESPN BET',
            'betus': 'BetUS',
            'mybookieag': 'MyBookie', 
            'betonlineag': 'BetOnline', 
            'lowvig': 'LowVig',
            'pointsbetus': 'PointsBet', # Bought by Fanatics, but key might persist
            'wynnbet': 'WynnBet',
            # International
            'bet365': 'Bet365', 'unibet': 'Unibet', 'unibet_eu': 'Unibet', 
            'unibet_uk': 'Unibet', 'pinnacle': 'Pinnacle', 'ladbrokes': 'Ladbrokes',
            'williamhill': 'William Hill', 'betfair': 'Betfair', 'matchbook': 'Matchbook',
            'nordicbet': 'NordicBet', 'betsson': 'Betsson', 'coolbet': 'Coolbet',
            'marathonbet': 'MarathonBet', 'onexbet': '1xBet', 'livescorebet': 'LiveScore Bet',
            'tipico_us': 'Tipico', 'windcreek': 'Wind Creek',
            'ballybet': 'Bally Bet', 'betparx': 'betPARX',
            'si_sportsbook': 'SI Sportsbook', 'hardrockbet': 'Hard Rock Bet'
        }
        
        game_name = f"{event_data.get('away_team', '')} @ {event_data.get('home_team', '')}"
        home_team = event_data.get('home_team', '')
        away_team = event_data.get('away_team', '')
        commence_time = event_data.get('commence_time', '')
        
        for bookmaker in event_data.get('bookmakers', []):
            book_key = bookmaker.get('key', 'unknown')
            book_name = book_display_names.get(book_key, bookmaker.get('title', book_key))
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                prop_type = market_names.get(market_key, market_key)
                
                player_outcomes = {}
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description', '')
                    if not player:
                        continue
                    if player not in player_outcomes:
                        player_outcomes[player] = {'line': outcome.get('point')}
                    if outcome.get('name') == 'Over':
                        player_outcomes[player]['over_odds'] = outcome.get('price')
                    elif outcome.get('name') == 'Under':
                        player_outcomes[player]['under_odds'] = outcome.get('price')
                
                for player, data in player_outcomes.items():
                    if data.get('line') is not None:
                        props.append({
                            'source': 'The Odds API',
                            'game': game_name,
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'player': player,
                            'prop_type': prop_type,
                            'line': data['line'],
                            'over_odds': data.get('over_odds'),
                            'under_odds': data.get('under_odds'),
                            'bookmaker': book_name,
                            'bookmaker_key': book_key
                        })
        
        return props
    
    def convert_to_standard_format(self, odds_data: List[Dict], team_name_mapping: Dict = None) -> List[Dict]:
        """Convert The Odds API format to our standard format"""
        converted = []
        
        for game in odds_data:
            game_id = self._match_game_id(game, team_name_mapping)
            
            for bookmaker in game.get('bookmakers', []):
                vendor = bookmaker.get('key', 'unknown')
                record = {
                    'game_id': game_id,
                    'vendor': vendor,
                    'spread_home_value': None, 'spread_away_value': None,
                    'moneyline_home_odds': None, 'moneyline_away_odds': None,
                    'total_value': None,
                }
                
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    
                    if market_key == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                record['moneyline_home_odds'] = outcome.get('price')
                            elif outcome.get('name') == away_team:
                                record['moneyline_away_odds'] = outcome.get('price')
                    
                    elif market_key == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                record['spread_home_value'] = outcome.get('point')
                            elif outcome.get('name') == away_team:
                                record['spread_away_value'] = outcome.get('point')
                    
                    elif market_key == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over':
                                record['total_value'] = outcome.get('point')
                
                converted.append(record)
        
        return converted
    
    def _match_game_id(self, game: Dict, team_name_mapping: Dict = None) -> str:
        """Match The Odds API game to our game IDs"""
        if team_name_mapping:
            home = game.get('home_team', '').lower()
            away = game.get('away_team', '').lower()
            
            for game_id, teams in team_name_mapping.items():
                bdl_home = teams.get('home', '').lower()
                bdl_away = teams.get('away', '').lower()
                
                if self._teams_match(home, bdl_home) and self._teams_match(away, bdl_away):
                    return game_id
        
        return game.get('id', 'unknown')
    
    def _teams_match(self, team1: str, team2: str) -> bool:
        """Check if two team names refer to the same team"""
        if not team1 or not team2:
            return False
        if team1 == team2:
            return True
        # Check if last word (mascot) matches
        words1 = team1.split()
        words2 = team2.split()
        if words1 and words2 and words1[-1] == words2[-1]:
            return True
        # Check substring match
        if team1 in team2 or team2 in team1:
            return True
        return False
    
    def check_usage(self) -> Dict:
        """Check API usage stats"""
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {"apiKey": self.api_key, "regions": "us", "markets": "h2h"}
        try:
            response = self.session.get(url, params=params)
            return {
                'remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                'used': response.headers.get('x-requests-used', 'Unknown'),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_espn_odds(self, league: str = "womens-college-basketball",
                      team_name_mapping: Dict = None) -> List[Dict]:
        """
        Fetch odds from ESPN's hidden header API (DraftKings provider).
        Used as fallback for sports The Odds API doesn't cover (e.g. WNCAAB).
        
        Returns data in the same standard format as convert_to_standard_format().
        """
        try:
            print(f"   Fetching {league} odds from ESPN (DraftKings)...")
            headers = {"User-Agent": "Mozilla/5.0"}
            
            # ESPN header API returns featured games with DraftKings odds
            url = "https://site.web.api.espn.com/apis/v2/scoreboard/header"
            resp = self.session.get(url, params={
                "sport": "basketball",
                "league": league
            }, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            converted = []
            for sport in data.get("sports", []):
                for lg in sport.get("leagues", []):
                    for ev in lg.get("events", []):
                        odds = ev.get("odds", {})
                        if not odds:
                            continue
                        
                        # Extract team names
                        home_team = away_team = ""
                        espn_home = espn_away = ""
                        for comp in ev.get("competitors", []):
                            if comp.get("homeAway") == "home":
                                espn_home = comp.get("displayName", comp.get("name", ""))
                                home_team = espn_home
                            else:
                                espn_away = comp.get("displayName", comp.get("name", ""))
                                away_team = espn_away
                        
                        # Match game ID
                        game_id = self._match_espn_game_id(espn_home, espn_away, team_name_mapping)
                        
                        spread = odds.get("spread")
                        total = odds.get("overUnder")
                        home_ml = odds.get("homeTeamOdds", {}).get("moneyLine")
                        away_ml = odds.get("awayTeamOdds", {}).get("moneyLine")
                        provider = odds.get("provider", {}).get("name", "ESPN")
                        
                        record = {
                            'game_id': game_id,
                            'vendor': provider.lower().replace(" ", ""),
                            'spread_home_value': spread,
                            'spread_away_value': -spread if spread is not None else None,
                            'moneyline_home_odds': home_ml,
                            'moneyline_away_odds': away_ml,
                            'total_value': total,
                        }
                        converted.append(record)
            
            if converted:
                print(f"   ESPN: Got odds for {len(converted)} games")
            else:
                print(f"   ESPN: No odds available for {league}")
            
            return converted
            
        except Exception as e:
            print(f"   ESPN odds fetch error: {e}")
            return []
    
    def _match_espn_game_id(self, espn_home: str, espn_away: str,
                            team_name_mapping: Dict = None) -> str:
        """Match ESPN team names to our game IDs"""
        if not team_name_mapping:
            return "unknown"
        
        espn_home_lower = espn_home.lower()
        espn_away_lower = espn_away.lower()
        
        for game_id, teams in team_name_mapping.items():
            bdl_home = teams.get('home', '').lower()
            bdl_away = teams.get('away', '').lower()
            
            if (self._teams_match(espn_home_lower, bdl_home) and 
                self._teams_match(espn_away_lower, bdl_away)):
                return game_id
        
        return "unknown"

