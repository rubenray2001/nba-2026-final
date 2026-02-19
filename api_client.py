"""
ESPN API Client for College Basketball
Handles all data fetching from ESPN's public API endpoints
Supports both Men's and Women's college basketball
"""
import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config


class ESPNCollegeBasketballClient:
    """API client for ESPN's college basketball data"""
    
    def __init__(self, gender: str = "mens"):
        """
        Args:
            gender: 'mens' or 'womens'
        """
        self.gender = gender
        self.gender_config = config.GENDER_CONFIG[gender]
        self.base_url = config.ESPN_BASE_URL
        self.sport_slug = self.gender_config["espn_slug"]
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.last_request_time = 0
        self.min_request_interval = 0.3  # Rate limiting
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make API request with error handling and retries"""
        self._rate_limit()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"ESPN API request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return {}
    
    def _build_url(self, endpoint: str) -> str:
        """Build full ESPN API URL"""
        return f"{self.base_url}/{self.sport_slug}/{endpoint}"
    
    # ==================== GAMES / SCOREBOARD ====================
    
    def get_games(self, dates: List[str] = None, limit: int = 357) -> List[Dict]:
        """
        Get college basketball games for specified dates
        
        Args:
            dates: List of dates in YYYY-MM-DD format
            limit: Max games per request
            
        Returns:
            List of game dicts with standardized fields
        """
        all_games = []
        
        if dates is None:
            dates = [datetime.now().strftime("%Y-%m-%d")]
        
        for date_str in dates:
            # ESPN expects YYYYMMDD format
            espn_date = date_str.replace("-", "")
            
            url = self._build_url("scoreboard")
            params = {
                "dates": espn_date,
                "limit": limit,
                "groups": 50  # Division I only
            }
            
            result = self._make_request(url, params)
            events = result.get("events", [])
            
            for event in events:
                game = self._parse_game(event, date_str)
                if game:
                    all_games.append(game)
        
        return all_games
    
    def _parse_game(self, event: Dict, date_str: str) -> Optional[Dict]:
        """Parse ESPN event into standardized game dict"""
        try:
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                return None
            
            # ESPN lists home team first (usually), check isHome flag
            home_team = None
            away_team = None
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home_team = comp
                else:
                    away_team = comp
            
            if not home_team or not away_team:
                return None
            
            home_team_info = home_team.get("team", {})
            away_team_info = away_team.get("team", {})
            
            # Get status
            status_obj = event.get("status", {}).get("type", {})
            status_name = status_obj.get("name", "STATUS_SCHEDULED")
            
            if status_name == "STATUS_FINAL":
                status = "Final"
            elif status_name == "STATUS_IN_PROGRESS":
                status = "In Progress"
            else:
                # Use raw ISO date string so the app can convert timezone
                raw_date = event.get("date", "")
                if raw_date and "T" in raw_date:
                    status = raw_date
                else:
                    status = event.get("status", {}).get("type", {}).get("detail", "Scheduled")
            
            # Parse scores
            home_score = int(home_team.get("score", 0) or 0)
            away_score = int(away_team.get("score", 0) or 0)
            
            # Get records
            home_records = home_team.get("records", [])
            away_records = away_team.get("records", [])
            home_record = home_records[0].get("summary", "0-0") if home_records else "0-0"
            away_record = away_records[0].get("summary", "0-0") if away_records else "0-0"
            
            game = {
                "id": int(event.get("id", 0)),
                "date": date_str,
                "status": status,
                "season": int(event.get("season", {}).get("year", 0)),
                "home_team_id": int(home_team_info.get("id", 0)),
                "home_team_name": home_team_info.get("displayName", "Unknown"),
                "home_team_abbreviation": home_team_info.get("abbreviation", "UNK"),
                "home_team_logo": home_team_info.get("logo", ""),
                "home_team_color": home_team_info.get("color", "000000"),
                "home_team_score": home_score,
                "home_team_record": home_record,
                "visitor_team_id": int(away_team_info.get("id", 0)),
                "visitor_team_name": away_team_info.get("displayName", "Unknown"),
                "visitor_team_abbreviation": away_team_info.get("abbreviation", "UNK"),
                "visitor_team_logo": away_team_info.get("logo", ""),
                "visitor_team_color": away_team_info.get("color", "000000"),
                "visitor_team_score": away_score,
                "visitor_team_record": away_record,
                "venue": competition.get("venue", {}).get("fullName", ""),
                "broadcast": "",
                "time": event.get("date", ""),
                "home_team": {"id": int(home_team_info.get("id", 0)), "full_name": home_team_info.get("displayName", "")},
                "visitor_team": {"id": int(away_team_info.get("id", 0)), "full_name": away_team_info.get("displayName", "")},
            }
            
            # Extract broadcast info
            broadcasts = competition.get("broadcasts", [])
            if broadcasts:
                names = [b.get("names", [""])[0] for b in broadcasts if b.get("names")]
                game["broadcast"] = ", ".join(names)
            
            return game
            
        except Exception as e:
            print(f"Error parsing game event: {e}")
            return None
    
    # ==================== BOX SCORES ====================
    
    def get_box_score(self, game_id: int) -> Dict:
        """
        Get detailed box score for a specific game
        
        Returns dict with player stats for both teams
        """
        url = self._build_url(f"summary")
        params = {"event": game_id}
        
        result = self._make_request(url, params)
        if not result:
            return {}
        
        box_score = {"game_id": game_id, "home": [], "away": []}
        
        # Parse box score data
        boxscore_data = result.get("boxscore", {})
        teams_data = boxscore_data.get("teams", [])
        players_data = boxscore_data.get("players", [])
        
        for team_data in players_data:
            team_info = team_data.get("team", {})
            team_id = int(team_info.get("id", 0))
            statistics = team_data.get("statistics", [])
            
            if not statistics:
                continue
            
            # Get stat labels
            stat_labels = statistics[0].get("labels", []) if statistics else []
            athletes = statistics[0].get("athletes", []) if statistics else []
            
            for athlete in athletes:
                player_info = athlete.get("athlete", {})
                stats = athlete.get("stats", [])
                
                # Map stats to labels
                player_stats = {}
                for i, label in enumerate(stat_labels):
                    if i < len(stats):
                        player_stats[label.lower()] = stats[i]
                
                player_record = {
                    "player_id": int(player_info.get("id", 0)),
                    "first_name": player_info.get("displayName", "").split(" ")[0] if player_info.get("displayName") else "",
                    "last_name": " ".join(player_info.get("displayName", "").split(" ")[1:]) if player_info.get("displayName") else "",
                    "position": player_info.get("position", {}).get("abbreviation", ""),
                    "team_id": team_id,
                    **player_stats
                }
                
                # Determine if home or away
                home_competitors = result.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
                is_home = any(
                    c.get("homeAway") == "home" and int(c.get("team", {}).get("id", 0)) == team_id
                    for c in home_competitors
                )
                
                if is_home:
                    box_score["home"].append(player_record)
                else:
                    box_score["away"].append(player_record)
        
        # Extract team stats
        for team_data in teams_data:
            team_info = team_data.get("team", {})
            statistics = team_data.get("statistics", [])
            
            team_stats = {}
            for stat in statistics:
                name = stat.get("name", "").lower()
                value = stat.get("displayValue", "0")
                try:
                    team_stats[name] = float(value.replace("-", "0").split("-")[0])
                except (ValueError, IndexError):
                    team_stats[name] = 0
            
            team_id = int(team_info.get("id", 0))
            home_competitors = result.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
            is_home = any(
                c.get("homeAway") == "home" and int(c.get("team", {}).get("id", 0)) == team_id
                for c in home_competitors
            )
            
            key = "home_stats" if is_home else "away_stats"
            box_score[key] = team_stats
        
        return box_score
    
    def get_team_stats_from_game(self, game_id: int) -> Dict:
        """
        Get aggregated team-level stats from a game's box score.
        Returns dict with home/visitor FGM, FGA, 3PM, FTM, FTA, OREB, DREB, TOV, etc.
        """
        url = self._build_url("summary")
        params = {"event": game_id}
        
        result = self._make_request(url, params)
        if not result:
            return {}
        
        boxscore = result.get("boxscore", {})
        teams = boxscore.get("teams", [])
        
        output = {}
        
        header = result.get("header", {})
        competitions = header.get("competitions", [{}])
        competitors = competitions[0].get("competitors", []) if competitions else []
        
        for team_data in teams:
            team_info = team_data.get("team", {})
            team_id = int(team_info.get("id", 0))
            stats = team_data.get("statistics", [])
            
            is_home = any(
                c.get("homeAway") == "home" and int(c.get("team", {}).get("id", 0)) == team_id
                for c in competitors
            )
            prefix = "home" if is_home else "visitor"
            
            stat_dict = {}
            for stat in stats:
                name = stat.get("name", "").lower().replace(" ", "_")
                display = stat.get("displayValue", "0")
                try:
                    # Handle stats like "25-55" (made-attempted)
                    if "-" in str(display) and name in ["fieldgoalsmade_fieldgoalsattempted",
                                                         "fieldgoals", "threepointers",
                                                         "freethrows"]:
                        parts = display.split("-")
                        stat_dict[f"{name}_made"] = float(parts[0])
                        stat_dict[f"{name}_attempted"] = float(parts[1]) if len(parts) > 1 else 0
                    else:
                        stat_dict[name] = float(display.replace(",", ""))
                except (ValueError, IndexError):
                    stat_dict[name] = 0
            
            output[prefix] = stat_dict
            output[f"{prefix}_team_id"] = team_id
        
        return output
    
    # ==================== TEAMS ====================
    
    def get_teams(self, page: int = 1, limit: int = 500) -> List[Dict]:
        """Get all Division I teams"""
        url = self._build_url("teams")
        params = {"limit": limit, "page": page, "groups": 50}
        
        result = self._make_request(url, params)
        teams = []
        
        for team_item in result.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = team_item.get("team", {})
            teams.append({
                "id": int(team.get("id", 0)),
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "logo": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                "color": team.get("color", "000000"),
                "conference": team.get("groups", {}).get("name", "Unknown") if isinstance(team.get("groups"), dict) else "Unknown",
                "location": team.get("location", ""),
                "nickname": team.get("nickname", ""),
            })
        
        return teams
    
    # ==================== STANDINGS ====================
    
    def get_standings(self, season: int = None, group: str = "50") -> List[Dict]:
        """
        Get conference standings
        
        Args:
            season: Season year
            group: Conference group ID (50 = all D1)
        """
        url = self._build_url("standings")
        params = {"group": group}
        if season:
            params["season"] = season
        
        result = self._make_request(url, params)
        standings = []
        
        for child in result.get("children", []):
            conf_name = child.get("name", "Unknown")
            for team_standing in child.get("standings", {}).get("entries", []):
                team_info = team_standing.get("team", {})
                stats = {}
                for stat in team_standing.get("stats", []):
                    stats[stat.get("name", "")] = stat.get("value", 0)
                
                standings.append({
                    "team_id": int(team_info.get("id", 0)),
                    "team_name": team_info.get("displayName", ""),
                    "team_abbreviation": team_info.get("abbreviation", ""),
                    "team_logo": team_info.get("logos", [{}])[0].get("href", "") if team_info.get("logos") else "",
                    "conference": conf_name,
                    "wins": int(stats.get("wins", 0)),
                    "losses": int(stats.get("losses", 0)),
                    "conference_wins": int(stats.get("conferenceWins", stats.get("vs_conf_wins", 0))),
                    "conference_losses": int(stats.get("conferenceLosses", stats.get("vs_conf_losses", 0))),
                    "win_pct": float(stats.get("winPercent", 0)),
                    "streak": int(stats.get("streak", 0)),
                    "ppg": float(stats.get("pointsFor", 0)) / max(1, int(stats.get("gamesPlayed", 1))),
                    "opp_ppg": float(stats.get("pointsAgainst", 0)) / max(1, int(stats.get("gamesPlayed", 1))),
                })
        
        return standings
    
    # ==================== RANKINGS ====================
    
    def get_rankings(self) -> List[Dict]:
        """Get AP Top 25 rankings"""
        url = self._build_url("rankings")
        result = self._make_request(url)
        
        rankings = []
        for poll in result.get("rankings", []):
            if poll.get("name") == "AP Top 25":
                for rank_entry in poll.get("ranks", []):
                    team_info = rank_entry.get("team", {})
                    rankings.append({
                        "rank": rank_entry.get("current", 0),
                        "previous_rank": rank_entry.get("previous", 0),
                        "team_id": int(team_info.get("id", 0)),
                        "team_name": team_info.get("location", "") + " " + team_info.get("name", ""),
                        "team_abbreviation": team_info.get("abbreviation", ""),
                        "team_logo": team_info.get("logo", ""),
                        "record": rank_entry.get("recordSummary", ""),
                        "points": rank_entry.get("points", 0),
                        "first_place_votes": rank_entry.get("firstPlaceVotes", 0),
                    })
                break
        
        return rankings
    
    # ==================== HISTORICAL GAME FETCH ====================
    
    def get_historical_games(self, season: int, date_range: tuple = None) -> List[Dict]:
        """
        Fetch all games for a season by iterating through dates.
        
        College basketball regular season runs roughly Nov 1 - mid March,
        with March Madness through early April.
        
        Args:
            season: Season starting year (e.g., 2024 for 2024-25 season)
            date_range: Optional (start_date, end_date) as YYYY-MM-DD strings
        """
        if date_range:
            start = datetime.strptime(date_range[0], "%Y-%m-%d")
            end = datetime.strptime(date_range[1], "%Y-%m-%d")
        else:
            # Default: Nov 1 to April 10
            start = datetime(season, 11, 1)
            end = datetime(season + 1, 4, 10)
        
        all_games = []
        current = start
        
        print(f"Fetching {self.gender} games from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                games = self.get_games(dates=[date_str])
                # Only keep completed games
                completed = [g for g in games if g.get("status") == "Final"]
                if completed:
                    all_games.extend(completed)
                    print(f"  {date_str}: {len(completed)} completed games (total: {len(all_games)})")
            except Exception as e:
                print(f"  {date_str}: Error - {e}")
            
            current += timedelta(days=1)
        
        print(f"Total games fetched: {len(all_games)}")
        return all_games


# Convenience instances
mens_client = ESPNCollegeBasketballClient("mens")
womens_client = ESPNCollegeBasketballClient("womens")
