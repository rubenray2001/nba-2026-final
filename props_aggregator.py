"""
Player Props Aggregator for College Basketball
Pulls props from multiple sources:
  - The Odds API (Bovada, DraftKings, FanDuel, BetMGM, etc.)
  - PrizePicks (DFS projections via Playwright browser)
  - Underdog Fantasy (pick'em projections)
"""
import requests
import time
import json
import os
import sys
import subprocess
import tempfile
from typing import List, Dict, Optional
from datetime import datetime
import config


# ==================== PRIZEPICKS CLIENT ====================

class PrizePicksClient:
    """
    Client for PrizePicks projections.
    
    PrizePicks uses PerimeterX bot protection that blocks all standard HTTP
    libraries (requests, cloudscraper, curl_cffi). The only 100% reliable
    approach is using Playwright with a persistent browser context:
    
    1. First run: Opens a real browser, auto-solves the press-and-hold captcha,
       and caches PerimeterX cookies to disk.
    2. Subsequent runs: Reuses cached cookies - instant API access (~3 sec),
       no captcha shown.
    3. If cookies expire: Automatically re-solves captcha (browser window pops
       up briefly off-screen then closes).
    """
    
    BASE_URL = "https://api.prizepicks.com"
    APP_URL = "https://app.prizepicks.com"
    
    LEAGUE_IDS = {
        "mens": "20",      # CBB / NCAAB
        "womens": "176",   # WCBB / WNCAAB
    }
    
    STAT_DISPLAY = {
        "Points": "Points",
        "Rebounds": "Rebounds",
        "Assists": "Assists",
        "3-Point Made": "3-Pointers",
        "3-PT Made": "3-Pointers",
        "3-Pointers Made": "3-Pointers",
        "Pts+Rebs+Asts": "PRA",
        "Pts+Rebs": "Pts+Rebs",
        "Pts+Asts": "Pts+Asts",
        "Rebs+Asts": "Rebs+Asts",
        "Steals": "Steals",
        "Blocks": "Blocks",
        "Blocked Shots": "Blocks",
        "Turnovers": "Turnovers",
        "Fantasy Score": "Fantasy",
        "Fantasy": "Fantasy",
        "Blks+Stls": "Blks+Stls",
        "1H Points": "1H Points",
        "1H Pts+Rebs+Asts": "1H PRA",
        "Double-Double": "Double-Double",
    }
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.league_id = self.LEAGUE_IDS.get(gender, "20")
        self.state_dir = os.path.join(config.DATA_DIR, "pp_browser_state")
        os.makedirs(self.state_dir, exist_ok=True)
    
    def get_projections(self) -> List[Dict]:
        """
        Fetch PrizePicks projections. Uses Playwright with persistent context
        for 100% reliability against PerimeterX bot protection.
        
        When called from Streamlit, Playwright's sync API conflicts with
        Streamlit's thread model. To avoid this, we run the Playwright
        fetch in a subprocess first.
        
        Falls back to in-process Playwright, cloudscraper, and requests.
        """
        # Primary: subprocess Playwright (works inside Streamlit)
        data = self._fetch_via_subprocess()
        if data:
            return self._parse_projections(data)
        
        # Fallback 1: in-process Playwright (works from CLI)
        data = self._fetch_with_playwright()
        if data:
            return self._parse_projections(data)
        
        # Fallback 2: cloudscraper (works sometimes)
        data = self._fetch_with_cloudscraper()
        if data:
            return self._parse_projections(data)
        
        # Fallback 3: standard requests (rarely works)
        data = self._fetch_with_requests()
        if data:
            return self._parse_projections(data)
        
        print("PrizePicks: all methods failed")
        return []
    
    def _fetch_via_subprocess(self) -> Optional[Dict]:
        """
        Run Playwright fetch in a clean subprocess.
        This avoids Streamlit's thread-model conflicts with Playwright's
        sync API / greenlet. The subprocess writes JSON to a temp file.
        """
        script = f'''
import json, sys, os
sys.path.insert(0, {repr(os.path.dirname(os.path.abspath(__file__)))})
os.chdir({repr(os.path.dirname(os.path.abspath(__file__)))})
from props_aggregator import PrizePicksClient
client = PrizePicksClient({repr(self.gender)})
data = client._fetch_with_playwright()
if data and "data" in data:
    outpath = sys.argv[1]
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"OK:{{len(data['data'])}}")
else:
    print("FAIL")
'''
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w", encoding="utf-8"
            )
            tmp_path = tmp.name
            tmp.close()
            
            result = subprocess.run(
                [sys.executable, "-c", script, tmp_path],
                capture_output=True, text=True, timeout=90,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            
            stdout = result.stdout.strip()
            ok_line = [l for l in stdout.splitlines() if l.startswith("OK:")]
            if ok_line:
                count = ok_line[0].split(":")[1]
                with open(tmp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"PrizePicks: {count} projections (subprocess)")
                return data
            else:
                stderr_preview = (result.stderr or "")[:200]
                print(f"PrizePicks subprocess: {stdout} {stderr_preview}")
        except subprocess.TimeoutExpired:
            print("PrizePicks subprocess: timed out after 90s")
        except Exception as e:
            print(f"PrizePicks subprocess error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        
        return None
    
    def _fetch_with_playwright(self) -> Optional[Dict]:
        """
        Use Playwright with persistent browser context.
        Cached cookies make subsequent calls instant (~3 sec).
        On first run or cookie expiry, briefly opens browser to solve captcha.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("PrizePicks: Playwright not installed, skipping browser method")
            return None
        
        api_url = (f"{self.BASE_URL}/projections"
                   f"?league_id={self.league_id}&per_page=250&single_stat=true")
        result = None
        
        try:
            with sync_playwright() as p:
                context = p.chromium.launch_persistent_context(
                    self.state_dir,
                    headless=False,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--window-size=1200,800",
                        "--window-position=-2400,-2400",
                    ],
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1200, "height": 800},
                    locale="en-US",
                )
                
                context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', "
                    "{get: () => undefined});"
                )
                
                page = context.pages[0] if context.pages else context.new_page()
                
                # Step 1: Try API directly (works with cached cookies)
                page.goto(api_url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                
                body = page.evaluate("() => document.body.innerText || ''")
                data = self._try_parse_json(body)
                if data and "data" in data:
                    print(f"PrizePicks: {len(data['data'])} projections (cached cookies)")
                    result = data
                    context.close()
                    return result
                
                # Step 2: Cookies expired - visit app to solve captcha
                print("PrizePicks: solving PerimeterX captcha...")
                page.goto(
                    f"{self.APP_URL}/board?league=CBB",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                
                for attempt in range(15):
                    time.sleep(5)
                    title = page.title()
                    
                    if "denied" not in title.lower():
                        print("PrizePicks: captcha cleared")
                        time.sleep(3)
                        break
                    
                    captcha = page.query_selector("#px-captcha")
                    if captcha and captcha.is_visible():
                        try:
                            box = captcha.bounding_box()
                            if box:
                                x = box["x"] + box["width"] / 2
                                y = box["y"] + box["height"] / 2
                                page.mouse.move(x, y)
                                time.sleep(0.3)
                                page.mouse.down()
                                time.sleep(8)
                                page.mouse.up()
                                time.sleep(2)
                        except Exception:
                            pass
                
                # Step 3: Retry API after captcha resolution
                page.goto(api_url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(2)
                
                body = page.evaluate("() => document.body.innerText || ''")
                data = self._try_parse_json(body)
                if data and "data" in data:
                    print(f"PrizePicks: {len(data['data'])} projections (after captcha)")
                    result = data
                else:
                    print("PrizePicks: API still blocked after captcha attempts")
                
                context.close()
        except Exception as e:
            print(f"PrizePicks Playwright error: {e}")
        
        return result
    
    def _fetch_with_cloudscraper(self) -> Optional[Dict]:
        """Fallback: try cloudscraper (works intermittently)"""
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            r = scraper.get(
                f"{self.BASE_URL}/projections",
                params={
                    "league_id": self.league_id,
                    "per_page": 250,
                    "single_stat": "true",
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if "data" in data and data["data"]:
                    print(f"PrizePicks: {len(data['data'])} projections (cloudscraper)")
                    return data
        except ImportError:
            pass
        except Exception as e:
            print(f"PrizePicks cloudscraper failed: {e}")
        return None
    
    def _fetch_with_requests(self) -> Optional[Dict]:
        """Last resort: standard requests"""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Referer": f"{self.APP_URL}/",
                "Origin": self.APP_URL,
            }
            r = requests.get(
                f"{self.BASE_URL}/projections",
                params={
                    "league_id": self.league_id,
                    "per_page": 250,
                    "single_stat": "true",
                },
                headers=headers,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if "data" in data and data["data"]:
                    print(f"PrizePicks: {len(data['data'])} projections (requests)")
                    return data
        except Exception as e:
            print(f"PrizePicks requests failed: {e}")
        return None
    
    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        """Safely try to parse JSON from text"""
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    
    def _parse_projections(self, data: Dict) -> List[Dict]:
        """Parse PrizePicks API response into standardized props"""
        props = []
        
        players_map = {}
        games_map = {}
        
        for item in data.get("included", []):
            item_type = item.get("type", "")
            item_id = item.get("id", "")
            attrs = item.get("attributes", {})
            
            if item_type == "new_player":
                players_map[item_id] = {
                    "name": attrs.get("display_name", attrs.get("name", "Unknown")),
                    "team": attrs.get("team", ""),
                    "position": attrs.get("position", ""),
                    "image_url": attrs.get("image_url", ""),
                }
            elif item_type == "game":
                away = attrs.get("away_team", "")
                home = attrs.get("home_team", "")
                games_map[item_id] = {
                    "game": f"{away} @ {home}" if away and home else attrs.get("name", ""),
                    "home_team": home,
                    "away_team": away,
                    "start_time": attrs.get("start_time", ""),
                }
        
        for proj in data.get("data", []):
            attrs = proj.get("attributes", {})
            relationships = proj.get("relationships", {})
            
            player_id = relationships.get("new_player", {}).get("data", {}).get("id", "")
            player_info = players_map.get(player_id, {})
            
            game_id = relationships.get("game", {}).get("data", {}).get("id", "")
            game_info = games_map.get(game_id, {})
            
            stat_type = attrs.get("stat_type", "")
            line_score = attrs.get("line_score")
            
            if not player_info.get("name") or line_score is None:
                continue
            
            if attrs.get("status") == "suspended":
                continue
            
            # Only use STANDARD lines — these match the actual PrizePicks board.
            # "demon" lines are boosted alternates (higher lines, harder to hit).
            # "goblin" lines are reduced alternates (lower lines, easier to hit).
            # Including demon/goblin produces wrong lines that don't match the
            # PrizePicks website and distort best-bets edge analysis.
            odds_type = attrs.get("odds_type", "standard")
            if odds_type != "standard":
                continue
            
            display_stat = self.STAT_DISPLAY.get(stat_type, stat_type)
            
            # Standard PrizePicks lines are ~50/50, maps to -110/-110
            over_odds = -110
            under_odds = -110
            
            props.append({
                "source": "PrizePicks",
                "game": game_info.get("game", ""),
                "home_team": game_info.get("home_team", ""),
                "away_team": game_info.get("away_team", ""),
                "commence_time": game_info.get("start_time", ""),
                "player": player_info.get("name", "Unknown"),
                "team": player_info.get("team", ""),
                "position": player_info.get("position", ""),
                "prop_type": display_stat,
                "line": float(line_score),
                "over_odds": over_odds,
                "under_odds": under_odds,
                "bookmaker": "PrizePicks",
                "bookmaker_key": "prizepicks",
                "image_url": player_info.get("image_url", ""),
            })
        
        return props


# ==================== UNDERDOG FANTASY CLIENT ====================

class UnderdogFantasyClient:
    """
    Client for Underdog Fantasy public pick'em API.
    Provides Higher/Lower projections with American odds for player stats.
    No API key required - public endpoint.
    
    API: https://api.underdogfantasy.com/beta/v5/over_under_lines
    Returns all sports. We filter to CBB (men's) or WCBB (women's).
    
    Data chain:
      over_under_lines -> over_under.appearance_stat.appearance_id -> appearances
      appearances.player_id -> players
      appearances.match_id -> games
    """
    
    BASE_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"
    
    SPORT_IDS = {
        "mens": "CBB",
        "womens": "WCBB",
    }
    
    STAT_DISPLAY = {
        "points": "Points",
        "rebounds": "Rebounds",
        "assists": "Assists",
        "three_pointers_made": "3-Pointers",
        "pts_rebs_asts": "PRA",
        "pts_rebs": "Pts+Rebs",
        "pts_asts": "Pts+Asts",
        "rebs_asts": "Rebs+Asts",
        "steals": "Steals",
        "blocks": "Blocks",
        "turnovers": "Turnovers",
        "fantasy_points": "Fantasy",
        "blks_stls": "Blks+Stls",
        "double_doubles": "Double-Double",
    }
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.sport_id = self.SPORT_IDS.get(gender, "CBB")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://underdogfantasy.com/",
        })
    
    def get_projections(self) -> List[Dict]:
        """Fetch Underdog Fantasy pick'em lines for college basketball"""
        try:
            response = self.session.get(self.BASE_URL, timeout=20)
            response.raise_for_status()
            data = response.json()
            return self._parse_projections(data)
        except requests.exceptions.RequestException as e:
            print(f"Underdog Fantasy request failed: {e}")
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Underdog Fantasy parse error: {e}")
            return []
    
    def _parse_projections(self, data: Dict) -> List[Dict]:
        """
        Parse Underdog Fantasy API response.
        
        Structure:
          players: [{id, first_name, last_name, sport_id, position_name, image_url}]
          appearances: [{id, player_id, match_id, match_type}]
          games: [{id, title, full_team_names_title, sport_id, scheduled_at}]
          over_under_lines: [{stat_value, status, options: [{choice, american_price}],
                              over_under: {appearance_stat: {appearance_id, stat}}}]
        """
        props = []
        
        players_map = {}
        for player in data.get("players", []):
            players_map[player.get("id", "")] = player
        
        appearances_map = {}
        for appearance in data.get("appearances", []):
            appearances_map[appearance.get("id", "")] = appearance
        
        games_map = {}
        for game in data.get("games", []):
            gid = game.get("id")
            games_map[gid] = game
            games_map[str(gid)] = game
        
        for line in data.get("over_under_lines", []):
            if line.get("status") == "suspended":
                continue
            
            stat_value = line.get("stat_value")
            if stat_value is None:
                continue
            
            over_under = line.get("over_under", {})
            if not over_under:
                continue
            
            appearance_stat = over_under.get("appearance_stat", {})
            appearance_id = appearance_stat.get("appearance_id", "")
            stat_name = appearance_stat.get("stat", "")
            display_stat_name = appearance_stat.get("display_stat", "")
            
            appearance = appearances_map.get(appearance_id, {})
            player_id = appearance.get("player_id", "")
            player = players_map.get(player_id, {})
            
            if player.get("sport_id", "") != self.sport_id:
                continue
            
            match_id = appearance.get("match_id", "")
            game = games_map.get(match_id) or games_map.get(str(match_id), {})
            
            higher_odds = None
            lower_odds = None
            for opt in line.get("options", []):
                price_str = opt.get("american_price")
                if price_str is not None:
                    try:
                        price = int(price_str) if "." not in str(price_str) else int(float(price_str))
                    except (ValueError, TypeError):
                        price = None
                    
                    if opt.get("choice") == "higher":
                        higher_odds = price
                    elif opt.get("choice") == "lower":
                        lower_odds = price
            
            player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            if not player_name:
                continue
            
            friendly_stat = self.STAT_DISPLAY.get(
                stat_name, display_stat_name or stat_name.replace("_", " ").title()
            )
            
            game_title = game.get("full_team_names_title") or game.get("title", "")
            
            props.append({
                "source": "Underdog",
                "game": game_title,
                "home_team": "",
                "away_team": "",
                "commence_time": game.get("scheduled_at", ""),
                "player": player_name,
                "team": "",
                "position": player.get("position_name", ""),
                "prop_type": friendly_stat,
                "line": float(stat_value),
                "over_odds": higher_odds,
                "under_odds": lower_odds,
                "bookmaker": "Underdog",
                "bookmaker_key": "underdog",
                "image_url": player.get("image_url", ""),
            })
        
        return props


# ==================== AGGREGATOR ====================

class PropsAggregator:
    """
    Aggregates player props from all available sources into a unified format.
    Sources: The Odds API (Bovada, DK, FD, etc.), PrizePicks, Underdog Fantasy
    """
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.sport = config.GENDER_CONFIG[gender]["odds_sport"]
    
    def fetch_all_props(self, include_odds_api: bool = True,
                        include_prizepicks: bool = True,
                        include_underdog: bool = True,
                        odds_api_max_events: int = 8) -> Dict[str, List[Dict]]:
        """
        Fetch props from all enabled sources.
        
        Returns dict with keys: odds_api, prizepicks, underdog, all, sources_status
        """
        results = {
            "odds_api": [],
            "prizepicks": [],
            "underdog": [],
            "all": [],
            "sources_status": {},
        }
        
        if include_odds_api:
            try:
                from odds_api_client import TheOddsAPIClient
                client = TheOddsAPIClient(sport=self.sport)
                odds_props = client.get_player_props(max_events=odds_api_max_events)
                results["odds_api"] = odds_props
                bookmakers = list(set(
                    p.get("bookmaker", "") for p in odds_props if p.get("bookmaker")
                ))
                results["sources_status"]["The Odds API"] = {
                    "status": "ok",
                    "count": len(odds_props),
                    "bookmakers": bookmakers,
                }
                print(f"   The Odds API: {len(odds_props)} props "
                      f"from {len(bookmakers)} bookmakers")
            except Exception as e:
                print(f"   The Odds API failed: {e}")
                results["sources_status"]["The Odds API"] = {
                    "status": "error", "error": str(e)
                }
        
        if include_prizepicks:
            try:
                pp_client = PrizePicksClient(gender=self.gender)
                pp_props = pp_client.get_projections()
                results["prizepicks"] = pp_props
                if pp_props:
                    results["sources_status"]["PrizePicks"] = {
                        "status": "ok",
                        "count": len(pp_props),
                    }
                else:
                    results["sources_status"]["PrizePicks"] = {
                        "status": "blocked",
                        "count": 0,
                        "message": "All methods failed",
                    }
            except Exception as e:
                print(f"   PrizePicks failed: {e}")
                results["sources_status"]["PrizePicks"] = {
                    "status": "error", "error": str(e)
                }
        
        if include_underdog:
            try:
                ud_client = UnderdogFantasyClient(gender=self.gender)
                ud_props = ud_client.get_projections()
                results["underdog"] = ud_props
                results["sources_status"]["Underdog Fantasy"] = {
                    "status": "ok",
                    "count": len(ud_props),
                }
                print(f"   Underdog Fantasy: {len(ud_props)} projections")
            except Exception as e:
                print(f"   Underdog Fantasy failed: {e}")
                results["sources_status"]["Underdog Fantasy"] = {
                    "status": "error", "error": str(e)
                }
        
        results["all"] = results["odds_api"] + results["prizepicks"] + results["underdog"]
        
        return results
    
    def get_line_comparison(self, all_props: List[Dict]) -> List[Dict]:
        """
        Build cross-platform line comparison for each player+prop_type combo.
        Shows how lines differ across sources.
        """
        import pandas as pd
        from collections import defaultdict
        
        player_props = defaultdict(lambda: defaultdict(list))
        
        for prop in all_props:
            player = prop.get("player", "").strip()
            prop_type = prop.get("prop_type", "").strip()
            if not player or not prop_type:
                continue
            
            key = (player, prop_type)
            source = prop.get("bookmaker", prop.get("source", "Unknown"))
            player_props[key][source].append(prop)
        
        comparisons = []
        for (player, prop_type), sources in player_props.items():
            comp = {
                "player": player,
                "prop_type": prop_type,
                "game": "",
                "sources": {},
                "line_range": {"min": float('inf'), "max": float('-inf')},
                "num_sources": len(sources),
            }
            
            for source_name, source_props in sources.items():
                best = source_props[0]
                line = best.get("line", 0)
                comp["sources"][source_name] = {
                    "line": line,
                    "over_odds": best.get("over_odds"),
                    "under_odds": best.get("under_odds"),
                }
                if not comp["game"] and best.get("game"):
                    comp["game"] = best["game"]
                comp["line_range"]["min"] = min(comp["line_range"]["min"], line)
                comp["line_range"]["max"] = max(comp["line_range"]["max"], line)
            
            comp["line_spread"] = comp["line_range"]["max"] - comp["line_range"]["min"]
            
            all_lines = [s["line"] for s in comp["sources"].values()]
            comp["consensus_line"] = sum(all_lines) / len(all_lines) if all_lines else 0
            
            comparisons.append(comp)
        
        comparisons.sort(key=lambda x: (-x["num_sources"], -x.get("line_spread", 0)))
        return comparisons
