"""
Data Manager - Orchestrates data collection and caching
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json
from api_client import BallDontLieClient
from odds_api_client import TheOddsAPIClient
import config


class DataManager:
    """Manages data fetching, caching, and aggregation"""
    
    def __init__(self):
        self.client = BallDontLieClient()
        self.odds_client = TheOddsAPIClient()
        self.data_dir = config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_cache_path(self, cache_name: str) -> str:
        """Get path for cache file"""
        return os.path.join(self.data_dir, f"{cache_name}.csv")
    
    def _is_cache_valid(self, cache_path: str, expiry_hours: int = None, 
                        required_cols: list = None) -> bool:
        """Check if cache file exists, is not expired, and has required columns.
        
        Args:
            cache_path: Path to cache CSV
            expiry_hours: Max age in hours before cache is stale
            required_cols: Optional list of columns that must exist in cached data
        """
        if not os.path.exists(cache_path):
            return False
        
        if expiry_hours is None:
            expiry_hours = config.CACHE_EXPIRY_HOURS
        
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_modified > timedelta(hours=expiry_hours):
            return False
        
        # Validate data integrity: check required columns exist and data isn't empty
        if required_cols:
            try:
                # Read just the header to check columns efficiently
                header_df = pd.read_csv(cache_path, nrows=0)
                missing = [c for c in required_cols if c not in header_df.columns]
                if missing:
                    print(f"Cache {cache_path} missing columns {missing[:5]}... - busting cache")
                    return False
            except Exception:
                return False
        
        return True
    
    def fetch_historical_games(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical games for specified seasons
        
        Args:
            seasons: List of seasons to fetch
            force_refresh: Force refresh from API instead of using cache
        """
        cache_path = self._get_cache_path("games_historical")
        
        # Required columns for valid historical data (including advanced stats)
        required_historical_cols = [
            'home_team_id', 'visitor_team_id', 'home_team_score', 'visitor_team_score',
            'home_fgm', 'home_fga', 'home_oreb', 'home_tov',
            'visitor_fgm', 'visitor_fga', 'visitor_oreb', 'visitor_tov'
        ]
        if not force_refresh and self._is_cache_valid(cache_path, expiry_hours=720, 
                                                       required_cols=required_historical_cols):
            print(f"Loading games from cache: {cache_path}")
            df = pd.read_csv(cache_path, low_memory=False)
            # Additional integrity check: ensure advanced stats aren't all zeros
            if 'home_fgm' in df.columns and (df['home_fgm'] == 0).mean() > 0.5:
                print(f"WARNING: >50% of games missing advanced stats in cache. Consider force_refresh=True.")
            return df
        
        print(f"Fetching games for seasons: {seasons}")
        all_games = []
        
        for season in seasons:
            print(f"Fetching season {season}...")
            games = self.client.get_games(seasons=[season], per_page=100)
            all_games.extend(games)
        
        df = pd.DataFrame(all_games)
        
        if not df.empty:
            # Flatten nested team data
            df['home_team_id'] = df['home_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
            df['home_team_name'] = df['home_team'].apply(lambda x: x['full_name'] if isinstance(x, dict) else None)
            df['home_team_abbreviation'] = df['home_team'].apply(lambda x: x['abbreviation'] if isinstance(x, dict) and 'abbreviation' in x else None)
            
            df['visitor_team_id'] = df['visitor_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
            df['visitor_team_name'] = df['visitor_team'].apply(lambda x: x['full_name'] if isinstance(x, dict) else None)
            df['visitor_team_abbreviation'] = df['visitor_team'].apply(lambda x: x['abbreviation'] if isinstance(x, dict) and 'abbreviation' in x else None)
            
            # --- ENRICH WITH ADVANCED STATS FROM NBA API ---
            print("Enriching dataset with Advanced Stats (Four Factors data) from nba_api...")
            
            from nba_api.stats.endpoints import leaguegamelog
            import time
            
            # We need to fetch data for ALL requested seasons to get the stats
            # Create a lookup: (Date, TeamAbbr) -> Stats Dict
            stats_lookup = {}
            
            # Columns to capture from NBA API
            # Map NBA API columns to our dataframe columns
            stat_map = {
                'FGM': 'fgm', 'FGA': 'fga', 'FG3M': 'fg3m', 
                'FTM': 'ftm', 'FTA': 'fta', 
                'OREB': 'oreb', 'DREB': 'dreb', 
                'TOV': 'tov', 'STL': 'stl', 'BLK': 'blk', 'PF': 'pf'
            }
            
            for season_yr in seasons:
                # Format season string (e.g., 2015 -> "2015-16")
                next_yr = str(season_yr + 1)[-2:]
                season_str = f"{season_yr}-{next_yr}"
                print(f"Fetching official NBA stats for {season_str}...")
                
                try:
                    # Fetch Regular Season AND Playoffs
                    for s_type in ['Regular Season', 'Playoffs']:
                        time.sleep(0.6) # Rate limit respect
                        log = leaguegamelog.LeagueGameLog(season=season_str, season_type_all_star=s_type)
                        nba_df = log.get_data_frames()[0]
                        
                        if not nba_df.empty:
                            for _, row in nba_df.iterrows():
                                date_str = str(row['GAME_DATE']) # YYYY-MM-DD
                                team_abbr = row['TEAM_ABBREVIATION']
                                
                                # Store all interesting stats
                                stats = {
                                    'pts': row['PTS'] # Official points
                                }
                                for nba_col, my_col in stat_map.items():
                                    if nba_col in row:
                                        stats[my_col] = row[nba_col]
                                
                                key = (date_str, team_abbr)
                                stats_lookup[key] = stats
                        
                except Exception as e:
                    print(f"Error fetching NBA API data for {season_str}: {e}")
            
            # Apply stats to dataframe
            print("Merging advanced stats into dataset...")
            
            # Initialize new columns with 0/None
            for col in stat_map.values():
                df[f'home_{col}'] = 0
                df[f'visitor_{col}'] = 0
            
            enriched_count = 0
            
            for idx, row in df.iterrows():
                # Extract date YYYY-MM-DD
                date_str = str(row['date']).split('T')[0]
                
                h_abbr = row.get('home_team_abbreviation')
                v_abbr = row.get('visitor_team_abbreviation')
                
                # Retrieve stats
                h_stats = stats_lookup.get((date_str, h_abbr))
                v_stats = stats_lookup.get((date_str, v_abbr))
                
                if h_stats and v_stats:
                    # Update Scores (Golden Source)
                    df.at[idx, 'home_team_score'] = h_stats['pts']
                    df.at[idx, 'visitor_team_score'] = v_stats['pts']
                    
                    # Update Advanced Stats
                    for col in stat_map.values():
                        df.at[idx, f'home_{col}'] = h_stats.get(col, 0)
                        df.at[idx, f'visitor_{col}'] = v_stats.get(col, 0)
                        
                    enriched_count += 1
            
            print(f"Successfully enriched {enriched_count}/{len(df)} games with advanced stats.")
            
            # Filter checks
            missing_stats = (df['home_fgm'] == 0).sum()
            if missing_stats > 0:
                print(f"Warning: {missing_stats} games missing advanced stats (likely name mismatch).")


            # Backfill missing scores using box_scores endpoint
            missing_scores_mask = (
                (df['home_team_score'] == 0) | (df['visitor_team_score'] == 0) |
                df['home_team_score'].isna() | df['visitor_team_score'].isna()
            )
            if missing_scores_mask.sum() > 0:
                print(f"Detected {missing_scores_mask.sum()} games with missing scores. Attempting backfill via box_scores...")
                df = self._enrich_from_box_scores(df)

            
            df.to_csv(cache_path, index=False)
            print(f"Saved {len(df)} games to cache")
        
        return df

    def _enrich_from_box_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch box scores for games with missing scores and aggregate player stats.
        This fixes the data starvation issue for historical games (2000-2015).
        """
        import time
        
        # Identify dates with missing scores
        missing_mask = (df['home_team_score'] == 0) | (df['visitor_team_score'] == 0) | df['home_team_score'].isna() | df['visitor_team_score'].isna()
        dates_to_check = pd.to_datetime(df[missing_mask]['date'], errors='coerce').dt.strftime('%Y-%m-%d').dropna().unique()
        
        print(f"Backfilling scores for {len(dates_to_check)} dates...")
        
        updates_count = 0
        
        # Process in batches to avoid overwhelming (though client handles rate limits)
        for date_str in dates_to_check:
            print(f"  Backfilling {date_str}...")
            
            try:
                # Fetch all box scores for this date
                box_scores = self.client.get_box_scores(dates=[date_str])
                
                # Group by game_id
                game_groups = {}
                for record in box_scores:
                    gid = record.get('game', {}).get('id')
                    if gid:
                        if gid not in game_groups:
                            game_groups[gid] = []
                        game_groups[gid].append(record)
                
                # Update dataframe
                for gid, records in game_groups.items():
                    # Find game in our dataframe
                    idx = df[df['id'] == gid].index
                    if len(idx) == 0:
                        continue
                        
                    idx = idx[0]
                    home_id = df.at[idx, 'home_team_id']
                    visitor_id = df.at[idx, 'visitor_team_id']
                    
                    # Aggregate stats
                    h_pts, v_pts = 0, 0
                    h_stats = {'fgm':0, 'fga':0, 'fg3m':0, 'ftm':0, 'fta':0, 'oreb':0, 'dreb':0, 'tov':0, 'stl':0, 'blk':0, 'pf':0}
                    v_stats = {'fgm':0, 'fga':0, 'fg3m':0, 'ftm':0, 'fta':0, 'oreb':0, 'dreb':0, 'tov':0, 'stl':0, 'blk':0, 'pf':0}
                    
                    for r in records:
                        tid = r.get('team', {}).get('id')
                        pts = r.get('pts') or 0
                        
                        # safely get other stats
                        def get_val(k): return r.get(k) or 0
                        
                        stats_delta = {
                            'fgm': get_val('fgm'), 'fga': get_val('fga'), 'fg3m': get_val('fg3m'),
                            'ftm': get_val('ftm'), 'fta': get_val('fta'), 
                            'oreb': get_val('oreb'), 'dreb': get_val('dreb'),
                            'tov': get_val('turnover'), 'stl': get_val('stl'), 'blk': get_val('blk'), 'pf': get_val('pf')
                        }
                        
                        if tid == home_id:
                            h_pts += pts
                            for k, v in stats_delta.items(): h_stats[k] += v
                        elif tid == visitor_id:
                            v_pts += pts
                            for k, v in stats_delta.items(): v_stats[k] += v
                    
                    # Update DataFrame if we found data
                    if h_pts > 0 or v_pts > 0:
                        df.at[idx, 'home_team_score'] = h_pts
                        df.at[idx, 'visitor_team_score'] = v_pts
                        
                        # Update advanced stat columns (if they are 0)
                        # We use a simple check: if fgm is 0, we assume advanced stats are missing
                        if df.at[idx, 'home_fgm'] == 0:
                            for k, v in h_stats.items(): df.at[idx, f'home_{k}'] = v
                        if df.at[idx, 'visitor_fgm'] == 0:
                            for k, v in v_stats.items(): df.at[idx, f'visitor_{k}'] = v
                            
                        updates_count += 1
                        
            except Exception as e:
                print(f"Error backfilling {date_str}: {e}")
                
        print(f"Backfill complete. Updated {updates_count} games.")
        return df
    
    def fetch_todays_games(self, target_date: str = None) -> pd.DataFrame:
        """
        Fetch games for a specific date (defaults to today)
        
        Args:
            target_date: Date in YYYY-MM-DD format, defaults to today
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching games for {target_date}")
        games = self.client.get_games(dates=[target_date])
        df = pd.DataFrame(games)
        
        if not df.empty:
            # Flatten nested team data
            df['home_team_id'] = df['home_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
            df['home_team_name'] = df['home_team'].apply(lambda x: x['full_name'] if isinstance(x, dict) else None)
            df['visitor_team_id'] = df['visitor_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
            df['visitor_team_name'] = df['visitor_team'].apply(lambda x: x['full_name'] if isinstance(x, dict) else None)
        
        return df
    
    def fetch_vegas_odds(self, dates: List[str] = None, game_ids: List[int] = None, 
                         games_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fetch betting odds for games with fallback to The Odds API
        
        Args:
            dates: Dates to fetch odds for
            game_ids: Specific game IDs
            games_df: DataFrame of games (for team name mapping)
        """
        # For today's odds, always refresh
        if dates and dates[0] == datetime.now().strftime("%Y-%m-%d"):
            print(f"Fetching fresh odds for today")
        
        # Try BallDontLie first (GOAT tier)
        try:
            print("   Trying BallDontLie API...")
            odds = self.client.get_betting_odds(dates=dates, game_ids=game_ids)
            df_bdl = pd.DataFrame(odds)
            
            if not df_bdl.empty:
                print(f"   BallDontLie: {len(df_bdl)} records, {df_bdl['game_id'].nunique()} games")
            else:
                print("   BallDontLie: No odds available")
        except Exception as e:
            print(f"   BallDontLie error: {e}")
            df_bdl = pd.DataFrame()
        
        # Check if we need fallback (missing games)
        if games_df is not None and not games_df.empty:
            total_games = len(games_df)
            games_with_odds = df_bdl['game_id'].nunique() if not df_bdl.empty else 0
            missing_count = total_games - games_with_odds
            
            if missing_count > 0:
                print(f"   Missing odds for {missing_count} games, trying The Odds API backup...")
                
                try:
                    # Fetch from The Odds API
                    odds_api_data = self.odds_client.get_odds()
                    
                    if odds_api_data:
                        # Create team name mapping for game ID matching
                        team_mapping = {}
                        for _, game in games_df.iterrows():
                            game_id = game.get('id')
                            home_name = game.get('home_team_name', '')
                            away_name = game.get('visitor_team_name', '')
                            team_mapping[game_id] = {'home': home_name, 'away': away_name}
                        
                        # Convert to BallDontLie format
                        converted_odds = self.odds_client.convert_to_balldontlie_format(
                            odds_api_data, 
                            team_mapping
                        )
                        
                        df_odds_api = pd.DataFrame(converted_odds)
                        
                        if not df_odds_api.empty:
                            # Get games not covered by BallDontLie
                            if not df_bdl.empty:
                                bdl_game_ids = set(df_bdl['game_id'].unique())
                                df_odds_api = df_odds_api[~df_odds_api['game_id'].isin(bdl_game_ids)]
                            
                            if not df_odds_api.empty:
                                print(f"   The Odds API: {len(df_odds_api)} additional records, {df_odds_api['game_id'].nunique()} games")
                                
                                # Combine both sources
                                df = pd.concat([df_bdl, df_odds_api], ignore_index=True)
                                print(f"   Combined: {df['game_id'].nunique()}/{total_games} games covered")
                                return df
                        else:
                            print("   The Odds API: No additional odds found")
                    else:
                        print("   The Odds API: No data returned")
                        
                except Exception as e:
                    print(f"   The Odds API error: {e}")
        
        # Return BallDontLie results (even if incomplete)
        if not df_bdl.empty:
            return df_bdl
            
        # No odds available from any source - return empty DataFrame.
        # Mock odds were removed because random fake data pollutes the model
        # when used in training. The model's smart NaN defaults (vegas_has_odds=0,
        # vegas_spread=0, vegas_total=220, vegas_implied_prob=0.5) handle missing
        # odds correctly as neutral signals.
        if games_df is not None and not games_df.empty:
            print(f"   No odds available for {len(games_df)} games. Model will use neutral defaults.")

        return pd.DataFrame()
    
    def fetch_advanced_stats(self, seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """Fetch advanced statistics for seasons"""
        cache_path = self._get_cache_path("advanced_stats")
        
        if not force_refresh and self._is_cache_valid(cache_path, expiry_hours=720):
            print(f"Loading advanced stats from cache: {cache_path}")
            return pd.read_csv(cache_path, low_memory=False)
        
        print(f"Fetching advanced stats for seasons: {seasons}")
        all_stats = []
        
        for season in seasons:
            print(f"Fetching advanced stats for season {season}...")
            stats = self.client.get_advanced_stats(seasons=[season], per_page=100)
            all_stats.extend(stats)
        
        df = pd.DataFrame(all_stats)
        
        if not df.empty:
            df.to_csv(cache_path, index=False)
            print(f"Saved {len(df)} advanced stats to cache")
        
        return df
    
    def fetch_box_scores(self, dates: List[str], game_ids: List[int] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch box scores for dates and return flattened player stats
        
        Args:
            dates: List of dates (YYYY-MM-DD)
            game_ids: Optional list of game IDs to filter
            force_refresh: Ignore cache
        """
        all_records = []
        dates_to_fetch = []
        
        # Check cache for each date
        for date in dates:
            cache_name = f"box_scores_{date}"
            cache_path = self._get_cache_path(cache_name)
            
            if not force_refresh and self._is_cache_valid(cache_path, expiry_hours=24):
                print(f"Loading box scores from cache: {cache_path}")
                try:
                    df_date = pd.read_csv(cache_path, low_memory=False)
                    all_records.append(df_date)
                except Exception as e:
                    print(f"Error reading cache {cache_path}: {e}, will re-fetch")
                    dates_to_fetch.append(date)
            else:
                dates_to_fetch.append(date)
        
        # Fetch missing dates
        if dates_to_fetch:
            print(f"Fetching box scores for {len(dates_to_fetch)} dates...")
            # We fetch by date using our modified client
            # The client handles looping through dates, but we want to process per date to cache per date
            
            for date in dates_to_fetch:
                print(f"Fetching box scores for {date}...")
                games_data = self.client.get_box_scores(dates=[date])
                
                date_records = []
                for game in games_data:
                    game_id = game.get('id')
                    game_date = game.get('date')
                    
                    # Process Home Team
                    home_team = game.get('home_team', {})
                    home_team_id = home_team.get('id')
                    home_team_name = home_team.get('full_name')
                    home_players = home_team.get('players', [])
                    
                    for player_stat in home_players:
                        player_info = player_stat.get('player', {})
                        record = {
                            'game_id': game_id,
                            'game_date': game_date,
                            'team_id': home_team_id,
                            'team_name': home_team_name,
                            'is_home': True,
                            'player_id': player_info.get('id'),
                            'first_name': player_info.get('first_name'),
                            'last_name': player_info.get('last_name'),
                            'position': player_info.get('position'),
                            **{k:v for k,v in player_stat.items() if k != 'player'}
                        }
                        date_records.append(record)
                        
                    # Process Visitor Team
                    visitor_team = game.get('visitor_team', {})
                    visitor_team_id = visitor_team.get('id')
                    visitor_team_name = visitor_team.get('full_name')
                    visitor_players = visitor_team.get('players', [])
                    
                    for player_stat in visitor_players:
                        player_info = player_stat.get('player', {})
                        record = {
                            'game_id': game_id,
                            'game_date': game_date,
                            'team_id': visitor_team_id,
                            'team_name': visitor_team_name,
                            'is_home': False,
                            'player_id': player_info.get('id'),
                            'first_name': player_info.get('first_name'),
                            'last_name': player_info.get('last_name'),
                            'position': player_info.get('position'),
                            **{k:v for k,v in player_stat.items() if k != 'player'}
                        }
                        date_records.append(record)
                
                if date_records:
                    df_date = pd.DataFrame(date_records)
                    cache_path = self._get_cache_path(f"box_scores_{date}")
                    df_date.to_csv(cache_path, index=False)
                    print(f"Saved {len(df_date)} player records to cache for {date}")
                    all_records.append(df_date)
                else:
                    print(f"No box scores found for {date}")

        if not all_records:
            return pd.DataFrame()
            
        final_df = pd.concat(all_records, ignore_index=True)
        
        # Filter by game_ids if provided
        if game_ids:
            final_df = final_df[final_df['game_id'].isin(game_ids)]
            
        return final_df
    def fetch_team_season_averages(self, season: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch all categories of team season averages
        Returns dataframe with general base and advanced stats
        """
        cache_path = self._get_cache_path(f"team_averages_{season}")
        
        if self._is_cache_valid(cache_path, expiry_hours=168):  # 1 week
            return pd.read_csv(cache_path)
        
        print(f"Fetching team season averages for {season}")
        
        # Fetch different categories
        all_averages = []
        
        categories = [
            ("general", "base"),
            ("general", "advanced"),
        ]
        
        for category, stat_type in categories:
            try:
                data = self.client.get_team_season_averages(
                    season=season,
                    category=category,
                    stat_type=stat_type
                )
                
                for team_data in data:
                    team_info = team_data.get('team', {})
                    stats = team_data.get('stats', {})
                    
                    record = {
                        'team_id': team_info.get('id'),
                        'team_name': team_info.get('full_name'),
                        'season': season,
                        'category': category,
                        'type': stat_type
                    }
                    record.update(stats)
                    all_averages.append(record)
            except Exception as e:
                print(f"Error fetching {category}/{stat_type}: {e}")
        
        df = pd.DataFrame(all_averages)
        
        if not df.empty:
            df.to_csv(cache_path, index=False)
        
        return df
    
    def fetch_standings(self, season: int) -> pd.DataFrame:
        """Fetch team standings"""
        cache_path = self._get_cache_path(f"standings_{season}")
        
        if self._is_cache_valid(cache_path, expiry_hours=24):
            return pd.read_csv(cache_path)
        
        print(f"Fetching standings for {season}")
        standings = self.client.get_standings(season=season)
        df = pd.DataFrame(standings)
        
        if not df.empty:
            # Flatten team data
            df['team_id'] = df['team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
            df['team_name'] = df['team'].apply(lambda x: x['full_name'] if isinstance(x, dict) else None)
            df.to_csv(cache_path, index=False)
        
        return df
    
    def get_complete_training_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Build complete training dataset with all features
        
        Args:
            seasons: List of seasons to include in training
        """
        print("=" * 50)
        print("BUILDING COMPLETE TRAINING DATASET")
        print("=" * 50)
        
        # Fetch games
        games_df = self.fetch_historical_games(seasons, force_refresh=False)
        
        if games_df.empty:
            print("No games found!")
            return pd.DataFrame()
        
        # Filter to completed games only
        games_df = games_df[games_df['status'] == 'Final'].copy()
        
        # Filter out games with missing scores (0) - Critical fix for training error
        total_games = len(games_df)
        games_df = games_df[
            (games_df['home_team_score'] > 0) & 
            (games_df['visitor_team_score'] > 0)
        ].copy()
        
        dropped_games = total_games - len(games_df)
        if dropped_games > 0:
            print(f"Dropped {dropped_games} games with missing/zero scores")
            
        print(f"Found {len(games_df)} valid completed games")
        
        # Fetch advanced stats
        advanced_df = self.fetch_advanced_stats(seasons, force_refresh=False)
        
        # Fetch team averages for each season
        all_team_stats = []
        for season in seasons:
            season_stats = self.fetch_team_season_averages(season)
            if not season_stats.empty:
                all_team_stats.append(season_stats)
        
        if all_team_stats:
            team_stats_df = pd.concat(all_team_stats, ignore_index=True)
        else:
            team_stats_df = pd.DataFrame()
        
        # Fetch standings
        all_standings = []
        for season in seasons:
            standings = self.fetch_standings(season)
            if not standings.empty:
                all_standings.append(standings)
        
        if all_standings:
            standings_df = pd.concat(all_standings, ignore_index=True)
        else:
            standings_df = pd.DataFrame()
        
        print(f"Fetched {len(team_stats_df)} team stat records")
        print(f"Fetched {len(standings_df)} standing records")
        
        # NOTE: Historical odds not fetched - no reliable source available
        # Vegas odds are fetched live for predictions and displayed in UI only
        
        return {
            'games': games_df,
            'advanced_stats': advanced_df,
            'team_stats': team_stats_df,
            'standings': standings_df
        }


# Convenience instance
data_manager = DataManager()
