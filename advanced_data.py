"""
Advanced Data Manager for NBA Elite Model
Fetches detailed advanced stats (PER, PIE, USG%, ratings) using nba_api
"""
import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from datetime import datetime
import os
import config

class AdvancedDataManager:
    def __init__(self):
        self.data_dir = config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_advanced_team_stats(self, seasons, measure_type='Advanced'):
        """
        Fetch advanced team stats (OffRtg, DefRtg, Pace, PIE)
        """
        all_stats = []
        
        for season in seasons:
            # Format season: 2023 -> '2023-24'
            next_yr = str(season + 1)[-2:]
            season_str = f"{season}-{next_yr}"
            
            print(f"Fetching {measure_type} team stats for {season_str}...")
            
            try:
                # Regular Season
                time.sleep(1.0) # Respect API limits
                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season_str,
                    measure_type_detailed_defense=measure_type
                ).get_data_frames()[0]
                
                stats['SEASON'] = season
                stats['SEASON_TYPE'] = 'Regular Season'
                all_stats.append(stats)
                
            except Exception as e:
                print(f"Error fetching {season_str}: {e}")
                
        if not all_stats:
            return pd.DataFrame()
            
        return pd.concat(all_stats, ignore_index=True)

    def process_for_lstm(self, lookback=10):
        """
        Prepare sequence data for LSTM training
        """
        # Placeholder for sequence generation logic
        pass

if __name__ == "__main__":
    adm = AdvancedDataManager()
    df = adm.fetch_advanced_team_stats(config.TRAINING_SEASONS)
    print(df.head())
    df.to_csv("data/advanced_team_stats_raw.csv", index=False)
