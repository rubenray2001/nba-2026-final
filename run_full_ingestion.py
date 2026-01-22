from data_manager import data_manager
import pandas as pd
import sys

def ingest_full_history():
    print("="*60)
    print("STARTING FULL HISTORICAL DATA INGESTION (2000-2026)")
    print("="*60)
    
    # Define season range: 2000 to 2026
    seasons = list(range(2000, 2027))
    print(f"Target Seasons: {seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)")
    
    # CLEAR CACHE TO FORCE FRESH FETCH
    import os
    import config
    cache_path = os.path.join(config.DATA_DIR, "games_historical.csv")
    if os.path.exists(cache_path):
        print(f"Removing existing cache: {cache_path}")
        os.remove(cache_path)
    
    try:
        # Call the data manager to build the complete dataset
        # force_refresh=False so we use existing cache where possible, 
        # but the logic inside fetch_historical_games will trigger the backfill fix 
        # for any files that have missing scores (which is all of them essentially if they were cached bad, 
        # but actually we probably want to FORCE REFRESH to be safe? 
        # The user said "Backfilling Missing NBA Scores", suggesting existing data is bad.
        # But wait, our 'nba_api' logic runs if it detects 0-0 scores in the dataframe.
        # If the cache has 0-0 scores, it loads them, detects them, and then fixes them using nba_api.
        # So force_refresh=False is fine, it will just load the bad data and then fix it.
        # However, to be absolutely sure we get a clean slate or if the logic requires it, we could confirm.
        # Let's check logic:
        # 1. Load from cache OR API.
        # 2. Check for 0-0.
        # 3. If 0-0 found, run nba_api backfill.
        # 4. Save to cache.
        # So simply calling get_complete_training_data is enough.
        
        data = data_manager.get_complete_training_data(seasons)
        
        if not data or data.get('games', pd.DataFrame()).empty:
            print("\nCRITICAL FAILURE: No data returned from manager.")
            return
            
        games = data['games']
        print("\n" + "="*60)
        print("INGESTION COMPLETE - VERIFICATION")
        print("="*60)
        
        # Verify 0-0
        completed_games = games[games['status'] == 'Final']
        zeros = completed_games[(completed_games['home_team_score'] == 0) & (completed_games['visitor_team_score'] == 0)]
        
        print(f"Total Games: {len(games)}")
        print(f"Completed Games: {len(completed_games)}")
        print(f"Games with 0-0 Score: {len(zeros)}")
        
        if len(zeros) == 0:
            print("\nSUCCESS: Dataset is CLEAN. No missing scores.")
        else:
            print(f"\nFAILURE: Dataset still has {len(zeros)} missing scores.")
            
    except Exception as e:
        print(f"\nERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ingest_full_history()
