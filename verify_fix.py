from data_manager import data_manager
import pandas as pd

def test_fix():
    print("Testing NBA API Backfill Fix on Season 2015...")
    
    # Force refresh implies fetching from BDL then attempting backfill
    # This ensures we exercise the new code path
    df = data_manager.fetch_historical_games(seasons=[2015], force_refresh=True)
    
    print(f"Total games fetched: {len(df)}")
    
    if df.empty:
        print("Error: No games fetched.")
        return

    # Check for missing scores
    missing = df[(df['status'] == 'Final') & (df['home_team_score'] == 0) & (df['visitor_team_score'] == 0)]
    missing_count = len(missing)
    
    print(f"Remaining 0-0 valid games: {missing_count}")
    
    if missing_count == 0:
        print("SUCCESS! All games have scores.")
        
        # Spot check our favorite 2015 game: CLE @ CHI, 2015-10-27
        # Note: BDL Dates are ISO strings
        date_mask = df['date'].astype(str).str.contains("2015-10-27")
        cle_mask = df['home_team_abbreviation'] == 'CHI' # CHI was home in the boxscore printed earlier? Or CLE?
        # In test_nba_api output: MATCHUP: CLE @ CHI. Usually @ means Home. So CHI is Home.
        # Let's check matching
        
        spot_check = df[date_mask]
        if not spot_check.empty:
            print("\nSpot Check (2015-10-27):")
            print(spot_check[['date', 'home_team_name', 'home_team_score', 'visitor_team_name', 'visitor_team_score']])
        else:
            print("Warning: Could not find 2015-10-27 game for spot check.")
            
    else:
        print("FAILURE: Some games still have 0-0 scores.")
        print(missing[['date', 'home_team_name', 'visitor_team_name']].head())

if __name__ == "__main__":
    test_fix()
