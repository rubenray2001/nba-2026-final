
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from data_manager import DataManager
from prediction_tracker import PredictionTracker

def debug_fetch():
    print("--- Debugging Past Games Fetch ---")
    
    # Simulate the logic in app.py
    target_date = datetime.now().date()
    # If user is running this "today" (Feb 12), we want to see if Feb 11 games are fetched.
    
    print(f"Simulating app run for date: {target_date}")
    
    dates_to_check = [(target_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    print(f"Dates being checked: {dates_to_check}")
    
    dm = DataManager()
    
    print("Calling dm.client.get_games...")
    past_games = dm.client.get_games(
        dates=dates_to_check,
        per_page=100
    )
    
    print(f"API returned {len(past_games)} games.")
    
    if not past_games:
        print("ERROR: No games returned!")
        return

    # Check for specific game 18447607 (OKC vs Bucks, Feb 11)
    # Note: I might have updated it manually in previous step, so it might have a result now.
    # But we want to see if it is in the FETCHED list.
    
    target_id = 18447607
    found = False
    for g in past_games:
        if g['id'] == target_id:
            found = True
            print(f"FOUND Game {target_id}: Status={g.get('status')}, Score={g.get('visitor_team_score')}-{g.get('home_team_score')}")
            break
            
    if not found:
        print(f"WARNING: Game {target_id} NOT found in fetched list!")
        # Print what WAS found for that date (Feb 11 is index 0 in dates_to_check? No, index 1 usually if today is 12)
        # 12-1 = 11.
        
        # Filter games by date 2026-02-11
        feb11_games = [g for g in past_games if g.get('date', '').startswith('2026-02-11')]
        print(f"Games found for 2026-02-11: {len(feb11_games)}")
        for g in feb11_games:
            print(f" - {g['id']}: {g['visitor_team']['abbreviation']} vs {g['home_team']['abbreviation']}")

    # Also check if prediction tracker has it
    tracker = PredictionTracker()
    if str(target_id) in tracker.predictions['games']:
        pred = tracker.predictions['games'][str(target_id)]
        print(f"Tracker status for {target_id}: Result is {pred.get('result')}")
    else:
        print(f"Game {target_id} not in tracker predictions.")

if __name__ == "__main__":
    debug_fetch()
