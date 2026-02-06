
import os
import sys
from api_client import BallDontLieClient
from datetime import datetime

# Initialize
client = BallDontLieClient()
target_date = "2026-01-26"

print(f"--- DEBUGGING VEGAS ODDS FOR {target_date} ---")

# 1. Check Games first (to ensure games exist)
print("\n[1] Fetching Games...")
try:
    games = client.get_games(dates=[target_date])
    print(f"Games found: {len(games)}")
    if games:
        print(f"First game ID: {games[0]['id']} - {games[0]['home_team']['abbreviation']} vs {games[0]['visitor_team']['abbreviation']}")
        game_ids = [g['id'] for g in games]
    else:
        print("CRITICAL: No games found for this date. Odds usually require games.")
        game_ids = []
except Exception as e:
    print(f"Error fetching games: {e}")
    game_ids = []

# 2. Check Odds
print("\n[2] Fetching Odds (Balldontlie v2)...")
try:
    # Try fetching by date
    odds_by_date = client.get_betting_odds(dates=[target_date])
    print(f"Odds by Date count: {len(odds_by_date)}")
    if odds_by_date:
        print("Sample Record:", odds_by_date[0])
    
    # Try fetching by game_ids if we have them
    if game_ids:
        print(f"\n[3] Fetching Odds by Game IDs ({len(game_ids)} games)...")
        odds_by_ids = client.get_betting_odds(game_ids=game_ids)
        print(f"Odds by ID count: {len(odds_by_ids)}")
        if odds_by_ids:
            print("Sample Record:", odds_by_ids[0])
            
except Exception as e:
    print(f"Error fetching odds: {e}")

print("\n--- DIAGNOSIS COMPLETE ---")
