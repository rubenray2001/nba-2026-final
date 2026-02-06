"""Quick test to see what fields the API returns for games"""
from api_client import client
import json

games = client.get_games(dates=["2026-01-25"])
print(f"Got {len(games)} games")

if games:
    print("\nFirst game keys:")
    print(list(games[0].keys()))
    
    print("\nFirst game data:")
    print(json.dumps(games[0], indent=2, default=str))
