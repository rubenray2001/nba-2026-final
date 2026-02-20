import os
import sys
import json
from odds_api_client import TheOddsAPIClient
import config

# Force ensure API key is present
if not config.ODDS_API_KEY:
    print("Error: ODDS_API_KEY not found in config.py")
    sys.exit(1)

def test_odds_api_sources():
    print("--- Testing The Odds API Sources ---")
    client = TheOddsAPIClient(sport="basketball_ncaab", api_key=config.ODDS_API_KEY)
    
    # 1. Check US, UK, EU, AU regions to see what books are returned
    print("\nFetching events...")
    events = client.get_events()
    if not events:
        print("No events found. Is the season active?")
        return

    print(f"Found {len(events)} events. Checking first 10 for props...")
    
    regions_to_test = ["us", "eu", "uk", "au"]
    
    games_with_props = 0
    
    for i, event in enumerate(events[:10]):
        game_id = event['id']
        game_name = f"{event.get('home_team')} vs {event.get('away_team')}"
        print(f"\n[{i+1}] Checking: {game_name} (ID: {game_id})")
        
        for region in regions_to_test:
            try:
                # 1. Check H2H first (basic odds)
                url = f"{client.base_url}/sports/{client.sport}/events/{game_id}/odds"
                params = {
                    "apiKey": client.api_key,
                    "regions": region,
                    "markets": "h2h",
                    "oddsFormat": "american"
                }
                response = client.session.get(url, params=params)
                data = response.json()
                h2h_books = data.get('bookmakers', [])
                
                # 2. Check Player Props
                params["markets"] = "player_points"
                response = client.session.get(url, params=params)
                data = response.json()
                prop_books = data.get('bookmakers', [])
                
                if prop_books:
                    print(f"  > REGION {region.upper()}: Found {len(prop_books)} books with props!")
                    for b in prop_books:
                        name = b['title']
                        key = b['key']
                        print(f"    - {name} ({key})")
                        games_with_props += 1
                        
                        # Check for requested books
                        target_keys = ['fanduel', 'draftkings', 'bovada', 'williamhill_us', 'superbook', 'fanatics']
                        if key in target_keys:
                            print(f"      *** FOUND TARGET BOOK: {name} ***")
                            
                elif h2h_books:
                     print(f"  > REGION {region.upper()}: Found {len(h2h_books)} books with H2H (No props)")
                     # Check h2h for target books too
                     for b in h2h_books:
                        if b['key'] in ['fanatics', 'bovada', 'williamhill_us', 'superbook']:
                            print(f"      (Found {b['title']} for H2H)")
                else:
                    pass # limit noise
                    
            except Exception as e:
                print(f"Error querying {region}: {e}")
                
        if games_with_props >= 3:
            print("\nFound enough games with props. Stopping search.")
            break

    if games_with_props == 0:
        print("\nWARNING: No player props found for any of the checked games.")
        print("This usually means currently available games are too small or lines aren't out yet.")

if __name__ == "__main__":
    test_odds_api_sources()
