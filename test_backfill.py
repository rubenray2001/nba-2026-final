import requests
import config

def verify_score_aggregation():
    headers = {"Authorization": config.API_KEY}
    
    # 1. Fetch one game from 2015
    print("Fetching one game from 2015...")
    games_url = "https://api.balldontlie.io/v1/games"
    params = {"seasons[]": [2015], "per_page": 1}
    
    try:
        resp = requests.get(games_url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        games = resp.json().get('data', [])
        
        if not games:
            print("No games found.")
            return
            
        game = games[0]
        game_id = game['id']
        date = game['date']
        print(f"Game ID: {game_id}, Date: {date}")
        print(f"Reported Scores - Home: {game['home_team_score']}, Visitor: {game['visitor_team_score']}")
        
        # 2. Fetch box scores for this game using DATE with different endpoints
        print(f"\nFetching box scores for Date {date}...")
        
        endpoints = [
            "https://api.balldontlie.io/v1/box_scores",
            "https://api.balldontlie.io/nba/v1/box_scores", # Attempt nba/v1
             # "https://api.balldontlie.io/v2/box_scores" # Attempt v2 (common pattern)
        ]
        
        success = False
        valid_records = []
        
        for box_url in endpoints:
            print(f"Testing {box_url}...")
            # v1 requires date
            box_params = {"date": date.split("T")[0]}
            
            try:
                resp_box = requests.get(box_url, headers=headers, params=box_params, timeout=10)
                if resp_box.status_code != 200:
                    print(f"  Failed: {resp_box.status_code}")
                    continue
                    
                records = resp_box.json().get('data', [])
                if not records:
                    print("  No records found.")
                    continue
                    
                first = records[0]
                print(f"  First record keys: {list(first.keys())}")
                if 'home_team_score' in first and 'visitor_team_score' in first:
                    print(f"  SUCCESS: Found score keys!")
                    print(f"  First game score: {first['home_team_score']} - {first['visitor_team_score']}")
                    
                    if first['home_team_score'] > 0:
                        print("  SUCCESS: Scores are NON-ZERO!")
                        success = True
                        break
                    else:
                        print("  FAILURE: Scores are ZERO.")
                    
            except Exception as e:
                print(f"  Error: {e}")
                
        if not success:
            print("Failed to find valid endpoint.")
            return

        # Filter for this game id
        records = [r for r in valid_records if r.get('game', {}).get('id') == game_id]

        print(f"Fetched {len(records)} player stats for this game.")
        
        # 3. Sum points
        home_team_id = game['home_team']['id']
        visitor_team_id = game['visitor_team']['id']
        
        home_pts = 0
        visitor_pts = 0
        
        for r in records:
            pts = r.get('pts') or 0
            # Check which team the player belongs to
            # The record has 'team' -> 'id'
            r_team_id = r.get('team', {}).get('id')
            
            if r_team_id == home_team_id:
                home_pts += pts
            elif r_team_id == visitor_team_id:
                visitor_pts += pts
            else:
                pass
                # print(f"Warning: Player team ID {r_team_id} matches neither home {home_team_id} nor visitor {visitor_team_id}")
        
        print("\n--- Aggregation Results ---")
        print(f"Home Team ({game['home_team']['full_name']}) Sum: {home_pts}")
        print(f"Visitor Team ({game['visitor_team']['full_name']}) Sum: {visitor_pts}")
        
        if home_pts > 0 or visitor_pts > 0:
            print("SUCCESS: Calculated scores are non-zero.")
        else:
            print("FAILURE: Calculated scores are still zero (or data missing).")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_score_aggregation()
