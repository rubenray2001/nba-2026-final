import requests
import config

def test_box_scores():
    url = "https://api.balldontlie.io/v1/box_scores"
    headers = {
        "Authorization": config.API_KEY
    }
    # Test for the date 2015-10-27 where we saw 0 scores
    params = {
        "date": "2015-10-27",
        "per_page": 100
    }
    
    print(f"Requesting {url} with params {params}...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            players = data.get('data', [])
            print(f"Fetched {len(players)} player records.")
            if players:
                print("First player record raw:")
                import json
                print(json.dumps(players[0], indent=2))
            
            # Group by game/team to calculate score
            games = {}
            for p in players:
                game_id = p.get('game', {}).get('id')
                team_id = p.get('team', {}).get('id')
                pts = p.get('pts') or 0
                
                if game_id not in games:
                    games[game_id] = {}
                if team_id not in games[game_id]:
                    games[game_id][team_id] = 0
                
                games[game_id][team_id] += pts
                
            print("\nCalculated Scores from Box Scores:")
            for game_id, teams in games.items():
                print(f"Game {game_id}: {teams}")
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_box_scores()
