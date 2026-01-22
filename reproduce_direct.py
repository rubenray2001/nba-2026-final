import requests
import config

def test_direct():
    url = "https://api.balldontlie.io/v1/games"
    headers = {
        "Authorization": config.API_KEY
    }
    params = {
        "seasons[]": [2015],
        "per_page": 5
    }
    
    print(f"Requesting {url} with params {params}...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('data', [])
            print(f"Fetched {len(games)} games.")
            for i, game in enumerate(games):
                print(f"\nGame {i+1}:")
                print(f"  Date: {game.get('date')}")
                print(f"  Home: {game.get('home_team', {}).get('full_name')} ({game.get('home_team_score')})")
                print(f"  Visitor: {game.get('visitor_team', {}).get('full_name')} ({game.get('visitor_team_score')})")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_direct()
