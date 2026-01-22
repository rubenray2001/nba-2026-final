import requests
import config

def test_v2_games():
    url = "https://api.balldontlie.io/v1/games" # Try v1 first to confirm failure
    url_v2 = "https://api.balldontlie.io/v2/games" # Try v2
    
    headers = {
        "Authorization": config.API_KEY
    }
    params = {
        "seasons[]": [2015],
        "per_page": 5
    }
    
    print(f"Testing V1: {url}")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json().get('data', [])
        if data:
            print(f"V1 Score: {data[0].get('home_team_score')}-{data[0].get('visitor_team_score')}")
        else:
            print("V1: No data")
    except Exception as e:
        print(f"V1 Error: {e}")

    print(f"\nTesting V2: {url_v2}")
    try:
        resp = requests.get(url_v2, headers=headers, params=params, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            if data:
                print(f"Fetched {len(data)} games.")
                for g in data:
                    print(f"Date: {g.get('date')}, Score: {g.get('home_team_score')}-{g.get('visitor_team_score')}")
            else:
                print("V2: No data")
        else:
            print(f"V2 Error: {resp.text}")
    except Exception as e:
        print(f"V2 Exception: {e}")

if __name__ == "__main__":
    test_v2_games()
