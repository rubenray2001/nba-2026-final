import requests
import config

def test_batch():
    url = "https://api.balldontlie.io/v1/box_scores"
    headers = {
        "Authorization": config.API_KEY
    }
    dates = ["2015-10-27", "2015-10-28", "2015-10-29"]
    params = {
        "dates[]": dates,
        "per_page": 100
    }
    
    print(f"Requesting {url} with params {params}...")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            print(f"Fetched {len(data)} records.")
            
            seen_dates = set()
            for d in data:
                seen_dates.add(d.get('date', '').split('T')[0])
                
            print(f"Dates returned: {sorted(list(seen_dates))}")
            
            if len(seen_dates) > 1:
                print("SUCCESS: Endpoint supports batch dates!")
            else:
                print("FAILURE: Endpoint only returned one date (or empty).")
        else:
            print(f"Error: {resp.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_batch()
