import requests
import config
import pandas as pd

def check_season(season):
    headers = {"Authorization": config.API_KEY}
    print(f"Checking season {season}...")
    
    # Get a game
    games_url = "https://api.balldontlie.io/v1/games"
    params = {"seasons[]": [season], "per_page": 5}
    resp = requests.get(games_url, headers=headers, params=params)
    games = resp.json().get('data', [])
    
    if not games:
        print(f"  No games found for {season}")
        return
        
    game = games[0]
    date = game['date'].split('T')[0]
    print(f"  Found game: {game['id']} on {date} (Score: {game['home_team_score']}-{game['visitor_team_score']})")
    
    # Check box scores
    box_url = "https://api.balldontlie.io/v1/box_scores"
    box_params = {"date": date}
    resp_box = requests.get(box_url, headers=headers, params=box_params)
    records = resp_box.json().get('data', [])
    
    print(f"  Found {len(records)} box score records for date {date}")
    
    # Check if we have stats for this game
    game_records = [r for r in records if r['game']['id'] == game['id']]
    print(f"  Records matching game ID {game['id']}: {len(game_records)}")
    
    if game_records:
        r = game_records[0]
        stats = {k: r.get(k) for k in ['pts', 'reb', 'ast', 'stl', 'blk', 'turnover', 'fgm', 'fga']}
        print(f"  Sample stats: {stats}")
        
        # Calculate team score from players
        home_id = game['home_team']['id']
        home_pts = sum(r['pts'] for r in game_records if r['team']['id'] == home_id)
        print(f"  Calculated Home Score: {home_pts}")

def main():
    seasons = [2000, 2010, 2018]
    for S in seasons:
        check_season(S)
        print("-" * 30)

if __name__ == "__main__":
    main()
