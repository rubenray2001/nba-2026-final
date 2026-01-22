from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv2
import pandas as pd
import time

def test_fetch_scores():
    print("Testing nba_api for historical data...")
    
    # 1. Get games from 2015-16 season
    print("Fetching 2015-16 Game Log...")
    game_log = leaguegamelog.LeagueGameLog(season='2015-16', season_type_all_star='Regular Season')
    games_df = game_log.get_data_frames()[0]
    
    if games_df.empty:
        print("No games found.")
        return

    print(f"Found {len(games_df)} games.")
    
    # Pick the first game
    game = games_df.iloc[0]
    game_id = game['GAME_ID']
    matchup = game['MATCHUP']
    date = game['GAME_DATE']
    
    print(f"\nChecking Game: {matchup} ({date}) [ID: {game_id}]")
    print(f"Scores from Game Log: {game['PTS']} pts (for one team)")
    
    # 2. Fetch Box Score for this game
    print(f"Fetching Box Score for {game_id}...")
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box.player_stats.get_data_frame()
        team_stats = box.team_stats.get_data_frame()
        
        print("\n--- Team Stats (Official Box Score) ---")
        print(team_stats[['TEAM_NAME', 'PTS']])
        
        home_score = team_stats.iloc[0]['PTS']
        visitor_score = team_stats.iloc[1]['PTS']
        
        if home_score > 0 and visitor_score > 0:
            print("\nSUCCESS: Found valid non-zero scores using nba_api!")
        else:
            print("\nFAILURE: Scores are zero.")
            
    except Exception as e:
        print(f"Error fetching box score: {e}")

if __name__ == "__main__":
    test_fetch_scores()
