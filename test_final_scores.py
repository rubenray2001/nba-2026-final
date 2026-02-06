"""Test the full data flow for final scores"""
from data_manager import DataManager

dm = DataManager()
games_df = dm.fetch_todays_games("2026-01-25")

print(f"Columns in games_df: {list(games_df.columns)}")
print(f"\nNumber of games: {len(games_df)}")

if not games_df.empty:
    print("\n--- First game sample ---")
    first = games_df.iloc[0]
    print(f"status: {first.get('status', 'MISSING')}")
    print(f"home_team_score: {first.get('home_team_score', 'MISSING')}")
    print(f"visitor_team_score: {first.get('visitor_team_score', 'MISSING')}")
    print(f"period: {first.get('period', 'MISSING')}")
    print(f"time: {first.get('time', 'MISSING')}")
    
    # Check all games status
    print("\n--- All games status ---")
    for _, g in games_df.iterrows():
        home = g.get('home_team_name', '?')
        away = g.get('visitor_team_name', '?')
        status = g.get('status', 'MISSING')
        hscore = g.get('home_team_score', 'MISSING')
        vscore = g.get('visitor_team_score', 'MISSING')
        print(f"{away} @ {home}: {status} | Score: {vscore}-{hscore}")
