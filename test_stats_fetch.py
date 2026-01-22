from nba_api.stats.endpoints import leaguegamelog
import pandas as pd

# Fetch one season to check columns
print("Fetching 2023-24 Season stats...")
log = leaguegamelog.LeagueGameLog(season='2023-24', season_type_all_star='Regular Season')
df = log.get_data_frames()[0]

print("Columns found:")
print(df.columns.tolist())

# Check for specific Four Factor components
required = ['FGM', 'FGA', 'FG3M', 'FTA', 'OREB', 'DREB', 'TOV', 'PTS']
available = [c for c in required if c in df.columns]

print(f"\nMissing columns: {set(required) - set(available)}")

if len(available) == len(required):
    print("\nSUCCESS: All required columns are available!")
    print(df[required].head())
else:
    print("\nFAILURE: Missing critical columns.")
