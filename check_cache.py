import pandas as pd

df = pd.read_csv('data/games_historical.csv')
df['date'] = pd.to_datetime(df['date'])

print(f'Total games: {len(df)}')
print(f'Earliest game: {df["date"].min()}')
print(f'Latest game: {df["date"].max()}')
print(f'\nGames from 2026-01-16 onward: {len(df[df["date"] >= "2026-01-16"])}')
print(f'Games from 2026-01-17 onward: {len(df[df["date"] >= "2026-01-17"])}')

# Show most recent games
print('\nMost recent 10 games:')
recent = df.nlargest(10, 'date')[['date', 'home_team_name', 'visitor_team_name', 'home_team_score', 'visitor_team_score']]
print(recent.to_string(index=False))
