import pandas as pd
import os
import config

data_dir = config.DATA_DIR
games_path = os.path.join(data_dir, "games_historical.csv")
training_path = os.path.join(data_dir, "training_data.csv")

print(f"Checking {games_path}...")
if os.path.exists(games_path):
    df = pd.read_csv(games_path, low_memory=False)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    if 'home_team_score' in df.columns:
        zeros = df[df['home_team_score'] == 0]
        print(f"Games with home_team_score == 0: {len(zeros)}")
        print(f"Unique home_team_score values: {df['home_team_score'].nunique()}")
        print(f"Value counts (top 5):\n{df['home_team_score'].value_counts().head(5)}")
    else:
        print("'home_team_score' column not found!")
else:
    print("games_historical.csv not found.")

print("\nChecking training_data.csv...")
if os.path.exists(training_path):
    df = pd.read_csv(training_path)
    print(f"Total rows: {len(df)}")
    if 'home_score' in df.columns:
        zeros = df[df['home_score'] == 0]
        print(f"Rows with home_score == 0: {len(zeros)}")
        print(f"Unique home_score values: {df['home_score'].nunique()}")
        print(f"Value counts (top 5):\n{df['home_score'].value_counts().head(5)}")
    else:
        print("'home_score' column not found!")
else:
    print("training_data.csv not found.")
