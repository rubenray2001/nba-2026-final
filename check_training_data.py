"""Quick check of training data feature count"""
import pandas as pd

df = pd.read_csv('data/training_data.csv')
print(f"Training data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Exclude target/meta columns
exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
           'home_score', 'visitor_score', 'home_won', 'season']
features = [c for c in df.columns if c not in exclude]
print(f"\nFeature columns: {len(features)}")
