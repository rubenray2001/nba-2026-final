
import pandas as pd
import json
import os
import config
from features import FeatureEngineer
from data_manager import DataManager

def test_feature_generation():
    print("Testing Feature Generation...")
    
    # 1. Load Metadata
    metadata_path = os.path.join('models', 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        expected_features = metadata.get('feature_names', [])
        print(f"Model expects {len(expected_features)} features.")
    else:
        print("ERROR: model_metadata.json not found!")
        expected_features = []

    # 2. Mock Game Data
    game = {
        'home_team_id': 1610612737, # Hawks
        'visitor_team_id': 1610612738, # Celtics
        'date': '2025-01-25'
    }
    
    # 3. Load Historical Data (just games)
    print("Loading historical data...")
    dm = DataManager()
    # We need some dummy historical data or real data
    # Let's try to fetch a small amount if cache exists, or just check 'games_historical.csv'
    # For speed, we might fail here if no data exists.
    # We'll assume the user has data since they trained.
    historical_data = dm.get_complete_training_data([2024, 2025])
    
    if 'games' not in historical_data or historical_data['games'].empty:
        print("No historical games found. Cannot test.")
        return

    # 4. Generate Features
    print("Generating features...")
    fe = FeatureEngineer()
    features = fe.build_features_for_game(game, historical_data, 2025)
    
    if features:
        print(f"Generated {len(features)} features.")
        
        # Check missing
        missing = [f for f in expected_features if f not in features]
        if missing:
            print(f"MISSING {len(missing)} FEATURES: {missing}")
        else:
            print("SUCCESS: All expected features present.")
            
        # Check extras
        extra = [f for f in features.keys() if f not in expected_features]
        # print(f"Extra features: {len(extra)}")
        
        # Verify DataFrame creation
        df = pd.DataFrame([features])
        # Reorder to match expectation
        if expected_features:
            df_final = df[expected_features]
            print(f"Final DataFrame Shape: {df_final.shape}")
    else:
        print("Failed to generate features.")

if __name__ == "__main__":
    test_feature_generation()
