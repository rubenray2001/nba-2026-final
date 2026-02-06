
import os
import json
import sys
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

import config
from features import FeatureEngineer

def debug_metadata_loading():
    print("--- DEBUGGING METADATA LOADING ---")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Config BASE_DIR: {config.BASE_DIR if hasattr(config, 'BASE_DIR') else 'NOT DEFINED'}")
    print(f"Config MODELS_DIR: {config.MODELS_DIR if hasattr(config, 'MODELS_DIR') else 'NOT DEFINED'}")

    expected_path = os.path.join(config.MODELS_DIR, 'model_metadata.json')
    print(f"Expected Metadata Path: {expected_path}")
    
    if os.path.exists(expected_path):
        print("[OK] Metadata file exists.")
        try:
            with open(expected_path, 'r') as f:
                data = json.load(f)
            feature_names = data.get('feature_names', [])
            print(f"[OK] Loaded {len(feature_names)} feature names from JSON.")
        except Exception as e:
            print(f"[ERROR] Error reading JSON: {e}")
    else:
        print("[ERROR] Metadata file NOT found at expected path.")
        # Try finding it relative to script
        alt_path = os.path.join('models', 'model_metadata.json')
        print(f"Checking alternative: {alt_path} -> Exists? {os.path.exists(alt_path)}")

def debug_feature_generation():
    print("\n--- DEBUGGING FEATURE GENERATION ---")
    fe = FeatureEngineer()
    
    # Mock data
    game = {'home_team_id': 1610612737, 'visitor_team_id': 1610612738, 'date': '2025-01-01'}
    # Minimal historical data
    hist_data = {
        'games': pd.DataFrame({
            'game_id': [1, 2],
            'home_team_id': [1610612737, 1610612738],
            'visitor_team_id': [1610612738, 1610612737],
            'date': ['2024-12-31', '2024-12-30'],
            'home_team_score': [100, 90],
            'visitor_team_score': [90, 100],
            'home_elo': [1500, 1500],
            'visitor_elo': [1500, 1500]
        })
    }
    
    print("Calling build_features_for_game...")
    try:
        features = fe.build_features_for_game(game, hist_data, 2025)
        if features:
            print(f"[OK] Generated {len(features)} features.")
            if len(features) == 93:
                print("[OK] MATCHES EXPECTED COUNT (93)")
            else:
                print(f"[FAIL] MISMATCH: Expected 93, Got {len(features)}")
        else:
            print("[FAIL] Features returned None/Empty")
    except Exception as e:
        print(f"[FAIL] Exception in build_features_for_game: {e}")

if __name__ == "__main__":
    debug_metadata_loading()
    debug_feature_generation()
