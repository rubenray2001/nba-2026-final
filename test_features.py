import pandas as pd
from features import FeatureEngineer
import config

def test_feature_generation():
    print("Loading historical games...")
    df = pd.read_csv("data/games_historical.csv")
    
    # Use a subset for speed, but ensure we have enough for rolling windows
    # df = df.head(2000) 
    
    print(f"Loaded {len(df)} games.")
    
    # Mock 'all_data' structure
    all_data = {'games': df, 'team_stats': pd.DataFrame()}
    
    fe = FeatureEngineer()
    
    # Run pipeline for a few seasons
    seasons = [2022, 2023]
    print(f"Generating features for seasons: {seasons}...")
    
    final_df = fe.build_training_dataset(all_data, seasons)
    
    print("\n--- Verification ---")
    print(f"Output Shape: {final_df.shape}")
    print("Columns:", final_df.columns.tolist()[:10], "...")
    
    # Check for new columns
    expected_cols = ['home_elo', 'visitor_elo', 'home_efg_pct_last10', 'visitor_rest_days']
    for col in expected_cols:
        if col in final_df.columns:
            print(f"[OK] Found column: {col}")
            print(f"   Sample values: {final_df[col].head(5).tolist()}")
        else:
            print(f"[FAIL] Missing column: {col}")
            
    # Check for NaN
    nan_counts = final_df.isna().sum().sum()
    if nan_counts == 0:
        print("[OK] No NaNs found in final dataset.")
    else:
        print(f"[WARN] Found {nan_counts} NaNs in final dataset.")

if __name__ == "__main__":
    test_feature_generation()
