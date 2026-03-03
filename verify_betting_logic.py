import pandas as pd
import numpy as np
import os
from betting_model import BettingModel

def verify_betting_logic():
    print("="*60)
    print("VERIFYING BETTING MODEL INFERENCE")
    print("="*60)
    
    # 1. Load Model
    model = BettingModel()
    success = model.load()
    if not success:
        print("[FAIL] Could not load betting models")
        return
        
    print(f"[OK] Models loaded. Type: {model.model_type}")
    
    # 2. Create Mock Data
    # We need a dataframe with the features expected by the model
    # We'll just create one row with average values
    print("\nCreating mock game data...")
    mock_data = {
        # Basic stats
        'home_points_scored_last10': 115,
        'visitor_points_scored_last10': 110,
        'home_points_allowed_last10': 110,
        'visitor_points_allowed_last10': 115,
        'home_win_pct_last10': 0.7,
        'visitor_win_pct_last10': 0.3,
        'elo_diff': 100, # Home favored
        
        # New Feature: Volatility
        'home_points_std_last10': 10,
        'visitor_points_std_last10': 12,
        
        # Vegas Data (Important for regression logic)
        'vegas_spread_home': -5.5,    # Home favored by 5.5
        'vegas_total': 225.5,         # Total 225.5
        'vegas_implied_home_prob': 0.65,
        'vegas_has_odds': 1
    }
    
    # Add other required columns with defaults
    for col in model.feature_names:
        if col not in mock_data:
            mock_data[col] = 0
            
    df = pd.DataFrame([mock_data])
    
    # 3. Predict
    print("\nRunning prediction...")
    preds = model.predict(df)
    
    if preds is None or preds.empty:
        print("[FAIL] Prediction returned None/Empty")
        return
        
    row = preds.iloc[0]
    
    # 4. Inspect Results
    print("\n[PREDICTION RESULTS]:")
    print(f"Moneyline Home Prob: {row['ml_home_prob']:.1%} (Pick: {row['ml_pick']})")
    
    print("\n[SPREAD REGRESSION]:")
    if 'pred_spread_margin' in row:
        margin = row['pred_spread_margin']
        cover_margin = row.get('cover_margin', 0)
        print(f"  Predicted Margin: {margin:.2f} (Positive = Home Wins)")
        print(f"  Vegas Spread:     {mock_data['vegas_spread_home']}")
        print(f"  Cover Margin:     {cover_margin:.2f} (Pred Margin + Vega Spread)")
        print(f"  Result:           {row['spread_pick']} (Conf: {row['spread_confidence']:.1%})")
        
        # Check sanity
        if abs(margin) > 50:
            print("[WARN] Margin seems extreme!")
    else:
        print("[FAIL] 'pred_spread_margin' not found in results (Old model logic?)")
        
    print("\n[TOTALS REGRESSION]:")
    if 'pred_total_points' in row:
        total = row['pred_total_points']
        total_diff = row.get('total_diff', 0)
        print(f"  Predicted Total:  {total:.2f}")
        print(f"  Vegas Total:      {mock_data['vegas_total']}")
        print(f"  Difference:       {total_diff:.2f}")
        print(f"  Result:           {row['total_pick']} (Conf: {row['total_confidence']:.1%})")
        
        # Check sanity
        if total < 150 or total > 300:
            print("[WARN] Total seems extreme!")
    else:
        print("[FAIL] 'pred_total_points' not found in results (Old model logic?)")

    # 5. Check Recommendation Logic
    rec = model.get_betting_recommendation(df)
    print("\n[RECOMMENDATION]:")
    print(rec)

if __name__ == "__main__":
    verify_betting_logic()
