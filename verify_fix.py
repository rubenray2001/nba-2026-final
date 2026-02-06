
import pandas as pd
import numpy as np
from model_engine import model

def verify():
    print("Loading model...")
    model.load_models()
    
    print("Model loaded. Feature count:", len(model.feature_names))
    
    # Create a dummy feature row
    features = pd.DataFrame(np.zeros((1, len(model.feature_names))), columns=model.feature_names)
    
    # Set reasonable values
    features['elo_diff'] = 100 
    features['home_elo'] = 1600
    features['visitor_elo'] = 1500
    features['home_win_pct_last10'] = 0.8
    features['visitor_win_pct_last10'] = 0.2
    
    # Critical: Add points so model sees a good team vs bad team
    features['home_points_scored_last10'] = 118.0
    features['visitor_points_scored_last10'] = 102.0
    features['home_points_allowed_last10'] = 105.0
    features['visitor_points_allowed_last10'] = 115.0
    
    # Net Rating Diff approx
    features['net_rating_diff'] = 13.0 - (-13.0) # 26.0
    
    # Remove unicode print to avoid crash
    print("\nRunning Prediction...")
    pred = model.predict(features)
    
    print("\nPrediction Result:")
    print(pred.iloc[0])
    
    prob = pred.iloc[0]['home_win_probability']
    print(f"\nHome Win Probability: {prob:.4f}")
    
    if prob > 0.6:
        print("SUCCESS: Model correctly favors the strong home team.")
    else:
        print("WARNING: Model did not strongly favor the home team (prob < 0.6).")

if __name__ == "__main__":
    verify()
