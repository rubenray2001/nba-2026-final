"""Verify the actual feature counts in saved model files"""
import joblib
import json
import os
import config

print("=== Verifying Model Files ===")
print(f"Models dir: {config.MODELS_DIR}")

# Check metadata
metadata_path = os.path.join(config.MODELS_DIR, 'model_metadata.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"\nMetadata says: {len(metadata['feature_names'])} features")
print(f"Trained at: {metadata['training_info']['trained_at']}")

# Check actual pkl files
for pkl_name in ['home_score_ensemble.pkl', 'visitor_score_ensemble.pkl', 'scaler.pkl']:
    pkl_path = os.path.join(config.MODELS_DIR, pkl_name)
    if os.path.exists(pkl_path):
        obj = joblib.load(pkl_path)
        
        # For VotingRegressor, check the first estimator
        if hasattr(obj, 'estimators_'):
            first_est = obj.estimators_[0]
            if hasattr(first_est, 'n_features_in_'):
                print(f"{pkl_name}: {first_est.n_features_in_} features (from first estimator)")
            else:
                print(f"{pkl_name}: estimator loaded but no n_features_in_")
        elif hasattr(obj, 'n_features_in_'):
            print(f"{pkl_name}: {obj.n_features_in_} features")
        else:
            print(f"{pkl_name}: loaded (no direct feature count)")
    else:
        print(f"{pkl_name}: NOT FOUND")

# File timestamps
import stat
from datetime import datetime

print("\n=== File Timestamps ===")
for f in os.listdir(config.MODELS_DIR):
    path = os.path.join(config.MODELS_DIR, f)
    if os.path.isfile(path):
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{f}: {mtime_str}")

# Actual Prediction Check
print("\n=== Functionality Check ===")
try:
    from model_engine import model
    model.load_models()
    
    # Check Metadata metrics
    metrics = model.training_info.get('metrics', {}) # fallback
    if 'hybrid_accuracy' in model.training_info:
         metrics = model.training_info # usage from train_elite
         print(f"Verified Hybrid Accuracy: {metrics['hybrid_accuracy']:.2%}")
    elif 'winner_test_accuracy' in metrics:
         print(f"Verified Test Accuracy: {metrics['winner_test_accuracy']:.2%}")
    
    # Dummy Prediction
    import pandas as pd
    import numpy as np
    
    dummy_features = pd.DataFrame(np.zeros((1, len(model.feature_names))), columns=model.feature_names)
    pred = model.predict(dummy_features)
    print("\nDummy Prediction Successful:")
    print(pred.iloc[0][['home_win_probability', 'predicted_spread']])
    
except Exception as e:
    print(f"ERROR LOAODING/PREDICTING: {e}")
