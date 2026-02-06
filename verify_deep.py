"""Deep verification of all estimators in the ensemble"""
import joblib
import os
import config
from sklearn.ensemble import VotingRegressor, VotingClassifier

print("=== Deep Model Verification ===")
model_dir = config.MODELS_DIR

def check_ensemble(name, filename):
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {name}: File not found")
        return
        
    try:
        ensemble = joblib.load(path)
        print(f"\nType: {type(ensemble)}")
        
        if hasattr(ensemble, 'estimators_'):
            print(f"[OK] {name} loaded. Checking {len(ensemble.estimators_)} estimators:")
            
            for i, est in enumerate(ensemble.estimators_):
                est_name = est.__class__.__name__
                if hasattr(est, 'n_features_in_'):
                    n_feat = est.n_features_in_
                    status = "OK" if n_feat == 93 else "BAD"
                    print(f"   {i}. {est_name}: Expects {n_feat} features [{status}]")
                else:
                    print(f"   {i}. {est_name}: No n_features_in_ attribute")
        else:
            print(f"⚠️ {name} has no estimators_ attribute")
            
        if hasattr(ensemble, 'n_features_in_'):
             print(f"   Ensemble itself expects: {ensemble.n_features_in_}")
             
    except Exception as e:
        print(f"[ERROR] Error loading {name}: {e}")

check_ensemble("Home Score Ensemble", "home_score_ensemble.pkl")
check_ensemble("Visitor Score Ensemble", "visitor_score_ensemble.pkl")
