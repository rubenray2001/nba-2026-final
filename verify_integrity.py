"""Comprehensive integrity check"""
import joblib
import os
import config
import hashlib

path = os.path.join(config.MODELS_DIR, 'home_score_ensemble.pkl')

print(f"Checking: {path}")

# Check Hash
with open(path, 'rb') as f:
    file_hash = hashlib.md5(f.read()).hexdigest()
print(f"MD5 Hash: {file_hash}")

# Check Content
model = joblib.load(path)
print(f"Loaded type: {type(model)}")

if hasattr(model, 'estimators_'):
    print(f"Estimators: {len(model.estimators_)}")
    est = model.estimators_[0]
    print(f"Est 0 type: {type(est)}")
    if hasattr(est, 'n_features_in_'):
        print(f"Est 0 n_features_in_: {est.n_features_in_}")
    else:
        print("Est 0 has no n_features_in_")
else:
    print("No estimators_")
