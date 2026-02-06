"""Check hash of model files"""
import hashlib
import os
import config

print("=== Model File Hash Check ===")
path = os.path.join(config.MODELS_DIR, 'home_score_ensemble.pkl')

if os.path.exists(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print(f"File: {path}")
    print(f"MD5 Hash: {file_hash}")
else:
    print(f"File not found: {path}")
