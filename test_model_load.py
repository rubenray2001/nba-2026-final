import sys
import os
import traceback

print("1. Starting test_model_load.py...")

try:
    from model_engine import EliteEnsembleModel
    print("2. Imported EliteEnsembleModel successfully.")
except Exception:
    print("FAILED to import model_engine:")
    traceback.print_exc()
    sys.exit(1)

try:
    print("3. Initializing model for 'mens'...")
    model = EliteEnsembleModel("mens")
    print("4. Model initialized.")
    
    print("5. Attempting to load models...")
    model.load_models()
    print("6. Models loaded successfully!")
    
    if hasattr(model, 'feature_names'):
        print(f"7. Feature names count: {len(model.feature_names)}")

except Exception:
    print("FAILED during model loading:")
    traceback.print_exc()
    sys.exit(1)

print("8. Test complete. No crash.")
