"""
Training script - Elite Ensemble Model (XGBoost/CatBoost)
Testing if more complex models perform better on the larger dataset (30k games)
"""
from data_manager import DataManager
from features import FeatureEngineer
from model_engine import EliteEnsembleModel
from training_history import TrainingHistoryTracker
import config
import pandas as pd

def main():
    print("ELITE MODEL TRAINING (XGBoost/CatBoost)")
    
    # Initialize
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    model = EliteEnsembleModel()
    
    # Load Data
    # Always rebuild to ensure we use the full 29k dataset
    path = "data/training_data.csv"
    print("Building training dataset from full history...")
    all_data = data_mgr.get_complete_training_data(config.TRAINING_SEASONS)
    training_df = feature_eng.build_training_dataset(all_data, config.TRAINING_SEASONS)
    
    # Save for future use
    training_df.to_csv(path, index=False)
    print(f"Saved training data to {path} ({len(training_df)} rows)")
    
    # Train
    training_info = model.train(training_df, test_size=0.2)
    
    # Save
    model.save_models()
    
    print("\n" + "=" * 60)
    print("ELITE TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {training_info['metrics']['winner_test_accuracy']:.1%}")
    print(f"Home Sc MAE: {training_info['metrics']['home_score_test_mae']:.2f}")
    
if __name__ == "__main__":
    main()
