"""
Training script - Elite Ensemble Model for College Basketball
Trains both Men's and Women's models
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from data_manager import DataManager
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from model_engine import EliteEnsembleModel
from training_history import TrainingHistoryTracker
import config


def train_gender(gender: str):
    """Train model for a specific gender"""
    print("=" * 60)
    print(f"TRAINING {gender.upper()} COLLEGE BASKETBALL MODEL")
    print("=" * 60)
    
    data_mgr = DataManager(gender)
    feature_eng = FeatureEngineer(gender)
    model = EliteEnsembleModel(gender)
    tracker = TrainingHistoryTracker(gender)
    
    # Load Data
    data_dir = os.path.join(config.DATA_DIR, gender)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "training_data.csv")
    
    print("Building training dataset from historical games...")
    all_data = data_mgr.get_complete_training_data(config.TRAINING_SEASONS)
    
    if all_data['games'].empty:
        print(f"No games found for {gender}. Skipping.")
        return
    
    training_df = feature_eng.build_training_dataset(all_data, config.TRAINING_SEASONS)
    
    if training_df.empty:
        print(f"No training data generated for {gender}. Skipping.")
        return
    
    # Save for future use
    training_df.to_csv(path, index=False)
    print(f"Saved training data to {path} ({len(training_df)} rows)")
    
    # Train
    print(f"\nTRAINING {gender.upper()} ELITE ENSEMBLE...")
    training_info = model.train(training_df, test_size=0.2)
    
    # Save
    model.save_models()
    
    # Track History
    session = tracker.add_training_session(training_info)
    print(f"Training session logged: #{len(tracker.get_all_sessions())}")
    
    print(f"\n{gender.upper()} TRAINING COMPLETE!")
    metrics = training_info['metrics']
    print(f"Test Accuracy: {metrics['winner_test_accuracy']:.1%}")
    print(f"Brier Score:   {metrics['winner_test_brier']:.4f} (lower is better)")
    print(f"Home Score MAE: {metrics['home_score_test_mae']:.2f}")
    
    if 'top_features' in training_info:
        print("\nTOP 5 MOST IMPORTANT FEATURES:")
        for i, feat in enumerate(training_info['top_features'][:5]):
            print(f"  {i+1}. {feat['feature']:30s} {feat['importance']:.4f}")


def main():
    """Train both Men's and Women's models"""
    # Check which gender to train (or both)
    if len(sys.argv) > 1:
        gender = sys.argv[1].lower()
        if gender in ['mens', 'womens']:
            train_gender(gender)
        else:
            print(f"Unknown gender: {gender}. Use 'mens' or 'womens'.")
    else:
        # Train both
        for gender in ['mens', 'womens']:
            train_gender(gender)
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
