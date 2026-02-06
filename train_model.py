"""
Training script - Elite Ensemble Model (XGBoost/CatBoost)
Uses EnhancedFeatureEngineer for CONSISTENT features between training and prediction.
"""
from data_manager import DataManager
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from model_engine import EliteEnsembleModel
from training_history import TrainingHistoryTracker
import config
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("=" * 60)
    print("ELITE MODEL TRAINING (Consistent Enhanced Features)")
    print("=" * 60)
    
    # Initialize - CRITICAL: Use EnhancedFeatureEngineer (same as app.py)
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    model = EliteEnsembleModel()
    tracker = TrainingHistoryTracker()
    
    # Load Data
    path = "data/training_data.csv"
    print("Building training dataset from full history...")
    all_data = data_mgr.get_complete_training_data(config.TRAINING_SEASONS)
    
    # --- Integration: Advanced Stats ---
    try:
        from advanced_data import AdvancedDataManager
        adv_mgr = AdvancedDataManager()
        adv_stats = adv_mgr.fetch_advanced_team_stats(config.TRAINING_SEASONS)
        if not adv_stats.empty:
            print(f"Enriching with {len(adv_stats)} advanced team records...")
            if 'team_stats' in all_data:
                all_data['team_stats'] = pd.concat([all_data['team_stats'], adv_stats], ignore_index=True)
            else:
                all_data['team_stats'] = adv_stats
    except Exception as e:
        print(f"Warning: Could not fetch advanced stats: {e}")
    # -----------------------------------

    training_df = feature_eng.build_training_dataset(all_data, config.TRAINING_SEASONS)
    
    # Save for future use
    training_df.to_csv(path, index=False)
    print(f"Saved training data to {path} ({len(training_df)} rows)")
    
    # Train with TimeSeriesSplit cross-validation
    print("\nTRAINING ELITE ENSEMBLE...")
    training_info = model.train(training_df, test_size=0.2)
    
    # Train Deep Learning Layer
    try:
        print("Training Deep Learning Layer...")
        model.train_deep_layer(training_df)
    except Exception as e:
        print(f"Deep Learning training failed: {e}")
    
    # Save
    model.save_models()
    
    # Track History
    session = tracker.add_training_session(training_info)
    print(f"Training session logged: #{len(tracker.get_all_sessions())}")
    
    print("\n" + "=" * 60)
    print("ELITE TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {training_info['metrics']['winner_test_accuracy']:.1%}")
    print(f"Home Sc MAE: {training_info['metrics']['home_score_test_mae']:.2f}")
    
if __name__ == "__main__":
    main()
