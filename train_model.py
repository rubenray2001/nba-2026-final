"""
Training script - Train the elite ensemble model
"""
from data_manager import DataManager
from features import FeatureEngineer
# Use regularized model to avoid sklearn version issues and overfitting
from model_engine_regularized import RegularizedEnsembleModel
from training_history import TrainingHistoryTracker
import config


def main():
    """Main training pipeline"""
    
    # Initialize components
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    import json
    import os
    
    # Load optimal hyperparameters if available
    param_path = os.path.join(config.MODELS_DIR, 'best_hyperparameters.json')
    params = None
    if os.path.exists(param_path):
        print(f"Loading tuned hyperparameters from {param_path}")
        with open(param_path, 'r') as f:
            params = json.load(f)
            
    model = RegularizedEnsembleModel(params=params)
    tracker = TrainingHistoryTracker()
    
    # Define training seasons
    training_seasons = config.TRAINING_SEASONS
    
    print(f"Training on seasons: {training_seasons}")
    
    # Step 1: Load data (using existing cache from ingestion)
    print("\n" + "=" * 60)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 60)
    
    # NOTE: Cache clearing removed to preserve the full ingestion we just ran.
    # We want to use the high-quality, backfilled data.
    
    all_data = data_mgr.get_complete_training_data(training_seasons)
    
    # Show latest game date
    if not all_data['games'].empty:
        import pandas as pd
        all_data['games']['date'] = pd.to_datetime(all_data['games']['date'])
        latest_game = all_data['games']['date'].max()
        print(f"\n*** LATEST GAME IN DATA: {latest_game.strftime('%Y-%m-%d')} ***")
        print(f"*** TOTAL GAMES: {len(all_data['games'])} ***")
    
    # Step 2: Build features
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    training_df = feature_eng.build_training_dataset(all_data, training_seasons)
    
    if training_df.empty:
        print("ERROR: No training data available!")
        return
    
    # Save training data
    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)
    training_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    training_df.to_csv(training_path, index=False)
    print(f"\nTraining data saved to: {training_path}")
    
    # Step 3: Train model
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING ELITE ENSEMBLE")
    print("=" * 60)
    # Verify data quality
    if training_df['home_score'].nunique() <= 1:
        print("ERROR: All training targets are the same! Cannot train regression model.")
        print(f"Unique home scores: {training_df['home_score'].unique()}")
        return

    training_info = model.train(training_df, test_size=0.2)
    
    # Step 4: Save models
    model.save_models()
    
    # Step 5: Track training history
    print("\n" + "=" * 60)
    print("TRACKING TRAINING HISTORY")
    print("=" * 60)
    session = tracker.add_training_session(training_info)
    print(f"Training session logged: #{len(tracker.get_all_sessions())}")
    
    # Show improvement stats
    improvement = tracker.get_improvement_stats()
    if improvement['has_improvement']:
        print(f"\nData Growth: +{improvement['data_growth']} games ({improvement['data_growth_pct']:+.1f}%)")
        print(f"From {improvement['first_samples']} to {improvement['latest_samples']} total games")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {training_info['metrics']['winner_test_accuracy']:.1%}")
    print(f"Home Score MAE: {training_info['metrics']['home_score_test_mae']:.2f}")
    print(f"Visitor Score MAE: {training_info['metrics']['visitor_score_test_mae']:.2f}")


if __name__ == "__main__":
    main()
