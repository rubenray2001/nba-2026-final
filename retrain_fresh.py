"""
Retrain model with FRESH data - forces cache refresh
"""
from data_manager import DataManager
from features import FeatureEngineer
from model_engine_regularized import RegularizedEnsembleModel
from training_history import TrainingHistoryTracker
import config
import os

def main():
    """Main training pipeline with forced data refresh"""
    
    # Initialize components
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    model = RegularizedEnsembleModel()
    tracker = TrainingHistoryTracker()
    
    # Define training seasons
    training_seasons = config.TRAINING_SEASONS
    
    print(f"Training on seasons: {training_seasons}")
    print("\n*** FORCING FRESH DATA FETCH - NO CACHE ***\n")
    
    # Step 1: DELETE OLD CACHE to force fresh fetch
    print("\n" + "=" * 60)
    print("STEP 1: CLEARING OLD CACHE")
    print("=" * 60)
    cache_file = os.path.join(config.DATA_DIR, "games_historical.csv")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Deleted old cache: {cache_file}")
    else:
        print("No cache to delete")
    
    # Step 2: Fetch all data (will be forced to fetch from API)
    print("\n" + "=" * 60)
    print("STEP 2: FETCHING FRESH DATA FROM API")
    print("=" * 60)
    all_data = data_mgr.get_complete_training_data(training_seasons)
    
    games_count_raw = len(all_data['games'])
    print(f"\n*** FETCHED {games_count_raw} TOTAL GAMES (RAW) ***")
    
    # Filter out unplayed games (score = 0)
    if not all_data['games'].empty:
        all_data['games']['date'] = pd.to_datetime(all_data['games']['date'])
        
        # Remove games where both teams scored 0 (scheduled but not played)
        played_mask = (all_data['games']['home_team_score'] > 0) | (all_data['games']['visitor_team_score'] > 0)
        unplayed_count = (~played_mask).sum()
        all_data['games'] = all_data['games'][played_mask].copy()
        
        print(f"*** REMOVED {unplayed_count} UNPLAYED GAMES (0-0 scores) ***")
        print(f"*** ACTUAL PLAYED GAMES: {len(all_data['games'])} ***")
        
        latest_game = all_data['games']['date'].max()
        print(f"*** LATEST GAME IN DATA: {latest_game.strftime('%Y-%m-%d')} ***")
    
    # Step 3: Build features
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    training_df = feature_eng.build_training_dataset(all_data, training_seasons)
    
    if training_df.empty:
        print("ERROR: No training data available!")
        return
    
    print(f"\n*** BUILT {len(training_df)} TRAINING SAMPLES ***")
    
    # Save training data
    os.makedirs(config.DATA_DIR, exist_ok=True)
    training_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    training_df.to_csv(training_path, index=False)
    print(f"Training data saved to: {training_path}")
    
    # Step 4: Train model
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING ELITE ENSEMBLE")
    print("=" * 60)
    
    # Verify data quality
    if training_df['home_score'].nunique() <= 1:
        print("ERROR: All training targets are the same! Cannot train regression model.")
        print(f"Unique home scores: {training_df['home_score'].unique()}")
        return

    training_info = model.train(training_df, test_size=0.2)
    
    # Step 5: Save models
    model.save_models()
    
    # Step 6: Track training history
    print("\n" + "=" * 60)
    print("TRACKING TRAINING HISTORY")
    print("=" * 60)
    session = tracker.add_training_session(training_info)
    print(f"Training session logged: #{len(tracker.get_all_sessions())}")
    
    # Show improvement stats
    improvement = tracker.get_improvement_stats()
    if improvement['has_improvement']:
        print(f"\n*** DATA GROWTH: +{improvement['data_growth']} games ({improvement['data_growth_pct']:+.1f}%) ***")
        print(f"*** From {improvement['first_samples']} to {improvement['latest_samples']} total games ***")
        print(f"*** Accuracy change: {improvement['accuracy_change']:+.1%} ***")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {training_info['metrics']['winner_test_accuracy']:.1%}")
    print(f"Home Score MAE: {training_info['metrics']['home_score_test_mae']:.2f}")
    print(f"Visitor Score MAE: {training_info['metrics']['visitor_score_test_mae']:.2f}")


if __name__ == "__main__":
    import pandas as pd
    main()
