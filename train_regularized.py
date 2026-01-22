"""
Train the REGULARIZED model (fixes overfitting)
Use this instead of train_model.py
"""
from data_manager import DataManager
from features import FeatureEngineer
from model_engine_regularized import RegularizedEnsembleModel
import config


def main():
    """Training pipeline with regularized model"""
    
    # Initialize components
    data_mgr = DataManager()
    feature_eng = FeatureEngineer()
    model = RegularizedEnsembleModel()
    
    # Define training seasons
    training_seasons = config.TRAINING_SEASONS
    
    print(f"Training on seasons: {training_seasons}")
    
    # Step 1: Fetch all data
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING DATA")
    print("=" * 60)
    all_data = data_mgr.get_complete_training_data(training_seasons)
    
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
    print("STEP 3: TRAINING REGULARIZED ENSEMBLE")
    print("=" * 60)
    
    # Verify data quality
    if training_df['home_score'].nunique() <= 1:
        print("ERROR: All training targets are the same!")
        return
    
    training_info = model.train(training_df, test_size=0.2)
    
    # Step 4: Save models
    model.save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {training_info['metrics']['winner_test_accuracy']:.1%}")
    print(f"Overfitting Gap: {training_info['metrics']['overfit_gap']:.1%}")
    print(f"Home Score MAE: {training_info['metrics']['home_score_test_mae']:.2f}")
    print(f"Visitor Score MAE: {training_info['metrics']['visitor_score_test_mae']:.2f}")


if __name__ == "__main__":
    main()
