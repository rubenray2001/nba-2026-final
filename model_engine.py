"""
Elite Ensemble Model Engine
6-model stacking ensemble with hyperparameter tuning
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, log_loss
import joblib
import os
from datetime import datetime
import config


class EliteEnsembleModel:
    """
    Elite stacking ensemble for NBA game predictions
    Predicts: home_score, visitor_score, winner, win_probability
    """
    
    def __init__(self):
        self.models_dir = config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Separate ensembles for score and classification
        self.score_ensemble = None
        self.winner_ensemble = None
        self.selector = None  # Feature selector support
        self.scaler = StandardScaler()
        
        self.feature_names = []
        self.training_info = {}
    
    def _create_score_ensemble(self):
        """Create ensemble for score prediction (regression) using pure sklearn"""
        
        # Pure sklearn models - guaranteed compatibility with sklearn 1.6+
        base_models = [
            ('hgb_1', HistGradientBoostingRegressor(
                max_iter=500, max_depth=8, learning_rate=0.05,
                max_leaf_nodes=31, random_state=42
            )),
            ('hgb_2', HistGradientBoostingRegressor(
                max_iter=500, max_depth=10, learning_rate=0.03,
                max_leaf_nodes=64, l2_regularization=5.0, random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                n_jobs=-1, random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                n_jobs=-1, random_state=42
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=42
            ))
        ]
        
        # VotingRegressor
        ensemble = VotingRegressor(estimators=base_models, n_jobs=-1)
        return ensemble
    
    def _create_winner_ensemble(self):
        """Create ensemble for winner prediction (classification) using pure sklearn"""
        
        base_models = [
            ('hgb_1', HistGradientBoostingClassifier(
                max_iter=500, max_depth=8, learning_rate=0.05,
                max_leaf_nodes=31, random_state=42
            )),
            ('hgb_2', HistGradientBoostingClassifier(
                max_iter=500, max_depth=10, learning_rate=0.03,
                max_leaf_nodes=64, l2_regularization=5.0, random_state=42
            )),
            ('hgb_3', HistGradientBoostingClassifier(
                max_iter=300, max_depth=5, learning_rate=0.02,
                max_leaf_nodes=20, random_state=42
            ))
        ]
        
        # Soft voting for probabilities
        ensemble = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
        return ensemble
    
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare features and targets from dataframe"""
        
        # Identify feature columns (exclude metadata and targets)
        exclude_cols = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
                       'home_score', 'visitor_score', 'home_won', 'season']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Targets
        y_home_score = df['home_score'].copy()
        y_visitor_score = df['visitor_score'].copy()
        y_winner = df['home_won'].copy()
        
        # Handle any remaining NaN
        X.fillna(0, inplace=True)
        
        self.feature_names = feature_cols
        
        return X, y_home_score, y_visitor_score, y_winner
    
    def train(self, training_df: pd.DataFrame, test_size: float = 0.2):
        """
        Train the elite ensemble models
        
        Args:
            training_df: DataFrame with features and targets
            test_size: Fraction of data to use for testing
        """
        print("=" * 60)
        print("TRAINING ELITE ENSEMBLE MODEL")
        print("=" * 60)
        
        # Prepare data
        X, y_home_score, y_visitor_score, y_winner = self.prepare_training_data(training_df)
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        
        # Split data (maintain temporal order)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home_score[:split_idx], y_home_score[split_idx:]
        y_visitor_train, y_visitor_test = y_visitor_score[:split_idx], y_visitor_score[split_idx:]
        y_winner_train, y_winner_test = y_winner[:split_idx], y_winner[split_idx:]
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train score prediction models
        print("\n" + "=" * 60)
        print("TRAINING SCORE PREDICTION ENSEMBLE (HOME)")
        print("=" * 60)
        
        self.home_score_ensemble = self._create_score_ensemble()
        self.home_score_ensemble.fit(X_train_scaled, y_home_train)
        
        # Evaluate home score predictions
        home_pred_train = self.home_score_ensemble.predict(X_train_scaled)
        home_pred_test = self.home_score_ensemble.predict(X_test_scaled)
        
        print(f"\nHome Score - Train MAE: {mean_absolute_error(y_home_train, home_pred_train):.2f}")
        print(f"Home Score - Test MAE: {mean_absolute_error(y_home_test, home_pred_test):.2f}")
        print(f"Home Score - Train R²: {r2_score(y_home_train, home_pred_train):.4f}")
        print(f"Home Score - Test R²: {r2_score(y_home_test, home_pred_test):.4f}")
        
        print("\n" + "=" * 60)
        print("TRAINING SCORE PREDICTION ENSEMBLE (VISITOR)")
        print("=" * 60)
        
        self.visitor_score_ensemble = self._create_score_ensemble()
        self.visitor_score_ensemble.fit(X_train_scaled, y_visitor_train)
        
        # Evaluate visitor score predictions
        visitor_pred_train = self.visitor_score_ensemble.predict(X_train_scaled)
        visitor_pred_test = self.visitor_score_ensemble.predict(X_test_scaled)
        
        print(f"\nVisitor Score - Train MAE: {mean_absolute_error(y_visitor_train, visitor_pred_train):.2f}")
        print(f"Visitor Score - Test MAE: {mean_absolute_error(y_visitor_test, visitor_pred_test):.2f}")
        print(f"Visitor Score - Train R²: {r2_score(y_visitor_train, visitor_pred_train):.4f}")
        print(f"Visitor Score - Test R²: {r2_score(y_visitor_test, visitor_pred_test):.4f}")
        
        # Train winner prediction model
        print("\n" + "=" * 60)
        print("TRAINING WINNER PREDICTION ENSEMBLE")
        print("=" * 60)
        
        self.winner_ensemble = self._create_winner_ensemble()
        self.winner_ensemble.fit(X_train_scaled, y_winner_train)
        
        # Evaluate winner predictions
        winner_pred_train = self.winner_ensemble.predict(X_train_scaled)
        winner_pred_test = self.winner_ensemble.predict(X_test_scaled)
        
        winner_proba_train = self.winner_ensemble.predict_proba(X_train_scaled)[:, 1]
        winner_proba_test = self.winner_ensemble.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nWinner - Train Accuracy: {accuracy_score(y_winner_train, winner_pred_train):.4f}")
        print(f"Winner - Test Accuracy: {accuracy_score(y_winner_test, winner_pred_test):.4f}")
        print(f"Winner - Train Log Loss: {log_loss(y_winner_train, winner_proba_train):.4f}")
        print(f"Winner - Test Log Loss: {log_loss(y_winner_test, winner_proba_test):.4f}")
        
        # Store training info
        self.training_info = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_names),
            'metrics': {
                'home_score_test_mae': float(mean_absolute_error(y_home_test, home_pred_test)),
                'visitor_score_test_mae': float(mean_absolute_error(y_visitor_test, visitor_pred_test)),
                'home_score_test_r2': float(r2_score(y_home_test, home_pred_test)),
                'visitor_score_test_r2': float(r2_score(y_visitor_test, visitor_pred_test)),
                'winner_test_accuracy': float(accuracy_score(y_winner_test, winner_pred_test))
            }
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return self.training_info
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for games
        
        Args:
            features_df: DataFrame with same features as training
        
        Returns:
            DataFrame with predictions
        """
        # Prepare features
        X = features_df[self.feature_names].copy()
        X.fillna(0, inplace=True)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection if available
        if self.selector:
            X_final = self.selector.transform(X_scaled)
        else:
            X_final = X_scaled
        
        # Predict scores
        home_scores = self.home_score_ensemble.predict(X_final)
        visitor_scores = self.visitor_score_ensemble.predict(X_final)
        
        # Predict winner and probabilities
        winner_probs = self.winner_ensemble.predict_proba(X_final)[:, 1]  # Probability home wins
        
        # Create predictions dataframe
        predictions = pd.DataFrame({
            'predicted_home_score': home_scores,
            'predicted_visitor_score': visitor_scores,
            'predicted_spread': home_scores - visitor_scores,
            'predicted_total': home_scores + visitor_scores,
            'home_win_probability': winner_probs,
            'visitor_win_probability': 1 - winner_probs
        })
        
        return predictions
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        joblib.dump(self.home_score_ensemble, os.path.join(self.models_dir, 'home_score_ensemble.pkl'))
        joblib.dump(self.visitor_score_ensemble, os.path.join(self.models_dir, 'visitor_score_ensemble.pkl'))
        joblib.dump(self.winner_ensemble, os.path.join(self.models_dir, 'winner_ensemble.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        if self.selector:
            joblib.dump(self.selector, os.path.join(self.models_dir, 'selector.pkl'))
        
        # Save metadata
        import json
        metadata = {
            'feature_names': self.feature_names,
            'training_info': self.training_info
        }
        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {self.models_dir}")
    
    def load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        self.home_score_ensemble = joblib.load(os.path.join(self.models_dir, 'home_score_ensemble.pkl'))
        self.visitor_score_ensemble = joblib.load(os.path.join(self.models_dir, 'visitor_score_ensemble.pkl'))
        self.winner_ensemble = joblib.load(os.path.join(self.models_dir, 'winner_ensemble.pkl'))
        self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
        
        # Load selector if exists
        selector_path = os.path.join(self.models_dir, 'selector.pkl')
        if os.path.exists(selector_path):
            self.selector = joblib.load(selector_path)
        
        # Load metadata
        import json
        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_info = metadata['training_info']
        
        print(f"Models loaded from {self.models_dir}")
        print(f"Model trained at: {self.training_info.get('trained_at', 'Unknown')}")
        print(f"Test Accuracy: {self.training_info.get('metrics', {}).get('winner_test_accuracy', 0):.1%}")


# Convenience instance
model = EliteEnsembleModel()
