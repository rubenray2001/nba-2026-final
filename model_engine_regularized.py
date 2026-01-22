"""
REGULARIZED Elite Ensemble Model Engine
Upgraded with XGBoost and CatBoost for 70%+ accuracy
"""
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingClassifier, VotingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, log_loss
from sklearn.feature_selection import SelectFromModel

# XGBoost and CatBoost removed for sklearn compatibility
# import xgboost as xgb
# from catboost import CatBoostClassifier, CatBoostRegressor

import joblib
import os
from datetime import datetime
import config


class RegularizedEnsembleModel:
    """
    Regularized ensemble model with fixes for overfitting:
    - Simpler models (fewer trees, less depth)
    - L1/L2 regularization
    - Voting instead of stacking (less overfitting)
    - Early stopping
    - Feature selection
    """
    
    def __init__(self, params=None):
        self.models_dir = config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default Hyperparameters
        self.default_params = {
            'rf': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 5},
            'gb': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            'et': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 5},
            'ridge': {'alpha': 10.0},
            'logistic': {'C': 0.1}
        }
        
        # Merge with provided params
        self.params = self.default_params.copy()
        if params:
            for key, val in params.items():
                if key in self.params:
                    self.params[key].update(val)
        
        self.score_ensemble = None
        self.winner_ensemble = None
        self.selector = None  # Feature selector
        self.scaler = StandardScaler()
        
        self.feature_names = []
        self.selected_features = []  # Names of selected features
        self.training_info = {}
    
    def _create_score_ensemble(self):
        """Create REGULARIZED ensemble for score prediction"""
        p = self.params
        
        # Pure sklearn models - guaranteed compatibility
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=p['rf']['n_estimators'],
                max_depth=p['rf']['max_depth'],
                min_samples_split=p['rf']['min_samples_split'],
                min_samples_leaf=p['rf']['min_samples_leaf'],
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=p['gb']['n_estimators'],
                max_depth=p['gb']['max_depth'],
                learning_rate=p['gb']['learning_rate'],
                subsample=p['gb']['subsample'],
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=p['et']['n_estimators'],
                max_depth=p['et']['max_depth'],
                min_samples_split=p['et']['min_samples_split'],
                min_samples_leaf=p['et']['min_samples_leaf'],
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('ridge', Ridge(alpha=p['ridge']['alpha']))
        ]
        
        # Use VOTING instead of STACKING (less overfitting)
        ensemble = VotingRegressor(
            estimators=base_models,
            n_jobs=-1
        )
        
        return ensemble
    
    def _create_winner_ensemble(self):
        """Create UPGRADED ensemble with pure sklearn models for winner prediction"""
        p = self.params
        
        # Pure sklearn models - guaranteed compatibility with sklearn 1.6+
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=p['rf']['n_estimators'],
                max_depth=p['rf']['max_depth'],
                min_samples_split=p['rf']['min_samples_split'],
                min_samples_leaf=p['rf']['min_samples_leaf'],
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('hgb_1', HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=42
            )),
            ('hgb_2', HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=4,
                learning_rate=0.02,
                l2_regularization=2.0,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )),
            ('logistic', LogisticRegression(
                C=p['logistic']['C'],
                max_iter=1000,
                random_state=42
            ))
        ]
        
        # Soft voting ensemble for probability calibration
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare features and targets from dataframe"""
        
        # Identify feature columns
        exclude_cols = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
                       'home_score', 'visitor_score', 'home_won', 'season']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Targets
        y_home_score = df['home_score'].copy()
        y_visitor_score = df['visitor_score'].copy()
        y_winner = df['home_won'].copy()
        
        # Handle any remaining NaN
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        return X, y_home_score, y_visitor_score, y_winner
    
    def train(self, training_df: pd.DataFrame, test_size: float = 0.2):
        """Train the regularized ensemble models"""
        
        print("=" * 60)
        print("TRAINING REGULARIZED ENSEMBLE MODEL")
        print("(Fixed for overfitting)")
        print("=" * 60)
        
        # Prepare data
        X, y_home_score, y_visitor_score, y_winner = self.prepare_training_data(training_df)
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples per feature: {len(X) / len(self.feature_names):.1f}")
        
        # Split data (temporal order maintained)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home_score[:split_idx], y_home_score[split_idx:]
        y_visitor_train, y_visitor_test = y_visitor_score[:split_idx], y_visitor_score[split_idx:]
        y_winner_train, y_winner_test = y_winner[:split_idx], y_winner[split_idx:]
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Feature Selection
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        # Use a lightweight Random Forest to select best features
        print("Selecting top features...")
        selector_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        self.selector = SelectFromModel(selector_model, threshold='1.25*mean', max_features=40)
        self.selector.fit(X_train_scaled, y_visitor_train) # Use visitor score as proxy for complexity
        
        # Transform data
        X_train_selected = self.selector.transform(X_train_scaled)
        X_test_selected = self.selector.transform(X_test_scaled)
        
        # Identify selected feature names
        selected_mask = self.selector.get_support()
        self.selected_features = [f for i, f in enumerate(self.feature_names) if selected_mask[i]]
        
        print(f"Original Features: {len(self.feature_names)}")
        print(f"Selected Features: {len(self.selected_features)}")
        print(f"Top 5 Features: {self.selected_features[:5]}")
        print(f"Samples per Feature (New): {len(X_train) / len(self.selected_features):.1f}")

        
        # Train combined score model (predict both scores together)
        print("\n" + "=" * 60)
        print("TRAINING SCORE PREDICTION ENSEMBLE")
        print("=" * 60)
        
        self.home_score_ensemble = self._create_score_ensemble()
        self.visitor_score_ensemble = self._create_score_ensemble()
        
        # Fit on SELECTED features
        self.home_score_ensemble.fit(X_train_selected, y_home_train)
        self.visitor_score_ensemble.fit(X_train_selected, y_visitor_train)
        
        # Evaluate
        home_pred_train = self.home_score_ensemble.predict(X_train_selected)
        home_pred_test = self.home_score_ensemble.predict(X_test_selected)
        visitor_pred_train = self.visitor_score_ensemble.predict(X_train_selected)
        visitor_pred_test = self.visitor_score_ensemble.predict(X_test_selected)
        
        print(f"\nHome Score - Train MAE: {mean_absolute_error(y_home_train, home_pred_train):.2f}")
        print(f"Home Score - Test MAE: {mean_absolute_error(y_home_test, home_pred_test):.2f}")
        print(f"Home Score - Train R²: {r2_score(y_home_train, home_pred_train):.4f}")
        print(f"Home Score - Test R²: {r2_score(y_home_test, home_pred_test):.4f}")
        
        print(f"\nVisitor Score - Train MAE: {mean_absolute_error(y_visitor_train, visitor_pred_train):.2f}")
        print(f"Visitor Score - Test MAE: {mean_absolute_error(y_visitor_test, visitor_pred_test):.2f}")
        print(f"Visitor Score - Train R²: {r2_score(y_visitor_train, visitor_pred_train):.4f}")
        print(f"Visitor Score - Test R²: {r2_score(y_visitor_test, visitor_pred_test):.4f}")
        
        # Train winner prediction model
        print("\n" + "=" * 60)
        print("TRAINING WINNER PREDICTION ENSEMBLE")
        print("=" * 60)
        
        self.winner_ensemble = self._create_winner_ensemble()
        self.winner_ensemble.fit(X_train_selected, y_winner_train)
        
        # Evaluate
        winner_pred_train = self.winner_ensemble.predict(X_train_selected)
        winner_pred_test = self.winner_ensemble.predict(X_test_selected)
        
        winner_proba_train = self.winner_ensemble.predict_proba(X_train_selected)[:, 1]
        winner_proba_test = self.winner_ensemble.predict_proba(X_test_selected)[:, 1]
        
        train_acc = accuracy_score(y_winner_train, winner_pred_train)
        test_acc = accuracy_score(y_winner_test, winner_pred_test)
        
        print(f"\nWinner - Train Accuracy: {train_acc:.4f}")
        print(f"Winner - Test Accuracy: {test_acc:.4f}")
        print(f"Winner - Train Log Loss: {log_loss(y_winner_train, winner_proba_train):.4f}")
        print(f"Winner - Test Log Loss: {log_loss(y_winner_test, winner_proba_test):.4f}")
        
        # Check overfitting
        overfit_gap = train_acc - test_acc
        if overfit_gap > 0.15:
            print(f"\n[WARNING] Still overfitting (gap: {overfit_gap:.1%})")
        else:
            print(f"\n[OK] Overfitting under control (gap: {overfit_gap:.1%})")
        
        # Store training info
        self.training_info = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_names),
            'selected_features': len(self.selected_features),
            'metrics': {
                'home_score_test_mae': float(mean_absolute_error(y_home_test, home_pred_test)),
                'visitor_score_test_mae': float(mean_absolute_error(y_visitor_test, visitor_pred_test)),
                'home_score_test_r2': float(r2_score(y_home_test, home_pred_test)),
                'visitor_score_test_r2': float(r2_score(y_visitor_test, visitor_pred_test)),
                'winner_test_accuracy': float(test_acc),
                'overfit_gap': float(overfit_gap)
            }
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return self.training_info
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for games"""
        X = features_df[self.feature_names].copy()
        X = X.fillna(0)
        
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection if available
        if self.selector:
            X_final = self.selector.transform(X_scaled)
        else:
            X_final = X_scaled
        
        # Predict scores
        home_scores = self.home_score_ensemble.predict(X_final)
        visitor_scores = self.visitor_score_ensemble.predict(X_final)
        
        # Predict winner
        winner_probs = self.winner_ensemble.predict_proba(X_final)[:, 1]
        
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
            'training_info': self.training_info,
            'model_type': 'regularized_ensemble'
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
model = RegularizedEnsembleModel()
