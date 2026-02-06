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
# Deep Learning Integration
try:
    from models.lstm_model import NBALSTMModel
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Warning: LSTM Model not available (tensorflow/models missing)")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, log_loss, brier_score_loss
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
        self.lstm_model = None  # Deep Learning Model
        self.selector = None  # Feature selector support
        self.scaler = StandardScaler()
        
        self.feature_names = []
        self.training_info = {}
    
    def _create_score_ensemble(self):
        """Create ensemble for score prediction (regression) with proper regularization"""
        
        base_models = [
            ('hgb_1', HistGradientBoostingRegressor(
                max_iter=500, max_depth=6, learning_rate=0.05,
                max_leaf_nodes=31, l2_regularization=1.0,
                min_samples_leaf=20, random_state=42
            )),
            ('hgb_2', HistGradientBoostingRegressor(
                max_iter=500, max_depth=5, learning_rate=0.03,
                max_leaf_nodes=32, l2_regularization=5.0,
                min_samples_leaf=30, random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt',
                n_jobs=-1, random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=300, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt',
                n_jobs=-1, random_state=42
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=10,
                subsample=0.8, random_state=42
            ))
        ]
        
        ensemble = VotingRegressor(estimators=base_models, n_jobs=-1)
        return ensemble
    
    def _create_winner_ensemble(self):
        """Create ensemble for winner prediction (classification) with proper regularization"""
        
        base_models = [
            ('hgb_1', HistGradientBoostingClassifier(
                max_iter=500, max_depth=6, learning_rate=0.05,
                max_leaf_nodes=31, l2_regularization=1.0,
                min_samples_leaf=20, random_state=42
            )),
            ('hgb_2', HistGradientBoostingClassifier(
                max_iter=500, max_depth=5, learning_rate=0.03,
                max_leaf_nodes=32, l2_regularization=5.0,
                min_samples_leaf=30, random_state=42
            )),
            ('hgb_3', HistGradientBoostingClassifier(
                max_iter=300, max_depth=4, learning_rate=0.02,
                max_leaf_nodes=20, l2_regularization=3.0,
                min_samples_leaf=25, random_state=42
            ))
        ]
        
        # Soft voting for calibrated probabilities
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
        
        # Smart NaN filling: use feature-appropriate defaults instead of blanket 0
        fill_defaults = {}
        for col in X.columns:
            if 'elo' in col.lower():
                fill_defaults[col] = 1500
            elif 'win_pct' in col.lower():
                fill_defaults[col] = 0.5
            elif 'efg_pct' in col.lower():
                fill_defaults[col] = 0.54
            elif 'tov_pct' in col.lower():
                fill_defaults[col] = 0.13
            elif 'oreb_pct' in col.lower():
                fill_defaults[col] = 0.25
            elif 'ftr' in col.lower() and 'last' in col.lower():
                fill_defaults[col] = 0.20
            elif 'points_scored' in col.lower():
                fill_defaults[col] = 110
            elif 'points_allowed' in col.lower():
                fill_defaults[col] = 110
            elif 'pace' in col.lower():
                fill_defaults[col] = 98
            elif 'rest_days' in col.lower():
                fill_defaults[col] = 2
            elif 'vegas_total' in col.lower():
                fill_defaults[col] = 220.0
            elif 'vegas_implied' in col.lower():
                fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower():
                fill_defaults[col] = 0.5
            else:
                fill_defaults[col] = 0
        X.fillna(fill_defaults, inplace=True)
        
        self.feature_names = feature_cols
        
        return X, y_home_score, y_visitor_score, y_winner
    
    def train(self, training_df: pd.DataFrame, test_size: float = 0.2):
        """
        Train the elite ensemble models with TimeSeriesSplit cross-validation.
        
        Args:
            training_df: DataFrame with features and targets
            test_size: Fraction of data to use for final holdout testing
        """
        print("=" * 60)
        print("TRAINING ELITE ENSEMBLE MODEL (with CV)")
        print("=" * 60)
        
        # Prepare data
        X, y_home_score, y_visitor_score, y_winner = self.prepare_training_data(training_df)
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Feature columns: {self.feature_names[:5]}... (total {len(self.feature_names)})")
        
        # Split data (maintain temporal order for holdout)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home_score[:split_idx], y_home_score[split_idx:]
        y_visitor_train, y_visitor_test = y_visitor_score[:split_idx], y_visitor_score[split_idx:]
        y_winner_train, y_winner_test = y_winner[:split_idx], y_winner[split_idx:]
        
        # --- TimeSeriesSplit Cross-Validation on training set ---
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        print(f"\nRunning 5-fold TimeSeriesSplit CV on {len(X_train)} training samples...")
        
        for fold, (cv_train_idx, cv_val_idx) in enumerate(tscv.split(X_train)):
            cv_X_train = X_train.iloc[cv_train_idx]
            cv_X_val = X_train.iloc[cv_val_idx]
            cv_y_train = y_winner_train.iloc[cv_train_idx]
            cv_y_val = y_winner_train.iloc[cv_val_idx]
            
            cv_scaler = StandardScaler()
            cv_X_train_s = cv_scaler.fit_transform(cv_X_train)
            cv_X_val_s = cv_scaler.transform(cv_X_val)
            
            cv_model = self._create_winner_ensemble()
            cv_model.fit(cv_X_train_s, cv_y_train)
            cv_pred = cv_model.predict(cv_X_val_s)
            fold_acc = accuracy_score(cv_y_val, cv_pred)
            cv_scores.append(fold_acc)
            print(f"  Fold {fold+1}: Accuracy = {fold_acc:.4f} (train={len(cv_train_idx)}, val={len(cv_val_idx)})")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"  CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Carve out calibration set from training data (last 15% of training)
        # This avoids data leakage from using the test set for calibration
        cal_split = int(len(X_train) * 0.85)
        X_train_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
        y_winner_fit, y_winner_cal = y_winner_train[:cal_split], y_winner_train[cal_split:]
        y_home_fit = y_home_train[:cal_split]
        y_visitor_fit = y_visitor_train[:cal_split]
        
        print(f"\nSplit: {len(X_train_fit)} train, {len(X_cal)} calibration, {len(X_test)} test")
        
        # Scale features for final training
        print("Scaling features for final training...")
        X_train_scaled = self.scaler.fit_transform(X_train_fit)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train score prediction models
        print("\n" + "=" * 60)
        print("TRAINING SCORE PREDICTION ENSEMBLE (HOME)")
        print("=" * 60)
        
        self.home_score_ensemble = self._create_score_ensemble()
        self.home_score_ensemble.fit(X_train_scaled, y_home_fit)
        
        home_pred_train = self.home_score_ensemble.predict(X_train_scaled)
        home_pred_test = self.home_score_ensemble.predict(X_test_scaled)
        
        print(f"\nHome Score - Train MAE: {mean_absolute_error(y_home_fit, home_pred_train):.2f}")
        print(f"Home Score - Test MAE: {mean_absolute_error(y_home_test, home_pred_test):.2f}")
        print(f"Home Score - Train R²: {r2_score(y_home_fit, home_pred_train):.4f}")
        print(f"Home Score - Test R²: {r2_score(y_home_test, home_pred_test):.4f}")
        
        print("\n" + "=" * 60)
        print("TRAINING SCORE PREDICTION ENSEMBLE (VISITOR)")
        print("=" * 60)
        
        self.visitor_score_ensemble = self._create_score_ensemble()
        self.visitor_score_ensemble.fit(X_train_scaled, y_visitor_fit)
        
        visitor_pred_train = self.visitor_score_ensemble.predict(X_train_scaled)
        visitor_pred_test = self.visitor_score_ensemble.predict(X_test_scaled)
        
        print(f"\nVisitor Score - Train MAE: {mean_absolute_error(y_visitor_fit, visitor_pred_train):.2f}")
        print(f"Visitor Score - Test MAE: {mean_absolute_error(y_visitor_test, visitor_pred_test):.2f}")
        print(f"Visitor Score - Train R²: {r2_score(y_visitor_fit, visitor_pred_train):.4f}")
        print(f"Visitor Score - Test R²: {r2_score(y_visitor_test, visitor_pred_test):.4f}")
        
        # Train winner prediction model
        print("\n" + "=" * 60)
        print("TRAINING WINNER PREDICTION ENSEMBLE")
        print("=" * 60)
        
        raw_winner_ensemble = self._create_winner_ensemble()
        raw_winner_ensemble.fit(X_train_scaled, y_winner_fit)
        
        # Calibrate probabilities using isotonic regression on SEPARATE calibration set
        # (NOT the test set — that would be data leakage)
        # sklearn 1.8+: cv='prefit' was removed — use FrozenEstimator wrapper instead
        print("Calibrating probabilities on held-out calibration set...")
        self.winner_ensemble = CalibratedClassifierCV(
            FrozenEstimator(raw_winner_ensemble), method='isotonic'
        )
        self.winner_ensemble.fit(X_cal_scaled, y_winner_cal)
        
        winner_pred_train = raw_winner_ensemble.predict(X_train_scaled)
        winner_pred_test = raw_winner_ensemble.predict(X_test_scaled)
        
        # Calibrated probabilities
        winner_proba_train = self.winner_ensemble.predict_proba(X_train_scaled)[:, 1]
        winner_proba_test = self.winner_ensemble.predict_proba(X_test_scaled)[:, 1]
        
        # Uncalibrated for comparison
        raw_proba_test = raw_winner_ensemble.predict_proba(X_test_scaled)[:, 1]
        
        train_acc = accuracy_score(y_winner_fit, winner_pred_train)
        test_acc = accuracy_score(y_winner_test, winner_pred_test)
        
        print(f"\nWinner - Train Accuracy: {train_acc:.4f}")
        print(f"Winner - Test Accuracy: {test_acc:.4f}")
        print(f"Winner - Train/Test Gap: {train_acc - test_acc:.4f} (lower = less overfitting)")
        print(f"Winner - CV Mean Accuracy: {cv_mean:.4f}")
        print(f"Winner - Train Log Loss: {log_loss(y_winner_fit, winner_proba_train):.4f}")
        print(f"Winner - Test Log Loss (calibrated): {log_loss(y_winner_test, winner_proba_test):.4f}")
        print(f"Winner - Test Log Loss (raw): {log_loss(y_winner_test, raw_proba_test):.4f}")
        print(f"Winner - Brier Score (calibrated): {brier_score_loss(y_winner_test, winner_proba_test):.4f}")
        print(f"Winner - Brier Score (raw): {brier_score_loss(y_winner_test, raw_proba_test):.4f}")
        
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
                'winner_test_accuracy': float(test_acc),
                'winner_cv_mean_accuracy': float(cv_mean),
                'winner_cv_std': float(cv_std),
                'winner_train_accuracy': float(train_acc),
                'train_test_gap': float(train_acc - test_acc)
            }
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"Overfit gap: {train_acc - test_acc:.4f}")
        print("=" * 60)
        
        return self.training_info

    def _prepare_sequences(self, X_scaled, lookback=10):
        """Helper to create sequences for LSTM"""
        Xs = []
        # Zero-pad beginning
        # Efficient approach: Just repeat first row or use 0s for initial games
        # For prediction, we need the actual last 10 games history for each row
        # This is complex in a stateless predict() call unless we have full history.
        # For MVP: We assume X_scaled is chronologically sorted and contiguous
        
        # Simple sliding window
        padding = np.zeros((lookback, X_scaled.shape[1]))
        X_padded = np.vstack((padding, X_scaled))
        
        for i in range(len(X_scaled)):
            # slice from padded
            seq = X_padded[i : i + lookback]
            Xs.append(seq)
            
        return np.array(Xs)

    def train_deep_layer(self, training_df):
        """Train the Deep Learning component"""
        if not DL_AVAILABLE: return
        
        print("Training Deep Learning Layer...")
        X, _, _, y_winner = self.prepare_training_data(training_df)
        
        # CRITICAL: Use transform() NOT fit_transform() — scaler was already
        # fit during train(). Re-fitting here would overwrite the ensemble's scaler.
        X_scaled = self.scaler.transform(X)
        
        X_seq = self._prepare_sequences(X_scaled)
        
        # Use proper temporal split for validation (last 20%)
        val_size = max(100, int(len(X_seq) * 0.2))
        X_train_seq = X_seq[:-val_size]
        y_train_seq = y_winner.iloc[:-val_size]
        X_val_seq = X_seq[-val_size:]
        y_val_seq = y_winner.iloc[-val_size:]
        
        self.lstm_model = NBALSTMModel(input_shape=(10, X_seq.shape[2]))
        self.lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=5)
        
        # Save locally
        self.lstm_model.save(os.path.join(self.models_dir, 'nba_lstm.keras'))
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for games
        
        Args:
            features_df: DataFrame with same features as training
        
        Returns:
            DataFrame with predictions
        """
        # Prepare features
        print(f"DEBUG: Helper 'predict' called. Model expects {len(self.feature_names)} features.")
        print(f"DEBUG: Input features_df shape: {features_df.shape}")
        
        # Check intersection — add missing columns with NaN (smart defaults will fill them)
        missing_cols = [c for c in self.feature_names if c not in features_df.columns]
        if missing_cols:
            print(f"DEBUG: MISSING COLUMNS IN INPUT: {len(missing_cols)} -> {missing_cols[:5]}...")
            for col in missing_cols:
                features_df[col] = np.nan
            
        X = features_df[self.feature_names].copy()
        print(f"DEBUG: X shape after filtering: {X.shape}")
        
        # Smart NaN filling consistent with training
        fill_defaults = {}
        for col in X.columns:
            if 'elo' in col.lower():
                fill_defaults[col] = 1500
            elif 'win_pct' in col.lower():
                fill_defaults[col] = 0.5
            elif 'efg_pct' in col.lower():
                fill_defaults[col] = 0.54
            elif 'tov_pct' in col.lower():
                fill_defaults[col] = 0.13
            elif 'oreb_pct' in col.lower():
                fill_defaults[col] = 0.25
            elif 'ftr' in col.lower() and 'last' in col.lower():
                fill_defaults[col] = 0.20
            elif 'points_scored' in col.lower():
                fill_defaults[col] = 110
            elif 'points_allowed' in col.lower():
                fill_defaults[col] = 110
            elif 'pace' in col.lower():
                fill_defaults[col] = 98
            elif 'rest_days' in col.lower():
                fill_defaults[col] = 2
            elif 'vegas_total' in col.lower():
                fill_defaults[col] = 220.0
            elif 'vegas_implied' in col.lower():
                fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower():
                fill_defaults[col] = 0.5
            else:
                fill_defaults[col] = 0
        X.fillna(fill_defaults, inplace=True)
        
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
        
        # Predict winner and probabilities (Ensemble)
        winner_probs = self.winner_ensemble.predict_proba(X_final)[:, 1]
        
        # NOTE: LSTM blending is DISABLED for prediction.
        # The LSTM expects chronologically ordered sequences of the SAME team's games.
        # At prediction time, features_df contains independent games (different matchups),
        # so sliding-window sequences across them are meaningless and add noise.
        # The LSTM is only useful during training where data IS chronologically ordered.
        if DL_AVAILABLE and self.lstm_model:
            print("INFO: LSTM model loaded but skipped for prediction (sequences not applicable to independent games)")
        
        # Create predictions dataframe - preserve original index for proper game matching
        predictions = pd.DataFrame({
            'predicted_home_score': home_scores,
            'predicted_visitor_score': visitor_scores,
            'predicted_spread': home_scores - visitor_scores,
            'predicted_total': home_scores + visitor_scores,
            'home_win_probability': winner_probs,
            'visitor_win_probability': 1 - winner_probs
        }, index=features_df.index)
        
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
        
        # Load DL
        if DL_AVAILABLE:
            dl_path = os.path.join(self.models_dir, 'nba_lstm.keras')
            if os.path.exists(dl_path):
                try:
                    self.lstm_model = NBALSTMModel.load(dl_path)
                    print("Loaded LSTM model")
                except Exception as e:
                    print(f"Failed to load LSTM: {e}")
        
        # Load metadata FIRST (needed for selector validation)
        import json
        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_info = metadata['training_info']
        
        print(f"DEBUG: Loaded {len(self.feature_names)} features from metadata")
        
        # Load selector if exists (after metadata is loaded)
        selector_path = os.path.join(self.models_dir, 'selector.pkl')
        if os.path.exists(selector_path):
            try:
                self.selector = joblib.load(selector_path)
                # Validate selector compatibility
                expected_features = len(self.feature_names)
                if hasattr(self.selector, 'n_features_in_') and self.selector.n_features_in_ != expected_features:
                    print(f"DEBUG: Selector mismatch (expects {self.selector.n_features_in_}, have {expected_features})")
                    self.selector = None
                else:
                    print(f"DEBUG: Loaded feature selector (SelectKBest with k={getattr(self.selector, 'k', 'unknown')})")
            except Exception as e:
                print(f"DEBUG: Could not load selector: {e}")
                self.selector = None
        else:
            self.selector = None
        
        print(f"Models loaded from {self.models_dir}")
        print(f"Model trained at: {self.training_info.get('trained_at', 'Unknown')}")
        
        # Display correct metric based on model type
        if 'hybrid_accuracy' in self.training_info:
             print(f"Test Accuracy (Hybrid): {self.training_info['hybrid_accuracy']:.1%}")
        else:
             print(f"Test Accuracy: {self.training_info.get('metrics', {}).get('winner_test_accuracy', 0):.1%}")


# Convenience instance
model = EliteEnsembleModel()
