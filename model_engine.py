"""
Elite Ensemble Model Engine for College Basketball
Identical architecture to NBA model - stacking ensemble
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingClassifier,
    VotingRegressor,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, log_loss, brier_score_loss
from sklearn.inspection import permutation_importance
import joblib
import os
from datetime import datetime
import config


class EliteEnsembleModel:
    """
    Elite stacking ensemble for college basketball predictions
    Predicts: home_score, visitor_score, winner, win_probability
    Identical architecture to NBA model
    """
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.gender_config = config.GENDER_CONFIG[gender]
        self.models_dir = os.path.join(config.MODELS_DIR, gender)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.home_score_ensemble = None
        self.visitor_score_ensemble = None
        self.winner_ensemble = None
        self.scaler = StandardScaler()
        self.selector = None
        
        self.feature_names = []
        self.training_info = {}
    
    def _create_score_ensemble(self):
        """Create ensemble for score prediction (regression)"""
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
        return VotingRegressor(estimators=base_models, n_jobs=-1)
    
    def _create_winner_ensemble(self):
        """Create ensemble for winner prediction (classification)"""
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
        return VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
    
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare features and targets"""
        exclude_cols = ['game_id', 'date', 'home_team_id', 'visitor_team_id',
                       'home_score', 'visitor_score', 'home_won', 'season']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        y_home_score = df['home_score'].copy()
        y_visitor_score = df['visitor_score'].copy()
        y_winner = df['home_won'].copy()
        
        avg_score = self.gender_config["avg_score"]
        avg_total = self.gender_config["avg_total"]
        
        fill_defaults = {}
        for col in X.columns:
            if 'elo' in col.lower(): fill_defaults[col] = 1500
            elif 'win_pct' in col.lower(): fill_defaults[col] = 0.5
            elif 'points_scored' in col.lower(): fill_defaults[col] = avg_score
            elif 'points_allowed' in col.lower(): fill_defaults[col] = avg_score
            elif 'rest_days' in col.lower(): fill_defaults[col] = 2
            elif 'vegas_total' in col.lower(): fill_defaults[col] = avg_total
            elif 'vegas_implied' in col.lower(): fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower(): fill_defaults[col] = 0.5
            else: fill_defaults[col] = 0
        X.fillna(fill_defaults, inplace=True)
        
        self.feature_names = feature_cols
        return X, y_home_score, y_visitor_score, y_winner
    
    def train(self, training_df: pd.DataFrame, test_size: float = 0.2):
        """Train the elite ensemble models with TimeSeriesSplit CV"""
        print("=" * 60)
        print(f"TRAINING {self.gender.upper()} ELITE ENSEMBLE MODEL")
        print("=" * 60)
        
        X, y_home_score, y_visitor_score, y_winner = self.prepare_training_data(training_df)
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home_score[:split_idx], y_home_score[split_idx:]
        y_visitor_train, y_visitor_test = y_visitor_score[:split_idx], y_visitor_score[split_idx:]
        y_winner_train, y_winner_test = y_winner[:split_idx], y_winner[split_idx:]
        
        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        print(f"\nRunning 5-fold TimeSeriesSplit CV on {len(X_train)} samples...")
        
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
            fold_acc = accuracy_score(cv_y_val, cv_model.predict(cv_X_val_s))
            cv_scores.append(fold_acc)
            print(f"  Fold {fold+1}: Accuracy = {fold_acc:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"  CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Calibration split
        cal_split = int(len(X_train) * 0.85)
        X_train_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
        y_winner_fit, y_winner_cal = y_winner_train[:cal_split], y_winner_train[cal_split:]
        y_home_fit = y_home_train[:cal_split]
        y_visitor_fit = y_visitor_train[:cal_split]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_fit)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train score models
        print("\nTraining HOME score ensemble...")
        self.home_score_ensemble = self._create_score_ensemble()
        self.home_score_ensemble.fit(X_train_scaled, y_home_fit)
        home_pred_test = self.home_score_ensemble.predict(X_test_scaled)
        print(f"Home Score Test MAE: {mean_absolute_error(y_home_test, home_pred_test):.2f}")
        
        print("Training VISITOR score ensemble...")
        self.visitor_score_ensemble = self._create_score_ensemble()
        self.visitor_score_ensemble.fit(X_train_scaled, y_visitor_fit)
        visitor_pred_test = self.visitor_score_ensemble.predict(X_test_scaled)
        print(f"Visitor Score Test MAE: {mean_absolute_error(y_visitor_test, visitor_pred_test):.2f}")
        
        # Train winner model
        print("Training WINNER ensemble...")
        raw_winner = self._create_winner_ensemble()
        raw_winner.fit(X_train_scaled, y_winner_fit)
        
        # Calibrate
        self.winner_ensemble = CalibratedClassifierCV(
            FrozenEstimator(raw_winner), method='isotonic'
        )
        self.winner_ensemble.fit(X_cal_scaled, y_winner_cal)
        
        winner_pred_test = raw_winner.predict(X_test_scaled)
        winner_proba_test = self.winner_ensemble.predict_proba(X_test_scaled)[:, 1]
        
        # QUALITY METRICS (Brier Score & Log Loss)
        test_brier = brier_score_loss(y_winner_test, winner_proba_test)
        test_logloss = log_loss(y_winner_test, winner_proba_test)
        
        train_acc = accuracy_score(y_winner_fit, raw_winner.predict(X_train_scaled))
        test_acc = accuracy_score(y_winner_test, winner_pred_test)
        
        print(f"\nWinner Train Accuracy: {train_acc:.4f}")
        print(f"Winner Test Accuracy: {test_acc:.4f}")
        print(f"Winner Test Brier Score: {test_brier:.4f} (lower is better)")
        print(f"Winner Test Log Loss: {test_logloss:.4f}")
        print(f"Overfit Gap: {train_acc - test_acc:.4f}")
        print(f"CV Mean: {cv_mean:.4f}")
        
        # FEATURE IMPORTANCE (Permutation Importance on test set for unbiased view)
        print("\nCalculating feature importance (this may take a moment)...")
        perm_importance = permutation_importance(
            self.winner_ensemble, X_test_scaled, y_winner_test, 
            n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Sort and map to names
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        top_features = []
        for i in sorted_idx[:15]:
            top_features.append({
                'feature': self.feature_names[i],
                'importance': float(perm_importance.importances_mean[i]),
                'std': float(perm_importance.importances_std[i])
            })
        
        self.training_info = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_names),
            'gender': self.gender,
            'top_features': top_features,
            'metrics': {
                'home_score_test_mae': float(mean_absolute_error(y_home_test, home_pred_test)),
                'visitor_score_test_mae': float(mean_absolute_error(y_visitor_test, visitor_pred_test)),
                'winner_test_accuracy': float(test_acc),
                'winner_test_brier': float(test_brier),
                'winner_test_logloss': float(test_logloss),
                'winner_cv_mean_accuracy': float(cv_mean),
                'winner_cv_std': float(cv_std),
                'winner_cv_folds': [float(s) for s in cv_scores],
                'winner_train_accuracy': float(train_acc),
                'train_test_gap': float(train_acc - test_acc)
            }
        }
        
        print(f"\nTRAINING COMPLETE!")
        return self.training_info
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for games"""
        missing_cols = [c for c in self.feature_names if c not in features_df.columns]
        if missing_cols:
            for col in missing_cols:
                features_df[col] = np.nan
        
        X = features_df[self.feature_names].copy()
        
        avg_score = self.gender_config["avg_score"]
        avg_total = self.gender_config["avg_total"]
        
        fill_defaults = {}
        for col in X.columns:
            if 'elo' in col.lower(): fill_defaults[col] = 1500
            elif 'win_pct' in col.lower(): fill_defaults[col] = 0.5
            elif 'points_scored' in col.lower(): fill_defaults[col] = avg_score
            elif 'points_allowed' in col.lower(): fill_defaults[col] = avg_score
            elif 'rest_days' in col.lower(): fill_defaults[col] = 2
            elif 'vegas_total' in col.lower(): fill_defaults[col] = avg_total
            elif 'vegas_implied' in col.lower(): fill_defaults[col] = 0.5
            elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower(): fill_defaults[col] = 0.5
            else: fill_defaults[col] = 0
        X.fillna(fill_defaults, inplace=True)
        
        X_scaled = self.scaler.transform(X)
        
        raw_home_scores = self.home_score_ensemble.predict(X_scaled)
        raw_visitor_scores = self.visitor_score_ensemble.predict(X_scaled)
        raw_winner_probs = self.winner_ensemble.predict_proba(X_scaled)[:, 1]
        
        # Vegas-anchored blending
        # College BB convention: predicted_spread positive = home wins by that much
        VEGAS_PROB_WEIGHT = 0.70
        MODEL_PROB_WEIGHT = 0.30
        VEGAS_SPREAD_WEIGHT = 0.75
        MODEL_SPREAD_WEIGHT = 0.25
        
        vegas_implied = features_df.get('vegas_implied_home_prob', pd.Series([0.5] * len(features_df), index=features_df.index)).fillna(0.5)
        vegas_spread_raw = features_df.get('vegas_spread_home', pd.Series([0.0] * len(features_df), index=features_df.index)).fillna(0.0)
        vegas_total = features_df.get('vegas_total', pd.Series([avg_total] * len(features_df), index=features_df.index)).fillna(avg_total)
        vegas_has_odds = features_df.get('vegas_has_odds', pd.Series([0] * len(features_df), index=features_df.index)).fillna(0)
        
        # Convert vegas spread from betting convention (neg=home fav) to
        # model convention (pos=home wins) by negating
        vegas_spread_model = -vegas_spread_raw.values
        
        blended_probs = np.where(
            vegas_has_odds.values > 0,
            VEGAS_PROB_WEIGHT * vegas_implied.values + MODEL_PROB_WEIGHT * raw_winner_probs,
            raw_winner_probs
        )
        blended_probs = np.clip(blended_probs, 0.05, 0.95)
        
        # raw_model_spread: positive = home scores more (natural convention)
        raw_model_spread = raw_home_scores - raw_visitor_scores
        
        # Blend in consistent convention (positive = home wins)
        blended_spread = np.where(
            vegas_has_odds.values > 0,
            VEGAS_SPREAD_WEIGHT * vegas_spread_model + MODEL_SPREAD_WEIGHT * raw_model_spread,
            raw_model_spread
        )
        
        blended_total = np.where(
            vegas_has_odds.values > 0,
            0.70 * vegas_total.values + 0.30 * (raw_home_scores + raw_visitor_scores),
            raw_home_scores + raw_visitor_scores
        )
        
        # Score derivation: spread positive = home scores more
        blended_home_scores = (blended_total + blended_spread) / 2
        blended_visitor_scores = (blended_total - blended_spread) / 2
        
        predictions = pd.DataFrame({
            'predicted_home_score': blended_home_scores,
            'predicted_visitor_score': blended_visitor_scores,
            'predicted_spread': blended_spread,  # positive = home wins
            'predicted_total': blended_total,
            'home_win_probability': blended_probs,
            'visitor_win_probability': 1 - blended_probs,
            'raw_model_home_prob': raw_winner_probs,
            'raw_model_spread': raw_model_spread,
        }, index=features_df.index)
        
        return predictions
    
    def save_models(self):
        """Save all trained models"""
        print(f"\nSaving {self.gender} models...")
        
        joblib.dump(self.home_score_ensemble, os.path.join(self.models_dir, 'home_score_ensemble.pkl'))
        joblib.dump(self.visitor_score_ensemble, os.path.join(self.models_dir, 'visitor_score_ensemble.pkl'))
        joblib.dump(self.winner_ensemble, os.path.join(self.models_dir, 'winner_ensemble.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
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
        print(f"Loading {self.gender} models...")
        
        self.home_score_ensemble = joblib.load(os.path.join(self.models_dir, 'home_score_ensemble.pkl'))
        self.visitor_score_ensemble = joblib.load(os.path.join(self.models_dir, 'visitor_score_ensemble.pkl'))
        self.winner_ensemble = joblib.load(os.path.join(self.models_dir, 'winner_ensemble.pkl'))
        self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
        
        import json
        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_info = metadata['training_info']
        
        print(f"Loaded {len(self.feature_names)} features for {self.gender}")
