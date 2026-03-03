"""
Betting Model - Specialized predictions for ML, Spread (Reg), Totals (Reg)
"""

import joblib
import json
import os
import numpy as np
import pandas as pd


class BettingModel:
    """Specialized model for betting predictions"""
    
    def __init__(self):
        self.ml_model = None
        self.spread_model = None
        self.totals_model = None
        self.scaler = None
        self.feature_names = []
        self.loaded = False
        self.results = {}
        self.model_type = 'classification'  # Default to old style
    
    def load(self, models_dir='models'):
        """Load betting models"""
        try:
            ml_path = os.path.join(models_dir, 'betting_moneyline.joblib')
            spread_path = os.path.join(models_dir, 'betting_spread.joblib')
            totals_path = os.path.join(models_dir, 'betting_totals.joblib')
            scaler_path = os.path.join(models_dir, 'betting_scaler.joblib')
            meta_path = os.path.join(models_dir, 'betting_metadata.json')
            
            if not all(os.path.exists(p) for p in [ml_path, spread_path, totals_path, scaler_path, meta_path]):
                print("Betting models not found. Run train_betting_model.py first.")
                return False
            
            self.ml_model = joblib.load(ml_path)
            self.spread_model = joblib.load(spread_path)
            self.totals_model = joblib.load(totals_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.results = metadata.get('results', {})
                self.model_type = metadata.get('model_type', 'classification')
            
            self.loaded = True
            print(f"Betting models loaded ({self.model_type}, {len(self.feature_names)} features)")
            return True
            
        except Exception as e:
            print(f"Error loading betting models: {e}")
            return False
    
    def predict(self, features_df):
        """
        Make betting predictions
        """
        
        if not self.loaded:
            return None
        
        # Ensure we have the right features with smart defaults
        data = {}
        for col in self.feature_names:
            if col in features_df.columns:
                data[col] = features_df[col].values
            else:
                # smart defaults
                if 'elo' in col.lower(): data[col] = [1500] * len(features_df)
                elif 'win_pct' in col.lower(): data[col] = [0.5] * len(features_df)
                elif 'points' in col.lower(): data[col] = [110] * len(features_df)
                elif 'pace' in col.lower(): data[col] = [0] * len(features_df)  # centered
                elif 'volatility' in col.lower(): data[col] = [10] * len(features_df)
                else: data[col] = [0] * len(features_df)
        
        X = pd.DataFrame(data, index=features_df.index)
        X.fillna(0, inplace=True)
        
        # Scale
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"Scaling error: {e}")
            return None
            
        results = {}
        
        # VEGAS DATA extraction (defaults if missing)
        vegas_has_odds = features_df.get('vegas_has_odds', pd.Series([0]*len(X))).fillna(0).values
        vegas_spread = features_df.get('vegas_spread_home', pd.Series([0.0]*len(X))).fillna(0.0).values
        vegas_total = features_df.get('vegas_total', pd.Series([220.0]*len(X))).fillna(220.0).values
        vegas_ml_prob = features_df.get('vegas_implied_home_prob', pd.Series([0.5]*len(X))).fillna(0.5).values
        
        # 1. MONEYLINE (Classifier) - UNTOUCHED
        # ---------------------------------------------------------------------
        try:
            raw_ml_probs = self.ml_model.predict_proba(X_scaled)[:, 1]
            
            # Blend ML with Vegas (70% Vegas, 30% Model)
            blended_ml = np.where(
                vegas_has_odds > 0,
                0.70 * vegas_ml_prob + 0.30 * raw_ml_probs,
                raw_ml_probs
            )
            results['ml_home_prob'] = blended_ml
            results['ml_pick'] = np.where(blended_ml > 0.5, 'HOME', 'AWAY')
            results['ml_confidence'] = np.maximum(blended_ml, 1 - blended_ml)
        except Exception as e:
            print(f"ML Prediction error: {e}")
            results['ml_home_prob'] = [0.5] * len(X)
            results['ml_pick'] = ['HOME'] * len(X)
            results['ml_confidence'] = [0.5] * len(X)
        
        # 2. SPREAD (Regressor)
        # ---------------------------------------------------------------------
        # Model predicts MARGIN (Home - Visitor). e.g., +8 means Home wins by 8.
        try:
            pred_margin = self.spread_model.predict(X_scaled)
            
            # Spread logic:
            # Vegas Spread is usually negative for favorites (e.g. -5.5)
            # We cover if (Actual Margin + Spread) > 0
            # So we predict cover if (Predicted Margin + Vegas Spread) > 0
            
            predicted_cover_margin = pred_margin + vegas_spread
            
            # Probability approximation (sigmoidish)
            # If cover margin is 0, prob is 50%. If +10, prob is high.
            results['spread_home_prob'] = 1 / (1 + np.exp(-0.15 * predicted_cover_margin))
            results['spread_pick'] = np.where(predicted_cover_margin > 0, 'HOME', 'AWAY')
            
            # Confidence is based on how far the margin is from 0
            # e.g. covering by 5 points is more confident than covering by 0.5
            results['spread_confidence'] = 0.50 + (np.abs(predicted_cover_margin) / 20.0)
            results['spread_confidence'] = np.clip(results['spread_confidence'], 0.5, 0.95)
            
            results['pred_spread_margin'] = pred_margin
            results['cover_margin'] = predicted_cover_margin
            
        except Exception as e:
            print(f"Spread Prediction error: {e}")
    
        # 3. TOTALS (Regressor)
        # ---------------------------------------------------------------------
        try:
            pred_total = self.totals_model.predict(X_scaled)
            
            # Total logic:
            # Diff = Predicted Total - Vegas Total
            total_diff = pred_total - vegas_total
            
            # Probability approximation
            results['over_prob'] = 1 / (1 + np.exp(-0.15 * total_diff))
            results['total_pick'] = np.where(total_diff > 0, 'OVER', 'UNDER')
            
            # Confidence based on point diff
            results['total_confidence'] = 0.50 + (np.abs(total_diff) / 20.0)
            results['total_confidence'] = np.clip(results['total_confidence'], 0.5, 0.95)
            
            results['pred_total_points'] = pred_total
            results['total_diff'] = total_diff
            
        except Exception as e:
            print(f"Totals Prediction error: {e}")
        
        return pd.DataFrame(results, index=features_df.index)
    
    def get_betting_recommendation(self, features, vegas_spread=None, vegas_total=None):
        """
        Get betting recommendation for a single game
        """
        if not self.loaded:
            return {'has_edge': False}
        
        # Convert to DataFrame
        if isinstance(features, pd.DataFrame):
            features_df = features
        elif isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.to_frame().T if hasattr(features, 'to_frame') else pd.DataFrame([features])
            
        preds = self.predict(features_df)
        if preds is None or preds.empty:
            return {'has_edge': False}
            
        row = preds.iloc[0]
        
        result = {
            'has_edge': False,
            'ml_pick': row.get('ml_pick'),
            'ml_confidence': row.get('ml_confidence', 0.5),
            'ml_is_confident': False,
            
            'spread_pick': row.get('spread_pick'),
            'spread_edge': 0.0,
            'spread_confidence': row.get('spread_confidence', 0.5),
            'spread_is_confident': False,
            
            'total_pick': row.get('total_pick'),
            'total_edge': 0.0,
            'total_confidence': row.get('total_confidence', 0.5),
            'total_is_confident': False
        }

        # Check edges (Regression Logic)
        if 'cover_margin' in row:
             result['spread_edge'] = abs(row['cover_margin'])
             
        if 'total_diff' in row:
             result['total_edge'] = abs(row['total_diff'])

        # Logic for "confident"
        # Moneyline: > 60% probability
        if result['ml_confidence'] >= 0.60:
            result['ml_is_confident'] = True
            
        # Spread: > 2.5 points edge (approx one possession)
        if result['spread_edge'] >= 2.5:
            result['spread_is_confident'] = True
            
        # Total: > 3.5 points edge
        if result['total_edge'] >= 3.5:
            result['total_is_confident'] = True
            
        if result['ml_is_confident'] or result['spread_is_confident'] or result['total_is_confident']:
            result['has_edge'] = True
        
        return result
    
    def get_accuracy_stats(self):
        """Return training accuracy stats"""
        return self.results
