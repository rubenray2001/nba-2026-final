"""
Betting Model - Specialized predictions for ML, Spread, Totals
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
            
            self.loaded = True
            print(f"Betting models loaded ({len(self.feature_names)} features)")
            return True
            
        except Exception as e:
            print(f"Error loading betting models: {e}")
            return False
    
    def predict(self, features_df):
        """
        Make betting predictions
        
        Returns dict with:
        - ml_home_prob: Probability home wins (moneyline)
        - spread_home_prob: Probability home covers spread
        - over_prob: Probability game goes over
        """
        
        if not self.loaded:
            return None
        
        # Ensure we have the right features with smart defaults
        data = {}
        for col in self.feature_names:
            if col in features_df.columns:
                data[col] = features_df[col].values
            else:
                # Use feature-appropriate defaults instead of blanket 0
                if 'elo' in col.lower():
                    data[col] = [1500] * len(features_df)
                elif 'win_pct' in col.lower():
                    data[col] = [0.5] * len(features_df)
                elif 'efg_pct' in col.lower():
                    data[col] = [0.54] * len(features_df)
                elif 'tov_pct' in col.lower():
                    data[col] = [0.13] * len(features_df)
                elif 'oreb_pct' in col.lower():
                    data[col] = [0.25] * len(features_df)
                elif 'ftr' in col.lower() and 'last' in col.lower():
                    data[col] = [0.20] * len(features_df)
                elif 'points_scored' in col.lower():
                    data[col] = [110] * len(features_df)
                elif 'points_allowed' in col.lower():
                    data[col] = [110] * len(features_df)
                elif 'pace' in col.lower():
                    data[col] = [98] * len(features_df)
                elif 'rest_days' in col.lower():
                    data[col] = [2] * len(features_df)
                elif 'vegas_total' in col.lower():
                    data[col] = [220.0] * len(features_df)
                elif 'vegas_implied' in col.lower():
                    data[col] = [0.5] * len(features_df)
                elif 'h2h_home_win_pct' in col.lower() or 'h2h_last3' in col.lower():
                    data[col] = [0.5] * len(features_df)
                else:
                    data[col] = [0] * len(features_df)
        
        X = pd.DataFrame(data, index=features_df.index)
        # Fill any remaining NaN with same smart logic
        fill_defaults = {}
        for col in X.columns:
            if 'elo' in col.lower(): fill_defaults[col] = 1500
            elif 'win_pct' in col.lower(): fill_defaults[col] = 0.5
            elif 'vegas_total' in col.lower(): fill_defaults[col] = 220.0
            elif 'vegas_implied' in col.lower(): fill_defaults[col] = 0.5
            else: fill_defaults[col] = 0
        X.fillna(fill_defaults, inplace=True)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Raw model predictions
        results = {}
        
        # =====================================================================
        # VEGAS-ANCHORED BLENDING for betting model
        # Same rationale as model_engine.py: model was trained on zero Vegas,
        # so we blend raw model output with Vegas implied probabilities.
        # =====================================================================
        VEGAS_WEIGHT = 0.70
        MODEL_WEIGHT = 0.30
        
        # Extract Vegas features from input
        vegas_implied = features_df.get('vegas_implied_home_prob', pd.Series([0.5] * len(features_df), index=features_df.index)).fillna(0.5).values
        vegas_has_odds = features_df.get('vegas_has_odds', pd.Series([0] * len(features_df), index=features_df.index)).fillna(0).values
        vegas_spread = features_df.get('vegas_spread_home', pd.Series([0.0] * len(features_df), index=features_df.index)).fillna(0.0).values
        vegas_total_val = features_df.get('vegas_total', pd.Series([220.0] * len(features_df), index=features_df.index)).fillna(220.0).values
        
        # Moneyline: blend with Vegas implied prob
        raw_ml_probs = self.ml_model.predict_proba(X_scaled)[:, 1]
        blended_ml = np.where(
            vegas_has_odds > 0,
            VEGAS_WEIGHT * vegas_implied + MODEL_WEIGHT * raw_ml_probs,
            raw_ml_probs
        )
        blended_ml = np.clip(blended_ml, 0.05, 0.95)
        
        results['ml_home_prob'] = blended_ml
        results['ml_pick'] = np.where(blended_ml > 0.5, 'HOME', 'AWAY')
        results['ml_confidence'] = np.maximum(blended_ml, 1 - blended_ml)
        
        # Spread: The Vegas spread line is SET so both sides have ~50% cover probability.
        # Don't convert spread to cover probability (0.5 + spread*0.03 is WRONG for ATS —
        # that formula estimates WIN probability, not cover probability).
        # Instead, anchor at 50% and let the model's signal adjust.
        raw_spread_probs = self.spread_model.predict_proba(X_scaled)[:, 1]
        
        # Anchor at 50% (the spread IS the 50/50 point), shrink model toward it
        # This gives: blended = 0.5 + 0.55 * (raw - 0.5) when Vegas odds exist
        blended_spread = np.where(
            vegas_has_odds > 0,
            0.45 * 0.5 + 0.55 * raw_spread_probs,
            raw_spread_probs
        )
        blended_spread = np.clip(blended_spread, 0.05, 0.95)
        
        results['spread_home_prob'] = blended_spread
        results['spread_pick'] = np.where(blended_spread > 0.5, 'HOME', 'AWAY')
        results['spread_confidence'] = np.maximum(blended_spread, 1 - blended_spread)
        
        # Totals: blend model with naive Vegas total anchor (50/50 baseline)
        raw_total_probs = self.totals_model.predict_proba(X_scaled)[:, 1]
        # Model output alone for totals (Vegas total is already in the features,
        # and there's no clean implied over/under probability from the line)
        results['over_prob'] = raw_total_probs
        results['total_pick'] = np.where(raw_total_probs > 0.5, 'OVER', 'UNDER')
        results['total_confidence'] = np.maximum(raw_total_probs, 1 - raw_total_probs)
        
        return pd.DataFrame(results, index=features_df.index)
    
    def get_betting_recommendation(self, features, vegas_spread=None, vegas_total=None):
        """
        Get betting recommendation for a single game
        
        Returns dict with picks and confidence
        """
        
        if not self.loaded:
            return {
                'ml_pick': None,
                'spread_pick': None,
                'total_pick': None,
                'has_edge': False
            }
        
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.to_frame().T if hasattr(features, 'to_frame') else pd.DataFrame([features])
        
        preds = self.predict(features_df)
        
        if preds is None or preds.empty:
            return {
                'ml_pick': None,
                'spread_pick': None, 
                'total_pick': None,
                'has_edge': False
            }
        
        row = preds.iloc[0]
        
        result = {
            # Moneyline
            'ml_pick': row['ml_pick'],
            'ml_home_prob': row['ml_home_prob'],
            'ml_confidence': row['ml_confidence'],
            'ml_is_confident': row['ml_confidence'] >= 0.60,
            
            # Spread
            'spread_pick': row['spread_pick'],
            'spread_home_prob': row['spread_home_prob'],
            'spread_confidence': row['spread_confidence'],
            'spread_is_confident': row['spread_confidence'] >= 0.55,
            
            # Totals
            'total_pick': row['total_pick'],
            'over_prob': row['over_prob'],
            'total_confidence': row['total_confidence'],
            'total_is_confident': row['total_confidence'] >= 0.55,
            
            # Has any edge?
            'has_edge': (
                row['ml_confidence'] >= 0.60 or
                row['spread_confidence'] >= 0.55 or
                row['total_confidence'] >= 0.55
            )
        }
        
        return result
    
    def get_accuracy_stats(self):
        """Return training accuracy stats"""
        return self.results
