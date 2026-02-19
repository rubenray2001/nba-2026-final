"""
Betting Model - Specialized predictions for ML, Spread (Reg), Totals (Reg)
Identical architecture to NBA betting model
"""
import joblib
import json
import os
import numpy as np
import pandas as pd
import config


class BettingModel:
    """Specialized model for betting predictions"""
    
    def __init__(self, gender: str = "mens"):
        self.gender = gender
        self.gender_config = config.GENDER_CONFIG[gender]
        self.models_dir = os.path.join(config.MODELS_DIR, gender)
        self.ml_model = None
        self.spread_model = None
        self.totals_model = None
        self.scaler = None
        self.feature_names = []
        self.loaded = False
        self.results = {}
        self.model_type = 'classification'
    
    def load(self, models_dir=None):
        """Load betting models"""
        if models_dir:
            self.models_dir = models_dir
        
        try:
            ml_path = os.path.join(self.models_dir, 'betting_moneyline.joblib')
            spread_path = os.path.join(self.models_dir, 'betting_spread.joblib')
            totals_path = os.path.join(self.models_dir, 'betting_totals.joblib')
            scaler_path = os.path.join(self.models_dir, 'betting_scaler.joblib')
            meta_path = os.path.join(self.models_dir, 'betting_metadata.json')
            
            if not all(os.path.exists(p) for p in [ml_path, spread_path, totals_path, scaler_path, meta_path]):
                print(f"Betting models not found for {self.gender}. Run train_betting_model.py first.")
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
            print(f"Betting models loaded for {self.gender} ({self.model_type}, {len(self.feature_names)} features)")
            return True
            
        except Exception as e:
            print(f"Error loading betting models: {e}")
            return False
    
    def predict(self, features_df):
        """Make betting predictions"""
        if not self.loaded:
            return None
        
        avg_score = self.gender_config["avg_score"]
        avg_total = self.gender_config["avg_total"]
        
        data = {}
        for col in self.feature_names:
            if col in features_df.columns:
                data[col] = features_df[col].values
            else:
                if 'elo' in col.lower(): data[col] = [1500] * len(features_df)
                elif 'win_pct' in col.lower(): data[col] = [0.5] * len(features_df)
                elif 'points' in col.lower(): data[col] = [avg_score] * len(features_df)
                else: data[col] = [0] * len(features_df)
        
        X = pd.DataFrame(data, index=features_df.index)
        X.fillna(0, inplace=True)
        
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"Scaling error: {e}")
            return None
        
        results = {}
        
        vegas_has_odds = features_df.get('vegas_has_odds', pd.Series([0]*len(X))).fillna(0).values
        vegas_spread = features_df.get('vegas_spread_home', pd.Series([0.0]*len(X))).fillna(0.0).values
        vegas_total = features_df.get('vegas_total', pd.Series([avg_total]*len(X))).fillna(avg_total).values
        vegas_ml_prob = features_df.get('vegas_implied_home_prob', pd.Series([0.5]*len(X))).fillna(0.5).values
        
        # Moneyline
        try:
            raw_ml_probs = self.ml_model.predict_proba(X_scaled)[:, 1]
            blended_ml = np.where(vegas_has_odds > 0, 0.70 * vegas_ml_prob + 0.30 * raw_ml_probs, raw_ml_probs)
            results['ml_home_prob'] = blended_ml
            results['ml_pick'] = np.where(blended_ml > 0.5, 'HOME', 'AWAY')
            results['ml_confidence'] = np.maximum(blended_ml, 1 - blended_ml)
        except Exception as e:
            print(f"ML error: {e}")
            results['ml_home_prob'] = [0.5] * len(X)
            results['ml_pick'] = ['HOME'] * len(X)
            results['ml_confidence'] = [0.5] * len(X)
        
        # Spread
        try:
            pred_margin = self.spread_model.predict(X_scaled)
            predicted_cover = pred_margin + vegas_spread
            results['spread_home_prob'] = 1 / (1 + np.exp(-0.15 * predicted_cover))
            results['spread_pick'] = np.where(predicted_cover > 0, 'HOME', 'AWAY')
            results['spread_confidence'] = np.clip(0.50 + (np.abs(predicted_cover) / 20.0), 0.5, 0.95)
            results['pred_spread_margin'] = pred_margin
            results['cover_margin'] = predicted_cover
        except Exception as e:
            print(f"Spread error: {e}")
        
        # Totals
        try:
            pred_total = self.totals_model.predict(X_scaled)
            total_diff = pred_total - vegas_total
            results['over_prob'] = 1 / (1 + np.exp(-0.15 * total_diff))
            results['total_pick'] = np.where(total_diff > 0, 'OVER', 'UNDER')
            results['total_confidence'] = np.clip(0.50 + (np.abs(total_diff) / 20.0), 0.5, 0.95)
            results['pred_total_points'] = pred_total
            results['total_diff'] = total_diff
        except Exception as e:
            print(f"Totals error: {e}")
        
        return pd.DataFrame(results, index=features_df.index)
    
    def get_betting_recommendation(self, features, vegas_spread=None, vegas_total=None):
        """Get betting recommendation for a single game"""
        if not self.loaded:
            return {'has_edge': False}
        
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
        
        if 'cover_margin' in row:
            result['spread_edge'] = abs(row['cover_margin'])
        if 'total_diff' in row:
            result['total_edge'] = abs(row['total_diff'])
        
        if result['ml_confidence'] >= 0.60:
            result['ml_is_confident'] = True
        if result['spread_edge'] >= 2.5:
            result['spread_is_confident'] = True
        if result['total_edge'] >= 3.5:
            result['total_is_confident'] = True
        
        if result['ml_is_confident'] or result['spread_is_confident'] or result['total_is_confident']:
            result['has_edge'] = True
        
        return result
