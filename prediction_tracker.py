"""
Prediction Tracker - Track actual prediction accuracy over time
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class PredictionTracker:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.json"
        self.predictions = self._load_predictions()
    
    def _load_predictions(self):
        """Load existing predictions from file"""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load predictions file: {e}")
                return {"games": {}, "stats": {}}
        return {"games": {}, "stats": {}}
    
    def _save_predictions(self):
        """Save predictions to file"""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
    
    def save_prediction(self, game_id, prediction_data):
        """Save a prediction for a game"""
        game_key = str(game_id)
        
        # Don't overwrite if we already have a prediction for this game
        if game_key in self.predictions["games"]:
            return
        
        self.predictions["games"][game_key] = {
            "predicted_at": datetime.now().isoformat(),
            "home_team": prediction_data.get("home_team"),
            "visitor_team": prediction_data.get("visitor_team"),
            "home_prob": prediction_data.get("home_prob"),
            "visitor_prob": prediction_data.get("visitor_prob"),
            "predicted_winner": prediction_data.get("predicted_winner"),
            "predicted_home_score": prediction_data.get("predicted_home_score"),
            "predicted_visitor_score": prediction_data.get("predicted_visitor_score"),
            "predicted_spread": prediction_data.get("predicted_spread"),
            "predicted_total": prediction_data.get("predicted_total"),
            "confidence": prediction_data.get("confidence"),
            "vegas_spread": prediction_data.get("vegas_spread"),
            "vegas_total": prediction_data.get("vegas_total"),
            # NEW: Track betting model picks separately
            "betting_ml_pick": prediction_data.get("betting_ml_pick"),  # 'HOME' or 'AWAY'
            "betting_ml_conf": prediction_data.get("betting_ml_conf"),
            "betting_spread_pick": prediction_data.get("betting_spread_pick"),  # 'HOME' or 'AWAY'
            "betting_spread_conf": prediction_data.get("betting_spread_conf"),
            "betting_total_pick": prediction_data.get("betting_total_pick"),  # 'OVER' or 'UNDER'
            "betting_total_conf": prediction_data.get("betting_total_conf"),
            "result": None  # Will be filled when game completes
        }
        self._save_predictions()
    
    def update_result(self, game_id, home_score, visitor_score):
        """Update with actual result"""
        game_key = str(game_id)
        
        if game_key not in self.predictions["games"]:
            return
        
        pred = self.predictions["games"][game_key]
        if pred["result"] is not None:
            return  # Already updated
        
        actual_winner = pred["home_team"] if home_score > visitor_score else pred["visitor_team"]
        actual_spread = home_score - visitor_score  # Positive = home won by X
        actual_total = home_score + visitor_score
        
        pred["result"] = {
            "home_score": home_score,
            "visitor_score": visitor_score,
            "actual_winner": actual_winner,
            "actual_spread": actual_spread,
            "actual_total": actual_total,
            "winner_correct": pred["predicted_winner"] == actual_winner,
            "spread_error": abs(pred["predicted_spread"] - actual_spread) if pred.get("predicted_spread") else None,
            "total_error": abs(pred["predicted_total"] - actual_total) if pred.get("predicted_total") else None,
            "updated_at": datetime.now().isoformat()
        }
        
        vegas_spread = pred.get("vegas_spread")
        vegas_total = pred.get("vegas_total")
        
        # Check BETTING MODEL spread pick (if available)
        betting_spread_pick = pred.get("betting_spread_pick")
        if betting_spread_pick and vegas_spread is not None:
            # Proper ATS: Did our pick cover?
            # Vegas spread is from home perspective (negative = home favored)
            # Push (actual == spread) counts as no-bet, not a win or loss
            if actual_spread == vegas_spread:
                pred["result"]["betting_spread_correct"] = None  # Push
            else:
                home_covered = actual_spread > vegas_spread
                if betting_spread_pick == "HOME":
                    pred["result"]["betting_spread_correct"] = home_covered
                else:  # AWAY
                    pred["result"]["betting_spread_correct"] = not home_covered
        
        # Check BETTING MODEL total pick (if available)
        betting_total_pick = pred.get("betting_total_pick")
        if betting_total_pick and vegas_total is not None:
            if actual_total == vegas_total:
                pred["result"]["betting_total_correct"] = None  # Push
            else:
                went_over = actual_total > vegas_total
                if betting_total_pick == "OVER":
                    pred["result"]["betting_total_correct"] = went_over
                else:  # UNDER
                    pred["result"]["betting_total_correct"] = not went_over
        
        # Check BETTING MODEL ML pick (if available)
        betting_ml_pick = pred.get("betting_ml_pick")
        if betting_ml_pick:
            home_won = home_score > visitor_score
            if betting_ml_pick == "HOME":
                pred["result"]["betting_ml_correct"] = home_won
            else:  # AWAY
                pred["result"]["betting_ml_correct"] = not home_won
        
        # Legacy ATS check (using predicted winner, not betting model)
        if vegas_spread is not None:
            if actual_spread == vegas_spread:
                pred["result"]["ats_correct"] = None  # Push
            else:
                home_covered = actual_spread > vegas_spread
                if pred["predicted_winner"] == pred["home_team"]:
                    pred["result"]["ats_correct"] = home_covered
                else:
                    pred["result"]["ats_correct"] = not home_covered
        
        self._save_predictions()
    
    def get_accuracy_stats(self, days=30):
        """Calculate accuracy over recent days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        stats = {
            "total_predictions": 0,
            "completed_games": 0,
            "winner_correct": 0,
            "winner_accuracy": 0.0,
            "high_conf_total": 0,
            "high_conf_correct": 0,
            "high_conf_accuracy": 0.0,
            "avg_spread_error": 0.0,
            "avg_total_error": 0.0,
            "ats_correct": 0,
            "ats_total": 0,
            "ats_accuracy": 0.0,
            # NEW: Betting model specific stats
            "betting_ml_correct": 0,
            "betting_ml_total": 0,
            "betting_ml_accuracy": 0.0,
            "betting_spread_correct": 0,
            "betting_spread_total": 0,
            "betting_spread_accuracy": 0.0,
            "betting_total_correct": 0,
            "betting_total_total": 0,
            "betting_total_accuracy": 0.0,
            "recent_picks": []
        }
        
        spread_errors = []
        total_errors = []
        recent = []
        
        for game_key, pred in self.predictions["games"].items():
            try:
                pred_time = datetime.fromisoformat(pred["predicted_at"])
                if pred_time < cutoff:
                    continue
            except (ValueError, KeyError):
                continue
            
            stats["total_predictions"] += 1
            
            if pred["result"] is None:
                continue
            
            stats["completed_games"] += 1
            result = pred["result"]
            
            # Winner accuracy (base model)
            if result.get("winner_correct"):
                stats["winner_correct"] += 1
            
            # High confidence (>= 65%)
            conf = pred.get("confidence", 0.5)
            if conf >= 0.65:
                stats["high_conf_total"] += 1
                if result.get("winner_correct"):
                    stats["high_conf_correct"] += 1
            
            # Spread/Total errors (base model)
            if result.get("spread_error") is not None:
                spread_errors.append(result["spread_error"])
            if result.get("total_error") is not None:
                total_errors.append(result["total_error"])
            
            # Legacy ATS (base model)
            if result.get("ats_correct") is not None:
                stats["ats_total"] += 1
                if result["ats_correct"]:
                    stats["ats_correct"] += 1
            
            # NEW: Betting model ML accuracy
            if result.get("betting_ml_correct") is not None:
                stats["betting_ml_total"] += 1
                if result["betting_ml_correct"]:
                    stats["betting_ml_correct"] += 1
            
            # NEW: Betting model spread accuracy
            if result.get("betting_spread_correct") is not None:
                stats["betting_spread_total"] += 1
                if result["betting_spread_correct"]:
                    stats["betting_spread_correct"] += 1
            
            # NEW: Betting model total accuracy
            if result.get("betting_total_correct") is not None:
                stats["betting_total_total"] += 1
                if result["betting_total_correct"]:
                    stats["betting_total_correct"] += 1
            
            # Recent picks for display
            recent.append({
                "home": pred["home_team"],
                "visitor": pred["visitor_team"],
                "predicted": pred["predicted_winner"],
                "actual": result.get("actual_winner"),
                "correct": result.get("winner_correct", False),
                "confidence": conf,
                "date": pred["predicted_at"][:10]
            })
        
        # Calculate percentages
        if stats["completed_games"] > 0:
            stats["winner_accuracy"] = stats["winner_correct"] / stats["completed_games"]
        if stats["high_conf_total"] > 0:
            stats["high_conf_accuracy"] = stats["high_conf_correct"] / stats["high_conf_total"]
        if stats["ats_total"] > 0:
            stats["ats_accuracy"] = stats["ats_correct"] / stats["ats_total"]
        if spread_errors:
            stats["avg_spread_error"] = sum(spread_errors) / len(spread_errors)
        if total_errors:
            stats["avg_total_error"] = sum(total_errors) / len(total_errors)
        
        # NEW: Betting model accuracies
        if stats["betting_ml_total"] > 0:
            stats["betting_ml_accuracy"] = stats["betting_ml_correct"] / stats["betting_ml_total"]
        if stats["betting_spread_total"] > 0:
            stats["betting_spread_accuracy"] = stats["betting_spread_correct"] / stats["betting_spread_total"]
        if stats["betting_total_total"] > 0:
            stats["betting_total_accuracy"] = stats["betting_total_correct"] / stats["betting_total_total"]
        
        # Sort recent by date descending, take last 10
        recent.sort(key=lambda x: x["date"], reverse=True)
        stats["recent_picks"] = recent[:10]
        
        return stats
    
    def check_and_update_results(self, games_df):
        """Check completed games and update results"""
        for _, game in games_df.iterrows():
            game_id = game.get('id')
            status = game.get('status', '')
            
            if status == 'Final' and game_id:
                home_score = game.get('home_team_score', 0)
                visitor_score = game.get('visitor_team_score', 0)
                
                if home_score and visitor_score:
                    self.update_result(game_id, home_score, visitor_score)
