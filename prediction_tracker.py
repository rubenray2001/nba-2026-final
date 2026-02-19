"""
Prediction Tracker - Track actual prediction accuracy over time
Identical to NBA model
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class PredictionTracker:
    def __init__(self, gender: str = "mens", data_dir: str = "data"):
        self.gender = gender
        self.data_dir = Path(data_dir) / gender
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.json"
        self.predictions = self._load_predictions()
    
    def _load_predictions(self):
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"games": {}, "stats": {}}
        return {"games": {}, "stats": {}}
    
    def _save_predictions(self):
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
    
    def save_prediction(self, game_id, prediction_data):
        game_key = str(game_id)
        existing = self.predictions["games"].get(game_key)

        # Always update confidence and betting fields so tier tracking matches
        # the game card badges, even for games that already have results.
        if existing and existing.get("result") is not None:
            # Only update confidence + betting fields, preserve everything else
            new_conf = prediction_data.get("confidence")
            if new_conf is not None:
                existing["confidence"] = new_conf
            for field in ["betting_ml_pick", "betting_ml_conf",
                          "betting_spread_pick", "betting_spread_conf",
                          "betting_total_pick", "betting_total_conf"]:
                val = prediction_data.get(field)
                if val is not None:
                    existing[field] = val
            self._save_predictions()
            return

        new_entry = {
            "predicted_at": existing["predicted_at"] if existing else datetime.now().isoformat(),
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
            "betting_ml_pick": prediction_data.get("betting_ml_pick"),
            "betting_ml_conf": prediction_data.get("betting_ml_conf"),
            "betting_spread_pick": prediction_data.get("betting_spread_pick"),
            "betting_spread_conf": prediction_data.get("betting_spread_conf"),
            "betting_total_pick": prediction_data.get("betting_total_pick"),
            "betting_total_conf": prediction_data.get("betting_total_conf"),
            "result": None
        }
        self.predictions["games"][game_key] = new_entry
        self._save_predictions()
    
    def update_result(self, game_id, home_score, visitor_score):
        game_key = str(game_id)
        if game_key not in self.predictions["games"]:
            return
        
        pred = self.predictions["games"][game_key]
        if pred["result"] is not None:
            return
        
        actual_winner = pred["home_team"] if home_score > visitor_score else pred["visitor_team"]
        actual_spread = home_score - visitor_score
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
        
        # Betting model spread check
        betting_spread_pick = pred.get("betting_spread_pick")
        if betting_spread_pick and vegas_spread is not None:
            adjusted = actual_spread + vegas_spread
            if adjusted == 0:
                pred["result"]["betting_spread_correct"] = None
            else:
                home_covered = adjusted > 0
                pred["result"]["betting_spread_correct"] = home_covered if betting_spread_pick == "HOME" else not home_covered
        
        # Betting model total check
        betting_total_pick = pred.get("betting_total_pick")
        if betting_total_pick and vegas_total is not None:
            if actual_total == vegas_total:
                pred["result"]["betting_total_correct"] = None
            else:
                went_over = actual_total > vegas_total
                pred["result"]["betting_total_correct"] = went_over if betting_total_pick == "OVER" else not went_over
        
        # Betting model ML check
        betting_ml_pick = pred.get("betting_ml_pick")
        if betting_ml_pick:
            home_won = home_score > visitor_score
            pred["result"]["betting_ml_correct"] = home_won if betting_ml_pick == "HOME" else not home_won
        
        self._save_predictions()
    
    def get_accuracy_stats(self, days=30):
        cutoff = datetime.now() - timedelta(days=days)
        
        # Tier thresholds (must match app.py game card badges)
        LOCK_PICK_THRESH = 0.70
        HIGH_CONF_THRESH = 0.62
        GOOD_VALUE_THRESH = 0.58
        
        stats = {
            "total_predictions": 0, "completed_games": 0,
            "winner_correct": 0, "winner_accuracy": 0.0,
            # Legacy high-conf bucket (kept for backward compat)
            "high_conf_total": 0, "high_conf_correct": 0, "high_conf_accuracy": 0.0,
            # Per-tier tracking
            "lock_pick_total": 0, "lock_pick_correct": 0, "lock_pick_accuracy": 0.0,
            "high_confidence_total": 0, "high_confidence_correct": 0, "high_confidence_accuracy": 0.0,
            "good_value_total": 0, "good_value_correct": 0, "good_value_accuracy": 0.0,
            "predicted_total": 0, "predicted_correct": 0, "predicted_accuracy": 0.0,
            "avg_spread_error": 0.0, "avg_total_error": 0.0,
            "ats_correct": 0, "ats_total": 0, "ats_accuracy": 0.0,
            "betting_ml_correct": 0, "betting_ml_total": 0, "betting_ml_accuracy": 0.0,
            "betting_spread_correct": 0, "betting_spread_total": 0, "betting_spread_accuracy": 0.0,
            "betting_total_correct": 0, "betting_total_total": 0, "betting_total_accuracy": 0.0,
            "recent_picks": []
        }
        
        spread_errors = []
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
            is_correct = result.get("winner_correct", False)
            
            if is_correct:
                stats["winner_correct"] += 1
            
            conf = pred.get("confidence", 0.5)
            
            # Per-tier bucketing
            if conf >= LOCK_PICK_THRESH:
                stats["lock_pick_total"] += 1
                if is_correct:
                    stats["lock_pick_correct"] += 1
            elif conf >= HIGH_CONF_THRESH:
                stats["high_confidence_total"] += 1
                if is_correct:
                    stats["high_confidence_correct"] += 1
            elif conf >= GOOD_VALUE_THRESH:
                stats["good_value_total"] += 1
                if is_correct:
                    stats["good_value_correct"] += 1
            else:
                stats["predicted_total"] += 1
                if is_correct:
                    stats["predicted_correct"] += 1
            
            # Legacy high-conf (≥65%) for backward compat
            if conf >= 0.65:
                stats["high_conf_total"] += 1
                if is_correct:
                    stats["high_conf_correct"] += 1
            
            if result.get("spread_error") is not None:
                spread_errors.append(result["spread_error"])
            
            if result.get("betting_ml_correct") is not None:
                stats["betting_ml_total"] += 1
                if result["betting_ml_correct"]:
                    stats["betting_ml_correct"] += 1
            
            if result.get("betting_spread_correct") is not None:
                stats["betting_spread_total"] += 1
                if result["betting_spread_correct"]:
                    stats["betting_spread_correct"] += 1
            
            if result.get("betting_total_correct") is not None:
                stats["betting_total_total"] += 1
                if result["betting_total_correct"]:
                    stats["betting_total_correct"] += 1
            
            recent.append({
                "home": pred["home_team"], "visitor": pred["visitor_team"],
                "predicted": pred["predicted_winner"],
                "actual": result.get("actual_winner"),
                "correct": result.get("winner_correct", False),
                "confidence": conf, "date": pred["predicted_at"][:10]
            })
        
        if stats["completed_games"] > 0:
            stats["winner_accuracy"] = stats["winner_correct"] / stats["completed_games"]
        if stats["high_conf_total"] > 0:
            stats["high_conf_accuracy"] = stats["high_conf_correct"] / stats["high_conf_total"]
        # Per-tier accuracies
        if stats["lock_pick_total"] > 0:
            stats["lock_pick_accuracy"] = stats["lock_pick_correct"] / stats["lock_pick_total"]
        if stats["high_confidence_total"] > 0:
            stats["high_confidence_accuracy"] = stats["high_confidence_correct"] / stats["high_confidence_total"]
        if stats["good_value_total"] > 0:
            stats["good_value_accuracy"] = stats["good_value_correct"] / stats["good_value_total"]
        if stats["predicted_total"] > 0:
            stats["predicted_accuracy"] = stats["predicted_correct"] / stats["predicted_total"]
        if spread_errors:
            stats["avg_spread_error"] = sum(spread_errors) / len(spread_errors)
        if stats["betting_ml_total"] > 0:
            stats["betting_ml_accuracy"] = stats["betting_ml_correct"] / stats["betting_ml_total"]
        if stats["betting_spread_total"] > 0:
            stats["betting_spread_accuracy"] = stats["betting_spread_correct"] / stats["betting_spread_total"]
        if stats["betting_total_total"] > 0:
            stats["betting_total_accuracy"] = stats["betting_total_correct"] / stats["betting_total_total"]
        
        recent.sort(key=lambda x: x["date"], reverse=True)
        stats["recent_picks"] = recent[:10]
        
        return stats
    
    def update_pending_games(self, data_manager):
        """Identify predictions without results and update from API.
        
        Checks the prediction date and up to 2 days after to handle
        late-night predictions and weekend scheduling gaps.
        """
        pending_ids = [int(gid) for gid, pred in self.predictions["games"].items() if pred.get("result") is None]
        if not pending_ids:
            return 0
        
        print(f"Found {len(pending_ids)} pending games...")
        updated = 0
        
        games_by_date = {}
        today = datetime.now().strftime("%Y-%m-%d")
        for gid in pending_ids:
            pred = self.predictions["games"].get(str(gid))
            if pred:
                d_str = pred['predicted_at'][:10]
                d_date = datetime.strptime(d_str, "%Y-%m-%d")
                for offset in range(3):
                    d = (d_date + timedelta(days=offset)).strftime("%Y-%m-%d")
                    if d > today:
                        break
                    if d not in games_by_date:
                        games_by_date[d] = set()
                    games_by_date[d].add(gid)
        
        for date_str in sorted(games_by_date.keys()):
            gids = games_by_date[date_str]
            try:
                daily_games = data_manager.client.get_games(dates=[date_str])
                for game in daily_games:
                    if game['id'] in gids and game['status'] == 'Final':
                        h_score = game.get('home_team_score', 0)
                        v_score = game.get('visitor_team_score', 0)
                        if h_score > 0 or v_score > 0:
                            self.update_result(game['id'], h_score, v_score)
                            updated += 1
            except Exception as e:
                print(f"Error updating games for {date_str}: {e}")
        
        if updated > 0:
            self._save_predictions()
        return updated
