"""
Training History Tracker
Logs each training session to show model improvement over time
Identical to NBA model
"""
import json
import os
from datetime import datetime
from typing import Dict, List


class TrainingHistoryTracker:
    def __init__(self, gender: str = "mens"):
        self.history_file = os.path.join("models", gender, "training_history.json")
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_training_session(self, training_info: Dict):
        session = {
            'timestamp': datetime.now().isoformat(),
            'training_samples': training_info.get('training_samples', 0),
            'test_samples': training_info.get('test_samples', 0),
            'total_samples': training_info.get('training_samples', 0) + training_info.get('test_samples', 0),
            'features': training_info.get('features', 0),
            'test_accuracy': training_info.get('metrics', {}).get('winner_test_accuracy', 0),
            'home_score_mae': training_info.get('metrics', {}).get('home_score_test_mae', 0),
            'visitor_score_mae': training_info.get('metrics', {}).get('visitor_score_test_mae', 0),
            'overfit_gap': training_info.get('metrics', {}).get('train_test_gap', 0),
        }
        self.history.append(session)
        self._save_history()
        return session
    
    def get_latest(self) -> Dict:
        return self.history[-1] if self.history else {}
    
    def get_improvement_stats(self) -> Dict:
        if len(self.history) < 2:
            return {'total_trainings': len(self.history), 'has_improvement': False}
        
        first = self.history[0]
        latest = self.history[-1]
        return {
            'total_trainings': len(self.history),
            'has_improvement': True,
            'data_growth': latest['total_samples'] - first['total_samples'],
            'data_growth_pct': ((latest['total_samples'] - first['total_samples']) / max(first['total_samples'], 1) * 100),
            'accuracy_change': latest['test_accuracy'] - first['test_accuracy'],
            'accuracy_change_pct': ((latest['test_accuracy'] - first['test_accuracy']) / max(first['test_accuracy'], 0.01) * 100),
            'mae_improvement': first['home_score_mae'] - latest['home_score_mae'],
            'first_trained': first['timestamp'][:10],
            'latest_trained': latest['timestamp'][:10],
            'first_samples': first['total_samples'],
            'latest_samples': latest['total_samples']
        }
    
    def get_all_sessions(self) -> List[Dict]:
        return self.history


def format_improvement_display(stats: Dict) -> str:
    if not stats.get('has_improvement'):
        return "Train the model multiple times to see improvement tracking."
    
    lines = [
        f"**Total Sessions:** {stats['total_trainings']}",
        f"**First:** {stats['first_trained']}  |  **Latest:** {stats['latest_trained']}",
        f"**Data:** {stats['first_samples']} -> {stats['latest_samples']} (+{stats['data_growth']})",
    ]
    if stats['accuracy_change'] >= 0:
        lines.append(f"**Accuracy:** {stats['accuracy_change_pct']:+.1f}%")
    else:
        lines.append(f"**Accuracy:** {stats['accuracy_change_pct']:+.1f}% (fluctuation)")
    
    return "\n".join(lines)
