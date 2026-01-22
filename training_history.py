"""
Training History Tracker
Logs each training session to show model improvement over time
"""
import json
import os
from datetime import datetime
from typing import Dict, List


class TrainingHistoryTracker:
    """Track model training history to show improvement"""
    
    def __init__(self, history_file: str = "models/training_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load training history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """Save training history to file"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_training_session(self, training_info: Dict):
        """Add a new training session to history"""
        session = {
            'timestamp': datetime.now().isoformat(),
            'training_samples': training_info.get('training_samples', 0),
            'test_samples': training_info.get('test_samples', 0),
            'total_samples': training_info.get('training_samples', 0) + training_info.get('test_samples', 0),
            'features': training_info.get('features', 0),
            'test_accuracy': training_info.get('metrics', {}).get('winner_test_accuracy', 0),
            'home_score_mae': training_info.get('metrics', {}).get('home_score_test_mae', 0),
            'visitor_score_mae': training_info.get('metrics', {}).get('visitor_score_test_mae', 0),
            'overfit_gap': training_info.get('metrics', {}).get('overfit_gap', 0),
            'model_type': training_info.get('model_type', 'unknown')
        }
        
        self.history.append(session)
        self._save_history()
        
        return session
    
    def get_latest(self) -> Dict:
        """Get the most recent training session"""
        if self.history:
            return self.history[-1]
        return {}
    
    def get_improvement_stats(self) -> Dict:
        """Calculate improvement from first to latest training"""
        if len(self.history) < 2:
            return {
                'total_trainings': len(self.history),
                'has_improvement': False
            }
        
        first = self.history[0]
        latest = self.history[-1]
        
        return {
            'total_trainings': len(self.history),
            'has_improvement': True,
            'data_growth': latest['total_samples'] - first['total_samples'],
            'data_growth_pct': ((latest['total_samples'] - first['total_samples']) / first['total_samples'] * 100) if first['total_samples'] > 0 else 0,
            'accuracy_change': latest['test_accuracy'] - first['test_accuracy'],
            'accuracy_change_pct': ((latest['test_accuracy'] - first['test_accuracy']) / first['test_accuracy'] * 100) if first['test_accuracy'] > 0 else 0,
            'mae_improvement': first['home_score_mae'] - latest['home_score_mae'],  # Lower is better
            'first_trained': first['timestamp'][:10],
            'latest_trained': latest['timestamp'][:10],
            'first_samples': first['total_samples'],
            'latest_samples': latest['total_samples']
        }
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all training sessions"""
        return self.history
    
    def clear_history(self):
        """Clear all training history"""
        self.history = []
        self._save_history()


def format_improvement_display(stats: Dict) -> str:
    """Format improvement stats for display"""
    if not stats.get('has_improvement'):
        return "Train the model multiple times to see improvement tracking."
    
    lines = []
    lines.append(f"**Total Training Sessions:** {stats['total_trainings']}")
    lines.append(f"**First Trained:** {stats['first_trained']}")
    lines.append(f"**Latest Trained:** {stats['latest_trained']}")
    lines.append("")
    lines.append(f"**Data Growth:**")
    lines.append(f"   - {stats['first_samples']} -> {stats['latest_samples']} games")
    lines.append(f"   - +{stats['data_growth']} games ({stats['data_growth_pct']:+.1f}%)")
    lines.append("")
    
    if stats['accuracy_change'] >= 0:
        lines.append(f"**Accuracy Improved:** {stats['accuracy_change_pct']:+.1f}%")
    else:
        lines.append(f"**Accuracy Change:** {stats['accuracy_change_pct']:+.1f}% (minor fluctuation)")
    
    if stats['mae_improvement'] > 0:
        lines.append(f"**Score Predictions Better:** -{stats['mae_improvement']:.2f} points MAE")
    elif stats['mae_improvement'] < 0:
        lines.append(f"**Score Predictions:** {-stats['mae_improvement']:.2f} points worse")
    else:
        lines.append(f"**Score Predictions:** Stable")
    
    return "\n".join(lines)
