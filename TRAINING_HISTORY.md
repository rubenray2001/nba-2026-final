# Training History Tracker - Feature Documentation

## What Was Added

A new **Training History Tracker** system that logs every training session and shows model improvement over time.

## New Files

### 1. `training_history.py`
- **TrainingHistoryTracker** class
- Logs each training session to `models/training_history.json`
- Tracks: samples, accuracy, MAE, overfitting gap
- Calculates improvement metrics

## Modified Files

### 1. `train_model.py`
- Added `TrainingHistoryTracker` integration
- Logs session after each training
- Displays improvement stats in terminal

### 2. `app.py` (Sidebar)
- New "Training History" section
- Shows improvement stats
- Line chart of data growth
- Expandable for details

## What You'll See in the UI

### Training History Section (Sidebar)

```
📈 Training History

**Total Training Sessions:** 3
**First Trained:** 2026-01-15
**Latest Trained:** 2026-01-16

**Data Growth:**
   - 3235 -> 3323 games
   - +88 games (+2.7%)

**Accuracy Improved:** +0.8%
**Score Predictions Better:** -0.12 points MAE

[📊 View Training Chart] (expandable)
```

### Training Chart
- Line chart showing total games over time
- Each point = one training session
- Shows the model learning from more data

## How It Works

1. **First Training**
   - Creates `models/training_history.json`
   - Logs: timestamp, samples, accuracy, etc.

2. **Subsequent Trainings**
   - Appends new session to history
   - Calculates improvement from first training
   - Shows data growth, accuracy changes

3. **UI Display**
   - Loads history on app start
   - Shows improvement summary
   - Optional chart view

## Example Training Session Data

```json
{
  "timestamp": "2026-01-16T04:12:35.194742",
  "training_samples": 2588,
  "test_samples": 647,
  "total_samples": 3235,
  "features": 72,
  "test_accuracy": 0.662,
  "home_score_mae": 9.76,
  "visitor_score_mae": 9.03,
  "overfit_gap": 0.086,
  "model_type": "regularized_ensemble"
}
```

## Key Metrics Tracked

| Metric | What It Shows |
|--------|---------------|
| **Total Samples** | How many games the model learned from |
| **Test Accuracy** | % of correct winner predictions |
| **Home/Visitor MAE** | Average point prediction error |
| **Overfit Gap** | Train-test accuracy difference |

## Proving the Model Gets Smarter

### Example Progression

```
Training #1: 3,235 games → 66.2% accuracy
Training #2: 3,279 games → 66.4% accuracy (+0.2%)
Training #3: 3,323 games → 66.7% accuracy (+0.5%)
```

**Evidence:**
- ✅ Data grows each training (+88 games in example)
- ✅ Accuracy trend improves
- ✅ Predictions get more accurate (lower MAE)
- ✅ Learns from newest games

## Benefits

1. **Transparency**: See exactly how much data the model has
2. **Confidence**: Watch accuracy improve over time
3. **Timing**: Know when to retrain (weekly optimal)
4. **Trust**: Proof the model is learning, not guessing

## Usage

### View History
- Look in sidebar under "Training History"
- Click "View Training Chart" for graph

### Clear History (if needed)
```python
from training_history import TrainingHistoryTracker
tracker = TrainingHistoryTracker()
tracker.clear_history()
```

### Manual Query
```python
from training_history import TrainingHistoryTracker
tracker = TrainingHistoryTracker()

# Get all sessions
sessions = tracker.get_all_sessions()

# Get improvement stats
stats = tracker.get_improvement_stats()
print(f"Data grew by {stats['data_growth']} games")
```

## File Location

Training history stored at:
```
models/training_history.json
```

Auto-created on first training, persists across retrains.

---

**Result**: You now have concrete proof that your model gets smarter with each retrain! 📈
