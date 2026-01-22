"""
Final CV validation to determine true model accuracy
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/training_data.csv")
exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 'home_score', 'visitor_score', 'home_won', 'season']
X = df[[c for c in df.columns if c not in exclude]].fillna(0).select_dtypes(include=[np.number])
y = df['home_won']

# Add key enhanced features
X['implied_home_prob'] = 1 / (1 + 10 ** (-X['elo_diff'] / 400))
X['prob_edge'] = np.abs(X['implied_home_prob'] - 0.5)
if 'momentum_diff_5' in X.columns and 'momentum_diff_10' in X.columns:
    X['weighted_momentum'] = X['momentum_diff_5'] * 0.6 + X['momentum_diff_10'] * 0.4
if 'home_is_b2b' in X.columns and 'visitor_is_b2b' in X.columns:
    X['b2b_advantage'] = X['visitor_is_b2b'].astype(float) - X['home_is_b2b'].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

print("Running 5-fold CV on full 2000-2026 dataset...")
clf = VotingClassifier([
    ('hgb1', HistGradientBoostingClassifier(max_iter=600, max_depth=5, learning_rate=0.03, l2_regularization=4.0, random_state=42)),
    ('hgb2', HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.04, l2_regularization=3.0, random_state=43)),
    ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=-1, random_state=42))
], voting='soft', n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y.values, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\nCV Scores: {[f'{s*100:.1f}%' for s in scores]}")
print(f"Mean CV Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
print(f"Best Fold: {scores.max()*100:.1f}%")
print(f"Worst Fold: {scores.min()*100:.1f}%")
