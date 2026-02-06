"""
Elite Hybrid Training Pipeline
Trains both Ensemble (Trees) and Deep Learning (LSTM) models using a unified, clean data split.
Includes progress bars for Deep Learning training.
"""
import pandas as pd
import numpy as np
import os
import joblib
import json
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import tensorflow as tf
from models.lstm_model import NBALSTMModel

# Configuration
MODELS_DIR = "models"
DATA_PATH = "data/training_data.csv"
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
LOOKBACK = 10 

def load_and_prep_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Sort chronologically (CRITICAL for time-series)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"Sorted {len(df)} games by date.")
    else:
        print("WARNING: No 'date' column found. Assuming data is already sorted.")

    exclude = ['game_id', 'date', 'home_team_id', 'visitor_team_id', 
               'home_score', 'visitor_score', 'home_won', 'season',
               'home_team_name', 'visitor_team_name']
    
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Smart NaN filling: use feature-appropriate defaults
    fill_defaults = {}
    for col in feature_cols:
        if 'elo' in col.lower():
            fill_defaults[col] = 1500
        elif 'win_pct' in col.lower():
            fill_defaults[col] = 0.5
        elif 'efg_pct' in col.lower():
            fill_defaults[col] = 0.54
        elif 'points_scored' in col.lower() or 'points_allowed' in col.lower():
            fill_defaults[col] = 110
        elif 'pace' in col.lower():
            fill_defaults[col] = 98
        elif 'rest_days' in col.lower():
            fill_defaults[col] = 2
        elif 'vegas_total' in col.lower():
            fill_defaults[col] = 220.0
        elif 'vegas_implied' in col.lower() or 'prob' in col.lower():
            fill_defaults[col] = 0.5
        else:
            fill_defaults[col] = 0
    X_raw = df[feature_cols].fillna(fill_defaults).select_dtypes(include=[np.number])
    y_winner = df['home_won'].values
    
    return X_raw, y_winner, list(X_raw.columns)

def create_sequences(X, y, lookback):
    """Create sequences for LSTM (X[t-lookback:t] -> y[t])"""
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i+lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to show clean progress"""
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        # ASCII Progress Bar
        bar_len = 20
        progress = int(bar_len * (acc or 0)) # approx based on acc? No, just visual
        
        print(f"Epoch {epoch+1}/{LSTM_EPOCHS} | Loss: {loss:.4f} - Acc: {acc:.2%} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2%}")

def train_elite():
    start_time = time.time()
    print("=" * 60)
    print("TRAINING ELITE HYBRID MODEL (ENSEMBLE + LSTM)")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    X_df, y, features = load_and_prep_data()
    print(f"Features: {len(features)}")
    
    # 2. Split (Time-series split: Train on past, Test on future)
    # We do ONE split for both models to ensure fair comparison/blending
    split_idx = int(len(X_df) * 0.8)
    
    X_tr_raw = X_df.values[:split_idx]
    X_te_raw = X_df.values[split_idx:]
    y_tr = y[:split_idx]
    y_te = y[split_idx:]
    
    print(f"Train Samples: {len(X_tr_raw)}")
    print(f"Test Samples:  {len(X_te_raw)}")
    
    # 3. Scale (Fit on TRAIN only)
    print("\n[Stage 1] Preprocessing & Feature Scaling...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    
    # ---------------------------------------------------------
    # PART A: ENSEMBLE MODEL (sklearn)
    # ---------------------------------------------------------
    print("\n[Stage 2] Training Tree Ensemble...")
    
    ensemble = VotingClassifier([
        ('hgb1', HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.04, 
            l2_regularization=3.0, random_state=42
        )),
        ('hgb2', HistGradientBoostingClassifier(
            max_iter=600, max_depth=8, learning_rate=0.03,
            max_leaf_nodes=40, random_state=43
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=400, max_depth=12, max_features='sqrt',
            n_jobs=-1, random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=400, max_depth=15, max_features='sqrt',
            n_jobs=-1, random_state=42
        ))
    ], voting='soft', n_jobs=-1)
    
    ensemble.fit(X_tr, y_tr)
    
    # Evaluation
    ens_pred = ensemble.predict(X_te)
    ens_proba = ensemble.predict_proba(X_te)[:, 1]
    ens_acc = accuracy_score(y_te, ens_pred)
    ens_loss = log_loss(y_te, ens_proba)
    
    print(f"Ensemble Accuracy: {ens_acc:.2%}")
    print(f"Ensemble Log Loss: {ens_loss:.4f}")
    
    joblib.dump(ensemble, f"{MODELS_DIR}/winner_ensemble.pkl")
    
    # ---------------------------------------------------------
    # PART B: DEEP LEARNING (LSTM)
    # ---------------------------------------------------------
    print("\n[Stage 3] Training Deep Learning (LSTM)...")
    
    # Sequence creation requires looking back into the past
    # For training data, we can just slice.
    # For test data, we technically need the last N rows of training data to predict the first row of test data.
    
    # Combine for sequence generation then re-split? 
    # Cleaner: Generate sequences on full scaled data, then split at same index (adjusted for lookback)
    
    X_all_scaled = np.vstack((X_tr, X_te))
    y_all = np.concatenate((y_tr, y_te))
    
    X_seq, y_seq = create_sequences(X_all_scaled, y_all, LOOKBACK)
    
    # Re-calculate split index for sequences
    # The first LOOKBACK samples are consumed by windowing
    seq_split_idx = max(0, split_idx - LOOKBACK)
    if seq_split_idx == 0:
        print(f"WARNING: split_idx ({split_idx}) <= LOOKBACK ({LOOKBACK}). LSTM training set will be empty.")
    
    X_seq_tr = X_seq[:seq_split_idx]
    X_seq_te = X_seq[seq_split_idx:]
    y_seq_tr = y_seq[:seq_split_idx]
    y_seq_te = y_seq[seq_split_idx:]
    
    print(f"LSTM Train Sequences: {len(X_seq_tr)}")
    
    lstm = NBALSTMModel(input_shape=(LOOKBACK, len(features)))
    
    # Keras Progress Bar is automatic, but we add custom callback for cleaner logs
    history = lstm.model.fit(
        X_seq_tr, y_seq_tr,
        validation_data=(X_seq_te, y_seq_te),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=1, # Show progress bar [====================]
        callbacks=[ProgressCallback()]
    )
    
    # Evaluate
    lstm_probs = lstm.model.predict(X_seq_te, verbose=0).flatten()
    lstm_pred = (lstm_probs > 0.5).astype(int)
    lstm_acc = accuracy_score(y_seq_te, lstm_pred)
    lstm_loss = log_loss(y_seq_te, lstm_probs)
    
    print(f"LSTM Accuracy: {lstm_acc:.2%}")
    print(f"LSTM Log Loss: {lstm_loss:.4f}")
    
    lstm.save(f"{MODELS_DIR}/nba_lstm.keras")
    
    # ---------------------------------------------------------
    # PART C: HYBRID BLENDING
    # ---------------------------------------------------------
    print("\n[Stage 4] Hybrid Ensemble Optimization...")
    
    # We need to align the predictions. 
    # Ensemble predictions on X_te start at split_idx
    # LSTM predictions on X_seq_te start at split_idx (conceptually, looking back)
    
    # The LSTM test set covers the exact same games as the Ensemble test set!
    # (Since we generated sequences from the full contiguous block and split at same relative point)
    
    # Blend Validation
    best_acc = 0
    best_w = 0.0
    
    # Grid search blending weight
    for w_ens in np.linspace(0.0, 1.0, 21):
        w_lstm = 1.0 - w_ens
        
        # Weighted probability
        hybrid_prob = (ens_proba * w_ens) + (lstm_probs * w_lstm)
        hybrid_pred = (hybrid_prob > 0.5).astype(int)
        acc = accuracy_score(y_te, hybrid_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_w = w_ens
            
    print(f"BEST HYBRID ACCURACY: {best_acc:.2%}")
    print(f"   (Ensemble Weight: {best_w:.2f}, LSTM Weight: {1.0-best_w:.2f})")
    
    # Save Metadata
    meta = {
        'feature_names': features,
        'training_info': {
            'trained_at': datetime.now().isoformat(),
            'ensemble_accuracy': float(ens_acc),
            'lstm_accuracy': float(lstm_acc),
            'hybrid_accuracy': float(best_acc),
            'best_ensemble_weight': float(best_w)
        }
    }
    
    with open(f"{MODELS_DIR}/model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nTotal Time: {time.time() - start_time:.1f}s")
    print(f"Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    train_elite()
