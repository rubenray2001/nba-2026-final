# 🎼 Orchestration Plan: NBA Elite Model Upgrade

**Goal:** Upgrade NBA prediction system to "Elite" status (70-90% accuracy target) using Deep Learning (LSTM) and Advanced Ensembling.

## 👥 Agent Roles
| Agent | Role | Responsibilities |
|-------|------|------------------|
| **Database Architect** | Data Engineering | Integrate `nba_api` for advanced stats (PER, BPM), optimize data storage, prepare time-series sequences. |
| **Backend Specialist** | AI/ML Implementation | Implement LSTM (Deep Learning), refine XGBoost/CatBoost, build Stacking Ensemble. |
| **Test Engineer** | Verification | Validation scripts, backtesting, accuracy verification, `checklist.py`. |
| **DevOps Engineer** | Deployment | Deploy to Cloud Run (if verified). |

## 📅 Phased Execution Plan

### Phase 1: Advanced Data Engineering (Database Architect)
*Objective: Move beyond basic box scores to "Elite" data.*
- [ ] **Dependency Update**: Add `nba_api`, `tensorflow` (or `torch`) to `requirements.txt`.
- [ ] **Advanced Stats Ingestion**:
    - Implement fetching of PER, BPM, USG%, ORtg/DRtg using `nba_api`.
    - Backfill data for 2015-2026.
- [ ] **Sequence Preparation**:
    - Create `SequenceBuilder` class to format data for LSTM (samples of N past games -> target).

### Phase 2: Deep Learning Implementation (Backend Specialist)
*Objective: Capture temporal patterns and momentum.*
- [ ] **LSTM Model**:
    - Create `models/lstm_model.py`.
    - Architecture: Bidirectional LSTM -> Dense/Dropout -> Output.
    - Input: Sequence of last 10 games (Advanced Stats).
- [ ] **Training Pipeline**:
    - Create `train_deep.py`.
    - Implement custom loss function (if needed) or simple Binary Crossentropy.

### Phase 3: Elite Ensemble (Backend Specialist)
*Objective: Combine Tree-based and Deep Learning models.*
- [ ] **Stacking Architecture**:
    - Level 0: XGBoost, CatBoost, Random Forest, LSTM.
    - Level 1 (Meta): Logistic Regression or simple Neural Net.
- [ ] **Integration**:
    - Update `model_engine.py` to support the hybrid ensemble.

### Phase 4: Verification & Deployment (Test Engineer)
*Objective: Prove it works.*
- [ ] **Backtesting**: Run `verify_accuracy.py` on held-out test set (2025-2026 season).
- [ ] **Code Quality**: Run `lint_runner.py`.
- [ ] **Deployment**: Update `app.py` to use new model, deploy to Cloud Run.

## 🧪 Verification Plan
- **Accuracy Check**: `python scripts/verify_accuracy.py` -> Must be > 65% (aiming for 70%).
- **Unit Tests**: `pytest tests/test_lstm.py`.
- **Integration**: Manual check of `app.py` prediction display.
