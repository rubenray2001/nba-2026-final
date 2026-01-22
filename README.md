# Elite NBA Prediction Model

An elite NBA game prediction system using ensemble machine learning and advanced statistics from balldontlie.io.

## Features

- **Elite Ensemble Model**: 6 base models (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting) with stacking
- **50+ Features**: Rolling stats, advanced metrics, form indicators, H2H records, rest days
- **Vegas Odds Integration**: Spread, moneyline, and over/under predictions
- **Beautiful Streamlit UI**: Matches user's design specifications
- **Detailed Analysis**: Elite play section with reasoning for each prediction
- **Historical Data**: Games from 2005-2026
- **GOAT Tier API**: Full access to advanced stats, betting odds, and more

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (first time only):
```bash
python train_model.py
```

This will:
- Fetch historical game data (2018-2024)
- Generate 50+ features per game
- Train the elite ensemble model
- Save trained models to `models/` directory

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will:
- Default to today's date
- Show all games with predictions ranked by confidence
- Display win probabilities, predicted scores, spreads, and totals
- Compare predictions to Vegas odds
- Provide detailed elite analysis for each game

## Project Structure

```
4 nba model/
├── app.py                  # Streamlit application
├── config.py              # Configuration and settings
├── api_client.py          # BallDontLie API wrapper
├── data_manager.py        # Data fetching and caching
├── features.py            # Feature engineering
├── model_engine.py        # Elite ensemble model
├── train_model.py         # Training pipeline
├── team_logos.py          # NBA team branding
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
├── data/                  # Cached data (auto-created)
└── models/                # Trained models (auto-created)
```

## Model Details

### Base Models
1. **XGBoost Regressor** - Gradient boosting
2. **LightGBM Regressor** - Fast tree-based learning
3. **CatBoost Regressor** - Categorical feature handling
4. **Random Forest** - Ensemble of decision trees
5. **Extra Trees** - Randomized trees
6. **Gradient Boosting** - Classic boosting

### Meta-Learner
- Ridge Regression with 5-fold cross-validation

### Features
- Rolling windows: Last 5, 10, 20 games
- Advanced metrics: OffRtg, DefRtg, NetRtg, Pace, eFG%, TS%
- Four Factors: Shooting, turnovers, rebounding, free throws
- Form indicators: Streaks, momentum
- Matchup features: H2H records, rest days
- Standings: Conference/division ranks, season W-L%

## API Access

This model uses the BallDontLie GOAT tier subscription which provides:
- Historical games (2005-2026)
- Advanced statistics
- Betting odds (spreads, totals, moneylines)
- Team season averages
- Standings
- Player injuries
- Box scores

## License

MIT License
