# Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
cd "C:\Users\ruben\OneDrive\Desktop\4 nba model"
pip install -r requirements.txt
```

## First Time Setup - Train the Model

**IMPORTANT**: You must train the model before running the app for the first time.

```bash
python train_model.py
```

This will:
- Fetch historical NBA games from 2018-2024 seasons
- Calculate 50+ features for each game
- Train the elite 6-model ensemble
- Save trained models to `models/` directory

**Expected time**: 10-30 minutes depending on your API rate limits and computer speed.

## Run the App

Once the model is trained, start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the App

### Features:
- **📅 Calendar Picker**: Select any date (defaults to today)
- **🎯 Game Predictions**: Ranked from best to worst by confidence
- **📊 Team Logos**: Visual NBA team branding
- **💰 Betting Lines**: Moneyline, Spread, and Total predictions
- **🔍 Elite Analysis**: Detailed reasoning for each prediction
- **✅ Model Consensus**: Shows agreement across all 6 base models

### Sidebar Controls:
- **Select Game Date**: Choose which day's games to view
- **Model Info**: See model accuracy and training date
- **🔄 Refresh Data**: Clear cache and fetch latest data

## Date Handling

The app correctly handles:
- **Default Date**: Always starts with today's date
- **Date Range**: Allows selection from 7 days ago to 30 days ahead
- **Season Detection**: Automatically detects current NBA season (Oct-Jun)
- **Historical Context**: Uses 2 seasons of data for predictions

## Troubleshooting

### "Model not trained" error
- Run `python train_model.py` first

### "No games found" message
- Check if there are NBA games scheduled for the selected date
- Try selecting today's date or a known game day

### API errors
- Verify your API key is correct in `config.py`
- Check your internet connection
- API may have rate limits - training will retry automatically

### Empty predictions
- Ensure model training completed successfully
- Check that `models/` directory contains .pkl files
- Try retraining: `python train_model.py`

## Retraining the Model

To update the model with more recent data:

```bash
python train_model.py
```

This will fetch fresh data and retrain from scratch. Recommended monthly during the NBA season.

## Files Structure

```
4 nba model/
├── app.py                 # Main Streamlit app (run this)
├── train_model.py         # Training script (run once)
├── config.py             # Settings (API key here)
├── api_client.py         # API wrapper
├── data_manager.py       # Data fetching
├── features.py           # Feature engineering
├── model_engine.py       # Elite ensemble
├── team_logos.py         # NBA branding
├── utils.py              # Helpers
├── requirements.txt      # Dependencies
├── data/                 # Auto-created cache
└── models/               # Auto-created models
```

## Support

For issues, check:
1. All dependencies installed: `pip list`
2. API key is valid in `config.py`
3. Model has been trained
4. Internet connection active
