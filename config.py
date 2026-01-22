"""
Elite NBA Model - Configuration
"""
import os

# Try to import Streamlit secrets (for cloud deployment)
try:
    import streamlit as st
    _secrets = st.secrets.get("api_keys", {})
except:
    _secrets = {}

# API Configuration (Streamlit Cloud secrets > Environment vars > Defaults)
API_KEY = _secrets.get("BALLDONTLIE_API_KEY", os.environ.get("BALLDONTLIE_API_KEY", "9deeba1d-acec-4e0a-86f6-c762a66a1f2e"))
API_BASE_URL = "https://api.balldontlie.io"
API_VERSION_V1 = "v1"
API_VERSION_V2 = "v2"

# The Odds API (Backup for complete odds coverage)
ODDS_API_KEY = _secrets.get("ODDS_API_KEY", os.environ.get("ODDS_API_KEY", "a3dd82c243fd40b0231a81777e360d83"))
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Data Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CACHE_EXPIRY_HOURS = 24

# Season Configuration
START_SEASON = 2005
END_SEASON = 2026
TRAINING_SEASONS = list(range(2000, 2027))  # Training from 2000 to 2026
VALIDATION_SEASON = 2025

# Model Configuration
MODEL_CONFIG = {
    "ensemble": {
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 50,
            "random_state": 42
        },
        "catboost": {
            "iterations": 300,
            "depth": 7,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": False
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42
        },
        "extra_trees": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42
        }
    },
    "meta_learner": {
        "alpha": 1.0
    }
}

# Feature Configuration
ROLLING_WINDOWS = [5, 10, 20]
MIN_GAMES_FOR_STATS = 5

# Team IDs (NBA teams)
TEAM_IDS = list(range(1, 31))

# Streamlit Configuration
PAGE_TITLE = "🏀 Elite NBA Predictions"
PAGE_ICON = "🏀"
LAYOUT = "wide"

# UI Configuration
UI_CONFIG = {
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e",
    "background_color": "#f8f9fa",
    "card_border_color": "#dee2e6",
    "confidence_high": "#28a745",
    "confidence_medium": "#ffc107",
    "confidence_low": "#dc3545"
}
