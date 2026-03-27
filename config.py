"""
Elite College Basketball Model - Configuration
Supports both Men's and Women's college basketball
"""
import os

# The Odds API (Primary odds source for college basketball)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "a3dd82c243fd40b0231a81777e360d83")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Preferred Bookmakers (for display sorting)
PREFERRED_BOOKS = [
    "FanDuel", "DraftKings", "Bovada", "Caesars", "BetMGM", 
    "Fanatics", "BetRivers", "Bet365", "Unibet", "William Hill"
]

# ESPN API (No key required - public endpoints)
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball"
ESPN_MENS = "mens-college-basketball"
ESPN_WOMENS = "womens-college-basketball"

# Data Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_EXPIRY_HOURS = 24

# Season Configuration - Last 2 years
# College basketball seasons span two calendar years (e.g., 2023-24 season)
# ESPN uses the starting year as the season identifier
CURRENT_SEASON = 2025  # 2024-25 season
TRAINING_SEASONS = [2024, 2025]  # Last 2 years: 2023-24 and 2024-25

# Model Configuration (same architecture as NBA model)
MODEL_CONFIG = {
    "ensemble": {
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42
        },
        "extra_trees": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42
        }
    }
}

# Feature Configuration
ROLLING_WINDOWS = [5, 10, 20]
MIN_GAMES_FOR_STATS = 5

# College Basketball Defaults (differ from NBA)
# Men's averages are roughly 70-75 PPG, Women's roughly 65-70 PPG
MENS_AVG_SCORE = 72.0
WOMENS_AVG_SCORE = 67.0
MENS_AVG_TOTAL = 144.0
WOMENS_AVG_TOTAL = 134.0
MENS_AVG_PACE = 68.0  # Possessions per game
WOMENS_AVG_PACE = 65.0

# Gender-specific config
GENDER_CONFIG = {
    "mens": {
        "espn_slug": ESPN_MENS,
        "odds_sport": "basketball_ncaab",
        "avg_score": MENS_AVG_SCORE,
        "avg_total": MENS_AVG_TOTAL,
        "avg_pace": MENS_AVG_PACE,
        "label": "Men's",
        "icon": "🏀",
    },
    "womens": {
        "espn_slug": ESPN_WOMENS,
        "odds_sport": "basketball_wncaab",
        "avg_score": WOMENS_AVG_SCORE,
        "avg_total": WOMENS_AVG_TOTAL,
        "avg_pace": WOMENS_AVG_PACE,
        "label": "Women's",
        "icon": "🏀",
    }
}

# Streamlit Configuration
PAGE_TITLE = "🏀 Elite College Basketball Predictions"
PAGE_ICON = "🏀"
LAYOUT = "wide"

# UI Configuration (identical to NBA)
UI_CONFIG = {
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e",
    "background_color": "#f8f9fa",
    "card_border_color": "#dee2e6",
    "confidence_high": "#28a745",
    "confidence_medium": "#ffc107",
    "confidence_low": "#dc3545"
}
