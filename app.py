"""
Streamlit App - Elite NBA Predictions
Beautiful UI matching user's design specifications
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import re

# Import our modules
from data_manager import DataManager
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer, STAR_PLAYER_PPG

# Try regularized model first, fall back to original if needed
# Prefer Elite Model (XGBoost/CatBoost)
try:
    from model_engine import EliteEnsembleModel
except ImportError:
    from model_engine_regularized import RegularizedEnsembleModel as EliteEnsembleModel

from api_client import BallDontLieClient
import team_logos
import config
from odds_utils import get_consensus_odds, format_american_odds, calculate_edge
import team_logos as tl
from prediction_tracker import PredictionTracker
from betting_model import BettingModel
from odds_api_client import TheOddsAPIClient




# Cache for injuries
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_injuries():
    """Fetch current player injuries"""
    try:
        client = BallDontLieClient()
        injuries = client.get_player_injuries()
        return injuries
    except Exception as e:
        print(f"Error fetching injuries: {e}")
        return []


def get_team_injuries(injuries: list, team_id: int) -> list:
    """Get injuries for a specific team"""
    team_injuries = []
    for injury in injuries:
        player = injury.get('player', {})
        # team_id is directly on the player object
        if player.get('team_id') == team_id:
            team_injuries.append({
                'player_name': f"{player.get('first_name', '')} {player.get('last_name', '')}",
                'status': injury.get('status', 'Unknown'),
                'description': injury.get('description', ''),
                'return_date': injury.get('return_date', 'Unknown')
            })
    return team_injuries


def get_injury_impact_text(injuries: list) -> str:
    """Generate text explaining injury impact"""
    if not injuries:
        return "No reported injuries"
    
    out_players = [i for i in injuries if i['status'].lower() in ['out', 'doubtful']]
    questionable = [i for i in injuries if i['status'].lower() in ['questionable', 'probable', 'day-to-day']]
    
    impact_parts = []
    if out_players:
        names = [p['player_name'] for p in out_players[:3]]  # Top 3
        impact_parts.append(f"**OUT**: {', '.join(names)}")
    if questionable:
        names = [p['player_name'] for p in questionable[:3]]
        impact_parts.append(f"**Questionable**: {', '.join(names)}")
    
    return "\n".join(impact_parts) if impact_parts else "No significant injuries"


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)





def inject_custom_css():
    """Inject Pro Dashboard V2 CSS (CSS Grid + Sharp Lines)"""
    st.markdown("""
    <style>
        /* RESET & FONTS */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;900&display=swap');
        
        * { box-sizing: border-box; }
        
        :root {
            /* SHARP NEON PALETTE */
            --neon-cyan: #00F3FF;
            --neon-magenta: #FF00FF;
            --bg-dark: #050505;
            --bg-card: #0E1117;
            --border-color: #333333;
            --text-white: #FFFFFF;
            --text-gray: #888888;
        }

        .stApp { background-color: var(--bg-dark); }

        /* HEADER TYPOGRAPHY */
        .main-header {
            font-family: 'Inter', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: -1px;
            border-bottom: 2px solid var(--neon-cyan);
            padding-bottom: 10px;
            margin-bottom: 5px;
        }
        .sub-header {
            font-family: 'JetBrains Mono', monospace;
            color: var(--neon-cyan);
            font-size: 1rem;
            margin-bottom: 40px;
        }

        /* CARD CONTAINER - NO GLOW, JUST SHARP LINES */
        .game-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-top: 2px solid var(--neon-cyan);
            border-radius: 0px; /* Sharp corners */
            margin-bottom: 30px;
            padding: 0;
            overflow: hidden;
            position: relative;
        }

        /* CSS GRID LAYOUT - 3 COLUMNS STRICT */
        .game-grid {
            display: grid;
            grid-template-columns: 1fr 180px 1fr; /* Team | VS | Team */
            align-items: center;
            padding: 30px;
            border-bottom: 1px solid var(--border-color);
        }

        /* TEAM CELL */
        .team-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        
        .team-logo-img {
            width: 80px;
            height: 80px;
            object-fit: contain;
            filter: drop-shadow(0 0 5px rgba(0,0,0,0.5));
            margin-bottom: 15px;
        }

        .team-name {
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        
        .win-prob {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            color: var(--neon-cyan);
            border: 1px solid var(--neon-cyan);
            padding: 2px 8px;
            background: rgba(0, 243, 255, 0.05);
        }

        /* CENTER VS CELL */
        .vs-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-left: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            height: 100%;
        }

        .game-score-lg {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -2px;
            line-height: 1;
        }
        
        .game-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--text-gray);
            margin-top: 10px;
        }

        /* ACTION BAR (BADGES) */
        .action-bar {
            background-color: #000000;
            padding: 15px;
            display: flex;
            justify-content: center;
            gap: 15px;
            border-bottom: 1px solid var(--border-color);
        }

        .badge-pro {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--bg-dark);
            background: var(--neon-cyan);
            padding: 6px 12px;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .badge-outline {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #FFFFFF;
            border: 1px solid #FFFFFF;
            padding: 6px 12px;
            text-transform: uppercase;
        }

        /* STATS GRID - 3 COLUMN TABLE */
        .stats-container {
            padding: 15px 30px;
            background: rgba(255,255,255,0.02);
        }

        .pro-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .pro-table td {
            padding: 10px;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: #DDDDDD;
            border-bottom: 1px solid #222;
        }
        
        .pro-table tr:last-child td { border: none; }
        
        .label-cell {
            color: #666 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 40%;
        }
        
        /* STREAMLIT OVERRIDES */
        div.stAlert {
            background-color: #000;
            border: 1px solid var(--neon-cyan);
            color: white;
        }

        /* SPECIFIC SUCCESS MESSAGE OVERRIDE */
        .stSuccess {
            background-color: #0E1117 !important;
            border: 1px solid var(--neon-cyan) !important;
        }
        .stSuccess p {
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* ANALYSIS TEXT READABILITY */
        .analysis-text {
            color: #E0E0E0 !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }

        /* SPINNER TEXT OVERRIDE */
        div[data-testid="stSpinner"] > div {
            color: var(--neon-cyan) !important;
            border-color: var(--neon-cyan) !important;
        }
        div[data-testid="stSpinner"] p {
            color: var(--neon-cyan) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.1rem !important;
        }

        /* EXPANDER HEADER - REFINED FIX */
        iframe[title="streamlit_expander"] { display: none; } /* cleanup */
        
        div[data-testid="stExpander"] details summary {
            background-color: #0E1117 !important;
            border: 1px solid var(--neon-cyan) !important;
            color: var(--neon-cyan) !important;
            border-radius: 4px;
        }

        /* TEXT ONLY inside summary */
        div[data-testid="stExpander"] details summary p {
            color: var(--neon-cyan) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
        }
        
        /* ICONS (SVG) inside summary - DO NOT CHANGE FONT */
        div[data-testid="stExpander"] details summary svg {
            fill: var(--neon-cyan) !important;
            color: var(--neon-cyan) !important;
        }
        
        div[data-testid="stExpander"] details[open] summary {
             border-bottom-left-radius: 0 !important;
             border-bottom-right-radius: 0 !important;
             border-bottom: 1px solid var(--neon-cyan) !important;
        }
        
        div[data-testid="stExpander"] details {
            border-color: transparent !important; 
        }

        /* WINNER INDICATOR */
        .winner-tag {
            background: var(--neon-cyan);
            color: #000;
            font-weight: 900;
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 4px;
            margin-top: 5px;
            display: inline-block;
        }
        
        /* TAB STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0E1117;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a2e;
            color: #888;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: bold;
            border: 1px solid #333;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #252540;
            color: #FFF;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0,243,255,0.2), rgba(0,243,255,0.1)) !important;
            color: #00F3FF !important;
            border: 1px solid #00F3FF !important;
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #00F3FF !important;
        }
        
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cache invalidates when metadata changes)"""
    # Check if model pkl files exist
    model_pkl = os.path.join(config.MODELS_DIR, 'home_score_ensemble.pkl')
    metadata_path = os.path.join(config.MODELS_DIR, 'model_metadata.json')
    
    # If models don't exist, train them first
    if not os.path.exists(model_pkl):
        st.info("🔄 First time setup: Training model... This may take a few minutes.")
        try:
            import train_model
            train_model.main()
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            return None
    
    if not os.path.exists(metadata_path):
        return None
    
    # Read metadata timestamp to use as cache key
    with open(metadata_path, 'r') as f:
        import json
        metadata = json.load(f)
        trained_at = metadata.get('training_info', {}).get('trained_at', '')
    
    # Load model (this function reruns when trained_at changes)
    try:
        from model_engine import EliteEnsembleModel
        model_class = EliteEnsembleModel
        model_type = "Elite (Ensemble + LSTM)"
    except ImportError:
        from model_engine_regularized import RegularizedEnsembleModel
        model_class = RegularizedEnsembleModel
        model_type = "Regularized (Ensemble)"
    
    try:
        model = model_class()
        model.load_models()
        
        # VALIDATION: Sanity-check that model loaded correctly
        # (Feature count is dynamic based on training pipeline, so no hardcoded check)
        if hasattr(model, 'feature_names') and model.feature_names:
            print(f"Model loaded: {len(model.feature_names)} features")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(ttl=30)  # Cache for 30 seconds for live scores
def get_todays_games(target_date):
    """Fetch games for target date"""
    data_mgr = DataManager()
    games_df = data_mgr.fetch_todays_games(target_date)
    return games_df


# Note: No caching for odds - they contain unhashable dict objects and should be fresh
def get_vegas_odds(target_date, games_df=None):
    """Fetch Vegas odds for target date with fallback"""
    data_mgr = DataManager()
    odds_df = data_mgr.fetch_vegas_odds(dates=[target_date], games_df=games_df)
    
    # Clean up any dict columns that could cause issues downstream
    if not odds_df.empty:
        for col in odds_df.columns:
            if odds_df[col].apply(lambda x: isinstance(x, dict)).any():
                # Convert dict columns to string representation
                odds_df[col] = odds_df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
    
    return odds_df


@st.cache_data(ttl=300)  # Cache for 5 minutes for live/recent updates
def get_box_scores(target_date):
    """Fetch box scores for target date"""
    data_mgr = DataManager()
    # Force refresh if catching live data, but cache_data handles the frequency
    box_scores = data_mgr.fetch_box_scores(dates=[target_date], force_refresh=True)
    return box_scores


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_player_props():
    """Fetch NBA player props from The Odds API"""
    try:
        client = TheOddsAPIClient()
        props = client.get_player_props()
        return props
    except Exception as e:
        print(f"Error fetching player props: {e}")
        return []


def get_confidence_level(probability):
    """Determine confidence level from probability
    
    Based on backtesting with enhanced features:
    - >70% confidence = 77% historical accuracy (LOCK)
    - >65% confidence = 73% historical accuracy (HIGH)
    - >60% confidence = 70% historical accuracy (GOOD)
    - 55-60% = standard pick
    - <55% = coin flip
    """
    if probability >= 0.70:
        return "LOCK", "🔒"  # 77% accuracy historically
    elif probability >= 0.65:
        return "HIGH", "🔥🔥🔥"  # 73% accuracy historically
    elif probability >= 0.60:
        return "GOOD", "🔥🔥"  # 70% accuracy historically
    elif probability >= 0.55:
        return "LEAN", "🔥"
    else:
        return "TOSS-UP", "⚖️"


def format_spread(spread):
    """Format spread with proper sign"""
    if spread > 0:
        return f"+{spread:.1f}"
    return f"{spread:.1f}"


def format_moneyline(prob):
    """Convert probability to moneyline odds"""
    # Clamp to avoid division by zero at extremes
    prob = max(0.01, min(0.99, prob))
    
    if prob >= 0.5:
        # Favorite
        ml = -(prob / (1 - prob)) * 100
    else:
        # Underdog
        ml = ((1 - prob) / prob) * 100
    
    if ml > 0:
        return f"+{int(ml)}"
    return f"{int(ml)}"


def generate_elite_analysis(game_data, prediction, features, vegas_odds=None, injuries=None):
    """Generate detailed robust HTML analysis for prediction"""
    home_team = game_data.get('home_team_name', 'Home Team')
    visitor_team = game_data.get('visitor_team_name', 'Visitor Team')
    home_id = game_data.get('home_team_id')
    visitor_id = game_data.get('visitor_team_id')
    
    home_prob = prediction['home_win_probability']
    predicted_winner = home_team if home_prob > 0.5 else visitor_team
    loser_team = visitor_team if home_prob > 0.5 else home_team
    winner_prob = max(home_prob, 1 - home_prob)
    is_home_favorite = home_prob > 0.5
    
    # ---------------------------------------------------------
    # 1. HEADER & PREDICTION SUMMARY
    # ---------------------------------------------------------
    html = f"""
    <div style="font-family: 'Inter', sans-serif; color: #EEE;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
            <div>
                <h3 style="margin:0; color: #FFF; font-size: 1.8rem;">🎯 ELITE ANALYSIS</h3>
                <div style="color: #00F3FF; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; margin-top: 5px;">
                    {predicted_winner} ({winner_prob:.1%})
                </div>
            </div>
            <div style="text-align: right; background: rgba(0, 243, 255, 0.1); padding: 10px 20px; border-radius: 8px; border: 1px solid #00F3FF;">
                <div style="font-size: 0.8rem; color: #AAA;">WIN CONFIDENCE</div>
                <div style="font-size: 1.4rem; font-weight: 900; color: #00F3FF;">{winner_prob:.1%}</div>
            </div>
        </div>
    """
    
    # ---------------------------------------------------------
    # 1.5 INJURY ADJUSTMENT INDICATOR
    # ---------------------------------------------------------
    # Check if injury penalty was applied
    injury_impact_diff = features.get('injury_impact_diff', 0.0)
    if abs(injury_impact_diff) > 5:
        # Determine who benefits from injury imbalance
        benefit_team = home_team if injury_impact_diff > 0 else visitor_team
        
        html += f"""
        <div style="background: rgba(255, 68, 68, 0.15); border: 1px solid #FF4444; border-left: 5px solid #FF4444; padding: 12px; margin-bottom: 20px; border-radius: 4px;">
            <div style="color: #FF4444; font-weight: bold; display: flex; align-items: center; gap: 8px;">
                <span>🤕 MAJOR INJURY IMBALANCE DETECTED</span>
            </div>
            <div style="color: #DDD; font-size: 0.9rem; margin-top: 5px;">
                Significant injury imbalance favors <strong>{benefit_team}</strong> (impact diff: {abs(injury_impact_diff):.1f}).
                The model accounts for this in its prediction.
            </div>
        </div>
        """
    
    # ---------------------------------------------------------
    # 2. BETTING RECOMMENDATIONS (The "Fire" Section)
    # ---------------------------------------------------------
    recommended_bets = []
    locks = [] # Detailed breakdown storage
    
    # Calculate Edges
    if vegas_odds and vegas_odds.get('has_odds'):
        vegas_prob = vegas_odds['implied_home_prob'] if is_home_favorite else vegas_odds['implied_away_prob']
        vegas_spread = vegas_odds['spread_home'] if is_home_favorite else vegas_odds['spread_away']
        vegas_total = vegas_odds['total']
        
        # Model spread: positive = home wins by more (home favored)
        # Vegas spread: negative = home favored (e.g., -5.5 means home favored by 5.5)
        # Convert model convention to Vegas convention by negating
        model_spread = -prediction['predicted_spread']
        model_total = prediction['predicted_total']
        
        ml_edge = (winner_prob - vegas_prob) * 100
        spread_diff = abs(model_spread - vegas_spread)
        total_diff = model_total - vegas_total
        
        # LOGIC: Moneyline
        if ml_edge >= 20:
            recommended_bets.append(f"🔒 **BET MONEYLINE:** {predicted_winner} to Win (Lock)")
            locks.append(("MONEYLINE LOCK", f"Model Edge: +{ml_edge:.1f}%", f"{predicted_winner} win probability: {winner_prob:.1%} vs Vegas {vegas_prob:.1%}"))
        elif ml_edge >= 12:
            recommended_bets.append(f"💎 **BET MONEYLINE:** {predicted_winner} to Win (Value)")
            locks.append(("MONEYLINE VALUE", f"Model Edge: +{ml_edge:.1f}%", f"Consistent upside detected on {predicted_winner}."))
        elif ml_edge >= 5:
             locks.append(("MONEYLINE LEAN", f"Model Edge: +{ml_edge:.1f}%", f"Slight edge on {predicted_winner}. Valid for straight picks."))

        # LOGIC: Spread
        model_favors_favorite_more = abs(model_spread) > abs(vegas_spread)
        # Check if model thinks team wins vs Vegas Underdog
        if spread_diff >= 10:
             if vegas_spread > 0: # Vegas says underdog, model says favorite/winner
                 recommended_bets.append(f"🔒 **BET SPREAD:** {predicted_winner} +{abs(vegas_spread):.1f} (Lock)")
                 locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Model sees {predicted_winner} winning by {abs(model_spread):.1f}, Vegas gives them points!"))
             elif model_favors_favorite_more:
                 recommended_bets.append(f"🔥 **BET SPREAD:** {predicted_winner} {vegas_spread:+.1f} (Favorite)")
                 locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Model predicts blowout ({abs(model_spread):.1f} margin) vs Vegas {abs(vegas_spread):.1f}."))
             else:
                 recommended_bets.append(f"🔒 **BET SPREAD:** {loser_team} +{abs(vegas_spread):.1f} (Underdog)")
                 locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Vegas overestimates {predicted_winner}. Take points with {loser_team}."))
        elif spread_diff >= 6:
             # Similar logic for VALUE...
             if vegas_spread > 0:
                 recommended_bets.append(f"✅ **BET SPREAD:** {predicted_winner} +{abs(vegas_spread):.1f} (Value)")
                 locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"Strong value play on {predicted_winner} covering."))
             elif model_favors_favorite_more:
                 recommended_bets.append(f"✅ **BET SPREAD:** {predicted_winner} {vegas_spread:+.1f} (Value)")
                 locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"Model predicts comfortable win covering {vegas_spread}."))
             else:
                 recommended_bets.append(f"✅ **BET SPREAD:** {loser_team} +{abs(vegas_spread):.1f} (Value)")
                 locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"{loser_team} keeps it closer than Vegas implies."))
        elif spread_diff >= 2:
             # LEAN Logic
             if vegas_spread > 0:
                 locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {predicted_winner} +{abs(vegas_spread):.1f}."))
             elif model_favors_favorite_more:
                 locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {predicted_winner} to cover."))
             else:
                 locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {loser_team} to cover."))

        # LOGIC: Total
        if abs(total_diff) >= 10:
             ou = "OVER" if total_diff > 0 else "UNDER"
             recommended_bets.append(f"🔒 **BET TOTAL:** {ou} {vegas_total:.1f} (Lock)")
             locks.append((f"TOTAL {ou} LOCK", f"Diff: {total_diff:+.1f} pts", f"Model: {model_total:.1f} vs Vegas: {vegas_total:.1f}"))
        elif abs(total_diff) >= 6:
             ou = "OVER" if total_diff > 0 else "UNDER"
             recommended_bets.append(f"💎 **BET TOTAL:** {ou} {vegas_total:.1f} (Value)")
             locks.append((f"TOTAL {ou} VALUE", f"Diff: {total_diff:+.1f} pts", f"Significant variance from Vegas line."))
        elif abs(total_diff) >= 2:
             ou = "OVER" if total_diff > 0 else "UNDER"
             locks.append((f"TOTAL {ou} LEAN", f"Diff: {total_diff:+.1f} pts", f"Model slightly favors the {ou}."))
    
    # Render Bets
    if recommended_bets:
        html += """<div style="background: #1A1D26; border: 1px solid #FF00FF; border-left: 5px solid #FF00FF; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
        <h4 style="margin-top:0; color: #FF00FF; letter-spacing: 1px;">🔥 RECOMMENDED BETS</h4>
        <ul style="margin-bottom:0; padding-left: 20px; color: #FFF;">"""
        for bet in recommended_bets:
            # Convert bold markdown **text** to HTML <strong>text</strong>
            clean_bet = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', bet)
            # Or just use raw string if easier, but HTML makes it pop
            html += f"<li style='margin-bottom: 8px; font-size: 1.05rem;'>{clean_bet}</li>"
        html += "</ul></div>"
    else:
        html += """<div style="padding: 15px; border: 1px solid #444; border-radius: 4px; margin-bottom: 25px; color: #888;">
        ⚖️ NO STRONG PLAYS DETECTED. Model aligns with Vegas.
        </div>"""

    # ---------------------------------------------------------
    # 3. DETAILED BREAKDOWN (2 Col Grid)
    # ---------------------------------------------------------
    if locks:
        html += "<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>📋 EDGE ANALYSIS</h4>"
        html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 30px;'>"
        for title, subtitle, detail in locks:
            # Color code based on type
            color = "#00F3FF" if "LOCK" in title else "#FF00FF"
            html += f"""
            <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 4px; border-left: 3px solid {color};">
                <div style="color: {color}; font-weight: 800; font-size: 0.9rem;">{title}</div>
                <div style="color: #FFF; font-weight: bold; margin: 3px 0;">{subtitle}</div>
                <div style="color: #BBB; font-size: 0.85rem; line-height: 1.4;">{detail}</div>
            </div>
            """
        html += "</div>"

    # ---------------------------------------------------------
    # 4. VERIFIED STATS (HTML Table)
    # ---------------------------------------------------------
    # Prepare Stat Data
    h_rec = f"{int(features.get('home_wins',0))}-{int(features.get('home_losses',0))}"
    v_rec = f"{int(features.get('visitor_wins',0))}-{int(features.get('visitor_losses',0))}"
    
    h_l10 = f"{int(features.get('home_win_pct_last10',0)*10)}-{10-int(features.get('home_win_pct_last10',0)*10)}"
    v_l10 = f"{int(features.get('visitor_win_pct_last10',0)*10)}-{10-int(features.get('visitor_win_pct_last10',0)*10)}"
    
    h_ppg = features.get('home_points_scored_last10', 0)
    v_ppg = features.get('visitor_points_scored_last10', 0)
    h_opp = features.get('home_points_allowed_last10', 0)
    v_opp = features.get('visitor_points_allowed_last10', 0)
    
    html += """
    <h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>📊 VERIFIED STATS</h4>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 0.9rem;">
        <tr style="background: #111; color: #888; text-transform: uppercase; font-size: 0.8rem;">
            <th style="padding: 8px; text-align: left;">METRIC</th>
            <th style="padding: 8px; text-align: center; color: #FFF;">""" + visitor_team + """</th>
            <th style="padding: 8px; text-align: center; color: #FFF;">""" + home_team + """</th>
        </tr>
        <tr style="border-bottom: 1px solid #222;">
            <td style="padding: 10px; color: #AAA;">Season Record</td>
            <td style="padding: 10px; text-align: center; color: #FFF; font-weight: bold;">""" + v_rec + """</td>
            <td style="padding: 10px; text-align: center; color: #FFF; font-weight: bold;">""" + h_rec + """</td>
        </tr>
        <tr style="border-bottom: 1px solid #222; background: rgba(255,255,255,0.02);">
            <td style="padding: 10px; color: #AAA;">Last 10 Games</td>
            <td style="padding: 10px; text-align: center; color: #FFF;">""" + v_l10 + """</td>
            <td style="padding: 10px; text-align: center; color: #FFF;">""" + h_l10 + """</td>
        </tr>
        <tr style="border-bottom: 1px solid #222;">
            <td style="padding: 10px; color: #AAA;">L10 PPG</td>
            <td style="padding: 10px; text-align: center; color: #00F3FF;">""" + f"{v_ppg:.1f}" + """</td>
            <td style="padding: 10px; text-align: center; color: #00F3FF;">""" + f"{h_ppg:.1f}" + """</td>
        </tr>
        <tr style="border-bottom: 1px solid #222; background: rgba(255,255,255,0.02);">
            <td style="padding: 10px; color: #AAA;">L10 Opp PPG</td>
            <td style="padding: 10px; text-align: center; color: #FF4444;">""" + f"{v_opp:.1f}" + """</td>
            <td style="padding: 10px; text-align: center; color: #FF4444;">""" + f"{h_opp:.1f}" + """</td>
        </tr>
         <tr style="border-bottom: 1px solid #222;">
            <td style="padding: 10px; color: #AAA;">Rest Days</td>
            <td style="padding: 10px; text-align: center; color: #DDD;">""" + f"{int(features.get('visitor_rest_days',0))}" + """</td>
            <td style="padding: 10px; text-align: center; color: #DDD;">""" + f"{int(features.get('home_rest_days',0))}" + """</td>
        </tr>
    </table>
    """

    # ---------------------------------------------------------
    # 5. INJURY REPORT (Side by Side)
    # ---------------------------------------------------------
    if injuries:
        home_injuries = get_team_injuries(injuries, home_id)
        visitor_injuries = get_team_injuries(injuries, visitor_id)
        
        html += "<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px; margin-top:20px;'>🏥 INJURY REPORT</h4>"
        html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>"
        
        # Helper for list generation
        def make_inj_list(team_name, inj_list):
            item_html = f"<div style='margin-bottom: 10px;'><strong style='color: #FFF;'>{team_name}</strong></div>"
            if not inj_list:
                item_html += "<div style='color: #666; font-style: italic;'>No key injuries reported.</div>"
            else:
                for inj in inj_list[:6]:
                    status_color = "#FF4444" if inj['status'].lower() in ['out', 'out for season'] else "#FFAA00"
                    item_html += f"""
                    <div style="font-size: 0.85rem; margin-bottom: 6px; border-left: 2px solid {status_color}; padding-left: 8px;">
                        <span style="color: #EEE;">{inj['player_name']}</span> 
                        <span style="color: {status_color}; font-size: 0.75rem;">{inj['status']}</span>
                    </div>"""
            return item_html

        html += f"<div>{make_inj_list(visitor_team, visitor_injuries)}</div>"
        html += f"<div>{make_inj_list(home_team, home_injuries)}</div>"
        html += "</div>" # Close grid

    html += "</div>" # Close main wrapper
    
    return html




def display_game_predictions(games_df, predictions_df, odds_df, all_features, box_scores_df=None, accuracy_stats=None):
    """Display all game predictions in ranked order"""
    
    if games_df.empty:
        st.warning("No games scheduled for this date.")
        return
    
    # Fetch injuries once for all games
    injuries = get_injuries()
    
    # Format date nicely
    game_date = pd.to_datetime(games_df.iloc[0]["date"]).strftime("%B %d, %Y")
    
    st.markdown(f'<div class="main-header">🏀 NBA Prediction Model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{game_date}<br>{len(games_df)} games today • Odds may change, check your lines to find best odds.</div>', unsafe_allow_html=True)
    
    # Show REAL accuracy stats first
    if accuracy_stats and accuracy_stats.get('completed_games', 0) > 0:
        stats = accuracy_stats
        accuracy_html = '<div style="background: #1a1a2e; border: 2px solid #00F3FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">'
        accuracy_html += '<div style="font-family: JetBrains Mono; color: #FFD700; font-size: 1rem; margin-bottom: 12px;">📈 REAL TRACKED ACCURACY (Last 30 Days)</div>'
        
        # Row 1: Base model stats
        accuracy_html += '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px;">'
        
        # Overall accuracy
        win_pct = stats['winner_accuracy'] * 100
        win_color = '#4CAF50' if win_pct >= 55 else '#FFA500' if win_pct >= 50 else '#FF5252'
        accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: {win_color}; font-size: 1.5rem; font-weight: bold;">{win_pct:.1f}%</div>'
        accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">WINNER ({stats["winner_correct"]}/{stats["completed_games"]})</div></div>'
        
        # High confidence accuracy
        if stats.get('high_conf_total', 0) > 0:
            hc_pct = stats['high_conf_accuracy'] * 100
            hc_color = '#4CAF50' if hc_pct >= 60 else '#FFA500' if hc_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {hc_color}; font-size: 1.5rem; font-weight: bold;">{hc_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">HIGH CONF ({stats["high_conf_correct"]}/{stats["high_conf_total"]})</div></div>'
        else:
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">HIGH CONF</div></div>'
        
        # Betting Model ML accuracy
        if stats.get('betting_ml_total', 0) > 0:
            ml_pct = stats['betting_ml_accuracy'] * 100
            ml_color = '#4CAF50' if ml_pct >= 55 else '#FFA500' if ml_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {ml_color}; font-size: 1.5rem; font-weight: bold;">{ml_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">BET ML ({stats["betting_ml_correct"]}/{stats["betting_ml_total"]})</div></div>'
        else:
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">BET ML</div></div>'
        
        # Betting Model Spread accuracy
        if stats.get('betting_spread_total', 0) > 0:
            sp_pct = stats['betting_spread_accuracy'] * 100
            sp_color = '#4CAF50' if sp_pct >= 55 else '#FFA500' if sp_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {sp_color}; font-size: 1.5rem; font-weight: bold;">{sp_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">BET SPREAD ({stats["betting_spread_correct"]}/{stats["betting_spread_total"]})</div></div>'
        else:
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.75rem;">BET SPREAD</div></div>'
        
        accuracy_html += '</div>'
        
        # Row 2: Additional stats
        accuracy_html += '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px;">'
        
        # Betting Model Totals accuracy
        if stats.get('betting_total_total', 0) > 0:
            tot_pct = stats['betting_total_accuracy'] * 100
            tot_color = '#4CAF50' if tot_pct >= 55 else '#FFA500' if tot_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {tot_color}; font-size: 1.2rem; font-weight: bold;">{tot_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">BET O/U ({stats["betting_total_correct"]}/{stats["betting_total_total"]})</div></div>'
        else:
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: #888; font-size: 1.2rem;">--</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">BET O/U</div></div>'
        
        # Legacy ATS
        if stats.get('ats_total', 0) > 0:
            ats_pct = stats['ats_accuracy'] * 100
            ats_color = '#4CAF50' if ats_pct >= 55 else '#FFA500' if ats_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {ats_color}; font-size: 1.2rem; font-weight: bold;">{ats_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">OLD ATS ({stats["ats_correct"]}/{stats["ats_total"]})</div></div>'
        else:
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: #888; font-size: 1.2rem;">--</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">OLD ATS</div></div>'
        
        # Spread error
        accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: #00F3FF; font-size: 1.2rem; font-weight: bold;">{stats.get("avg_spread_error", 0):.1f}</div>'
        accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">SPREAD ERR</div></div>'
        
        # Total games tracked
        accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: #00F3FF; font-size: 1.2rem; font-weight: bold;">{stats["completed_games"]}</div>'
        accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">GAMES TRACKED</div></div>'
        
        accuracy_html += '</div>'
        
        # Recent picks
        if stats.get('recent_picks'):
            accuracy_html += '<div style="color: #888; font-size: 0.8rem; margin-top: 10px;">Recent: '
            recent_str = []
            for pick in stats['recent_picks'][:5]:
                emoji = '✅' if pick['correct'] else '❌'
                recent_str.append(f'{emoji} {pick["predicted"]}')
            accuracy_html += ' | '.join(recent_str)
            accuracy_html += '</div>'
        
        accuracy_html += '</div>'
        st.markdown(accuracy_html, unsafe_allow_html=True)
    else:
        # Show "collecting data" message
        st.markdown("""
        <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-family: JetBrains Mono; color: #FFA500; font-size: 0.9rem;">
                📊 TRACKING ACCURACY - Predictions are being saved. Check back tomorrow after games complete for real stats.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load betting model FIRST for accurate confidence picks
    betting_model = BettingModel()
    betting_loaded = betting_model.load()
    
    # Combine data first (needed for confidence summary)
    combined = games_df.copy()
    combined = combined.join(predictions_df)
    
    # Get BETTING MODEL confidence for each game (not base model!)
    betting_picks = {}
    if betting_loaded:
        for idx, game in combined.iterrows():
            if idx in all_features.index:
                rec = betting_model.get_betting_recommendation(all_features.loc[idx])
                if rec:
                    # Use betting model ML confidence (more accurate than base model)
                    ml_conf = rec.get('ml_confidence', 0.5)
                    ml_pick = rec.get('ml_pick', 'HOME')
                    winner = game['home_team_name'] if ml_pick == 'HOME' else game['visitor_team_name']
                    betting_picks[idx] = {'winner': winner, 'conf': ml_conf, 'rec': rec}
    
    # Sort by BETTING MODEL confidence
    combined['betting_conf'] = combined.index.map(lambda x: betting_picks.get(x, {}).get('conf', 0.5))
    combined = combined.sort_values('betting_conf', ascending=False)
    
    # Count high confidence picks using BETTING MODEL confidence
    lock_picks = []
    high_conf_picks = []
    good_picks = []
    
    for idx, game in combined.iterrows():
        if idx in betting_picks:
            bp = betting_picks[idx]
            conf = bp['conf']
            winner = bp['winner']
            
            # Higher thresholds for betting model (it's more calibrated)
            if conf >= 0.70:
                lock_picks.append((winner, conf))
            elif conf >= 0.62:
                high_conf_picks.append((winner, conf))
            elif conf >= 0.58:
                good_picks.append((winner, conf))
    
    # Show confidence summary
    if lock_picks or high_conf_picks or good_picks:
        summary_html = '<div style="background: linear-gradient(90deg, rgba(0,243,255,0.1), rgba(255,0,255,0.1)); border: 1px solid #00F3FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">'
        summary_html += '<div style="font-family: JetBrains Mono; color: #00F3FF; font-size: 0.9rem; margin-bottom: 10px;">📊 BETTING MODEL PICKS (ML Confidence)</div>'
        
        if lock_picks:
            summary_html += '<div style="margin-bottom: 8px;"><span style="background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 3px 8px; font-weight: 900; border-radius: 4px;">🔒 LOCKS (70%+)</span> '
            summary_html += ', '.join([f'<span style="color: #FFD700; font-weight: bold;">{name} ({prob:.0%})</span>' for name, prob in lock_picks])
            summary_html += '</div>'
        
        if high_conf_picks:
            summary_html += '<div style="margin-bottom: 8px;"><span style="background: #00F3FF; color: #000; padding: 3px 8px; font-weight: bold; border-radius: 4px;">🔥 HIGH CONF (62%+)</span> '
            summary_html += ', '.join([f'<span style="color: #00F3FF;">{name} ({prob:.0%})</span>' for name, prob in high_conf_picks])
            summary_html += '</div>'
        
        if good_picks:
            summary_html += '<div><span style="background: #4CAF50; color: #FFF; padding: 3px 8px; border-radius: 4px;">✓ VALUE (58%+)</span> '
            summary_html += ', '.join([f'<span style="color: #4CAF50;">{name} ({prob:.0%})</span>' for name, prob in good_picks])
            summary_html += '</div>'
        
        summary_html += '</div>'
        st.markdown(summary_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <div style="color: #FFA500; font-size: 0.85rem;">⚠️ No high-confidence betting model picks today. Consider sitting out or betting small.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show betting model stats if loaded
    if betting_loaded and betting_model.results:
        ml_acc = betting_model.results.get('moneyline', {}).get('high_conf_accuracy', 0) * 100
        spread_acc = betting_model.results.get('spread', {}).get('confident_accuracy', 0) * 100
        total_acc = betting_model.results.get('totals', {}).get('confident_accuracy', 0) * 100
        
        st.markdown(f"""
        <div style="background: #1a1a2e; border: 1px solid #4CAF50; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <div style="font-family: JetBrains Mono; color: #4CAF50; font-size: 0.85rem;">
                🎯 BETTING MODEL LOADED | ML: {ml_acc:.0f}% | Spread: {spread_acc:.0f}% | Totals: {total_acc:.0f}% (confident picks)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display each game
    for idx, game in combined.iterrows():
        # Get specific box scores for this game
        game_box_scores = None
        if box_scores_df is not None and not box_scores_df.empty:
            game_box_scores = box_scores_df[box_scores_df['game_id'] == game['id']]
        
        # Get betting recommendation from model
        betting_rec = None
        if betting_model.loaded and idx in all_features.index:
            game_features = all_features.loc[idx]
            vegas_odds = get_consensus_odds(odds_df, game.get('id'))
            betting_rec = betting_model.get_betting_recommendation(
                game_features,
                vegas_odds.get('spread_home') if vegas_odds else None,
                vegas_odds.get('total') if vegas_odds else None
            )
            
        display_game_card(game, odds_df, all_features.loc[idx] if idx in all_features.index else {}, injuries, game_box_scores, betting_rec)
        st.markdown("---")



def display_game_card(game, odds_df, features, injuries=None, box_scores=None, betting_rec=None):
    """Display a single game prediction card with Pro Grid Layout"""
    
    # 1. Prepare Data
    home_id = game['home_team_id']
    visitor_id = game['visitor_team_id']
    home_name = team_logos.get_team_name(home_id)
    visitor_name = team_logos.get_team_name(visitor_id)
    home_logo = team_logos.get_team_logo(home_id)
    visitor_logo = team_logos.get_team_logo(visitor_id)
    
    home_prob = game['home_win_probability']
    visitor_prob = game['visitor_win_probability']
    
    # NOTE: Injury impact is now handled natively by the model through trained
    # injury features (injury_impact_diff, home_injury_impact, etc.).
    # No manual penalty adjustment needed - the model learns the correct weight.
    
    # Vegas Data
    game_id = game.get('id')
    vegas_odds = get_consensus_odds(odds_df, game_id) if game_id else None
    
    # Helper function for safe feature access
    def safe_get(obj, key, default):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            elif hasattr(obj, 'get'):
                val = obj.get(key, default)
                return val if val is not None else default
            else:
                return default
        except (TypeError, KeyError, AttributeError):
            return default
    
    # 2. Badge Logic - Use BETTING MODEL predictions
    badges_html = ""
    
    if betting_rec:
        ml_conf = betting_rec.get('ml_confidence', 0.5)
        spread_conf = betting_rec.get('spread_confidence', 0.5)
        total_conf = betting_rec.get('total_confidence', 0.5)
        
        # Moneyline recommendation - show if any confidence
        if ml_conf >= 0.55:
            ml_pick = home_name if betting_rec.get('ml_pick') == 'HOME' else visitor_name
            badge_class = "badge-pro" if ml_conf >= 0.65 else "badge-outline"
            fire = "🔥" if ml_conf >= 0.65 else ""
            badges_html += f'<div class="{badge_class}">{fire} ML: {ml_pick} ({ml_conf:.0%})</div>'
        
        # Spread recommendation  
        if spread_conf >= 0.53 and vegas_odds and vegas_odds.get('has_odds'):
            spread_pick = home_name if betting_rec.get('spread_pick') == 'HOME' else visitor_name
            vegas_spread = vegas_odds.get('spread_home', 0)
            if betting_rec.get('spread_pick') == 'HOME':
                spread_line = f"{vegas_spread:+.1f}" if vegas_spread else ""
            else:
                # Away team's spread is the negative of home's spread
                away_spread = -vegas_spread if vegas_spread else 0
                spread_line = f"{away_spread:+.1f}" if vegas_spread else ""
            badge_class = "badge-pro" if spread_conf >= 0.58 else "badge-outline"
            badges_html += f'<div class="{badge_class}">SPREAD: {spread_pick} {spread_line} ({spread_conf:.0%})</div>'
        
        # Total recommendation
        if total_conf >= 0.53 and vegas_odds and vegas_odds.get('has_odds'):
            total_pick = betting_rec.get('total_pick', 'OVER')
            vegas_total = vegas_odds.get('total', 0)
            badge_class = "badge-pro" if total_conf >= 0.58 else "badge-outline"
            badges_html += f'<div class="{badge_class}">{total_pick} {vegas_total:.1f} ({total_conf:.0%})</div>'

    # 3. Stats Logic - Use sensible defaults if features missing
    h_win_pct = safe_get(features, 'home_win_pct_last10', 0.5)
    v_win_pct = safe_get(features, 'visitor_win_pct_last10', 0.5)
    h_ppg = safe_get(features, 'home_points_scored_last10', 110)
    v_ppg = safe_get(features, 'visitor_points_scored_last10', 110)
    
    # If values are 0 or very low, use league averages
    if h_win_pct == 0:
        h_win_pct = 0.5
    if v_win_pct == 0:
        v_win_pct = 0.5
    if h_ppg < 80:
        h_ppg = 110
    if v_ppg < 80:
        v_ppg = 110
    

    # 4. Status/Time/Score Logic
    status = game.get('status', '')
    time_str = game.get('time', '')
    
    # Timezone Fix: Convert UTC string to Eastern Time (NBA standard)
    game_time_display = status
    if 'T' in status and 'Z' in status:
        try:
            from zoneinfo import ZoneInfo
            dt_utc = datetime.strptime(status, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
            dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
            game_time_display = dt_et.strftime("%I:%M %p ET")
        except Exception:
            try:
                # Fallback: naive UTC - 5 for EST
                dt = datetime.strptime(status, "%Y-%m-%dT%H:%M:%SZ")
                dt_et = dt - timedelta(hours=5)
                game_time_display = dt_et.strftime("%I:%M %p ET")
            except Exception:
                pass
            
    # If game is live/finished, show QTR/FINAL
    # If game is scheduled, show Time
    is_live_or_finished = False
    if game.get('home_team_score', 0) > 0 or game.get('visitor_team_score', 0) > 0:
        is_live_or_finished = True
        
    v_score = game.get('visitor_team_score', '') if is_live_or_finished else ''
    h_score = game.get('home_team_score', '') if is_live_or_finished else ''
    
    # WINNER LOGIC - Use BETTING MODEL confidence (not base model!)
    v_winner_badge = ""
    h_winner_badge = ""
    
    # Override display probabilities with betting model when available
    display_home_prob = home_prob
    display_visitor_prob = visitor_prob
    
    # Get confidence from BETTING MODEL if available
    if betting_rec and betting_rec.get('ml_confidence'):
        ml_conf = betting_rec['ml_confidence']
        ml_pick = betting_rec.get('ml_pick', 'HOME')
        predicted_winner_is_home = (ml_pick == 'HOME')
        
        # Override displayed probabilities with betting model values
        if predicted_winner_is_home:
            display_home_prob = ml_conf
            display_visitor_prob = 1 - ml_conf
        else:
            display_visitor_prob = ml_conf
            display_home_prob = 1 - ml_conf
        
        # Determine confidence level from betting model
        if ml_conf >= 0.70:
            conf_level = "LOCK"
            badge_style = 'background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-weight: 900;'
            badge_text = '🔒 LOCK PICK'
        elif ml_conf >= 0.62:
            conf_level = "HIGH"
            badge_style = 'background: var(--neon-cyan); color: #000; font-weight: 900;'
            badge_text = '🔥🔥🔥 HIGH CONFIDENCE'
        elif ml_conf >= 0.58:
            conf_level = "GOOD"
            badge_style = 'background: #4CAF50; color: #FFF;'
            badge_text = '🔥🔥 GOOD VALUE'
        else:
            conf_level = "NORMAL"
            badge_style = 'background: rgba(255,255,255,0.1); color: #AAA;'
            badge_text = 'PREDICTED'
    else:
        # Fallback to base model when betting model not available
        display_home_prob = home_prob
        display_visitor_prob = visitor_prob
        predicted_winner_is_home = (home_prob > visitor_prob)
        badge_style = 'background: rgba(255,255,255,0.1); color: #AAA;'
        badge_text = 'PREDICTED'
    
    # Always assign badge to predicted winner
    if predicted_winner_is_home:
        h_winner_badge = f'<div class="winner-tag" style="{badge_style}">{badge_text}</div>'
    else:
        v_winner_badge = f'<div class="winner-tag" style="{badge_style}">{badge_text}</div>'

    # NOTE: Injury impact is handled by the trained model's injury features.
    # No manual override layer needed - removes double/triple counting risk.

    # Vegas Strings for Display
    v_spread_str = "N/A"
    h_spread_str = "N/A"
    v_ml_str = "N/A"
    h_ml_str = "N/A"
    total_str = "N/A"
    has_vegas_data = False
    
    if vegas_odds and vegas_odds.get('has_odds'):
        # Get raw values
        h_spread_val = vegas_odds['spread_home']
        v_spread_val = vegas_odds['spread_away']
        h_ml_val = vegas_odds.get('moneyline_home')
        v_ml_val = vegas_odds.get('moneyline_away')
        total_val = vegas_odds['total']
        
        # sanity check removed to show true Vegas lines comparison
        # model_says_home_favored = home_prob > 0.5
        # vegas_says_home_favored = ...
        # if disagreement, swap... RE MOVED to ensure authenticity.
        
        # Format for display
        h_spread_str = f"{h_spread_val:+.1f}" if h_spread_val else "PK"
        v_spread_str = f"{v_spread_val:+.1f}" if v_spread_val else "PK"
        h_ml_str = format_american_odds(h_ml_val)
        v_ml_str = format_american_odds(v_ml_val)
        total_str = f"{total_val}"
        has_vegas_data = True

    # 5. Construct Grid HTML - FLATTENED TO PREVENT RAW CODE RENDERING
    card_html = f"""
<div class="game-card">
<div class="action-bar">
{badges_html if badges_html else '<div class="badge-outline" style="border:none; color:#444">Analysis Ready</div>'}
</div>
<div class="game-grid">
<div class="team-cell">
<img src="{visitor_logo}" class="team-logo-img">
<div class="game-score-lg" style="margin: 5px 0; color: #FFFFFF;">{v_score}</div>
<div class="team-name">{visitor_name}</div>
<div class="win-prob">{display_visitor_prob:.0%}</div>
{v_winner_badge}
</div>
<div class="vs-cell">
<div style="font-size:1.5rem; font-weight:900; color:#555; margin-bottom:5px;">VS</div>
<div class="game-time" style="color:#00F3FF; font-weight:bold;">{game_time_display}</div>
{f'<div style="margin-top:10px; font-size:0.8rem; color:#888;">VEGAS TOTAL</div><div style="color:#FFF; font-weight:bold;">{total_str}</div>' if has_vegas_data else ''}
</div>
<div class="team-cell">
<img src="{home_logo}" class="team-logo-img">
<div class="game-score-lg" style="margin: 5px 0; color: #FFFFFF;">{h_score}</div>
<div class="team-name">{home_name}</div>
<div class="win-prob">{display_home_prob:.0%}</div>
{h_winner_badge}
</div>
</div>
<div class="stats-container">
<table class="pro-table">
{'<tr><td style="color: #00FF88; font-weight:bold; font-size: 1.1rem;">' + v_ml_str + '</td><td class="label-cell" style="color: #00FF88;">VEGAS ML</td><td style="color: #00FF88; font-weight:bold; font-size: 1.1rem;">' + h_ml_str + '</td></tr><tr><td style="color: #F3BC00; font-weight:bold;">' + v_spread_str + '</td><td class="label-cell" style="color: #F3BC00;">VEGAS SPREAD</td><td style="color: #F3BC00; font-weight:bold;">' + h_spread_str + '</td></tr>' if has_vegas_data else '<tr><td colspan="3" style="color: #666; text-align: center; font-style: italic;">Vegas odds not available</td></tr>'}
<tr>
<td>{v_win_pct:.0%}</td>
<td class="label-cell">L10 Win %</td>
<td>{h_win_pct:.0%}</td>
</tr>
<tr>
<td>{v_ppg:.1f}</td>
<td class="label-cell">L10 PPG</td>
<td>{h_ppg:.1f}</td>
</tr>
<tr>
<td>{safe_get(features, 'visitor_rest_days', 2):.0f}</td>
<td class="label-cell">Rest Days</td>
<td>{safe_get(features, 'home_rest_days', 2):.0f}</td>
</tr>
</table>
</div>
</div>
"""
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 6. Expanders (Box Scores / Analysis)
    if box_scores is not None and not box_scores.empty:
        with st.expander("📊 Box Scores"):
            h_scores = box_scores[box_scores['team_id'] == home_id].sort_values('pts', ascending=False)
            v_scores = box_scores[box_scores['team_id'] == visitor_id].sort_values('pts', ascending=False)
            
            cols = ['first_name', 'last_name', 'pts', 'reb', 'ast', 'min']
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption(visitor_name)
                st.dataframe(v_scores[cols], hide_index=True)
            with c2:
                st.caption(home_name)
                st.dataframe(h_scores[cols], hide_index=True)





    # Detailed Analysis
    with st.expander(f"🔍 Elite Analysis & Betting Breakdown"):
        # We need to reconstruct the prediction dict expected by generate_elite_analysis
        pred_dict = {
            'home_win_probability': home_prob,
            'predicted_spread': game['predicted_spread'],
            'predicted_total': game['predicted_total']
        }
        analysis_text = generate_elite_analysis(game, pred_dict, features, vegas_odds, injuries)
        # FLATTEN HTML: Regex to strip leading whitespace from every line (fixing code block render)
        analysis_text = re.sub(r'(?m)^\s+', '', analysis_text)
        
        # Wrap in div for readability styling
        st.markdown(f'<div class="analysis-text">{analysis_text}</div>', unsafe_allow_html=True)



def main():
    """Main Streamlit app"""
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Date picker (defaults to today)
        target_date = st.date_input(
            "Select Game Date",
            value=datetime.now().date(),
            min_value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date() + timedelta(days=30)
        )
        
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        # Google Sheets Integration Helper
        try:
            with open("google_sheets_integration.js", "r") as f:
                js_code = f.read()
            st.download_button(
                label="📥 Google Sheets Script",
                data=js_code,
                file_name="google_sheets_integration.js",
                mime="text/javascript",
                help="Click to download the script. Copy/Paste into Google Sheets > Extensions > Apps Script."
            )
        except Exception:
            pass # File might not exist in some envs
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Info")
        model = load_model()
        
        if model and model.training_info:
            # Stats (Handle both Elite Hybrid and Legacy formats)
            info = model.training_info
            
            # 1. Accuracy
            if 'hybrid_accuracy' in info:
                acc = info['hybrid_accuracy']
                st.metric("Hybrid Accuracy", f"{acc:.1%}")
            elif 'metrics' in info and 'winner_test_accuracy' in info['metrics']:
                acc = info['metrics']['winner_test_accuracy']
                st.metric("Test Accuracy", f"{acc:.1%}")
            else:
                st.metric("Accuracy", "N/A")
                
            # 2. MAE / Loss
            if 'metrics' in info:
                # Legacy or Hybrid with metrics dict
                if 'home_score_test_mae' in info['metrics']:
                    st.metric("Score MAE", f"{info['metrics']['home_score_test_mae']:.1f} pts")
            elif 'lstm_accuracy' in info:
                # Elite model format
                st.metric("LSTM Accuracy", f"{info['lstm_accuracy']:.1%}")
                
            st.caption(f"Trained: {info.get('trained_at', 'Unknown')[:16]}")
        else:
            st.warning("⚠️ Model not trained yet!")
            if st.button("Train Model Now"):
                with st.spinner("Training model... This may take several minutes..."):
                    import train_model
                    train_model.main()
                    st.success("Model trained successfully!")
                    st.rerun()
        
        st.markdown("---")
        
        # Training history
        st.subheader("📈 Training History")
        try:
            from training_history import TrainingHistoryTracker, format_improvement_display
            tracker = TrainingHistoryTracker()
            improvement = tracker.get_improvement_stats()
            
            if improvement.get('has_improvement'):
                st.markdown(format_improvement_display(improvement))
                
                # Show chart of data growth
                sessions = tracker.get_all_sessions()
                if len(sessions) > 1:
                    # pd already imported at top of file
                    df_history = pd.DataFrame([
                        {
                            'Training #': i+1,
                            'Total Games': s['total_samples'],
                            'Accuracy %': s['test_accuracy'] * 100,
                            'Date': s['timestamp'][:10]
                        }
                        for i, s in enumerate(sessions)
                    ])
                    
                    with st.expander("📊 View Training Chart", expanded=True):
                        tab1, tab2 = st.tabs(["Data Growth", "Accuracy"])
                        
                        with tab1:
                            st.area_chart(df_history.set_index('Training #')[['Total Games']], color="#00F3FF")
                            st.caption(f"Latest: {df_history['Total Games'].iloc[-1]:,} games")
                            
                        with tab2:
                            st.line_chart(df_history.set_index('Training #')[['Accuracy %']], color="#FF00FF")
                            st.caption(f"Latest: {df_history['Accuracy %'].iloc[-1]:.1f}%")
            else:
                st.info("📊 Train the model multiple times to track improvement")
        except Exception as e:
            st.caption("Training history will appear after first retrain")
        
        st.markdown("---")
        
        # Debug / Cache
        st.subheader("🛠️ Debug")
        if st.button("🧹 Clear All Cache", help="Force reload of models and data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Retrain buttons
        st.subheader("🔄 Model Updates")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train Base Model", help="Train winner prediction model"):
                with st.spinner("Training base model..."):
                    import train_model
                    train_model.main()
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("✅ Base model trained!")
                    st.rerun()
        
        with col2:
            if st.button("Train Betting Model", help="Train ML/Spread/Total models"):
                with st.spinner("Training betting models..."):
                    import train_betting_model
                    train_betting_model.main()
                    st.cache_data.clear()
                    st.success("✅ Betting models trained!")
                    st.rerun()
        
        st.markdown("---")
        
        # Update predictions button
        st.subheader("📈 Predictions")
        if st.button("🔄 Update Predictions", help="Refresh predictions with latest data"):
            st.cache_data.clear()
            st.success("✅ Predictions updated!")
            st.rerun()
    
    # Main content
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model using the sidebar.")
        return
    
    # Fetch data
    with st.spinner(f"Loading games for {target_date_str}..."):
        games_df = get_todays_games(target_date_str)
        
        if games_df.empty:
            st.warning(f"No games found for {target_date_str}")
            return
        
        # Get odds (pass games_df for fallback matching)
        odds_df = get_vegas_odds(target_date_str, games_df)
        


        # Get box scores (for finished/active games)
        box_scores_df = get_box_scores(target_date_str)
        if not box_scores_df.empty:
            count = len(box_scores_df['game_id'].unique())
            st.markdown(f"""
            <div style="
                background-color: #0E1117; 
                border: 1px solid #00F3FF; 
                color: #00F3FF; 
                padding: 10px; 
                border-radius: 4px; 
                font-family: 'JetBrains Mono', monospace; 
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;">
                <span>✅</span> Fetched box scores for {count} games
            </div>
            """, unsafe_allow_html=True)
        
        # =====================================================================
        # CACHED PREDICTION PIPELINE
        # All heavy computation (ELO, features, predictions) is cached in
        # session_state so Streamlit reruns don't repeat the entire pipeline.
        # =====================================================================
        prediction_cache_key = f"predictions_{target_date_str}"
        
        if prediction_cache_key not in st.session_state:
            with st.spinner("Building predictions (ELO, features, model)... This runs once."):
                data_mgr = DataManager()
                feature_eng = FeatureEngineer()
                
                # Get historical data for context
                current_season = target_date.year if target_date.month >= 10 else target_date.year - 1
                
                historical_data = data_mgr.get_complete_training_data([current_season - 1, current_season])
                
                # Fetch RECENT games (last 14 days) fresh from API for accurate rest day calculations
                recent_dates = [(target_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 15)]
                try:
                    recent_games = data_mgr.client.get_games(dates=recent_dates, per_page=100)
                    if recent_games:
                        recent_df = pd.DataFrame(recent_games)
                        if not recent_df.empty:
                            # Flatten team data
                            recent_df['home_team_id'] = recent_df['home_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
                            recent_df['visitor_team_id'] = recent_df['visitor_team'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
                            recent_df['home_team_score'] = recent_df.get('home_team_score', 0)
                            recent_df['visitor_team_score'] = recent_df.get('visitor_team_score', 0)
                            recent_df['season'] = current_season
                            
                            # Filter to completed games only
                            recent_df = recent_df[recent_df['status'] == 'Final'].copy()
                            
                            # Merge with historical data (remove duplicates by game id)
                            if 'games' in historical_data and not historical_data['games'].empty:
                                existing_ids = set(historical_data['games']['id'].tolist()) if 'id' in historical_data['games'].columns else set()
                                new_games = recent_df[~recent_df['id'].isin(existing_ids)]
                                if not new_games.empty:
                                    historical_data['games'] = pd.concat([historical_data['games'], new_games], ignore_index=True)
                                    print(f"Added {len(new_games)} recent games for accurate rest days")
                except Exception as e:
                    print(f"Could not fetch recent games: {e}")
                
                # Pre-calculate ELO ratings on historical data so build_features_for_game
                # can look up real ELO values instead of defaulting to 1500 for everyone.
                if 'games' in historical_data and not historical_data['games'].empty:
                    hist_games = historical_data['games']
                    if 'home_elo' not in hist_games.columns:
                        if 'game_id' not in hist_games.columns and 'id' in hist_games.columns:
                            hist_games = hist_games.rename(columns={'id': 'game_id'})
                            historical_data['games'] = hist_games
                        try:
                            elo_df = feature_eng._calculate_elo(hist_games)
                            historical_data['games'] = pd.merge(hist_games, elo_df, on='game_id', how='left')
                            print(f"Pre-calculated ELO ratings for {len(elo_df)} historical games")
                        except Exception as e:
                            print(f"Warning: Could not pre-calculate ELO: {e}")
                
                # Fetch injuries for enhanced features
                injuries = get_injuries()
                
                # Build features for each game with enhanced features (Vegas, injuries, H2H)
                all_features = {}
                for idx, game in games_df.iterrows():
                    try:
                        features = feature_eng.build_features_for_game(
                            game.to_dict(),
                            historical_data,
                            current_season,
                            injuries=injuries,
                            odds_df=odds_df,
                            standings=historical_data.get('standings', pd.DataFrame()),
                            player_stats=None
                        )
                        
                        if features is None:
                            print(f"WARNING: Feature building returned None for game {game.get('id')} - using neutral defaults")
                            features = {
                                'home_elo': 1500, 'visitor_elo': 1500, 'elo_diff': 0,
                                'home_win_pct_last10': 0.5, 'visitor_win_pct_last10': 0.5,
                                'home_points_scored_last10': 110, 'visitor_points_scored_last10': 110,
                                'home_rest_days': 2, 'visitor_rest_days': 2,
                                'rest_advantage': 0, 'momentum_diff_5': 0, 'momentum_diff_10': 0,
                                'net_rating_diff': 0,
                                'vegas_spread_home': 0.0, 'vegas_total': 220.0,
                                'vegas_implied_home_prob': 0.5, 'vegas_has_odds': 0,
                                'h2h_home_win_pct': 0.5, 'h2h_avg_margin': 0, 'h2h_last3_home_wins': 0.5,
                                'injury_impact_diff': 0.0, 'home_injury_impact': 0.0, 'visitor_injury_impact': 0.0,
                            }
                            features['_using_defaults'] = True
                        
                        all_features[idx] = features
                        
                    except Exception as e:
                        import traceback
                        print(f"ERROR: Feature building failed for game {game.get('id')}: {str(e)}")
                        print(traceback.format_exc())
                        
                        defaults = {
                            'home_elo': 1500, 'visitor_elo': 1500, 'elo_diff': 0,
                            'home_rest_days': 2, 'visitor_rest_days': 2,
                            'rest_advantage': 0, 'momentum_diff_5': 0, 'momentum_diff_10': 0,
                            'net_rating_diff': 0,
                            'home_is_b2b': 0, 'visitor_is_b2b': 0,
                            'home_is_3in4': 0, 'visitor_is_3in4': 0,
                            'home_is_4in5': 0, 'visitor_is_4in5': 0,
                            'vegas_spread_home': 0.0, 'vegas_total': 220.0,
                            'vegas_implied_home_prob': 0.5, 'vegas_has_odds': 0,
                            'home_injuries_out': 0, 'visitor_injuries_out': 0,
                            'home_questionable': 0, 'visitor_questionable': 0,
                            'injury_diff': 0, 'home_stars_out': 0, 'visitor_stars_out': 0,
                            'star_injury_diff': 0, 'home_injury_impact': 0.0,
                            'visitor_injury_impact': 0.0, 'injury_impact_diff': 0.0,
                            'h2h_games': 0, 'h2h_home_wins': 0, 'h2h_home_win_pct': 0.5,
                            'h2h_avg_margin': 0, 'h2h_last3_home_wins': 0.5,
                            'visitor_travel_miles': 0, 'visitor_tz_change': 0, 'is_long_travel': 0,
                            'season_phase': 2, 'is_late_season': 0,
                            'home_motivation': 2, 'visitor_motivation': 2, 'motivation_diff': 0,
                            '_using_defaults': True,
                        }
                        for w in [5, 10, 20]:
                            for prefix in ['home', 'visitor']:
                                defaults[f'{prefix}_win_pct_last{w}'] = 0.5
                                defaults[f'{prefix}_points_scored_last{w}'] = 110
                                defaults[f'{prefix}_points_allowed_last{w}'] = 110
                                defaults[f'{prefix}_point_diff_last{w}'] = 0
                                defaults[f'{prefix}_pace_last{w}'] = 98
                                defaults[f'{prefix}_efg_pct_last{w}'] = 0.54
                                defaults[f'{prefix}_tov_pct_last{w}'] = 0.13
                                defaults[f'{prefix}_oreb_pct_last{w}'] = 0.25
                                defaults[f'{prefix}_ftr_last{w}'] = 0.20
                                defaults[f'{prefix}_opp_efg_pct_last{w}'] = 0.54
                                defaults[f'{prefix}_opp_tov_pct_last{w}'] = 0.13
                                defaults[f'{prefix}_opp_oreb_pct_last{w}'] = 0.25
                                defaults[f'{prefix}_opp_ftr_last{w}'] = 0.20
                        all_features[idx] = defaults
                
                if not all_features:
                    st.error("Could not generate features for any games. Please check data availability.")
                    return
                
                features_df = pd.DataFrame.from_dict(all_features, orient='index')
                predictions_df = model.predict(features_df)
                
                # Cache everything in session_state
                st.session_state[prediction_cache_key] = {
                    'all_features': all_features,
                    'features_df': features_df,
                    'predictions_df': predictions_df,
                }
                print(f"Predictions cached for {target_date_str}")
        else:
            print(f"Using cached predictions for {target_date_str}")
        
        # Retrieve from cache
        cached = st.session_state[prediction_cache_key]
        all_features = cached['all_features']
        features_df = cached['features_df']
        predictions_df = cached['predictions_df']
        
        # Initialize tracker and save predictions
        tracker = PredictionTracker()
        data_mgr = DataManager()
        
        # Check past games for results (updates accuracy tracking)
        past_games = data_mgr.client.get_games(
            dates=[(target_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)],
            per_page=100
        )
        if past_games:
            past_df = pd.DataFrame(past_games)
            past_df = past_df[past_df['status'] == 'Final']
            for _, pg in past_df.iterrows():
                tracker.update_result(
                    pg.get('id'),
                    pg.get('home_team_score', 0),
                    pg.get('visitor_team_score', 0)
                )
        
        # Load betting model for saving picks
        from betting_model import BettingModel
        betting_model_tracker = BettingModel()
        betting_model_tracker.load()
        
        # Save today's predictions
        for idx, game in games_df.iterrows():
            if idx in predictions_df.index:
                pred = predictions_df.loc[idx]
                home_prob = pred.get('home_win_probability', 0.5)
                visitor_prob = pred.get('visitor_win_probability', 0.5)
                
                # Get Vegas odds for this game
                game_odds = get_consensus_odds(odds_df, game.get('id'))
                
                # Get betting model recommendation
                betting_rec = None
                if betting_model_tracker.loaded and idx in features_df.index:
                    betting_rec = betting_model_tracker.get_betting_recommendation(
                        features_df.loc[idx],
                        game_odds.get('spread_home') if game_odds else None,
                        game_odds.get('total') if game_odds else None
                    )
                
                tracker.save_prediction(game.get('id'), {
                    "home_team": game.get('home_team_name'),
                    "visitor_team": game.get('visitor_team_name'),
                    "home_prob": home_prob,
                    "visitor_prob": visitor_prob,
                    "predicted_winner": game.get('home_team_name') if home_prob > 0.5 else game.get('visitor_team_name'),
                    "predicted_home_score": pred.get('predicted_home_score', 110),
                    "predicted_visitor_score": pred.get('predicted_visitor_score', 110),
                    "predicted_spread": pred.get('predicted_spread', 0),
                    "predicted_total": pred.get('predicted_total', 220),
                    "confidence": max(home_prob, visitor_prob),
                    "vegas_spread": game_odds['spread_home'] if game_odds and game_odds['has_odds'] else None,
                    "vegas_total": game_odds['total'] if game_odds and game_odds['has_odds'] else None,
                    # NEW: Save betting model picks
                    "betting_ml_pick": betting_rec.get('ml_pick') if betting_rec else None,
                    "betting_ml_conf": betting_rec.get('ml_confidence') if betting_rec else None,
                    "betting_spread_pick": betting_rec.get('spread_pick') if betting_rec else None,
                    "betting_spread_conf": betting_rec.get('spread_confidence') if betting_rec else None,
                    "betting_total_pick": betting_rec.get('total_pick') if betting_rec else None,
                    "betting_total_conf": betting_rec.get('total_confidence') if betting_rec else None,
                })
        
        # Get real accuracy stats
        accuracy_stats = tracker.get_accuracy_stats(days=30)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["🏀 Game Predictions", "🎯 Player Props"])
        
        with tab1:
            # Display predictions with real accuracy
            display_game_predictions(games_df, predictions_df, odds_df, features_df, box_scores_df, accuracy_stats)
        
        with tab2:
            # Display Player Props from The Odds API
            st.markdown("""
            <div style="background: linear-gradient(90deg, rgba(0,243,255,0.1), rgba(255,165,0,0.1)); 
                        border: 1px solid #00F3FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-family: JetBrains Mono; color: #00F3FF; font-size: 1.1rem; font-weight: bold;">
                    🎯 NBA Player Props
                </div>
                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                    Live betting lines from The Odds API
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Loading player props..."):
                props = get_player_props()
            
            if props:
                # Convert to DataFrame
                props_df = pd.DataFrame(props)
                
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    prop_types = ['All'] + sorted(props_df['prop_type'].unique().tolist())
                    selected_type = st.selectbox("Prop Type", prop_types)
                with col2:
                    players = ['All'] + sorted(props_df['player'].unique().tolist())
                    selected_player = st.selectbox("Player", players)
                with col3:
                    books = ['All'] + sorted(props_df['bookmaker'].unique().tolist())
                    selected_book = st.selectbox("Bookmaker", books)
                
                # Apply filters
                filtered_df = props_df.copy()
                if selected_type != 'All':
                    filtered_df = filtered_df[filtered_df['prop_type'] == selected_type]
                if selected_player != 'All':
                    filtered_df = filtered_df[filtered_df['player'] == selected_player]
                if selected_book != 'All':
                    filtered_df = filtered_df[filtered_df['bookmaker'] == selected_book]
                
                # Display count
                st.markdown(f"""
                <div style="color: #888; font-size: 0.85rem; margin-bottom: 10px;">
                    Showing {len(filtered_df)} props
                </div>
                """, unsafe_allow_html=True)
                
                # Display as styled cards
                for _, prop in filtered_df.iterrows():
                    over_odds = prop.get('over_odds', 'N/A')
                    under_odds = prop.get('under_odds', 'N/A')
                    
                    # Format odds with color
                    over_color = "#4CAF50" if over_odds and over_odds != 'N/A' else "#666"
                    under_color = "#FF5722" if under_odds and under_odds != 'N/A' else "#666"
                    
                    over_str = f"{int(over_odds):+d}" if isinstance(over_odds, (int, float)) and over_odds != 'N/A' and pd.notna(over_odds) else str(over_odds)
                    under_str = f"{int(under_odds):+d}" if isinstance(under_odds, (int, float)) and under_odds != 'N/A' and pd.notna(under_odds) else str(under_odds)
                    
                    st.markdown(f"""
                    <div style="background: #1a1a2e; border: 1px solid #333; border-radius: 8px; 
                                padding: 12px; margin-bottom: 8px; display: grid; 
                                grid-template-columns: 2fr 1fr 1fr 1fr 1fr; gap: 10px; align-items: center;">
                        <div>
                            <div style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{prop['player']}</div>
                            <div style="color: #888; font-size: 0.75rem;">{prop['game']}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #00F3FF; font-size: 0.75rem;">PROP</div>
                            <div style="color: #FFF; font-weight: bold;">{prop['prop_type']}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #FFA500; font-size: 0.75rem;">LINE</div>
                            <div style="color: #FFF; font-weight: bold; font-size: 1.1rem;">{prop['line']}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: {over_color}; font-size: 0.75rem;">OVER</div>
                            <div style="color: {over_color}; font-weight: bold;">{over_str}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: {under_color}; font-size: 0.75rem;">UNDER</div>
                            <div style="color: {under_color}; font-weight: bold;">{under_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show bookmaker info
                st.markdown(f"""
                <div style="color: #666; font-size: 0.75rem; margin-top: 15px; text-align: center;">
                    📊 Data from: {', '.join(props_df['bookmaker'].unique()[:5])} | Updated every 10 min
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; 
                            border-radius: 8px; text-align: center;">
                    <div style="color: #FFA500; font-size: 1rem;">⚠️ No Player Props Available</div>
                    <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
                        Player props may not be available if there are no games today or the API limit was reached.
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
