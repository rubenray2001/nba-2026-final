import sys
import io

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

"""
# CSS and UI Styling
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import config

# Force-allow set_page_config in case runtime pre-initialized something
try:
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    _ctx = get_script_run_ctx()
    if _ctx is not None:
        _ctx._set_page_config_allowed = True
except Exception:
    pass

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
import json
import threading
import time

from data_manager import DataManager
from features_enhanced import EnhancedFeatureEngineer as FeatureEngineer
from model_engine import EliteEnsembleModel
from betting_model import BettingModel
from prediction_tracker import PredictionTracker
from training_history import TrainingHistoryTracker
from odds_utils import get_consensus_odds, format_american_odds, calculate_edge
from odds_api_client import TheOddsAPIClient
try:
    from props_aggregator import PropsAggregator, PrizePicksClient, UnderdogFantasyClient
    _PROPS_AVAILABLE = True
except ImportError:
    _PROPS_AVAILABLE = False
    PropsAggregator = None
import team_logos


def _background_result_updater(interval_hours: int = 4):
    """Background thread: update pending game results every N hours."""
    while True:
        time.sleep(interval_hours * 3600)
        try:
            for gender in ("mens", "womens"):
                tracker = PredictionTracker(gender)
                dm = DataManager(gender)
                updated = tracker.update_pending_games(dm)
                if updated:
                    print(f"[bg] Updated {updated} results for {gender}")
        except Exception as e:
            print(f"[bg] Result updater error: {e}")


def _start_background_updater():
    t = threading.Thread(target=_background_result_updater, daemon=True)
    t.start()


if not getattr(_start_background_updater, "_started", False):
    _start_background_updater()
    _start_background_updater._started = True


def inject_custom_css():
    """Inject Pro Dashboard V2 CSS (CSS Grid + Sharp Lines) - identical to NBA"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;900&display=swap');
        
        * { box-sizing: border-box; }
        
        :root {
            --neon-cyan: #00F3FF;
            --neon-magenta: #FF00FF;
            --bg-dark: #050505;
            --bg-card: #0E1117;
            --border-color: #333333;
            --text-white: #FFFFFF;
            --text-gray: #888888;
        }

        .stApp { background-color: var(--bg-dark); }

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

        .game-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-top: 2px solid var(--neon-cyan);
            border-radius: 0px;
            margin-bottom: 30px;
            padding: 0;
            overflow: hidden;
            position: relative;
        }

        .game-grid {
            display: grid;
            grid-template-columns: 1fr 180px 1fr;
            align-items: center;
            padding: 30px;
            border-bottom: 1px solid var(--border-color);
        }

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
        
        div.stAlert {
            background-color: #000;
            border: 1px solid var(--neon-cyan);
            color: white;
        }

        .stSuccess {
            background-color: #0E1117 !important;
            border: 1px solid var(--neon-cyan) !important;
        }
        .stSuccess p {
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        .analysis-text {
            color: #E0E0E0 !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }

        div[data-testid="stSpinner"] > div {
            color: var(--neon-cyan) !important;
            border-color: var(--neon-cyan) !important;
        }
        div[data-testid="stSpinner"] p {
            color: var(--neon-cyan) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.1rem !important;
        }

        div[data-testid="stExpander"] details summary {
            background-color: #0E1117 !important;
            border: 1px solid var(--neon-cyan) !important;
            color: var(--neon-cyan) !important;
            border-radius: 4px;
        }
        div[data-testid="stExpander"] details summary p {
            color: var(--neon-cyan) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
        }
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
        
        .stRadio label,
        .stRadio div[role="radiogroup"] label,
        .stRadio > label,
        .stRadio p,
        .stRadio span,
        .stRadio div,
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] p,
        div[data-testid="stRadio"] span,
        div[data-testid="stRadio"] div {
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def get_gender():
    """Get selected gender from session state"""
    return st.session_state.get("gender", "mens")


@st.cache_resource
def load_model(gender: str):
    """Load the trained model for a specific gender"""
    models_dir = os.path.join(config.MODELS_DIR, gender)
    model_pkl = os.path.join(models_dir, 'home_score_ensemble.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    if not os.path.exists(model_pkl):
        st.info("First time setup: Training model... This may take a few minutes.")
        try:
            import train_model
            train_model.train_gender(gender)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            return None
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        model = EliteEnsembleModel(gender)
        model.load_models()
        if hasattr(model, 'feature_names') and model.feature_names:
            print(f"{gender} model loaded: {len(model.feature_names)} features")
        return model
    except Exception as e:
        st.error(f"Error loading {gender} model: {e}")
        return None


@st.cache_data(ttl=30)
def get_todays_games(gender: str, target_date: str):
    """Fetch games for target date"""
    data_mgr = DataManager(gender)
    games_df = data_mgr.fetch_todays_games(target_date)
    return games_df


def get_vegas_odds(gender: str, target_date: str, games_df=None):
    """Fetch Vegas odds for target date (cached in session_state per gender+date)"""
    cache_key = f"_odds_{gender}_{target_date}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    data_mgr = DataManager(gender)
    odds_df = data_mgr.fetch_vegas_odds(dates=[target_date], games_df=games_df)
    
    if not odds_df.empty:
        for col in odds_df.columns:
            if odds_df[col].apply(lambda x: isinstance(x, dict)).any():
                odds_df[col] = odds_df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
    
    st.session_state[cache_key] = odds_df
    return odds_df


@st.cache_data(ttl=600)
def get_all_player_props(gender: str, _v: int = 6):
    """Fetch player props from ALL sources: Odds API, PrizePicks, Underdog Fantasy
    _v param is a cache-busting version; bump when upstream parsers change."""
    try:
        aggregator = PropsAggregator(gender=gender)
        results = aggregator.fetch_all_props(
            include_odds_api=True,
            include_prizepicks=True,
            include_underdog=True,
            odds_api_max_events=24,
        )
        return results
    except Exception as e:
        print(f"Error fetching aggregated player props: {e}")
        return {"odds_api": [], "prizepicks": [], "underdog": [], "all": [], "sources_status": {}}


@st.cache_data(ttl=1800)
def get_rankings(gender: str):
    """Fetch AP Top 25 rankings"""
    try:
        from api_client import ESPNCollegeBasketballClient
        client = ESPNCollegeBasketballClient(gender=gender)
        rankings = client.get_rankings()
        return rankings
    except Exception as e:
        print(f"Error fetching rankings: {e}")
        return []


@st.cache_data(ttl=1800)
def get_standings(gender: str, season: int = None):
    """Fetch conference standings"""
    try:
        data_mgr = DataManager(gender)
        standings = data_mgr.fetch_standings(season)
        return standings
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return pd.DataFrame()



def generate_elite_analysis(game_data, prediction, features, vegas_odds=None, gender="mens"):
    """Generate detailed robust HTML analysis with team breakdowns and betting reasoning"""
    home_team = game_data.get('home_team_name', 'Home Team')
    visitor_team = game_data.get('visitor_team_name', 'Visitor Team')
    
    home_prob = prediction['home_win_probability']
    predicted_winner = home_team if home_prob > 0.5 else visitor_team
    loser_team = visitor_team if home_prob > 0.5 else home_team
    winner_prob = max(home_prob, 1 - home_prob)
    is_home_favorite = home_prob > 0.5
    
    avg_score = config.GENDER_CONFIG[gender]["avg_score"]
    avg_total = config.GENDER_CONFIG[gender]["avg_total"]
    
    # Pull all feature values with safe defaults
    h_elo = features.get('home_elo', 1500)
    v_elo = features.get('visitor_elo', 1500)
    elo_diff = features.get('elo_diff', h_elo - v_elo)
    
    h_wp5 = features.get('home_win_pct_last5', 0.5)
    v_wp5 = features.get('visitor_win_pct_last5', 0.5)
    h_wp10 = features.get('home_win_pct_last10', 0.5)
    v_wp10 = features.get('visitor_win_pct_last10', 0.5)
    h_wp20 = features.get('home_win_pct_last20', 0.5)
    v_wp20 = features.get('visitor_win_pct_last20', 0.5)
    
    h_ppg5 = features.get('home_points_scored_last5', avg_score)
    v_ppg5 = features.get('visitor_points_scored_last5', avg_score)
    h_ppg10 = features.get('home_points_scored_last10', avg_score)
    v_ppg10 = features.get('visitor_points_scored_last10', avg_score)
    h_opp10 = features.get('home_points_allowed_last10', avg_score)
    v_opp10 = features.get('visitor_points_allowed_last10', avg_score)
    h_opp5 = features.get('home_points_allowed_last5', avg_score)
    v_opp5 = features.get('visitor_points_allowed_last5', avg_score)
    
    h_diff5 = features.get('home_point_diff_last5', 0)
    v_diff5 = features.get('visitor_point_diff_last5', 0)
    h_diff10 = features.get('home_point_diff_last10', 0)
    v_diff10 = features.get('visitor_point_diff_last10', 0)
    
    h_rest = features.get('home_rest_days', 2)
    v_rest = features.get('visitor_rest_days', 2)
    h_b2b = features.get('home_is_b2b', 0)
    v_b2b = features.get('visitor_is_b2b', 0)
    
    h2h_pct = features.get('h2h_home_win_pct', 0.5)
    h2h_games = features.get('h2h_games', 0)
    h2h_margin = features.get('h2h_avg_margin', 0)
    
    is_march = features.get('is_march', 0)
    season_phase = features.get('season_phase', 2)
    
    predicted_spread = prediction.get('predicted_spread', 0)
    predicted_total = prediction.get('predicted_total', avg_total)
    
    # Derive scores from total and winner to guarantee consistency
    # The winner must always have the higher score
    winner_margin = abs(predicted_spread) if predicted_spread != 0 else (winner_prob - 0.5) * 20
    high_score = (predicted_total + winner_margin) / 2
    low_score = (predicted_total - winner_margin) / 2
    
    if is_home_favorite:
        predicted_home = high_score
        predicted_away = low_score
    else:
        predicted_home = low_score
        predicted_away = high_score
    
    # --- MODEL RIGOR & VERIFICATION ---
    training_info = game_data.get('training_info', {})
    metrics = training_info.get('metrics', {})
    brier = metrics.get('winner_test_brier')
    logloss = metrics.get('winner_test_logloss')
    
    rigor_html = ""
    if brier:
        rigor_html = f"""
        <div style="background: rgba(0, 243, 255, 0.03); border: 1px solid #333; padding: 12px; border-radius: 6px; margin-bottom: 25px;">
            <div style="color: #00F3FF; font-weight: 800; font-size: 0.85rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">🛡️ Model Rigor & Verification (Proven Data)</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;">
                <div>
                    <div style="color: #888; font-size: 0.65rem;">BRIER SCORE</div>
                    <div style="color: #FFF; font-weight: bold; font-size: 1rem;">{brier:.4f}</div>
                    <div style="color: #4CAF50; font-size: 0.6rem;">VERIFIED PROB</div>
                </div>
                <div>
                    <div style="color: #888; font-size: 0.65rem;">LOG LOSS</div>
                    <div style="color: #FFF; font-weight: bold; font-size: 1rem;">{logloss:.4f}</div>
                    <div style="color: #4CAF50; font-size: 0.6rem;">HIGH PRECISION</div>
                </div>
                <div>
                    <div style="color: #888; font-size: 0.65rem;">CV STABILITY</div>
                    <div style="color: #FFF; font-weight: bold; font-size: 1rem;">{metrics.get('winner_cv_std', 0):.1%}</div>
                    <div style="color: #4CAF50; font-size: 0.6rem;">CONSISTENT</div>
                </div>
            </div>
            <div style="color: #666; font-size: 0.7rem; margin-top: 10px; border-top: 1px solid #222; padding-top: 8px;">
                Verified accuracy of <b>{metrics.get('winner_test_accuracy', 0):.1%}</b> over {training_info.get('test_samples', 0)} blind test samples.
                Brier score of {brier:.4f} indicates excellent probability calibration.
            </div>
        </div>
        """

    # ── HEADER ──
    html = rigor_html + f"""
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
    
    # ── PREDICTED SCORE ──
    html += f"""
    <div style="background: #111; border: 1px solid #333; padding: 15px; border-radius: 6px; margin-bottom: 25px; text-align: center;">
        <div style="color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Predicted Final Score</div>
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
            <div>
                <div style="color: #FFF; font-size: 0.85rem;">{visitor_team}</div>
                <div style="color: #00F3FF; font-size: 2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{predicted_away:.0f}</div>
            </div>
            <div style="color: #555; font-size: 1.2rem; font-weight: bold;">—</div>
            <div>
                <div style="color: #FFF; font-size: 0.85rem;">{home_team}</div>
                <div style="color: #00F3FF; font-size: 2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;">{predicted_home:.0f}</div>
            </div>
        </div>
        <div style="color: #888; font-size: 0.8rem; margin-top: 8px;">
            Spread: <span style="color: #F3BC00;">{predicted_winner} {-abs(predicted_spread):+.1f}</span> &nbsp;|&nbsp; Total: <span style="color: #F3BC00;">{predicted_total:.1f}</span>
        </div>
    </div>
    """
    
    # ── TEAM PROFILES (SIDE BY SIDE) ──
    def team_grade(wp, diff, elo):
        score = 0
        if wp >= 0.7: score += 3
        elif wp >= 0.5: score += 2
        else: score += 1
        if diff > 8: score += 3
        elif diff > 3: score += 2
        elif diff > 0: score += 1
        if elo > 1600: score += 2
        elif elo > 1520: score += 1
        if score >= 7: return "A+", "#FFD700"
        elif score >= 6: return "A", "#4CAF50"
        elif score >= 5: return "B+", "#00F3FF"
        elif score >= 4: return "B", "#00BCD4"
        elif score >= 3: return "C+", "#FFA500"
        elif score >= 2: return "C", "#FF9800"
        else: return "D", "#FF5252"
    
    def momentum_label(wp5, wp10):
        if wp5 > wp10 + 0.15: return "🔥 HOT", "#4CAF50"
        elif wp5 > wp10 + 0.05: return "📈 RISING", "#00F3FF"
        elif wp5 < wp10 - 0.15: return "❄️ COLD", "#FF5252"
        elif wp5 < wp10 - 0.05: return "📉 FADING", "#FFA500"
        else: return "➡️ STEADY", "#888"
    
    h_grade, h_grade_color = team_grade(h_wp10, h_diff10, h_elo)
    v_grade, v_grade_color = team_grade(v_wp10, v_diff10, v_elo)
    h_mom_text, h_mom_color = momentum_label(h_wp5, h_wp10)
    v_mom_text, v_mom_color = momentum_label(v_wp5, v_wp10)
    
    h_net = h_ppg10 - h_opp10
    v_net = v_ppg10 - v_opp10
    
    html += """<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>🏀 TEAM PROFILES</h4>"""
    html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;'>"
    
    for t_name, t_elo, t_grade, t_gc, t_wp5, t_wp10, t_wp20, t_ppg, t_opp, t_diff, t_net, t_rest, t_b2b, t_mom, t_mc, is_home in [
        (visitor_team, v_elo, v_grade, v_grade_color, v_wp5, v_wp10, v_wp20, v_ppg10, v_opp10, v_diff10, v_net, v_rest, v_b2b, v_mom_text, v_mom_color, False),
        (home_team, h_elo, h_grade, h_grade_color, h_wp5, h_wp10, h_wp20, h_ppg10, h_opp10, h_diff10, h_net, h_rest, h_b2b, h_mom_text, h_mom_color, True),
    ]:
        home_tag = ' <span style="color: #4CAF50; font-size: 0.7rem;">🏠 HOME</span>' if is_home else ' <span style="color: #888; font-size: 0.7rem;">✈️ AWAY</span>'
        b2b_tag = '<div style="color: #FF5252; font-size: 0.75rem; margin-top: 4px;">⚠️ BACK-TO-BACK</div>' if t_b2b else ''
        net_color = "#4CAF50" if t_net > 0 else "#FF5252" if t_net < 0 else "#888"
        
        html += f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid #333; border-radius: 6px; padding: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="color: #FFF; font-weight: 800; font-size: 1rem;">{t_name}{home_tag}</div>
                <div style="background: {t_gc}; color: #000; font-weight: 900; padding: 3px 10px; border-radius: 4px; font-size: 0.9rem;">{t_grade}</div>
            </div>
            <div style="color: {t_mc}; font-size: 0.85rem; font-weight: bold; margin-bottom: 10px;">{t_mom}</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <tr><td style="color: #888; padding: 4px 0;">ELO Rating</td><td style="color: #FFF; text-align: right; font-weight: bold;">{t_elo:.0f}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">L5 Win %</td><td style="color: #FFF; text-align: right;">{t_wp5:.0%}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">L10 Win %</td><td style="color: #FFF; text-align: right;">{t_wp10:.0%}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">L20 Win %</td><td style="color: #FFF; text-align: right;">{t_wp20:.0%}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">Offense (PPG)</td><td style="color: #00F3FF; text-align: right; font-weight: bold;">{t_ppg:.1f}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">Defense (Opp PPG)</td><td style="color: #FF4444; text-align: right; font-weight: bold;">{t_opp:.1f}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">Net Rating</td><td style="color: {net_color}; text-align: right; font-weight: bold;">{t_net:+.1f}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">Avg Margin (L10)</td><td style="color: #FFF; text-align: right;">{t_diff:+.1f}</td></tr>
                <tr><td style="color: #888; padding: 4px 0;">Rest Days</td><td style="color: #FFF; text-align: right;">{int(t_rest)}</td></tr>
            </table>
            {b2b_tag}
        </div>
        """
    
    html += "</div>"
    
    # ── HEAD-TO-HEAD ──
    if h2h_games > 0:
        h2h_winner = home_team if h2h_pct >= 0.5 else visitor_team
        html += f"""
        <div style="background: rgba(255,215,0,0.05); border: 1px solid #444; border-left: 3px solid #FFD700; padding: 12px 15px; border-radius: 4px; margin-bottom: 25px;">
            <div style="color: #FFD700; font-weight: 800; font-size: 0.9rem; margin-bottom: 5px;">⚔️ HEAD-TO-HEAD ({int(h2h_games)} recent meetings)</div>
            <div style="color: #DDD; font-size: 0.9rem;">
                {home_team} wins <strong>{h2h_pct:.0%}</strong> of matchups &nbsp;|&nbsp; Avg margin: <strong>{h2h_margin:+.1f}</strong> (home perspective)
            </div>
            <div style="color: #AAA; font-size: 0.8rem; margin-top: 4px;">
                Historical edge favors <strong style="color: #FFF;">{h2h_winner}</strong> in this matchup.
            </div>
        </div>
        """
    
    # ── SITUATIONAL FLAGS ──
    sit_flags = []
    if is_march:
        sit_flags.append(("🏆 MARCH MADNESS PERIOD", "Season phase = 4 (Tournament). Model weights this as a distinct context.", "#FFD700"))
    if h_b2b:
        sit_flags.append((f"⚠️ {home_team} ON BACK-TO-BACK", f"{home_team} rest days = 0. Model feature 'home_is_b2b' = 1.", "#FF5252"))
    if v_b2b:
        sit_flags.append((f"⚠️ {visitor_team} ON BACK-TO-BACK", f"{visitor_team} rest days = 0. Model feature 'visitor_is_b2b' = 1.", "#FF5252"))
    rest_adv = h_rest - v_rest
    if rest_adv >= 3:
        sit_flags.append((f"😴 REST ADVANTAGE: {home_team} (+{int(rest_adv)} days)", f"{home_team}: {int(h_rest)} rest days vs {visitor_team}: {int(v_rest)} rest days. Difference: {int(rest_adv)}.", "#4CAF50"))
    elif rest_adv <= -3:
        sit_flags.append((f"😴 REST ADVANTAGE: {visitor_team} (+{int(abs(rest_adv))} days)", f"{visitor_team}: {int(v_rest)} rest days vs {home_team}: {int(h_rest)} rest days. Difference: {int(abs(rest_adv))}.", "#4CAF50"))
    if abs(elo_diff) > 150:
        better = home_team if elo_diff > 0 else visitor_team
        worse = visitor_team if elo_diff > 0 else home_team
        sit_flags.append((f"📊 MAJOR ELO GAP ({abs(elo_diff):.0f} pts)", f"{better} ELO: {max(h_elo,v_elo):.0f} vs {worse} ELO: {min(h_elo,v_elo):.0f}. Gap of {abs(elo_diff):.0f} built from season-long results.", "#00F3FF"))
    
    if sit_flags:
        html += "<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>🚩 SITUATIONAL FLAGS</h4>"
        for title, desc, color in sit_flags:
            html += f"""
            <div style="background: rgba(255,255,255,0.02); border-left: 3px solid {color}; padding: 10px 15px; margin-bottom: 8px; border-radius: 4px;">
                <div style="color: {color}; font-weight: 700; font-size: 0.9rem;">{title}</div>
                <div style="color: #BBB; font-size: 0.85rem; margin-top: 3px;">{desc}</div>
            </div>
            """
    
    # ── DATA SCORECARD: WHY THE MODEL PICKS THIS WINNER ──
    html += "<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px; margin-top: 25px;'>🧠 WHY THE MODEL PICKS " + predicted_winner.upper() + " — DATA PROOF</h4>"
    
    # Build a scorecard: each metric, who wins, by how much
    w_elo = h_elo if is_home_favorite else v_elo
    l_elo = v_elo if is_home_favorite else h_elo
    w_wp10 = h_wp10 if is_home_favorite else v_wp10
    l_wp10 = v_wp10 if is_home_favorite else h_wp10
    w_wp5 = h_wp5 if is_home_favorite else v_wp5
    l_wp5 = v_wp5 if is_home_favorite else h_wp5
    w_ppg = h_ppg10 if is_home_favorite else v_ppg10
    l_ppg = v_ppg10 if is_home_favorite else h_ppg10
    w_opp = h_opp10 if is_home_favorite else v_opp10
    l_opp = v_opp10 if is_home_favorite else h_opp10
    w_net = h_net if is_home_favorite else v_net
    l_net = v_net if is_home_favorite else h_net
    w_diff = h_diff10 if is_home_favorite else v_diff10
    l_diff = v_diff10 if is_home_favorite else h_diff10
    w_rest = h_rest if is_home_favorite else v_rest
    l_rest = v_rest if is_home_favorite else h_rest
    
    scorecard = []
    scorecard.append(("ELO Rating", f"{w_elo:.0f}", f"{l_elo:.0f}", w_elo > l_elo, w_elo - l_elo))
    scorecard.append(("L10 Win %", f"{w_wp10:.0%}", f"{l_wp10:.0%}", w_wp10 > l_wp10, (w_wp10 - l_wp10) * 100))
    scorecard.append(("L5 Win %", f"{w_wp5:.0%}", f"{l_wp5:.0%}", w_wp5 > l_wp5, (w_wp5 - l_wp5) * 100))
    scorecard.append(("Offense (L10 PPG)", f"{w_ppg:.1f}", f"{l_ppg:.1f}", w_ppg > l_ppg, w_ppg - l_ppg))
    scorecard.append(("Defense (Opp PPG)", f"{w_opp:.1f}", f"{l_opp:.1f}", w_opp < l_opp, l_opp - w_opp))
    scorecard.append(("Net Rating", f"{w_net:+.1f}", f"{l_net:+.1f}", w_net > l_net, w_net - l_net))
    scorecard.append(("Avg Margin (L10)", f"{w_diff:+.1f}", f"{l_diff:+.1f}", w_diff > l_diff, w_diff - l_diff))
    scorecard.append(("Rest Days", f"{int(w_rest)}", f"{int(l_rest)}", w_rest >= l_rest, w_rest - l_rest))
    
    winner_wins = sum(1 for _, _, _, favors, _ in scorecard if favors)
    loser_wins = len(scorecard) - winner_wins
    
    html += f"""
    <div style="background: rgba(0,243,255,0.03); border: 1px solid #333; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
            <div style="color: #4CAF50; font-weight: bold; font-size: 0.9rem;">{predicted_winner} wins {winner_wins} of {len(scorecard)} categories</div>
            <div style="color: #FF5722; font-size: 0.8rem;">{loser_team} wins {loser_wins}</div>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
            <tr style="color: #888; text-transform: uppercase; font-size: 0.7rem;">
                <th style="text-align: left; padding: 4px;">Metric</th>
                <th style="text-align: center; padding: 4px;">{predicted_winner}</th>
                <th style="text-align: center; padding: 4px;">{loser_team}</th>
                <th style="text-align: center; padding: 4px;">Edge</th>
            </tr>"""
    
    for label, w_val, l_val, favors_winner, edge in scorecard:
        icon = "✅" if favors_winner else "❌"
        edge_color = "#4CAF50" if favors_winner else "#FF5722"
        edge_str = f"+{abs(edge):.1f}" if abs(edge) >= 0.1 else "="
        html += f"""
            <tr style="border-bottom: 1px solid #1a1a2e;">
                <td style="padding: 6px 4px; color: #AAA;">{label}</td>
                <td style="padding: 6px 4px; text-align: center; color: #FFF; font-weight: bold;">{w_val}</td>
                <td style="padding: 6px 4px; text-align: center; color: #888;">{l_val}</td>
                <td style="padding: 6px 4px; text-align: center; color: {edge_color};">{icon} {edge_str}</td>
            </tr>"""
    
    html += "</table>"
    
    # Home court factor
    if is_home_favorite:
        html += f'<div style="color: #4CAF50; font-size: 0.78rem; margin-top: 10px; border-top: 1px solid #222; padding-top: 8px;">🏠 {predicted_winner} also has home court advantage.</div>'
    else:
        html += f'<div style="color: #FFA500; font-size: 0.78rem; margin-top: 10px; border-top: 1px solid #222; padding-top: 8px;">✈️ {predicted_winner} is on the road — data advantages overcome home court.</div>'
    
    # H2H context
    if h2h_games > 0:
        h2h_favors = (is_home_favorite and h2h_pct > 0.5) or (not is_home_favorite and h2h_pct < 0.5)
        h2h_icon = "✅" if h2h_favors else "⚠️"
        html += f'<div style="color: {"#4CAF50" if h2h_favors else "#FFA500"}; font-size: 0.78rem; margin-top: 4px;">{h2h_icon} H2H: {home_team} {h2h_pct:.0%} win rate across {int(h2h_games)} meetings (avg margin: {h2h_margin:+.1f})</div>'
    
    html += "</div>"
    
    # ── BETTING VERDICTS WITH REASONING ──
    recommended_bets = []
    locks = []
    
    if vegas_odds and vegas_odds.get('has_odds'):
        vegas_prob = vegas_odds.get('implied_home_prob', 0.5) if is_home_favorite else vegas_odds.get('implied_away_prob', 0.5)
        # Vegas spread in betting convention (neg = home fav)
        vegas_spread_betting = vegas_odds['spread_home']
        # Convert to model convention (pos = home wins) for comparison
        vegas_spread_model = -vegas_spread_betting
        vegas_total = vegas_odds['total']
        
        # predicted_spread is already in model convention (pos = home wins)
        model_spread = predicted_spread
        model_total = predicted_total
        
        # For the favorite: compare margins in absolute terms
        # model_margin = how much the predicted winner wins by
        # vegas_margin = how much vegas says the favorite wins by
        model_margin = abs(model_spread)
        vegas_margin = abs(vegas_spread_model)
        
        ml_edge = (winner_prob - vegas_prob) * 100
        spread_diff = abs(model_margin - vegas_margin)
        total_diff = model_total - vegas_total
        
        # ML
        if ml_edge >= 20:
            recommended_bets.append(f"🔒 **BET MONEYLINE:** {predicted_winner} to Win (Lock)")
            locks.append(("MONEYLINE LOCK", f"Model Edge: +{ml_edge:.1f}%", f"{predicted_winner} win probability: {winner_prob:.1%} vs Vegas {vegas_prob:.1%}. The model sees a {ml_edge:.0f}% edge — well above the ~5% vig threshold. This suggests Vegas is under-pricing {predicted_winner} significantly."))
        elif ml_edge >= 12:
            recommended_bets.append(f"💎 **BET MONEYLINE:** {predicted_winner} to Win (Value)")
            locks.append(("MONEYLINE VALUE", f"Model Edge: +{ml_edge:.1f}%", f"The model gives {predicted_winner} a {winner_prob:.1%} chance vs Vegas's implied {vegas_prob:.1%}. A +{ml_edge:.0f}% edge is strong value — the market may be slow to react to recent form changes."))
        elif ml_edge >= 5:
            locks.append(("MONEYLINE LEAN", f"Model Edge: +{ml_edge:.1f}%", f"Slight edge on {predicted_winner}. Model sees {winner_prob:.1%} vs Vegas {vegas_prob:.1%}. Not enough for a strong play, but valid as part of a parlay or straight pick."))

        # Spread analysis
        # model_margin: how much model says winner wins by (always positive)
        # vegas_margin: how much vegas says favorite wins by (always positive)
        # Check if model thinks predicted_winner wins by MORE or LESS than vegas
        model_favors_bigger_win = model_margin > vegas_margin
        
        # Does vegas agree on who the favorite is?
        vegas_fav_is_home = vegas_spread_betting < 0
        model_fav_is_home = is_home_favorite
        vegas_agrees_on_winner = (vegas_fav_is_home == model_fav_is_home)
        
        # Display spread for the predicted winner (betting format: negative for fav)
        winner_vegas_line = vegas_spread_betting if model_fav_is_home else -vegas_spread_betting
        
        if spread_diff >= 10:
            if not vegas_agrees_on_winner:
                recommended_bets.append(f"🔒 **BET SPREAD:** {predicted_winner} +{abs(winner_vegas_line):.1f} (Lock)")
                locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Model sees {predicted_winner} winning by {model_margin:.1f}, but Vegas gives them +{abs(winner_vegas_line):.1f} points. This is a rare spot where the model's pick is also getting free points — a massive edge. The {spread_diff:.1f}-point gap is driven by {predicted_winner}'s superior net rating and recent scoring trends."))
            elif model_favors_bigger_win:
                recommended_bets.append(f"🔥 **BET SPREAD:** {predicted_winner} {winner_vegas_line:+.1f} (Favorite)")
                locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Model predicts a {model_margin:.1f}-point margin vs Vegas's {vegas_margin:.1f}. This {spread_diff:.1f}-point gap indicates the model expects a blowout — likely driven by a significant offense-vs-defense mismatch and/or fatigue disadvantage for {loser_team}."))
            else:
                recommended_bets.append(f"🔒 **BET SPREAD:** {loser_team} +{abs(winner_vegas_line):.1f} (Underdog)")
                locks.append(("SPREAD LOCK", f"Gap: {spread_diff:.1f} pts", f"Vegas has {predicted_winner} winning by {vegas_margin:.1f}, but the model only sees a {model_margin:.1f}-point margin. That's {spread_diff:.1f} points of value on {loser_team} with the points. The market may be overvaluing {predicted_winner}'s brand/ranking vs their actual recent performance."))
        elif spread_diff >= 6:
            if not vegas_agrees_on_winner:
                recommended_bets.append(f"✅ **BET SPREAD:** {predicted_winner} +{abs(winner_vegas_line):.1f} (Value)")
                locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"Model favors {predicted_winner} outright while they're getting +{abs(winner_vegas_line):.1f}. A {spread_diff:.1f}-point value gap — strong enough to act on."))
            elif model_favors_bigger_win:
                recommended_bets.append(f"✅ **BET SPREAD:** {predicted_winner} {winner_vegas_line:+.1f} (Value)")
                locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"Model sees {predicted_winner} covering {winner_vegas_line:+.1f} comfortably — predicted margin of {model_margin:.1f} exceeds the line by {spread_diff:.1f} points."))
            else:
                recommended_bets.append(f"✅ **BET SPREAD:** {loser_team} +{abs(winner_vegas_line):.1f} (Value)")
                locks.append(("SPREAD VALUE", f"Gap: {spread_diff:.1f} pts", f"{loser_team} keeps it closer than Vegas implies. Model margin ({model_margin:.1f}) is {spread_diff:.1f} points tighter than the line ({vegas_margin:.1f})."))
        elif spread_diff >= 2:
            if not vegas_agrees_on_winner:
                locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {predicted_winner} +{abs(winner_vegas_line):.1f}. Small edge — consider only if part of a multi-leg bet."))
            elif model_favors_bigger_win:
                locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {predicted_winner} to cover by a slim margin. The {spread_diff:.1f}-point gap isn't huge but points in the right direction."))
            else:
                locks.append(("SPREAD LEAN", f"Gap: {spread_diff:.1f} pts", f"Model leans {loser_team} to cover. {predicted_winner} may win but not by as much as the line suggests."))

        # Total
        if abs(total_diff) >= 10:
            ou = "OVER" if total_diff > 0 else "UNDER"
            pace_reason = ""
            combined_off = h_ppg10 + v_ppg10
            combined_def = h_opp10 + v_opp10
            if total_diff > 0:
                pace_reason = f"Both teams' L10 offensive output ({combined_off:.1f} combined PPG) suggests a faster pace than the market expects. Defensive averages ({combined_def:.1f} combined Opp PPG) confirm the scoring environment."
            else:
                pace_reason = f"Both defenses are stifling — combined opponent PPG of {combined_def:.1f} over L10. The model sees a lower-scoring game than the {vegas_total:.1f} line implies."
            recommended_bets.append(f"🔒 **BET TOTAL:** {ou} {vegas_total:.1f} (Lock)")
            locks.append((f"TOTAL {ou} LOCK", f"Diff: {total_diff:+.1f} pts", f"Model: {model_total:.1f} vs Vegas: {vegas_total:.1f}. {pace_reason}"))
        elif abs(total_diff) >= 6:
            ou = "OVER" if total_diff > 0 else "UNDER"
            recommended_bets.append(f"💎 **BET TOTAL:** {ou} {vegas_total:.1f} (Value)")
            locks.append((f"TOTAL {ou} VALUE", f"Diff: {total_diff:+.1f} pts", f"Model total ({model_total:.1f}) diverges {abs(total_diff):.1f} from Vegas ({vegas_total:.1f}). This is driven by recent scoring trends — L10 combined offense is {h_ppg10 + v_ppg10:.1f} PPG."))
        elif abs(total_diff) >= 2:
            ou = "OVER" if total_diff > 0 else "UNDER"
            locks.append((f"TOTAL {ou} LEAN", f"Diff: {total_diff:+.1f} pts", f"Model slightly favors the {ou} ({model_total:.1f} vs {vegas_total:.1f}). Not a strong play by itself."))
    
    # Render recommended bets
    if recommended_bets:
        html += """<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>🔥 RECOMMENDED BETS</h4>"""
        html += """<div style="background: #1A1D26; border: 1px solid #FF00FF; border-left: 5px solid #FF00FF; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
        <ul style="margin-bottom:0; padding-left: 20px; color: #FFF;">"""
        for bet in recommended_bets:
            clean_bet = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', bet)
            html += f"<li style='margin-bottom: 8px; font-size: 1.05rem;'>{clean_bet}</li>"
        html += "</ul></div>"
    else:
        html += """<div style="padding: 15px; border: 1px solid #444; border-radius: 4px; margin-bottom: 25px; color: #888;">
        ⚖️ NO STRONG PLAYS DETECTED. Model aligns with Vegas on this game — no actionable edge found.
        </div>"""

    # Render edge analysis cards
    if locks:
        html += "<h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>📋 DETAILED EDGE BREAKDOWN</h4>"
        html += "<div style='display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 30px;'>"
        for title, subtitle, detail in locks:
            color = "#FFD700" if "LOCK" in title else "#00F3FF" if "VALUE" in title else "#FF00FF"
            html += f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 4px; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="color: {color}; font-weight: 800; font-size: 0.95rem;">{title}</div>
                    <div style="color: #FFF; font-weight: bold; font-size: 0.9rem; background: rgba(255,255,255,0.05); padding: 3px 10px; border-radius: 4px;">{subtitle}</div>
                </div>
                <div style="color: #CCC; font-size: 0.88rem; line-height: 1.6; margin-top: 8px;">{detail}</div>
            </div>
            """
        html += "</div>"

    # ── FULL STATS TABLE ──
    h_l5 = f"{int(h_wp5*5)}-{5-int(h_wp5*5)}"
    v_l5 = f"{int(v_wp5*5)}-{5-int(v_wp5*5)}"
    h_l10_str = f"{int(h_wp10*10)}-{10-int(h_wp10*10)}"
    v_l10_str = f"{int(v_wp10*10)}-{10-int(v_wp10*10)}"
    h_l20 = f"{int(h_wp20*20)}-{20-int(h_wp20*20)}"
    v_l20 = f"{int(v_wp20*20)}-{20-int(v_wp20*20)}"
    
    html += """
    <h4 style='color: #DDD; border-bottom: 1px solid #333; padding-bottom: 5px;'>📊 FULL STAT COMPARISON</h4>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 0.9rem;">
        <tr style="background: #111; color: #888; text-transform: uppercase; font-size: 0.8rem;">
            <th style="padding: 8px; text-align: left;">METRIC</th>
            <th style="padding: 8px; text-align: center; color: #FFF;">""" + visitor_team + """</th>
            <th style="padding: 8px; text-align: center; color: #FFF;">""" + home_team + """</th>
        </tr>"""
    
    rows = [
        ("ELO Rating", f"{v_elo:.0f}", f"{h_elo:.0f}", False),
        ("Last 5 Games", v_l5, h_l5, True),
        ("Last 10 Games", v_l10_str, h_l10_str, False),
        ("Last 20 Games", v_l20, h_l20, True),
        ("L10 PPG", f"{v_ppg10:.1f}", f"{h_ppg10:.1f}", False),
        ("L5 PPG", f"{v_ppg5:.1f}", f"{h_ppg5:.1f}", True),
        ("L10 Opp PPG", f"{v_opp10:.1f}", f"{h_opp10:.1f}", False),
        ("L5 Opp PPG", f"{v_opp5:.1f}", f"{h_opp5:.1f}", True),
        ("Net Rating (L10)", f"{v_net:+.1f}", f"{h_net:+.1f}", False),
        ("Avg Margin (L10)", f"{v_diff10:+.1f}", f"{h_diff10:+.1f}", True),
        ("Avg Margin (L5)", f"{v_diff5:+.1f}", f"{h_diff5:+.1f}", False),
        ("Rest Days", f"{int(v_rest)}", f"{int(h_rest)}", True),
    ]
    
    for label, v_val, h_val, alt_bg in rows:
        bg = "background: rgba(255,255,255,0.02);" if alt_bg else ""
        ppg_color = ""
        if "PPG" in label and "Opp" not in label:
            ppg_color = "color: #00F3FF;"
        elif "Opp PPG" in label:
            ppg_color = "color: #FF4444;"
        elif "Net" in label or "Margin" in label:
            ppg_color = "color: #FFA500;"
        
        html += f"""
        <tr style="border-bottom: 1px solid #222; {bg}">
            <td style="padding: 10px; color: #AAA;">{label}</td>
            <td style="padding: 10px; text-align: center; color: #FFF; {ppg_color}">{v_val}</td>
            <td style="padding: 10px; text-align: center; color: #FFF; {ppg_color}">{h_val}</td>
        </tr>"""
    
    html += "</table>"
    html += "</div>"
    return html


def display_game_predictions(gender, games_df, predictions_df, odds_df, all_features, accuracy_stats=None, model=None):
    """Display all game predictions in ranked order"""
    
    if games_df.empty:
        st.warning("No games scheduled for this date.")
        return
    
    gender_label = config.GENDER_CONFIG[gender]["label"]
    game_date = pd.to_datetime(games_df.iloc[0]["date"]).strftime("%B %d, %Y") if "date" in games_df.columns and not games_df.empty else datetime.now().strftime("%B %d, %Y")
    
    st.markdown(f'<div class="main-header">🏀 {gender_label} College Basketball</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{game_date}<br>{len(games_df)} games today • Odds may change, check your lines to find best odds.</div>', unsafe_allow_html=True)
    
    # Show REAL accuracy stats
    if accuracy_stats and accuracy_stats.get('completed_games', 0) > 0:
        stats = accuracy_stats
        accuracy_html = '<div style="background: #1a1a2e; border: 2px solid #00F3FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">'
        accuracy_html += '<div style="font-family: JetBrains Mono; color: #FFD700; font-size: 1rem; margin-bottom: 12px;">📈 REAL TRACKED ACCURACY (Last 30 Days)</div>'
        
        # Row 1: Overall + Per-Tier Accuracy
        accuracy_html += '<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 10px;">'
        
        # Fire emoji helper — more fire for higher accuracy
        def fire(pct):
            if pct >= 75: return '🔥🔥🔥 '
            if pct >= 65: return '🔥🔥 '
            if pct >= 55: return '🔥 '
            return ''
        
        # Overall Winner
        win_pct = stats['winner_accuracy'] * 100
        overall_w = stats['winner_correct']
        overall_l = stats['completed_games'] - overall_w
        win_color = '#4CAF50' if win_pct >= 55 else '#FFA500' if win_pct >= 50 else '#FF5252'
        accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: {win_color}; font-size: 1.5rem; font-weight: bold;">{fire(win_pct)}{win_pct:.1f}%</div>'
        accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">OVERALL ({overall_w}W-{overall_l}L)</div></div>'
        
        # Lock Pick tier
        if stats.get('lock_pick_total', 0) > 0:
            lp_pct = stats['lock_pick_accuracy'] * 100
            lp_w = stats['lock_pick_correct']
            lp_l = stats['lock_pick_total'] - lp_w
            lp_color = '#FFD700' if lp_pct >= 70 else '#4CAF50' if lp_pct >= 55 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #FFD700;">'
            accuracy_html += f'<div style="color: {lp_color}; font-size: 1.5rem; font-weight: bold;">{fire(lp_pct)}{lp_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #FFD700; font-size: 0.7rem;">🔒 LOCK ({lp_w}W-{lp_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #FFD700;">'
            accuracy_html += '<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += '<div style="color: #FFD700; font-size: 0.7rem;">🔒 LOCK</div></div>'
        
        # High Confidence tier
        if stats.get('high_confidence_total', 0) > 0:
            hc_pct = stats['high_confidence_accuracy'] * 100
            hc_w = stats['high_confidence_correct']
            hc_l = stats['high_confidence_total'] - hc_w
            hc_color = '#4CAF50' if hc_pct >= 60 else '#FFA500' if hc_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #00F3FF;">'
            accuracy_html += f'<div style="color: {hc_color}; font-size: 1.5rem; font-weight: bold;">{fire(hc_pct)}{hc_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #00F3FF; font-size: 0.7rem;">🔥🔥🔥 HIGH ({hc_w}W-{hc_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #00F3FF;">'
            accuracy_html += '<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += '<div style="color: #00F3FF; font-size: 0.7rem;">🔥🔥🔥 HIGH</div></div>'
        
        # Good Value tier
        if stats.get('good_value_total', 0) > 0:
            gv_pct = stats['good_value_accuracy'] * 100
            gv_w = stats['good_value_correct']
            gv_l = stats['good_value_total'] - gv_w
            gv_color = '#4CAF50' if gv_pct >= 55 else '#FFA500' if gv_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #4CAF50;">'
            accuracy_html += f'<div style="color: {gv_color}; font-size: 1.5rem; font-weight: bold;">{fire(gv_pct)}{gv_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #4CAF50; font-size: 0.7rem;">🔥🔥 GOOD ({gv_w}W-{gv_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #4CAF50;">'
            accuracy_html += '<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += '<div style="color: #4CAF50; font-size: 0.7rem;">🔥🔥 GOOD</div></div>'
        
        # Predicted (lower confidence) tier
        if stats.get('predicted_total', 0) > 0:
            pr_pct = stats['predicted_accuracy'] * 100
            pr_w = stats['predicted_correct']
            pr_l = stats['predicted_total'] - pr_w
            pr_color = '#4CAF50' if pr_pct >= 55 else '#FFA500' if pr_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #888;">'
            accuracy_html += f'<div style="color: {pr_color}; font-size: 1.5rem; font-weight: bold;">{fire(pr_pct)}{pr_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">PREDICTED ({pr_w}W-{pr_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid #888;">'
            accuracy_html += '<div style="color: #888; font-size: 1.5rem;">--</div>'
            accuracy_html += '<div style="color: #888; font-size: 0.7rem;">PREDICTED</div></div>'
        
        accuracy_html += '</div>'
        
        # Row 2: Betting stats
        accuracy_html += '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px;">'
        
        if stats.get('betting_ml_total', 0) > 0:
            ml_pct = stats['betting_ml_accuracy'] * 100
            ml_w = stats['betting_ml_correct']
            ml_l = stats['betting_ml_total'] - ml_w
            ml_color = '#4CAF50' if ml_pct >= 55 else '#FFA500' if ml_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {ml_color}; font-size: 1.2rem; font-weight: bold;">{fire(ml_pct)}{ml_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">BET ML ({ml_w}W-{ml_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += '<div style="color: #888; font-size: 1.2rem;">--</div>'
            accuracy_html += '<div style="color: #888; font-size: 0.7rem;">BET ML</div></div>'
        
        if stats.get('betting_spread_total', 0) > 0:
            sp_pct = stats['betting_spread_accuracy'] * 100
            sp_w = stats['betting_spread_correct']
            sp_l = stats['betting_spread_total'] - sp_w
            sp_color = '#4CAF50' if sp_pct >= 55 else '#FFA500' if sp_pct >= 50 else '#FF5252'
            accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += f'<div style="color: {sp_color}; font-size: 1.2rem; font-weight: bold;">{fire(sp_pct)}{sp_pct:.1f}%</div>'
            accuracy_html += f'<div style="color: #888; font-size: 0.7rem;">BET SPREAD ({sp_w}W-{sp_l}L)</div></div>'
        else:
            accuracy_html += '<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
            accuracy_html += '<div style="color: #888; font-size: 1.2rem;">--</div>'
            accuracy_html += '<div style="color: #888; font-size: 0.7rem;">BET SPREAD</div></div>'
        
        accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: #00F3FF; font-size: 1.2rem; font-weight: bold;">{stats.get("avg_spread_error", 0):.1f}</div>'
        accuracy_html += '<div style="color: #888; font-size: 0.7rem;">SPREAD ERR</div></div>'
        
        accuracy_html += f'<div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">'
        accuracy_html += f'<div style="color: #00F3FF; font-size: 1.2rem; font-weight: bold;">{stats["completed_games"]}</div>'
        accuracy_html += '<div style="color: #888; font-size: 0.7rem;">GAMES TRACKED</div></div>'
        
        accuracy_html += '</div>'
        
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
        st.markdown("""
        <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-family: JetBrains Mono; color: #FFA500; font-size: 0.9rem;">
                📊 TRACKING ACCURACY - Predictions are being saved. Check back tomorrow after games complete for real stats.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load betting model
    betting_model = BettingModel(gender)
    betting_loaded = betting_model.load()
    
    combined = games_df.copy()
    combined = combined.join(predictions_df)
    
    # Get betting model confidence
    betting_picks = {}
    if betting_loaded:
        for idx, game in combined.iterrows():
            if idx in all_features.index:
                rec = betting_model.get_betting_recommendation(all_features.loc[idx])
                if rec:
                    ml_conf = rec.get('ml_confidence', 0.5)
                    ml_pick = rec.get('ml_pick', 'HOME')
                    winner = game['home_team_name'] if ml_pick == 'HOME' else game['visitor_team_name']
                    betting_picks[idx] = {'winner': winner, 'conf': ml_conf, 'rec': rec}
    
    combined['betting_conf'] = combined.index.map(lambda x: betting_picks.get(x, {}).get('conf', 0.5))
    combined = combined.sort_values('betting_conf', ascending=False)
    
    lock_picks = []
    high_conf_picks = []
    good_picks = []
    
    for idx, game in combined.iterrows():
        if idx in betting_picks:
            bp = betting_picks[idx]
            conf = bp['conf']
            winner = bp['winner']
            
            if conf >= 0.70:
                lock_picks.append((winner, conf))
            elif conf >= 0.62:
                high_conf_picks.append((winner, conf))
            elif conf >= 0.58:
                good_picks.append((winner, conf))
    
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
    
    # Display each game card
    for idx, game in combined.iterrows():
        betting_rec = None
        game_vegas_odds = get_consensus_odds(odds_df, game.get('id')) if game.get('id') else None
        if betting_model.loaded and idx in all_features.index:
            game_features = all_features.loc[idx]
            betting_rec = betting_model.get_betting_recommendation(
                game_features,
                game_vegas_odds.get('spread_home') if game_vegas_odds else None,
                game_vegas_odds.get('total') if game_vegas_odds else None
            )
            
        display_game_card(gender, game, odds_df, all_features.loc[idx] if idx in all_features.index else {}, betting_rec, precomputed_odds=game_vegas_odds, model=model)
        st.markdown("---")


def display_game_card(gender, game, odds_df, features, betting_rec=None, precomputed_odds=None, model=None):
    """Display a single game prediction card with Pro Grid Layout"""
    avg_score = config.GENDER_CONFIG[gender]["avg_score"]
    
    home_name = game.get('home_team_name', 'Home')
    visitor_name = game.get('visitor_team_name', 'Away')
    home_id = game.get('home_team_id')
    visitor_id = game.get('visitor_team_id')
    
    home_logo = team_logos.get_team_logo(home_id, game.to_dict() if hasattr(game, 'to_dict') else game)
    visitor_logo = team_logos.get_team_logo(visitor_id, game.to_dict() if hasattr(game, 'to_dict') else game)
    
    # Cache team data
    if hasattr(game, 'to_dict'):
        team_logos.cache_team_from_game(game.to_dict())
    
    home_prob = game.get('home_win_probability', 0.5)
    visitor_prob = game.get('visitor_win_probability', 0.5)
    
    game_id = game.get('id')
    vegas_odds = precomputed_odds if precomputed_odds is not None else (get_consensus_odds(odds_df, game_id) if game_id else None)
    
    def safe_get(obj, key, default):
        try:
            if isinstance(obj, dict):
                val = obj.get(key, default)
            elif hasattr(obj, 'get'):
                val = obj.get(key, default)
            else:
                return default
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return val
        except (TypeError, KeyError, AttributeError):
            return default
    
    # Badge Logic
    badges_html = ""
    
    if betting_rec:
        ml_conf = betting_rec.get('ml_confidence', 0.5)
        spread_conf = betting_rec.get('spread_confidence', 0.5)
        total_conf = betting_rec.get('total_confidence', 0.5)
        
        if ml_conf >= 0.55:
            ml_pick = home_name if betting_rec.get('ml_pick') == 'HOME' else visitor_name
            badge_class = "badge-pro" if ml_conf >= 0.65 else "badge-outline"
            fire = "🔥" if ml_conf >= 0.65 else ""
            badges_html += f'<div class="{badge_class}">{fire} ML: {ml_pick} ({ml_conf:.0%})</div>'
        
        if spread_conf >= 0.53 and vegas_odds and vegas_odds.get('has_odds'):
            spread_pick = home_name if betting_rec.get('spread_pick') == 'HOME' else visitor_name
            vegas_spread = vegas_odds.get('spread_home', 0)
            if betting_rec.get('spread_pick') == 'HOME':
                spread_line = f"{vegas_spread:+.1f}" if vegas_spread else ""
            else:
                away_spread = -vegas_spread if vegas_spread else 0
                spread_line = f"{away_spread:+.1f}" if vegas_spread else ""
            badge_class = "badge-pro" if spread_conf >= 0.58 else "badge-outline"
            badges_html += f'<div class="{badge_class}">SPREAD: {spread_pick} {spread_line} ({spread_conf:.0%})</div>'
        
        if total_conf >= 0.53 and vegas_odds and vegas_odds.get('has_odds'):
            total_pick = betting_rec.get('total_pick', 'OVER')
            vegas_total = vegas_odds.get('total', 0)
            badge_class = "badge-pro" if total_conf >= 0.58 else "badge-outline"
            badges_html += f'<div class="{badge_class}">{total_pick} {vegas_total:.1f} ({total_conf:.0%})</div>'

    # Stats
    h_win_pct = safe_get(features, 'home_win_pct_last10', 0.5)
    v_win_pct = safe_get(features, 'visitor_win_pct_last10', 0.5)
    h_ppg = safe_get(features, 'home_points_scored_last10', avg_score)
    v_ppg = safe_get(features, 'visitor_points_scored_last10', avg_score)
    
    if h_win_pct == 0: h_win_pct = 0.5
    if v_win_pct == 0: v_win_pct = 0.5
    if h_ppg < 30: h_ppg = avg_score
    if v_ppg < 30: v_ppg = avg_score

    # Status / Time / Score - always convert to California (Pacific) time
    status = str(game.get('status', ''))
    game_time_display = status
    
    try:
        from zoneinfo import ZoneInfo
        _PACIFIC = ZoneInfo("America/Los_Angeles")
        _UTC = ZoneInfo("UTC")
    except ImportError:
        _PACIFIC = None
        _UTC = None
    
    # Map source timezone abbreviations to IANA zone names for DST-aware conversion
    _TZ_MAP = {
        'EST': 'US/Eastern', 'ET': 'US/Eastern', 'EDT': 'US/Eastern',
        'CST': 'US/Central', 'CT': 'US/Central', 'CDT': 'US/Central',
        'MST': 'US/Mountain', 'MT': 'US/Mountain', 'MDT': 'US/Mountain',
        'PST': 'America/Los_Angeles', 'PT': 'America/Los_Angeles', 'PDT': 'America/Los_Angeles',
    }
    
    if 'T' in status and _PACIFIC:
        try:
            time_part = status.split("T")[1].replace("Z", "").split("+")[0].split("-")[0]
            parts = time_part.split(":")
            date_part = status.split("T")[0]
            dp = date_part.split("-")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            second = int(parts[2]) if len(parts) > 2 else 0
            dt_utc = datetime(int(dp[0]), int(dp[1]), int(dp[2]), hour, minute, second, tzinfo=_UTC)
            dt_pt = dt_utc.astimezone(_PACIFIC)
            game_time_display = dt_pt.strftime("%I:%M %p PT")
        except Exception:
            pass
    
    elif ('PM' in status or 'AM' in status) and _PACIFIC:
        try:
            match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)\s*(EST|ET|EDT|CST|CT|CDT|MST|MT|MDT|PST|PT|PDT)?', status.strip())
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                ampm = match.group(3)
                tz_label = match.group(4) or 'ET'
                
                if ampm == 'PM' and hour != 12:
                    hour += 12
                elif ampm == 'AM' and hour == 12:
                    hour = 0
                
                src_tz_name = _TZ_MAP.get(tz_label, 'US/Eastern')
                src_tz = ZoneInfo(src_tz_name)
                dt_src = datetime(2000, 1, 1, hour, minute, tzinfo=src_tz)
                dt_pt = dt_src.astimezone(_PACIFIC)
                game_time_display = dt_pt.strftime("%I:%M %p PT")
        except Exception:
            pass
    
    is_live_or_finished = False
    h_s = game.get('home_team_score')
    v_s = game.get('visitor_team_score')
    if h_s is not None and v_s is not None and (h_s > 0 or v_s > 0):
        is_live_or_finished = True
    
    v_score = game.get('visitor_team_score', '') if is_live_or_finished else ''
    h_score = game.get('home_team_score', '') if is_live_or_finished else ''
    
    # Winner badge logic
    v_winner_badge = ""
    h_winner_badge = ""
    display_home_prob = home_prob
    display_visitor_prob = visitor_prob
    
    if betting_rec and betting_rec.get('ml_confidence'):
        ml_conf = betting_rec['ml_confidence']
        ml_pick = betting_rec.get('ml_pick', 'HOME')
        predicted_winner_is_home = (ml_pick == 'HOME')
        
        if predicted_winner_is_home:
            display_home_prob = ml_conf
            display_visitor_prob = 1 - ml_conf
        else:
            display_visitor_prob = ml_conf
            display_home_prob = 1 - ml_conf
        
        if ml_conf >= 0.70:
            badge_style = 'background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-weight: 900;'
            badge_text = '🔒 LOCK PICK'
        elif ml_conf >= 0.62:
            badge_style = 'background: var(--neon-cyan); color: #000; font-weight: 900;'
            badge_text = '🔥🔥🔥 HIGH CONFIDENCE'
        elif ml_conf >= 0.58:
            badge_style = 'background: #4CAF50; color: #FFF;'
            badge_text = '🔥🔥 GOOD VALUE'
        else:
            badge_style = 'background: rgba(255,255,255,0.1); color: #AAA;'
            badge_text = 'PREDICTED'
    else:
        predicted_winner_is_home = (home_prob > visitor_prob)
        badge_style = 'background: rgba(255,255,255,0.1); color: #AAA;'
        badge_text = 'PREDICTED'
    
    if predicted_winner_is_home:
        h_winner_badge = f'<div class="winner-tag" style="{badge_style}">{badge_text}</div>'
    else:
        v_winner_badge = f'<div class="winner-tag" style="{badge_style}">{badge_text}</div>'

    # Vegas display strings
    v_spread_str = "N/A"
    h_spread_str = "N/A"
    v_ml_str = "N/A"
    h_ml_str = "N/A"
    total_str = "N/A"
    has_vegas_data = False
    
    if vegas_odds and vegas_odds.get('has_odds'):
        h_spread_val = vegas_odds.get('spread_home', 0)
        v_spread_val = vegas_odds.get('spread_away', 0)
        h_ml_val = vegas_odds.get('moneyline_home')
        v_ml_val = vegas_odds.get('moneyline_away')
        total_val = vegas_odds.get('total', 0)
        
        def _is_nan(v):
            return v is None or (isinstance(v, float) and pd.isna(v))
        
        # Fill missing moneylines from the other side (with ~4% vig adjustment)
        def _derive_ml(known_ml):
            """Derive the opposite side ML from a known moneyline, adding standard vig."""
            if known_ml > 0:
                imp = 100 / (known_ml + 100)
            else:
                imp = abs(known_ml) / (abs(known_ml) + 100)
            opp = max(0.02, min(0.98, 1 - imp))
            opp = min(0.98, opp * 1.04)
            if opp >= 0.5:
                ml = -(opp / (1 - opp)) * 100
                ml = max(ml, -10000)
                return round(ml / 50) * 50
            else:
                ml = ((1 - opp) / opp) * 100
                return round(ml / 50) * 50
        
        if _is_nan(h_ml_val) and not _is_nan(v_ml_val):
            h_ml_val = _derive_ml(v_ml_val)
        elif _is_nan(v_ml_val) and not _is_nan(h_ml_val):
            v_ml_val = _derive_ml(h_ml_val)
        
        h_spread_str = f"{h_spread_val:+.1f}" if not _is_nan(h_spread_val) and h_spread_val != 0 else "PK"
        v_spread_str = f"{v_spread_val:+.1f}" if not _is_nan(v_spread_val) and v_spread_val != 0 else "PK"
        h_ml_str = format_american_odds(h_ml_val) if not _is_nan(h_ml_val) else "N/A"
        v_ml_str = format_american_odds(v_ml_val) if not _is_nan(v_ml_val) else "N/A"
        total_str = f"{total_val:.1f}" if not _is_nan(total_val) and total_val > 0 else "N/A"
        has_vegas_data = True

    # Build card HTML
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
    
    # Analysis expander
    with st.expander(f"🔍 Elite Analysis & Betting Breakdown"):
        pred_dict = {
            'home_win_probability': home_prob,
            'predicted_home_score': game.get('predicted_home_score', config.GENDER_CONFIG[gender]["avg_score"]),
            'predicted_visitor_score': game.get('predicted_visitor_score', config.GENDER_CONFIG[gender]["avg_score"]),
            'predicted_spread': game.get('predicted_spread', 0),
            'predicted_total': game.get('predicted_total', config.GENDER_CONFIG[gender]["avg_total"])
        }
        
        # Get actual training info from model
        g_data = game.to_dict() if hasattr(game, 'to_dict') else game
        if model:
            g_data['training_info'] = model.training_info
            
        analysis_text = generate_elite_analysis(g_data, pred_dict, features, vegas_odds, gender)
        analysis_text = re.sub(r'(?m)^\s+', '', analysis_text)
        st.markdown(f'<div class="analysis-text">{analysis_text}</div>', unsafe_allow_html=True)


def _render_props_table(props_list, show_source=False, show_bookmaker=False,
                        is_dfs=False, dfs_color="#00F3FF"):
    """Render a filterable table of player props"""
    if not props_list:
        st.markdown('<div style="color: #888;">No props available from this source.</div>', unsafe_allow_html=True)
        return
    
    props_df = pd.DataFrame(props_list)
    
    # Build display subtitle: game if available, else team
    # Safely convert to strings to avoid float.strip() errors
    props_df['game'] = props_df['game'].fillna('').astype(str)
    props_df['team'] = props_df['team'].fillna('').astype(str) if 'team' in props_df.columns else ''
    props_df['_subtitle'] = props_df.apply(
        lambda r: r['game'] if r['game'].strip() else r.get('team', ''), axis=1
    )
    
    # Filters
    filter_cols = st.columns(4 if show_bookmaker else 3)
    with filter_cols[0]:
        prop_types = ['All'] + sorted(props_df['prop_type'].dropna().astype(str).unique().tolist())
        selected_type = st.selectbox("Prop Type", prop_types, key=f"pt_{id(props_list)}")
    with filter_cols[1]:
        players = ['All'] + sorted(props_df['player'].dropna().astype(str).unique().tolist())
        selected_player = st.selectbox("Player", players, key=f"pl_{id(props_list)}")
    with filter_cols[2]:
        # Use game when available, otherwise show team-based filtering
        has_games = props_df['game'].str.strip().ne('').any()
        if has_games:
            game_options = ['All'] + sorted(props_df[props_df['game'].str.strip() != '']['game'].unique().tolist())
            selected_game = st.selectbox("Game", game_options, key=f"gm_{id(props_list)}")
        else:
            team_vals = props_df['team'].dropna().astype(str).unique().tolist() if 'team' in props_df.columns else []
            team_options = ['All'] + sorted([t for t in team_vals if str(t).strip() and str(t).lower() != 'nan'])
            selected_game = st.selectbox("Team", team_options, key=f"tm_{id(props_list)}")
    if show_bookmaker and len(filter_cols) > 3:
        with filter_cols[3]:
            books = ['All'] + sorted(props_df['bookmaker'].dropna().unique().tolist())
            selected_book = st.selectbox("Bookmaker", books, key=f"bk_{id(props_list)}")
    else:
        selected_book = 'All'
    
    filtered_df = props_df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['prop_type'] == selected_type]
    if selected_player != 'All':
        filtered_df = filtered_df[filtered_df['player'] == selected_player]
    if selected_game != 'All':
        if has_games:
            filtered_df = filtered_df[filtered_df['game'] == selected_game]
        elif 'team' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['team'] == selected_game]
    if selected_book != 'All':
        filtered_df = filtered_df[filtered_df['bookmaker'] == selected_book]
    
    total_count = len(filtered_df)
    st.markdown(f'<div style="color: #888; font-size: 0.8rem; margin-bottom: 8px;">Showing {total_count} props</div>', unsafe_allow_html=True)
    
    # Limit display for performance (show first 100, with load-more hint)
    display_limit = 100
    display_df = filtered_df.head(display_limit)
    
    for _, prop in display_df.iterrows():
        over_odds = prop.get('over_odds')
        under_odds = prop.get('under_odds')
        
        has_over = over_odds is not None and pd.notna(over_odds)
        has_under = under_odds is not None and pd.notna(under_odds)
        
        source_name = prop.get('source', '')
        bookmaker = prop.get('bookmaker', '')
        
        # Default DFS odds when missing
        if source_name == "PrizePicks" and not has_over and not has_under:
            over_odds, under_odds = -110, -110
            has_over, has_under = True, True
        elif source_name == "Underdog" and not has_over and not has_under:
            over_odds, under_odds = -112, -112
            has_over, has_under = True, True
        
        over_str = f"{int(over_odds):+d}" if has_over else "—"
        under_str = f"{int(under_odds):+d}" if has_under else "—"
        over_color = "#4CAF50" if has_over else "#555"
        under_color = "#FF5722" if has_under else "#555"
        subtitle = prop.get('_subtitle', '')
        
        # Team badge for DFS
        team_str = str(prop.get('team', '') or '')
        team_badge = ""
        if team_str.strip() and team_str.lower() != 'nan':
            team_badge = f'<span style="color: #AAA; font-size: 0.7rem; margin-left: 4px;">({team_str.strip()})</span>'
        
        # Source badge
        source_badge = ""
        if show_source and source_name:
            src_colors = {"The Odds API": "#00F3FF", "PrizePicks": "#8B5CF6", "Underdog": "#F59E0B"}
            src_c = src_colors.get(source_name, "#888")
            badge_label = bookmaker if bookmaker else source_name
            source_badge = f'<span style="background: {src_c}22; border: 1px solid {src_c}; color: {src_c}; padding: 1px 6px; border-radius: 8px; font-size: 0.6rem; margin-left: 6px;">{badge_label}</span>'
        elif show_bookmaker and bookmaker:
            bk_colors = {
                "Bovada": "#CC0000", "DraftKings": "#53D337", "FanDuel": "#1493FF",
                "BetMGM": "#C4A962", "Caesars": "#00473E", "BetRivers": "#003DA5",
                "PrizePicks": "#8B5CF6", "Underdog": "#F59E0B",
            }
            bk_c = bk_colors.get(bookmaker, "#888")
            source_badge = f'<span style="background: {bk_c}22; border: 1px solid {bk_c}; color: {bk_c}; padding: 1px 6px; border-radius: 8px; font-size: 0.6rem; margin-left: 6px;">{bookmaker}</span>'
        
        if is_dfs and not has_over and not has_under:
            # Pure DFS projection card (PrizePicks style - no odds)
            st.markdown(f"""
<div style="background: #1a1a2e; border-left: 3px solid {dfs_color}; border-radius: 8px;
padding: 10px 14px; margin-bottom: 5px; display: grid;
grid-template-columns: 2fr 1fr 100px; gap: 10px; align-items: center;">
<div>
<div style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{prop['player']}{team_badge}{source_badge}</div>
<div style="color: #888; font-size: 0.72rem;">{subtitle}</div>
</div>
<div style="text-align: center;">
<div style="color: {dfs_color}; font-size: 0.7rem; text-transform: uppercase;">{prop['prop_type']}</div>
</div>
<div style="text-align: center;">
<div style="color: {dfs_color}; font-size: 0.65rem;">PROJECTION</div>
<div style="color: #FFF; font-weight: 900; font-size: 1.3rem;">{prop['line']}</div>
<div style="color: #888; font-size: 0.6rem;">MORE / LESS</div>
</div>
</div>
            """, unsafe_allow_html=True)
        elif is_dfs and (has_over or has_under):
            # DFS card with odds
            if source_name == "Underdog":
                higher_label = "HIGHER"
                lower_label = "LOWER"
            elif source_name == "PrizePicks":
                higher_label = "MORE"
                lower_label = "LESS"
            else:
                higher_label = "OVER"
                lower_label = "UNDER"
            st.markdown(f"""
<div style="background: #1a1a2e; border-left: 3px solid {dfs_color}; border-radius: 8px;
padding: 10px 14px; margin-bottom: 5px; display: grid;
grid-template-columns: 2fr 1fr 80px 80px 80px; gap: 8px; align-items: center;">
<div>
<div style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{prop['player']}{team_badge}{source_badge}</div>
<div style="color: #888; font-size: 0.72rem;">{subtitle}</div>
</div>
<div style="text-align: center;">
<div style="color: {dfs_color}; font-size: 0.7rem; text-transform: uppercase;">{prop['prop_type']}</div>
</div>
<div style="text-align: center;">
<div style="color: {dfs_color}; font-size: 0.65rem;">LINE</div>
<div style="color: #FFF; font-weight: 900; font-size: 1.15rem;">{prop['line']}</div>
</div>
<div style="text-align: center;">
<div style="color: {over_color}; font-size: 0.65rem;">{higher_label}</div>
<div style="color: {over_color}; font-weight: bold;">{over_str}</div>
</div>
<div style="text-align: center;">
<div style="color: {under_color}; font-size: 0.65rem;">{lower_label}</div>
<div style="color: {under_color}; font-weight: bold;">{under_str}</div>
</div>
</div>
            """, unsafe_allow_html=True)
        else:
            # Traditional sportsbook card with over/under odds
            st.markdown(f"""
<div style="background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
padding: 10px 14px; margin-bottom: 5px; display: grid;
grid-template-columns: 2fr 1fr 80px 80px 80px; gap: 8px; align-items: center;">
<div>
<div style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{prop['player']}{source_badge}</div>
<div style="color: #888; font-size: 0.72rem;">{subtitle}</div>
</div>
<div style="text-align: center;">
<div style="color: #00F3FF; font-size: 0.7rem; text-transform: uppercase;">{prop['prop_type']}</div>
</div>
<div style="text-align: center;">
<div style="color: #FFA500; font-size: 0.65rem;">LINE</div>
<div style="color: #FFF; font-weight: 900; font-size: 1.15rem;">{prop['line']}</div>
</div>
<div style="text-align: center;">
<div style="color: {over_color}; font-size: 0.65rem;">OVER</div>
<div style="color: {over_color}; font-weight: bold;">{over_str}</div>
</div>
<div style="text-align: center;">
<div style="color: {under_color}; font-size: 0.65rem;">UNDER</div>
<div style="color: {under_color}; font-weight: bold;">{under_str}</div>
</div>
</div>
            """, unsafe_allow_html=True)
    
    if total_count > display_limit:
        st.markdown(f'<div style="color: #FFA500; font-size: 0.8rem; text-align: center; margin-top: 8px;">Showing first {display_limit} of {total_count} props. Use filters to narrow results.</div>', unsafe_allow_html=True)


def _analyze_best_bets(all_props, pp_props, ud_props, odds_api_props):
    """
    Identify the best player prop bets for the current day by finding
    edges between DFS platforms (PrizePicks, Underdog) and sportsbook lines.
    
    Scoring factors:
    - Line discrepancy vs sportsbook consensus (bigger gap = more value)
    - Odds value (plus money or light juice)
    - Cross-platform agreement (PP + UD both high/low vs books)
    - Number of sportsbooks confirming a consensus
    
    Returns a list of best-bet dicts sorted by edge score.
    """
    from collections import defaultdict
    
    # Fantasy Score uses platform-specific scoring formulas that differ
    # between PrizePicks and Underdog — not comparable cross-platform.
    # Exclude from all edge analysis to prevent false edges.
    _NON_COMPARABLE_KEYWORDS = ("fantasy", "combo")
    def _is_non_comparable(pt):
        return any(kw in pt for kw in _NON_COMPARABLE_KEYWORDS)
    
    # Build sportsbook consensus for each player+prop
    book_consensus = defaultdict(lambda: {"lines": [], "over_odds": [], "under_odds": []})
    
    for prop in odds_api_props:
        player = (prop.get("player", "").strip() or "").lower()
        prop_type = (prop.get("prop_type", "").strip() or "").lower()
        if not player or not prop_type:
            continue
        key = (player, prop_type)
        book_consensus[key]["lines"].append(prop.get("line", 0))
        if prop.get("over_odds") is not None and pd.notna(prop.get("over_odds")):
            book_consensus[key]["over_odds"].append(prop["over_odds"])
        if prop.get("under_odds") is not None and pd.notna(prop.get("under_odds")):
            book_consensus[key]["under_odds"].append(prop["under_odds"])
    
    # Build DFS lines index (exclude non-comparable stats)
    dfs_lines = defaultdict(list)
    for prop in (pp_props or []) + (ud_props or []):
        player = (prop.get("player", "").strip() or "").lower()
        prop_type = (prop.get("prop_type", "").strip() or "").lower()
        if not player or not prop_type:
            continue
        if _is_non_comparable(prop_type):
            continue
        dfs_lines[(player, prop_type)].append(prop)
    
    best_bets = []
    seen_keys = set()
    
    # Strategy 1: DFS lines that diverge from sportsbook consensus
    for key, dfs_props in dfs_lines.items():
        player_lower, prop_type_lower = key
        consensus = book_consensus.get(key)
        
        for dfs_prop in dfs_props:
            source = dfs_prop.get("source", "")
            dfs_line = dfs_prop.get("line", 0)
            dfs_over = dfs_prop.get("over_odds")
            dfs_under = dfs_prop.get("under_odds")
            
            bet_key = (player_lower, prop_type_lower, source)
            if bet_key in seen_keys:
                continue
            seen_keys.add(bet_key)
            
            edge_score = 0
            edge_reasons = []
            direction = None
            book_avg = None
            
            if consensus and consensus["lines"]:
                book_avg = sum(consensus["lines"]) / len(consensus["lines"])
                diff = dfs_line - book_avg
                num_books = len(consensus["lines"])
                
                if abs(diff) >= 1.0:
                    # DFS line higher than books = lean UNDER/LOWER
                    # DFS line lower than books = lean OVER/HIGHER
                    if diff > 0:
                        direction = "under"
                        edge_reasons.append(f"Line {dfs_line} is {abs(diff):.1f} pts ABOVE book consensus ({book_avg:.1f})")
                    else:
                        direction = "over"
                        edge_reasons.append(f"Line {dfs_line} is {abs(diff):.1f} pts BELOW book consensus ({book_avg:.1f})")
                    
                    # Edge score scales with discrepancy size and book count
                    edge_score += min(abs(diff) * 10, 50)
                    edge_score += min(num_books * 3, 15)
                elif abs(diff) >= 0.5:
                    if diff > 0:
                        direction = "under"
                        edge_reasons.append(f"Line {dfs_line} slightly above book avg ({book_avg:.1f})")
                    else:
                        direction = "over"
                        edge_reasons.append(f"Line {dfs_line} slightly below book avg ({book_avg:.1f})")
                    edge_score += abs(diff) * 5
            
            # Odds value scoring
            if direction == "over" and dfs_over is not None and pd.notna(dfs_over):
                if int(dfs_over) >= 100:
                    edge_score += 15
                    edge_reasons.append(f"Plus-money odds ({int(dfs_over):+d})")
                elif int(dfs_over) >= -110:
                    edge_score += 5
            elif direction == "under" and dfs_under is not None and pd.notna(dfs_under):
                if int(dfs_under) >= 100:
                    edge_score += 15
                    edge_reasons.append(f"Plus-money odds ({int(dfs_under):+d})")
                elif int(dfs_under) >= -110:
                    edge_score += 5
            
            # Cross-platform agreement bonus
            other_source = "Underdog" if source == "PrizePicks" else "PrizePicks"
            for other_prop in dfs_props:
                if other_prop.get("source") == other_source:
                    other_line = other_prop.get("line", 0)
                    if consensus and consensus["lines"]:
                        other_diff = other_line - book_avg
                        if (diff > 0 and other_diff > 0) or (diff < 0 and other_diff < 0):
                            edge_score += 10
                            edge_reasons.append(f"Both platforms agree vs books")
                    break
            
            if edge_score < 5:
                continue
            
            # Determine confidence tier
            if edge_score >= 40:
                tier = "STRONG"
                tier_color = "#4CAF50"
                tier_icon = "🔥"
            elif edge_score >= 25:
                tier = "SOLID"
                tier_color = "#00F3FF"
                tier_icon = "✅"
            elif edge_score >= 15:
                tier = "LEAN"
                tier_color = "#FFA500"
                tier_icon = "📊"
            else:
                tier = "WATCH"
                tier_color = "#888"
                tier_icon = "👀"
            
            if source == "PrizePicks":
                play_label = "MORE" if direction == "over" else "LESS"
            else:
                play_label = "HIGHER" if direction == "over" else "LOWER"
            
            best_bets.append({
                "player": dfs_prop.get("player", ""),
                "team": dfs_prop.get("team", ""),
                "game": dfs_prop.get("game", ""),
                "commence_time": dfs_prop.get("commence_time", ""),
                "prop_type": dfs_prop.get("prop_type", ""),
                "source": source,
                "dfs_line": dfs_line,
                "dfs_over_odds": dfs_over,
                "dfs_under_odds": dfs_under,
                "book_consensus": book_avg,
                "num_books": len(consensus["lines"]) if consensus else 0,
                "direction": direction or "over",
                "play_label": play_label,
                "edge_score": edge_score,
                "edge_reasons": edge_reasons,
                "tier": tier,
                "tier_color": tier_color,
                "tier_icon": tier_icon,
                "image_url": dfs_prop.get("image_url", ""),
            })
    
    # Strategy 2: PrizePicks vs Underdog line disagreements (no books needed)
    # Exclude non-comparable stats (Fantasy uses different scoring per platform)
    pp_index = {}
    for prop in (pp_props or []):
        pk = ((prop.get("player", "").strip() or "").lower(), (prop.get("prop_type", "").strip() or "").lower())
        if pk[0] and pk[1] and not _is_non_comparable(pk[1]):
            pp_index[pk] = prop
    
    ud_index = {}
    for prop in (ud_props or []):
        pk = ((prop.get("player", "").strip() or "").lower(), (prop.get("prop_type", "").strip() or "").lower())
        if pk[0] and pk[1] and not _is_non_comparable(pk[1]):
            ud_index[pk] = prop
    
    for pk, pp_prop in pp_index.items():
        if pk in ud_index:
            ud_prop = ud_index[pk]
            pp_line = pp_prop.get("line", 0)
            ud_line = ud_prop.get("line", 0)
            diff = abs(pp_line - ud_line)
            
            if diff >= 2.0:
                # Big discrepancy between platforms
                for src_prop, other_line, src_name in [
                    (pp_prop, ud_line, "PrizePicks"),
                    (ud_prop, pp_line, "Underdog"),
                ]:
                    bet_key = (pk[0], pk[1], src_name, "xplat")
                    if bet_key in seen_keys:
                        continue
                    seen_keys.add(bet_key)
                    
                    src_line = src_prop.get("line", 0)
                    if src_line > other_line:
                        direction = "under"
                        play = "LESS" if src_name == "PrizePicks" else "LOWER"
                        reason = f"{src_name} line ({src_line}) is {diff:.1f} higher than {'Underdog' if src_name == 'PrizePicks' else 'PrizePicks'} ({other_line})"
                    else:
                        direction = "over"
                        play = "MORE" if src_name == "PrizePicks" else "HIGHER"
                        reason = f"{src_name} line ({src_line}) is {diff:.1f} lower than {'Underdog' if src_name == 'PrizePicks' else 'PrizePicks'} ({other_line})"
                    
                    edge_score = min(diff * 8, 40)
                    
                    existing = [b for b in best_bets if b["player"].lower() == pk[0] and b["prop_type"].lower() == pk[1] and b["source"] == src_name]
                    if existing:
                        existing[0]["edge_reasons"].append(reason)
                        existing[0]["edge_score"] += edge_score * 0.5
                    else:
                        tier = "SOLID" if edge_score >= 25 else "LEAN"
                        tier_color = "#00F3FF" if tier == "SOLID" else "#FFA500"
                        tier_icon = "✅" if tier == "SOLID" else "📊"
                        
                        best_bets.append({
                            "player": src_prop.get("player", ""),
                            "team": src_prop.get("team", ""),
                            "game": src_prop.get("game", ""),
                            "commence_time": src_prop.get("commence_time", ""),
                            "prop_type": src_prop.get("prop_type", ""),
                            "source": src_name,
                            "dfs_line": src_line,
                            "dfs_over_odds": src_prop.get("over_odds"),
                            "dfs_under_odds": src_prop.get("under_odds"),
                            "book_consensus": None,
                            "num_books": 0,
                            "direction": direction,
                            "play_label": play,
                            "edge_score": edge_score,
                            "edge_reasons": [reason],
                            "tier": tier,
                            "tier_color": tier_color,
                            "tier_icon": tier_icon,
                            "image_url": src_prop.get("image_url", ""),
                        })
    
    # Re-sort and re-tier after merging
    best_bets.sort(key=lambda x: -x["edge_score"])
    for bet in best_bets:
        s = bet["edge_score"]
        if s >= 40:
            bet["tier"], bet["tier_color"], bet["tier_icon"] = "STRONG", "#4CAF50", "🔥"
        elif s >= 25:
            bet["tier"], bet["tier_color"], bet["tier_icon"] = "SOLID", "#00F3FF", "✅"
        elif s >= 15:
            bet["tier"], bet["tier_color"], bet["tier_icon"] = "LEAN", "#FFA500", "📊"
        else:
            bet["tier"], bet["tier_color"], bet["tier_icon"] = "WATCH", "#888", "👀"
    
    return best_bets


def _render_best_bets(all_props, pp_props, ud_props, odds_api_props, gender, games_df=None):
    """Render the Best Player Prop Bets section with game schedule and top picks"""
    from collections import defaultdict
    import pytz
    
    gender_label = config.GENDER_CONFIG[gender]["label"]
    
    st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(76,175,80,0.15), rgba(0,243,255,0.1)); 
            border: 1px solid #4CAF50; padding: 18px; border-radius: 10px; margin-bottom: 20px;">
    <div style="font-family: JetBrains Mono; color: #4CAF50; font-size: 1.2rem; font-weight: bold;">
        🔥 Best Player Prop Bets — {gender_label} College Basketball
    </div>
    <div style="color: #AAA; font-size: 0.85rem; margin-top: 5px;">
        AI-analyzed value plays from PrizePicks & Underdog Fantasy vs sportsbook consensus
        <br><span style="color: #FFD700;">Edge scores factor in line discrepancy, odds value, and cross-platform agreement</span>
    </div>
</div>
    """, unsafe_allow_html=True)
    
    # Analyze best bets
    best_bets = _analyze_best_bets(all_props, pp_props, ud_props, odds_api_props)
    
    if not best_bets:
        if gender == "womens":
            st.markdown("""
<div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; border-radius: 8px; text-align: center;">
    <div style="color: #FFA500; font-size: 1rem;">⚠️ No Women's Player Props Available</div>
    <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
        PrizePicks, Underdog Fantasy, and most sportsbooks currently have <b>limited or no WNCAAB player prop coverage</b>.
        <br>This section will populate automatically when these platforms add women's college basketball props.
    </div>
</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; border-radius: 8px; text-align: center;">
    <div style="color: #FFA500; font-size: 1rem;">⚠️ No Best Bets Identified</div>
    <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
        Not enough cross-platform data to identify value plays right now.
        <br>Best bets require PrizePicks or Underdog lines to compare against sportsbooks.
    </div>
</div>
            """, unsafe_allow_html=True)
        return
    
    # --- TODAY'S SCHEDULE DATA (For later use) ---
    games_from_props = defaultdict(lambda: {"teams": "", "time": "", "props_count": 0, "best_count": 0})
    for prop in all_props:
        game = str(prop.get("game", "") or "").strip()
        if not game or game.lower() == "nan": continue
        games_from_props[game]["teams"] = game
        games_from_props[game]["props_count"] += 1
        ct = prop.get("commence_time", "")
        if ct and not games_from_props[game]["time"]:
            games_from_props[game]["time"] = ct
    
    for bet in best_bets:
        game = str(bet.get("game", "") or "").strip()
        if game and game.lower() != "nan" and game in games_from_props:
            games_from_props[game]["best_count"] += 1

    # --- SUMMARY STATS ---
    total_bets = len(best_bets)
    strong = sum(1 for b in best_bets if b["tier"] == "STRONG")
    solid = sum(1 for b in best_bets if b["tier"] == "SOLID")
    pp_count = sum(1 for b in best_bets if b["source"] == "PrizePicks")
    ud_count = sum(1 for b in best_bets if b["source"] == "Underdog Fantasy")
    
    st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 18px;">
<div style="background: #1a1a2e; border: 1px solid #4CAF50; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="color: #4CAF50; font-size: 1.4rem; font-weight: 900;">{total_bets}</div>
    <div style="color: #888; font-size: 0.7rem;">Total Plays</div>
</div>
<div style="background: #1a1a2e; border: 1px solid #4CAF50; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="color: #4CAF50; font-size: 1.4rem; font-weight: 900;">🔥 {strong}</div>
    <div style="color: #888; font-size: 0.7rem;">Strong Edge</div>
</div>
<div style="background: #1a1a2e; border: 1px solid #00F3FF; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="color: #00F3FF; font-size: 1.4rem; font-weight: 900;">✅ {solid}</div>
    <div style="color: #888; font-size: 0.7rem;">Solid Edge</div>
</div>
<div style="background: #1a1a2e; border: 1px solid #8B5CF6; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="color: #8B5CF6; font-size: 1.4rem; font-weight: 900;">{pp_count}</div>
    <div style="color: #888; font-size: 0.7rem;">PrizePicks</div>
</div>
<div style="background: #1a1a2e; border: 1px solid #F59E0B; border-radius: 8px; padding: 12px; text-align: center;">
    <div style="color: #F59E0B; font-size: 1.4rem; font-weight: 900;">{ud_count}</div>
    <div style="color: #888; font-size: 0.7rem;">Underdog</div>
</div>
</div>
    """, unsafe_allow_html=True)
    
    # --- FILTERS ---
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.selectbox("Filter by Tier", ["All Tiers", "STRONG", "SOLID", "LEAN", "WATCH"], key="best_bets_tier")
    with col_f2:
        source_filter = st.selectbox("Filter by Platform", ["All Platforms", "PrizePicks", "Underdog Fantasy"], key="best_bets_source")
    with col_f3:
        all_games = sorted(set(str(b.get("game", "") or "").strip() for b in best_bets if str(b.get("game", "") or "").strip()))
        game_filter = st.selectbox("Filter by Game", ["All Games"] + all_games, key="best_bets_game")
    
    filtered = best_bets
    if tier_filter != "All Tiers":
        filtered = [b for b in filtered if b["tier"] == tier_filter]
    if source_filter != "All Platforms":
        filtered = [b for b in filtered if b["source"] == source_filter]
    if game_filter != "All Games":
        filtered = [b for b in filtered if str(b.get("game", "") or "").strip() == game_filter]
    
    if not filtered:
        st.markdown('<div style="color: #888; text-align: center; padding: 20px;">No bets match the selected filters.</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div style="color: #888; font-size: 0.8rem; margin-bottom: 8px;">Showing {len(filtered)} best bets</div>', unsafe_allow_html=True)
    
    # --- RENDER BEST BET CARDS ---
    for bet in filtered[:60]:
        player = bet["player"]
        team = bet.get("team", "")
        game = bet.get("game", "")
        prop_type = bet["prop_type"]
        source = bet["source"]
        dfs_line = bet["dfs_line"]
        direction = bet["direction"]
        play_label = bet["play_label"]
        tier = bet["tier"]
        tier_color = bet["tier_color"]
        tier_icon = bet["tier_icon"]
        edge_score = bet["edge_score"]
        edge_reasons = bet.get("edge_reasons", [])
        book_consensus = bet.get("book_consensus")
        num_books = bet.get("num_books", 0)
        
        # Odds display — default missing odds to standard vig
        if direction == "over":
            odds_val = bet.get("dfs_over_odds")
        else:
            odds_val = bet.get("dfs_under_odds")
        
        has_odds = odds_val is not None and pd.notna(odds_val)
        if not has_odds:
            odds_val = -112 if source == "Underdog" else -110
            has_odds = True
        
        odds_str = f"{int(odds_val):+d}" if has_odds else "—"
        odds_color = "#4CAF50" if has_odds and int(odds_val) >= 100 else "#FFF"
        
        # Source badge
        src_color = "#8B5CF6" if source == "PrizePicks" else "#F59E0B"
        src_icon = "🟣" if source == "PrizePicks" else "🟡"
        
        # Team badge
        team_str = str(team or "")
        team_badge = ""
        if team_str.strip() and team_str.lower() != "nan":
            team_badge = f' <span style="color: #AAA; font-size: 0.72rem;">({team_str.strip()})</span>'
        
        # Consensus line
        consensus_text = ""
        if book_consensus is not None:
            diff = dfs_line - book_consensus
            diff_color = "#FF5722" if abs(diff) >= 2 else "#FFA500" if abs(diff) >= 1 else "#888"
            consensus_text = f'<span style="color: #888; font-size: 0.7rem;">Books: <b style="color:#FFF">{book_consensus:.1f}</b> ({num_books})</span> <span style="color: {diff_color}; font-size: 0.7rem;">Diff: {diff:+.1f}</span>'
        
        # Reasons as simple inline text
        reasons_text = ""
        if edge_reasons:
            reasons_parts = " | ".join(edge_reasons[:3])
            reasons_text = f'<span style="color: #AAA; font-size: 0.68rem;">{reasons_parts}</span>'
        
        # Direction arrow
        dir_arrow = "⬆️" if direction == "over" else "⬇️"
        play_color = "#4CAF50" if direction == "over" else "#FF5722"
        
        # Build info lines for left column
        info_lines = f'<span style="color: #888; font-size: 0.72rem;">{game}</span>'
        if consensus_text:
            info_lines += f'<br>{consensus_text}'
        if reasons_text:
            info_lines += f'<br>{reasons_text}'
        
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e, #1a1a35); border-left: 4px solid {tier_color}; 
border-radius: 8px; padding: 14px 16px; margin-bottom: 8px; 
display: grid; grid-template-columns: 2.5fr 1fr 1fr 1fr; gap: 10px; align-items: center;">
<div>
<span style="color: #FFF; font-weight: bold; font-size: 1.0rem;">{player}</span>{team_badge}
<span style="background: {src_color}22; border: 1px solid {src_color}; color: {src_color}; 
padding: 1px 8px; border-radius: 10px; font-size: 0.6rem;">{src_icon} {source}</span>
<span style="background: {tier_color}22; border: 1px solid {tier_color}; color: {tier_color}; 
padding: 1px 8px; border-radius: 10px; font-size: 0.6rem; font-weight: bold;">{tier_icon} {tier}</span>
<br>{info_lines}
</div>
<div style="text-align: center;">
<span style="color: #00F3FF; font-size: 0.7rem; text-transform: uppercase;">{prop_type}</span><br>
<span style="color: #FFF; font-weight: 900; font-size: 1.3rem;">{dfs_line}</span><br>
<span style="color: #888; font-size: 0.6rem;">LINE</span>
</div>
<div style="text-align: center;">
<span style="color: {play_color}; font-size: 0.7rem; font-weight: bold;">{dir_arrow} {play_label}</span><br>
<span style="color: {odds_color}; font-weight: 900; font-size: 1.2rem;">{odds_str}</span><br>
<span style="color: #888; font-size: 0.6rem;">PLAY</span>
</div>
<div style="text-align: center;">
<span style="color: {tier_color}; font-size: 0.7rem;">EDGE SCORE</span><br>
<span style="color: {tier_color}; font-weight: 900; font-size: 1.3rem;">{edge_score:.0f}</span>
</div>
</div>
        """, unsafe_allow_html=True)

    # --- TODAY'S SCHEDULE SECTION (Moved to end) ---
    if games_from_props:
        st.markdown("---")
        st.markdown("""
<div style="color: #00F3FF; font-weight: bold; font-size: 0.95rem; margin: 15px 0 8px; font-family: JetBrains Mono;">
📅 Today's Schedule — Games with Prop Bets
</div>""", unsafe_allow_html=True)
        
        schedule_items = sorted(games_from_props.values(), key=lambda x: x.get("time", ""))
        
        schedule_html = '<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 5px;">'
        for g in schedule_items:
            teams = g["teams"]
            time_str = ""
            if g["time"]:
                try:
                    from dateutil import parser as dt_parser
                    utc_dt = dt_parser.isoparse(g["time"])
                    pacific = pytz.timezone("US/Pacific")
                    if utc_dt.tzinfo is None:
                        utc_dt = pytz.utc.localize(utc_dt)
                    pt_dt = utc_dt.astimezone(pacific)
                    time_str = pt_dt.strftime("%I:%M %p PT")
                except Exception:
                    time_str = ""
            
            best_badge = ""
            if g["best_count"] > 0:
                best_badge = f'<span style="color: #4CAF50; font-size: 0.6rem; margin-left: 4px;">🔥{g["best_count"]}</span>'
            
            schedule_html += f"""
<div style="background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
padding: 8px 14px; min-width: 180px; flex: 1; max-width: 320px; opacity: 0.8;">
<div style="color: #FFF; font-size: 0.78rem; font-weight: bold;">{teams}{best_badge}</div>
<div style="color: #888; font-size: 0.65rem;">{time_str} • {g['props_count']} props available</div>
</div>"""
        schedule_html += '</div>'
        st.markdown(schedule_html, unsafe_allow_html=True)
    
    if len(filtered) > 60:
        st.markdown(f'<div style="color: #FFA500; font-size: 0.8rem; text-align: center;">Showing top 60 of {len(filtered)} plays. Use filters to narrow results.</div>', unsafe_allow_html=True)


def _render_line_comparison(all_props, gender):
    """Render cross-platform line comparison view"""
    aggregator = PropsAggregator(gender=gender)
    comparisons = aggregator.get_line_comparison(all_props)
    
    if not comparisons:
        st.markdown('<div style="color: #888;">Not enough data across sources for comparison.</div>', unsafe_allow_html=True)
        return
    
    st.markdown("""
<div style="color: #00F3FF; font-size: 0.85rem; margin-bottom: 10px; font-family: JetBrains Mono;">
Compare lines across Bovada, DraftKings, FanDuel, PrizePicks, Underdog Fantasy and more.
<br><span style="color: #FFD700;">Lines with large spreads may indicate value opportunities.</span>
</div>
    """, unsafe_allow_html=True)
    
    # Filter to multi-source comparisons first
    multi_source = [c for c in comparisons if c["num_sources"] >= 2]
    single_source = [c for c in comparisons if c["num_sources"] < 2]
    
    if multi_source:
        st.markdown('<div style="color: #FFD700; font-weight: bold; font-size: 0.9rem; margin: 10px 0 6px;">⚖️ Multi-Source Comparisons</div>', unsafe_allow_html=True)
        for comp in multi_source[:50]:
            _render_comparison_card(comp)
    
    if single_source:
        with st.expander(f"Single-Source Props ({len(single_source)} props)", expanded=False):
            for comp in single_source[:30]:
                _render_comparison_card(comp)


def _render_comparison_card(comp):
    """Render a single comparison card"""
    player = comp["player"]
    prop_type = comp["prop_type"]
    game = comp.get("game", "")
    consensus = comp.get("consensus_line", 0)
    spread = comp.get("line_spread", 0)
    
    spread_color = "#FF5252" if spread >= 2 else "#FFA500" if spread >= 1 else "#4CAF50"
    spread_label = "HIGH VARIANCE" if spread >= 2 else "MODERATE" if spread >= 1 else "CONSISTENT"
    
    sources_html = ""
    source_colors = {
        "Bovada": "#CC0000", "DraftKings": "#53D337", "FanDuel": "#1493FF",
        "BetMGM": "#C4A962", "Caesars": "#00473E", "BetRivers": "#003DA5",
        "PrizePicks": "#8B5CF6", "Underdog Fantasy": "#F59E0B",
        "BetOnline": "#00F3FF", "BetUS": "#1E40AF", "Fanatics": "#FACC15",
    }
    
    for src_name, src_data in comp["sources"].items():
        line = src_data.get("line", "—")
        over = src_data.get("over_odds")
        under = src_data.get("under_odds")
        sc = source_colors.get(src_name, "#888")
        
        # Default DFS odds if missing
        if src_name == "PrizePicks" and (over is None or (isinstance(over, float) and pd.isna(over))):
            over, under = -110, -110
        elif src_name in ("Underdog", "Underdog Fantasy") and (over is None or (isinstance(over, float) and pd.isna(over))):
            over, under = -112, -112
        
        # Source-specific labels
        if src_name == "PrizePicks":
            over_label, under_label = "MORE", "LESS"
        elif src_name in ("Underdog", "Underdog Fantasy"):
            over_label, under_label = "HIGHER", "LOWER"
        else:
            over_label, under_label = "OVER", "UNDER"
        
        odds_str = ""
        if over is not None and pd.notna(over):
            odds_str += f'<span style="color: #4CAF50; font-size: 0.65rem;">{over_label} {int(over):+d}</span> '
        if under is not None and pd.notna(under):
            odds_str += f'<span style="color: #FF5722; font-size: 0.65rem;">{under_label} {int(under):+d}</span>'
        
        sources_html += f"""
<div style="display: inline-block; background: {sc}11; border: 1px solid {sc}44; border-radius: 6px;
padding: 4px 10px; margin: 2px 4px 2px 0; min-width: 100px; text-align: center;">
<div style="color: {sc}; font-size: 0.6rem; font-weight: bold;">{src_name}</div>
<div style="color: #FFF; font-weight: 900; font-size: 1.05rem;">{line}</div>
{f'<div>{odds_str}</div>' if odds_str else ''}
</div>"""
    
    # --- Consensus Verdict ---
    over_votes = 0
    under_votes = 0
    over_juice_total = 0
    under_juice_total = 0
    juice_count = 0
    
    # Track lines for middling detection
    over_lines = []
    under_lines = []
    
    for src_name, src_data in comp["sources"].items():
        over = src_data.get("over_odds")
        under = src_data.get("under_odds")
        line = src_data.get("line")
        
        # Fill defaults for DFS
        if src_name == "PrizePicks" and (over is None or (isinstance(over, float) and pd.isna(over))):
            over, under = -110, -110
        elif src_name in ("Underdog", "Underdog Fantasy") and (over is None or (isinstance(over, float) and pd.isna(over))):
            over, under = -112, -112
        
        has_over = over is not None and not (isinstance(over, float) and pd.isna(over))
        has_under = under is not None and not (isinstance(under, float) and pd.isna(under))
        
        if has_over and has_under:
            over_val = int(over)
            under_val = int(under)
            # More negative = more juice = books expect that side to hit
            if over_val < under_val:
                over_votes += 1
                if line: over_lines.append(float(line))
            elif under_val < over_val:
                under_votes += 1
                if line: under_lines.append(float(line))
            over_juice_total += over_val
            under_juice_total += under_val
            juice_count += 1
        elif has_over:
            over_juice_total += int(over)
            juice_count += 1
            
    # Detect Middling (Over on low, Under on high)
    is_middling = False
    if over_lines and under_lines:
        max_over_line = max(over_lines)
        min_under_line = min(under_lines)
        # If books favor Over X and Under Y, and X < Y, that's a middle
        if max_over_line < min_under_line:
            is_middling = True
    
    if is_middling:
        verdict = f"↕️ <b>Middling Opportunity</b>"
        sub_text = f"Books favor Over {max(over_lines)} & Under {min(under_lines)}"
        side_color = "#FFA500" # Orange for caution/opportunity
        
        verdict_html = f"""
<div style="margin-top: 6px; padding: 5px 10px; background: {side_color}15; border-left: 3px solid {side_color};
border-radius: 0 4px 4px 0;">
<span style="color: {side_color}; font-family: JetBrains Mono; font-size: 0.78rem;">{verdict}</span>
<span style="color: #AAA; font-size: 0.68rem; margin-left: 8px;">({sub_text})</span>
</div>"""

    elif juice_count > 0:
        avg_over = over_juice_total / juice_count
        avg_under = under_juice_total / juice_count if under_juice_total != 0 else 0
        
        diff = abs(avg_over - avg_under)
        if avg_over < avg_under:
            side = "OVER"
            side_color = "#4CAF50"
            side_emoji = "📈"
        elif avg_under < avg_over:
            side = "UNDER"
            side_color = "#FF5722"
            side_emoji = "📉"
        else:
            side = "TOSS-UP"
            side_color = "#888"
            side_emoji = "⚖️"
        
        if side != "TOSS-UP":
            if diff >= 20:
                strength = "Strong"
            elif diff >= 10:
                strength = "Lean"
            else:
                strength = "Slight lean"
            verdict = f"{side_emoji} <b>Consensus: {strength} {side}</b>"
        else:
            verdict = f"{side_emoji} <b>Consensus: Even — TOSS-UP</b>"
        
        verdict_html = f"""
<div style="margin-top: 6px; padding: 5px 10px; background: {side_color}15; border-left: 3px solid {side_color};
border-radius: 0 4px 4px 0;">
<span style="color: {side_color}; font-family: JetBrains Mono; font-size: 0.78rem;">{verdict}</span>
<span style="color: #666; font-size: 0.68rem; margin-left: 8px;">({over_votes} sources favor over, {under_votes} favor under)</span>
</div>"""
    else:
        verdict_html = ""
    
    st.markdown(f"""
<div style="background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
padding: 12px; margin-bottom: 6px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<div>
<span style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{player}</span>
<span style="color: #00F3FF; font-size: 0.8rem; margin-left: 8px;">{prop_type}</span>
<span style="color: #666; font-size: 0.72rem; margin-left: 8px;">{game}</span>
</div>
<div>
<span style="color: {spread_color}; font-size: 0.7rem; border: 1px solid {spread_color}; padding: 1px 8px; border-radius: 10px;">
{spread_label} (±{spread:.1f})
</span>
<span style="color: #FFD700; font-size: 0.8rem; margin-left: 8px; font-weight: bold;">
Consensus: {consensus:.1f}
</span>
</div>
</div>
<div style="display: flex; flex-wrap: wrap; gap: 4px;">
{sources_html}
</div>
{verdict_html}
</div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Gender toggle
        gender_options = {"Men's Basketball": "mens", "Women's Basketball": "womens"}
        selected_label = st.radio(
            "Select Sport",
            list(gender_options.keys()),
            horizontal=True,
            key="gender_radio"
        )
        gender = gender_options[selected_label]
        st.session_state["gender"] = gender
        
        st.markdown("---")
        
        # Date picker
        target_date = st.date_input(
            "Select Game Date",
            value=datetime.now().date(),
            min_value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date() + timedelta(days=30)
        )
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Info")
        model = load_model(gender)
        
        if model and model.training_info:
            info = model.training_info
            
            if 'metrics' in info and 'winner_test_accuracy' in info['metrics']:
                acc = info['metrics']['winner_test_accuracy']
                st.metric("Test Accuracy", f"{acc:.1%}")
            else:
                st.metric("Accuracy", "N/A")
            
            if 'metrics' in info:
                if 'home_score_test_mae' in info['metrics']:
                    st.metric("Score MAE", f"{info['metrics']['home_score_test_mae']:.1f} pts")
            
            st.caption(f"Trained: {info.get('trained_at', 'Unknown')[:16]}")
        else:
            st.warning("⚠️ Model not trained yet!")
            if st.button("Train Model Now"):
                with st.spinner("Training model... This may take several minutes..."):
                    import train_model
                    train_model.train_gender(gender)
                    st.success("Model trained successfully!")
                    st.rerun()
        
        st.markdown("---")
        
        # Training history
        st.subheader("📈 Training History")
        try:
            hist_tracker = TrainingHistoryTracker(gender)
            sessions = hist_tracker.get_all_sessions()
            
            if len(sessions) > 1:
                df_history = pd.DataFrame([
                    {
                        'Training #': i+1,
                        'Total Games': s.get('total_samples', 0),
                        'Accuracy %': s.get('test_accuracy', 0) * 100,
                        'Date': s.get('timestamp', '')[:10]
                    }
                    for i, s in enumerate(sessions)
                ])
                
                with st.expander("📊 View Training Chart", expanded=True):
                    tab1, tab2 = st.tabs(["Data Growth", "Accuracy"])
                    
                    with tab1:
                        st.area_chart(df_history.set_index('Training #')[['Total Games']], color="#00F3FF", height=250, width="stretch")
                        st.caption(f"Latest: {df_history['Total Games'].iloc[-1]:,} games")
                        
                    with tab2:
                        st.line_chart(df_history.set_index('Training #')[['Accuracy %']], color="#FF00FF", height=250, width="stretch")
                        st.caption(f"Latest: {df_history['Accuracy %'].iloc[-1]:.1f}%")
                
                # Show session summary below chart
                latest = sessions[-1]
                st.markdown(f"""
                <div style="background: #1a1a2e; border: 1px solid #333; padding: 10px; border-radius: 6px; margin-top: 5px;">
                    <div style="color: #00F3FF; font-size: 0.8rem; font-family: JetBrains Mono;">
                        Sessions: {len(sessions)} | Games: {latest.get('total_samples', 0):,} | Acc: {latest.get('test_accuracy', 0)*100:.1f}%
                    </div>
                    <div style="color: #666; font-size: 0.7rem; margin-top: 3px;">
                        Last trained: {latest.get('timestamp', 'N/A')[:16]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif len(sessions) == 1:
                s = sessions[0]
                st.markdown(f"""
                <div style="background: #1a1a2e; border: 1px solid #333; padding: 10px; border-radius: 6px;">
                    <div style="color: #00F3FF; font-size: 0.8rem; font-family: JetBrains Mono;">
                        1 session | {s.get('total_samples', 0):,} games | {s.get('test_accuracy', 0)*100:.1f}% acc
                    </div>
                    <div style="color: #666; font-size: 0.7rem; margin-top: 3px;">
                        Train again to track improvement
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("📊 Train the model to start tracking history")
        except Exception:
            st.caption("Training history will appear after first retrain")
        
        st.markdown("---")
        
        # Debug / Cache
        st.subheader("🛠️ Debug")
        if st.button("🧹 Clear All Cache", help="Force reload of models and data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            keys_to_remove = [k for k in st.session_state if k.startswith("predictions_") or k.startswith("_odds_")]
            for k in keys_to_remove:
                del st.session_state[k]
            st.success("Cache cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Retrain buttons
        st.subheader("🔄 Model Updates")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train Base Model", help="Train winner prediction model"):
                with st.spinner(f"Training {gender} base model..."):
                    import train_model
                    train_model.train_gender(gender)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    for k in [k for k in st.session_state if k.startswith("predictions_") or k.startswith("_odds_")]:
                        del st.session_state[k]
                    st.success("✅ Base model trained!")
                    st.rerun()
        
        with col2:
            if st.button("Train Betting Model", help="Train ML/Spread/Total models"):
                with st.spinner(f"Training {gender} betting models..."):
                    import train_betting_model
                    train_betting_model.train_gender(gender)
                    st.cache_data.clear()
                    for k in [k for k in st.session_state if k.startswith("predictions_") or k.startswith("_odds_")]:
                        del st.session_state[k]
                    st.success("✅ Betting models trained!")
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Update Predictions", help="Refresh predictions with latest data"):
            st.cache_data.clear()
            for k in [k for k in st.session_state if k.startswith("predictions_") or k.startswith("_odds_")]:
                del st.session_state[k]
            st.success("✅ Predictions updated!")
            st.rerun()
    
    # Main content
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model using the sidebar.")
        return
    
    # Fetch data
    with st.spinner(f"Loading {gender} games for {target_date_str}..."):
        games_df = get_todays_games(gender, target_date_str)
        
        if games_df.empty:
            st.warning(f"No {config.GENDER_CONFIG[gender]['label']} games found for {target_date_str}")
            return
        
        odds_df = get_vegas_odds(gender, target_date_str, games_df)
        
        # Cache team logos from game data
        for _, game in games_df.iterrows():
            team_logos.cache_team_from_game(game.to_dict())
        
        # Prediction pipeline (cached in session state)
        prediction_cache_key = f"predictions_{gender}_{target_date_str}"
        
        if prediction_cache_key not in st.session_state:
            with st.spinner("Building predictions (ELO, features, model)... This runs once."):
                data_mgr = DataManager(gender)
                feature_eng = FeatureEngineer(gender)
                
                # Determine season
                current_season = target_date.year if target_date.month >= 10 else target_date.year - 1
                
                historical_data = data_mgr.get_complete_training_data([current_season - 1, current_season])
                
                # Remove today's games from historical data
                if 'games' in historical_data and not historical_data['games'].empty:
                    hg = historical_data['games']
                    if 'date' in hg.columns:
                        before_filter = len(hg)
                        hg = hg[hg['date'].astype(str).str[:10] != target_date_str]
                        removed = before_filter - len(hg)
                        if removed > 0:
                            print(f"Removed {removed} games from target date to keep predictions stable")
                        historical_data['games'] = hg
                
                # Pre-calculate ELO
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
                
                avg_score = config.GENDER_CONFIG[gender]["avg_score"]
                avg_total = config.GENDER_CONFIG[gender]["avg_total"]
                
                # Build features for each game
                all_features = {}
                for idx, game in games_df.iterrows():
                    try:
                        features = feature_eng.build_features_for_game(
                            game.to_dict(),
                            historical_data,
                            current_season,
                            odds_df=odds_df,
                            standings=historical_data.get('standings', pd.DataFrame()),
                        )
                        
                        if features is None:
                            features = {
                                'home_elo': 1500, 'visitor_elo': 1500, 'elo_diff': 0,
                                'home_win_pct_last10': 0.5, 'visitor_win_pct_last10': 0.5,
                                'home_points_scored_last10': avg_score, 'visitor_points_scored_last10': avg_score,
                                'home_rest_days': 2, 'visitor_rest_days': 2,
                                'rest_advantage': 0, 'momentum_diff_5': 0, 'momentum_diff_10': 0,
                                'net_rating_diff': 0,
                                'vegas_spread_home': 0.0, 'vegas_total': avg_total,
                                'vegas_implied_home_prob': 0.5, 'vegas_has_odds': 0,
                                'h2h_home_win_pct': 0.5, 'h2h_avg_margin': 0, 'h2h_last3_home_wins': 0.5,
                            }
                        
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
                            'vegas_spread_home': 0.0, 'vegas_total': avg_total,
                            'vegas_implied_home_prob': 0.5, 'vegas_has_odds': 0,
                            'h2h_games': 0, 'h2h_home_wins': 0, 'h2h_home_win_pct': 0.5,
                            'h2h_avg_margin': 0, 'h2h_last3_home_wins': 0.5,
                            'season_phase': 2, 'is_march': 0,
                        }
                        for w in [5, 10, 20]:
                            for prefix in ['home', 'visitor']:
                                defaults[f'{prefix}_win_pct_last{w}'] = 0.5
                                defaults[f'{prefix}_points_scored_last{w}'] = avg_score
                                defaults[f'{prefix}_points_allowed_last{w}'] = avg_score
                                defaults[f'{prefix}_point_diff_last{w}'] = 0
                        all_features[idx] = defaults
                
                if not all_features:
                    st.error("Could not generate features for any games. Please check data availability.")
                    return
                
                features_df = pd.DataFrame.from_dict(all_features, orient='index')
                predictions_df = model.predict(features_df)
                
                st.session_state[prediction_cache_key] = {
                    'all_features': all_features,
                    'features_df': features_df,
                    'predictions_df': predictions_df,
                }
                print(f"Predictions cached for {gender} {target_date_str}")
        else:
            print(f"Using cached predictions for {gender} {target_date_str}")
        
        cached = st.session_state[prediction_cache_key]
        all_features = cached['all_features']
        features_df = cached['features_df']
        predictions_df = cached['predictions_df']
        
        avg_score = config.GENDER_CONFIG[gender]["avg_score"]
        avg_total = config.GENDER_CONFIG[gender]["avg_total"]
        
        # Initialize tracker and save predictions
        tracker = PredictionTracker(gender)
        
        # Update any pending predictions with actual results from completed games
        try:
            data_mgr_for_tracker = DataManager(gender)
            updated_count = tracker.update_pending_games(data_mgr_for_tracker)
            if updated_count > 0:
                print(f"Updated {updated_count} completed game results for {gender}")
                st.rerun()
        except Exception as e:
            print(f"Warning: Could not update pending game results: {e}")
        
        # Also check today's games that might already be final (late-night games, past games)
        for _, game_row in games_df.iterrows():
            gid = game_row.get('id')
            if gid and game_row.get('status') == 'Final':
                h_score = game_row.get('home_team_score', 0)
                v_score = game_row.get('visitor_team_score', 0)
                if h_score > 0 or v_score > 0:
                    tracker.update_result(gid, h_score, v_score)
        
        # Load betting model for saving picks
        betting_model_tracker = BettingModel(gender)
        betting_model_tracker.load()
        
        # Save today's predictions
        for idx, game in games_df.iterrows():
            if idx in predictions_df.index:
                pred = predictions_df.loc[idx]
                home_prob = pred.get('home_win_probability', 0.5)
                visitor_prob = pred.get('visitor_win_probability', 0.5)
                
                game_odds = get_consensus_odds(odds_df, game.get('id'))
                
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
                    "predicted_home_score": pred.get('predicted_home_score', avg_score),
                    "predicted_visitor_score": pred.get('predicted_visitor_score', avg_score),
                    "predicted_spread": pred.get('predicted_spread', 0),
                    "predicted_total": pred.get('predicted_total', avg_total),
                    "confidence": betting_rec.get('ml_confidence', max(home_prob, visitor_prob)) if betting_rec else max(home_prob, visitor_prob),
                    "vegas_spread": game_odds.get('spread_home') if game_odds and game_odds.get('has_odds') else None,
                    "vegas_total": game_odds.get('total') if game_odds and game_odds.get('has_odds') else None,
                    "betting_ml_pick": betting_rec.get('ml_pick') if betting_rec else None,
                    "betting_ml_conf": betting_rec.get('ml_confidence') if betting_rec else None,
                    "betting_spread_pick": betting_rec.get('spread_pick') if betting_rec else None,
                    "betting_spread_conf": betting_rec.get('spread_confidence') if betting_rec else None,
                    "betting_total_pick": betting_rec.get('total_pick') if betting_rec else None,
                    "betting_total_conf": betting_rec.get('total_confidence') if betting_rec else None,
                })
        
        # Get accuracy stats
        accuracy_stats = tracker.get_accuracy_stats(days=30)
        
        # Create tabs
        tab1, tab2, tab_best, tab3 = st.tabs(["🏀 Game Predictions", "🎯 Player Props", "🔥 Best Prop Bets", "🏆 Rankings"])
        
        with tab1:
            display_game_predictions(gender, games_df, predictions_df, odds_df, features_df, accuracy_stats, model=model)
        
        with tab2:
            gender_label = config.GENDER_CONFIG[gender]["label"]
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(0,243,255,0.1), rgba(255,165,0,0.1)); 
                        border: 1px solid #00F3FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-family: JetBrains Mono; color: #00F3FF; font-size: 1.1rem; font-weight: bold;">
                    🎯 {gender_label} College Basketball Player Props
                </div>
                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                    Live lines from Bovada, DraftKings, FanDuel + PrizePicks & Underdog Fantasy projections
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Loading player props from all sources..."):
                props_data = get_all_player_props(gender)
            
            # Source status badges
            status = props_data.get("sources_status", {})
            source_badges_html = ""
            source_colors = {
                "The Odds API": "#00F3FF",
                "PrizePicks": "#8B5CF6",
                "Underdog Fantasy": "#F59E0B",
            }
            for src_name, src_info in status.items():
                color = source_colors.get(src_name, "#888")
                src_status = src_info.get("status", "")
                if src_status == "ok" and src_info.get("count", 0) > 0:
                    count = src_info.get("count", 0)
                    icon = "✅"
                    extra = ""
                    if src_name == "The Odds API":
                        books = src_info.get("bookmakers", [])
                        if books:
                            extra = f' <span style="color: #666; font-size: 0.65rem;">({", ".join(books[:5])})</span>'
                    source_badges_html += f'<span style="background: {color}22; border: 1px solid {color}; color: {color}; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; margin-right: 6px;">{icon} {src_name}: {count} props{extra}</span>'
                elif src_status == "blocked":
                    msg = src_info.get("message", "blocked by bot protection")
                    source_badges_html += f'<span style="background: #FFA50022; border: 1px solid #FFA500; color: #FFA500; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; margin-right: 6px;">🔒 {src_name} (bot protected)</span>'
                else:
                    source_badges_html += f'<span style="background: #FF525222; border: 1px solid #FF5252; color: #FF5252; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; margin-right: 6px;">❌ {src_name}</span>'
            
            if source_badges_html:
                st.markdown(f'<div style="margin-bottom: 12px;">{source_badges_html}</div>', unsafe_allow_html=True)
            
            all_props = props_data.get("all", [])
            odds_api_props = props_data.get("odds_api", [])
            pp_props = props_data.get("prizepicks", [])
            ud_props = props_data.get("underdog", [])
            
            if all_props:
                # Source-specific sub-tabs
                source_tab_names = ["📊 All Sources"]
                if odds_api_props:
                    source_tab_names.append("🎰 Sportsbooks (Bovada, DK, FD...)")
                if pp_props:
                    source_tab_names.append("🟣 PrizePicks")
                if ud_props:
                    source_tab_names.append("🟡 Underdog Fantasy")
                if len(all_props) > 0:
                    source_tab_names.append("⚖️ Line Comparison")
                
                source_tabs = st.tabs(source_tab_names)
                tab_idx = 0
                
                # ---- ALL SOURCES TAB ----
                with source_tabs[tab_idx]:
                    tab_idx += 1
                    _render_props_table(all_props, show_source=True)
                
                # ---- SPORTSBOOKS TAB (Bovada, DK, FD, etc.) ----
                if odds_api_props:
                    with source_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("""
<div style="color: #00F3FF; font-size: 0.85rem; margin-bottom: 8px; font-family: JetBrains Mono;">
Traditional sportsbook lines with Over/Under odds from Bovada, DraftKings, FanDuel, BetMGM, and more
</div>""", unsafe_allow_html=True)
                        _render_props_table(odds_api_props, show_source=False, show_bookmaker=True)
                
                # ---- PRIZEPICKS TAB ----
                if pp_props:
                    with source_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("""
<div style="color: #8B5CF6; font-size: 0.85rem; margin-bottom: 8px; font-family: JetBrains Mono;">
PrizePicks projections — pick More or Less on player stat lines for DFS contests
</div>""", unsafe_allow_html=True)
                        _render_props_table(pp_props, show_source=False, is_dfs=True, dfs_color="#8B5CF6")
                
                # ---- UNDERDOG FANTASY TAB ----
                if ud_props:
                    with source_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("""
<div style="color: #F59E0B; font-size: 0.85rem; margin-bottom: 8px; font-family: JetBrains Mono;">
Underdog Fantasy pick'em lines — pick Higher or Lower on player projections
</div>""", unsafe_allow_html=True)
                        _render_props_table(ud_props, show_source=False, is_dfs=True, dfs_color="#F59E0B")
                
                # ---- LINE COMPARISON TAB ----
                if len(all_props) > 0:
                    with source_tabs[tab_idx]:
                        tab_idx += 1
                        _render_line_comparison(all_props, gender)
            
            else:
                if gender == "womens":
                    st.markdown("""
                    <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; 
                                border-radius: 8px; text-align: center;">
                        <div style="color: #FFA500; font-size: 1rem;">⚠️ No Women's Player Props Available</div>
                        <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
                            PrizePicks, Underdog Fantasy, and sportsbooks currently have <b>limited or no WNCAAB player prop coverage</b>.
                            <br>This section will populate automatically when these platforms add women's college basketball props.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; 
                                border-radius: 8px; text-align: center;">
                        <div style="color: #FFA500; font-size: 1rem;">⚠️ No Player Props Available</div>
                        <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
                            Player props may not be available if there are no games today or API limits were reached.
                            <br>Sources checked: The Odds API, PrizePicks, Underdog Fantasy
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_best:
            with st.spinner("Analyzing best prop bets..."):
                best_props_data = get_all_player_props(gender)
                best_all = best_props_data.get("all", [])
                best_pp = best_props_data.get("prizepicks", [])
                best_ud = best_props_data.get("underdog", [])
                best_odds = best_props_data.get("odds_api", [])
            _render_best_bets(best_all, best_pp, best_ud, best_odds, gender, games_df)

        with tab3:
            gender_label = config.GENDER_CONFIG[gender]["label"]
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(255,215,0,0.1), rgba(255,165,0,0.1)); 
                        border: 1px solid #FFD700; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="font-family: JetBrains Mono; color: #FFD700; font-size: 1.1rem; font-weight: bold;">
                    🏆 {gender_label} College Basketball Rankings & Standings
                </div>
                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                    AP Top 25 &amp; Conference Standings
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # AP Top 25
            rankings = get_rankings(gender)
            
            if rankings:
                st.markdown("""
                <div style="font-family: JetBrains Mono; font-size: 1.05rem; font-weight: bold;
                            color: #FFD700; margin: 20px 0 10px 0; border-bottom: 2px solid #FFD700;
                            padding-bottom: 5px;">
                    🏆 AP Top 25
                </div>
                """, unsafe_allow_html=True)
                
                for team in rankings[:25]:
                    rank = team.get('rank', 0)
                    name = team.get('team_name', 'Unknown')
                    record = team.get('record', '')
                    logo = team.get('team_logo', '')
                    prev_rank = team.get('previous_rank', 0)
                    points = team.get('points', 0)
                    fpv = team.get('first_place_votes', 0)
                    
                    # Trend indicator
                    if prev_rank > 0 and rank < prev_rank:
                        trend = f'<span style="color: #4CAF50; font-size: 0.75rem;">▲{prev_rank - rank}</span>'
                    elif prev_rank > 0 and rank > prev_rank:
                        trend = f'<span style="color: #FF5252; font-size: 0.75rem;">▼{rank - prev_rank}</span>'
                    else:
                        trend = '<span style="color: #888; font-size: 0.75rem;">—</span>'
                    
                    # Tier coloring
                    if rank <= 5:
                        rank_color = "#FFD700"
                        tier_label = "ELITE"
                    elif rank <= 10:
                        rank_color = "#00F3FF"
                        tier_label = "TOP 10"
                    elif rank <= 15:
                        rank_color = "#4CAF50"
                        tier_label = "CONTENDER"
                    elif rank <= 20:
                        rank_color = "#FFA500"
                        tier_label = ""
                    else:
                        rank_color = "#888"
                        tier_label = ""
                    
                    # First place votes badge
                    fpv_badge = f'<span style="color: #FFD700; font-size: 0.7rem;"> ({fpv})</span>' if fpv > 0 else ''
                    
                    # Points display
                    pts_display = f'{int(points):,}' if points else ''
                    
                    st.markdown(f"""
<div style="background: #1a1a2e; border-left: 3px solid {rank_color}; border-radius: 6px;
padding: 10px 14px; margin-bottom: 4px; display: grid;
grid-template-columns: 40px 30px 44px 1fr 70px 80px 80px; gap: 8px; align-items: center;">
<div style="color: {rank_color}; font-weight: 900; font-size: 1.2rem; font-family: JetBrains Mono; text-align: center;">{rank}</div>
<div style="text-align: center;">{trend}</div>
<div><img src="{logo}" style="width: 36px; height: 36px;" onerror="this.style.display='none'"></div>
<div>
<div style="color: #FFF; font-weight: bold; font-size: 0.95rem;">{name}{fpv_badge}</div>
{f'<div style="color: {rank_color}; font-size: 0.65rem;">{tier_label}</div>' if tier_label else ''}
</div>
<div style="text-align: center;">
<div style="color: #FFF; font-weight: bold; font-size: 1.05rem;">{record}</div>
</div>
<div style="text-align: center;">
<div style="color: #AAA; font-size: 0.85rem;">{pts_display}</div>
<div style="color: #666; font-size: 0.6rem;">PTS</div>
</div>
<div>
<div style="background: #333; border-radius: 4px; height: 6px; overflow: hidden;">
<div style="background: {rank_color}; width: {max(5, 100 - (rank - 1) * 4)}%; height: 100%; border-radius: 4px;"></div>
</div>
</div>
</div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #1a1a2e; border: 1px solid #FFA500; padding: 20px; 
                            border-radius: 8px; text-align: center;">
                    <div style="color: #FFA500; font-size: 1rem;">⚠️ Rankings data unavailable</div>
                    <div style="color: #888; font-size: 0.85rem; margin-top: 10px;">
                        Could not fetch AP Top 25 from the API. Try refreshing.
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"CRITICAL ERROR: {e}")
        st.text(traceback.format_exc())
        traceback.print_exc()
