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

# Import our modules
from data_manager import DataManager
from features import FeatureEngineer

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

# Custom CSS for beautiful UI matching the design EXACTLY
st.markdown("""
<style>
    /* Reset and base */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
        text-align: left;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: left;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    /* Game container */
    .game-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Team section */
    .team-section {
        text-align: center;
        padding: 1rem;
    }
    
    .team-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .team-logo {
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin: 1rem 0;
    }
    
    .team-name {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .win-probability {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .prob-red {
        color: #dc3545;
    }
    
    .prob-green {
        color: #28a745;
    }
    
    /* Best bet banner */
    .best-bet-banner {
        background: #4a5fd9;
        color: white;
        padding: 1.25rem;
        border-radius: 8px;
        text-align: center;
        font-size: 0.95rem;
        line-height: 1.4;
        margin: 1.5rem 0;
    }
    
    .best-bet-title {
        font-weight: 600;
    }
    
    /* Prediction cards */
    .pred-cards-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .prediction-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.25rem;
        flex: 1;
        border: 2px solid #e9ecef;
    }
    
    .pred-icon {
        font-size: 1.4rem;
        margin-right: 0.5rem;
    }
    
    .pred-title {
        font-weight: 600;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .pred-team {
        font-size: 1.2rem;
        color: #333;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .pred-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #28a745;
        margin: 0.5rem 0;
    }
    
    .pred-consensus {
        font-size: 0.95rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .checkmark {
        color: #28a745;
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
        model = EliteEnsembleModel()
    except:
        from model_engine_regularized import RegularizedEnsembleModel
        model = RegularizedEnsembleModel()
    
    model.load_models()
    return model


@st.cache_data(ttl=60)  # Cache for 1 minute for live scores
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


def get_confidence_level(probability):
    """Determine confidence level from probability"""
    if probability >= 0.65:
        return "HIGH", "🔥🔥🔥"
    elif probability >= 0.55:
        return "MEDIUM", "🔥🔥"
    else:
        return "LOW", "🔥"


def format_spread(spread):
    """Format spread with proper sign"""
    if spread > 0:
        return f"+{spread:.1f}"
    return f"{spread:.1f}"


def format_moneyline(prob):
    """Convert probability to moneyline odds"""
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
    """Generate detailed reasoning for prediction"""
    home_team = game_data.get('home_team_name', 'Home Team')
    visitor_team = game_data.get('visitor_team_name', 'Visitor Team')
    home_id = game_data.get('home_team_id')
    visitor_id = game_data.get('visitor_team_id')
    
    home_prob = prediction['home_win_probability']
    predicted_winner = home_team if home_prob > 0.5 else visitor_team
    loser_team = visitor_team if home_prob > 0.5 else home_team
    winner_prob = max(home_prob, 1 - home_prob)
    is_home_favorite = home_prob > 0.5
    
    analysis = f"### 🎯 Elite Analysis: {predicted_winner} ({winner_prob:.1%})\n\n"
    
    # LOCK ANALYSIS - Compare model vs Vegas
    if vegas_odds and vegas_odds.get('has_odds'):
        vegas_prob = vegas_odds['implied_home_prob'] if is_home_favorite else vegas_odds['implied_away_prob']
        vegas_spread = vegas_odds['spread_home'] if is_home_favorite else vegas_odds['spread_away']
        vegas_total = vegas_odds['total']
        
        model_spread = -abs(prediction['predicted_spread'])
        model_total = prediction['predicted_total']
        
        ml_edge = (winner_prob - vegas_prob) * 100
        spread_diff = abs(model_spread - vegas_spread)
        total_diff = model_total - vegas_total
        
        # Collect all recommended bets for summary
        recommended_bets = []
        locks = []
        
        # MONEYLINE analysis
        ml_bet = None
        if ml_edge >= 10:
            ml_bet = f"🔒 **{predicted_winner} ML** (Moneyline) - LOCK"
            locks.append(('🔒 MONEYLINE LOCK', f"""
**Model says:** {predicted_winner} wins {winner_prob:.1%} of the time
**Vegas says:** {predicted_winner} wins {vegas_prob:.1%} of the time
**Edge:** +{ml_edge:.1f}% in our favor

*Why this is a lock:* Our model sees {predicted_winner} as significantly more likely to win than Vegas does. A 10%+ edge is rare and suggests the market is undervaluing this team."""))
            recommended_bets.append(ml_bet)
        elif ml_edge >= 5:
            ml_bet = f"💎 **{predicted_winner} ML** (Moneyline) - VALUE"
            locks.append(('💎 MONEYLINE VALUE', f"""
**Model says:** {predicted_winner} wins {winner_prob:.1%} of the time
**Vegas says:** {predicted_winner} wins {vegas_prob:.1%} of the time
**Edge:** +{ml_edge:.1f}% in our favor

*Why this has value:* Our model sees more upside than Vegas. Consider this bet if the odds are favorable."""))
            recommended_bets.append(ml_bet)
        
        # Determine which side has value based on spread comparison
        # If model spread is SMALLER (closer to 0) than Vegas → underdog value
        # If model spread is LARGER (further from 0) than Vegas → favorite value
        model_favors_favorite_more = abs(model_spread) > abs(vegas_spread)
        
        # SPREAD analysis
        # Check if model and Vegas disagree on who is favorite
        vegas_has_team_as_favorite = vegas_spread < 0  # Negative spread = favorite
        model_has_team_as_favorite = True  # We always show predicted_winner's spread as negative
        
        spread_bet = None
        if spread_diff >= 5:
            # Case 1: Model thinks team wins, Vegas has them as underdog (+spread)
            if vegas_spread > 0:
                spread_bet = f"🔒 **{predicted_winner} +{abs(vegas_spread):.1f}** (Spread) - LOCK"
                reasoning = f"Model says {predicted_winner} WINS by {abs(model_spread):.1f}, but Vegas has them as +{abs(vegas_spread):.1f} UNDERDOGS! Take the FREE points on a team that should win outright."
            # Case 2: Both agree team is favorite, but model likes them more
            elif model_favors_favorite_more:
                spread_bet = f"🔒 **{predicted_winner} {vegas_spread:+.1f}** (Spread) - LOCK"
                reasoning = f"Model says {predicted_winner} wins by {abs(model_spread):.1f}, Vegas only has them at {abs(vegas_spread):.1f}. They should cover the Vegas spread easily."
            # Case 3: Both agree team is favorite, but Vegas likes them more (take underdog)
            else:
                spread_bet = f"🔒 **{loser_team} +{abs(vegas_spread):.1f}** (Spread) - LOCK"
                reasoning = f"Model says {predicted_winner} wins by only {abs(model_spread):.1f}, but Vegas has them at {abs(vegas_spread):.1f}. Take the underdog + points."
            
            locks.append(('🔒 SPREAD LOCK', f"""
**Model:** {predicted_winner} wins by {abs(model_spread):.1f} points
**Vegas:** {predicted_winner} {vegas_spread:+.1f}
**Gap:** {spread_diff:.1f} points

{reasoning}"""))
            recommended_bets.append(spread_bet)
        elif spread_diff >= 3:
            if vegas_spread > 0:
                spread_bet = f"💎 **{predicted_winner} +{abs(vegas_spread):.1f}** (Spread) - VALUE"
                reasoning = f"Model says {predicted_winner} wins, but Vegas has them getting +{abs(vegas_spread):.1f} points. Free points on a winner."
            elif model_favors_favorite_more:
                spread_bet = f"💎 **{predicted_winner} {vegas_spread:+.1f}** (Spread) - VALUE"
                reasoning = f"Model likes {predicted_winner} to cover more than Vegas does."
            else:
                spread_bet = f"💎 **{loser_team} +{abs(vegas_spread):.1f}** (Spread) - VALUE"
                reasoning = f"Model says {predicted_winner} wins by less than Vegas line. Underdog value."
            
            locks.append(('💎 SPREAD VALUE', f"""
**Model:** {predicted_winner} wins by {abs(model_spread):.1f} points
**Vegas:** {predicted_winner} {vegas_spread:+.1f}
**Gap:** {spread_diff:.1f} points

{reasoning}"""))
            recommended_bets.append(spread_bet)
        
        # TOTAL analysis
        total_bet = None
        if abs(total_diff) >= 6:
            ou_pick = "OVER" if total_diff > 0 else "UNDER"
            total_bet = f"🔒 **{ou_pick} {vegas_total:.1f}** (Total) - LOCK"
            locks.append((f'🔒 {ou_pick} LOCK', f"""
**Model total:** {model_total:.1f} points
**Vegas total:** {vegas_total:.1f} points
**Difference:** {total_diff:+.1f} points

Model projects {abs(total_diff):.1f} {"more" if total_diff > 0 else "fewer"} points than Vegas."""))
            recommended_bets.append(total_bet)
        elif abs(total_diff) >= 4:
            ou_pick = "OVER" if total_diff > 0 else "UNDER"
            total_bet = f"💎 **{ou_pick} {vegas_total:.1f}** (Total) - VALUE"
            locks.append((f'💎 {ou_pick} VALUE', f"""
**Model total:** {model_total:.1f} points
**Vegas total:** {vegas_total:.1f} points
**Difference:** {total_diff:+.1f} points

Model sees the total going {ou_pick.lower()} Vegas's line."""))
            recommended_bets.append(total_bet)
        
        # ADD CLEAR BETTING SUMMARY AT TOP
        if recommended_bets:
            analysis += "## 🎰 BETS TO MAKE:\n\n"
            for bet in recommended_bets:
                analysis += f"{bet}\n\n"
            analysis += "---\n\n"
        else:
            analysis += "## ⚖️ NO CLEAR BETS\n\n"
            analysis += "Model and Vegas are aligned. No strong edge detected.\n\n"
            analysis += "---\n\n"
        
        # Detailed breakdown
        if locks:
            analysis += "#### 📋 Detailed Breakdown:\n\n"
            for lock_title, lock_detail in locks:
                analysis += f"**{lock_title}**\n{lock_detail}\n\n"
    
    # Key factors - HARD FACTS WITH ACTUAL NUMBERS
    analysis += "#### 📊 Verified Stats:\n\n"
    
    # Season Records
    home_wins = features.get('home_wins', 0)
    home_losses = features.get('home_losses', 0)
    visitor_wins = features.get('visitor_wins', 0)
    visitor_losses = features.get('visitor_losses', 0)
    
    if home_wins + home_losses > 0 and visitor_wins + visitor_losses > 0:
        analysis += f"| Team | Season Record | Win % |\n"
        analysis += f"|------|---------------|-------|\n"
        home_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0
        visitor_pct = visitor_wins / (visitor_wins + visitor_losses) if (visitor_wins + visitor_losses) > 0 else 0
        analysis += f"| {home_team} | **{int(home_wins)}-{int(home_losses)}** | {home_pct:.1%} |\n"
        analysis += f"| {visitor_team} | **{int(visitor_wins)}-{int(visitor_losses)}** | {visitor_pct:.1%} |\n\n"
    
    # Last 10 Games Performance
    home_l10_pct = features.get('home_win_pct_last10', 0)
    visitor_l10_pct = features.get('visitor_win_pct_last10', 0)
    home_l10_wins = int(home_l10_pct * 10)
    visitor_l10_wins = int(visitor_l10_pct * 10)
    
    analysis += f"**Last 10 Games:**\n"
    analysis += f"- {home_team}: **{home_l10_wins}-{10-home_l10_wins}** ({home_l10_pct:.0%})\n"
    analysis += f"- {visitor_team}: **{visitor_l10_wins}-{10-visitor_l10_wins}** ({visitor_l10_pct:.0%})\n\n"
    
    # Current Streak
    home_streak = features.get('home_streak', 0)
    visitor_streak = features.get('visitor_streak', 0)
    
    home_streak_str = f"{abs(int(home_streak))}W" if home_streak > 0 else f"{abs(int(home_streak))}L"
    visitor_streak_str = f"{abs(int(visitor_streak))}W" if visitor_streak > 0 else f"{abs(int(visitor_streak))}L"
    
    analysis += f"**Current Streak:**\n"
    analysis += f"- {home_team}: **{home_streak_str}** streak\n"
    analysis += f"- {visitor_team}: **{visitor_streak_str}** streak\n\n"
    
    # Points Per Game (Last 10)
    home_ppg = features.get('home_avg_points_for_last10', 0)
    visitor_ppg = features.get('visitor_avg_points_for_last10', 0)
    home_opp_ppg = features.get('home_avg_points_against_last10', 0)
    visitor_opp_ppg = features.get('visitor_avg_points_against_last10', 0)
    
    if home_ppg > 0 and visitor_ppg > 0:
        analysis += f"**Scoring (Last 10 Games):**\n"
        analysis += f"| Team | PPG | Opp PPG | Diff |\n"
        analysis += f"|------|-----|---------|------|\n"
        analysis += f"| {home_team} | **{home_ppg:.1f}** | {home_opp_ppg:.1f} | {home_ppg - home_opp_ppg:+.1f} |\n"
        analysis += f"| {visitor_team} | **{visitor_ppg:.1f}** | {visitor_opp_ppg:.1f} | {visitor_ppg - visitor_opp_ppg:+.1f} |\n\n"
    
    # Rest Days
    home_rest = features.get('home_rest_days', 0)
    visitor_rest = features.get('visitor_rest_days', 0)
    
    analysis += f"**Rest:**\n"
    analysis += f"- {home_team}: **{int(home_rest)} days** rest\n"
    analysis += f"- {visitor_team}: **{int(visitor_rest)} days** rest\n\n"
    
    # Head-to-Head (if meaningful data exists)
    h2h_pct = features.get('h2h_win_pct', 0.5)
    h2h_margin = features.get('h2h_avg_margin', 0)
    
    if h2h_pct != 0.5 or abs(h2h_margin) > 0:
        analysis += f"**Head-to-Head (This Season):**\n"
        analysis += f"- {home_team} wins {h2h_pct:.0%} of matchups (avg margin: {h2h_margin:+.1f})\n\n"
    
    # Conference Rankings
    home_conf_rank = features.get('home_conference_rank', 0)
    visitor_conf_rank = features.get('visitor_conference_rank', 0)
    
    if home_conf_rank > 0 and visitor_conf_rank > 0:
        analysis += f"**Conference Rank:**\n"
        analysis += f"- {home_team}: **#{int(home_conf_rank)}**\n"
        analysis += f"- {visitor_team}: **#{int(visitor_conf_rank)}**\n\n"
    
    # Injury Report Section
    if injuries:
        home_injuries = get_team_injuries(injuries, home_id)
        visitor_injuries = get_team_injuries(injuries, visitor_id)
        
        analysis += "\n#### 🏥 Injury Report:\n\n"
        analysis += "*🔴 Out | 🟠 Doubtful (~25%) | 🟡 Questionable (~50%) | 🟢 Probable (~75%)*\n\n"
        
        def get_status_emoji(status: str) -> str:
            """Get emoji based on injury status"""
            status_lower = status.lower()
            if status_lower in ['out', 'out for season']:
                return "🔴"  # Definitely out
            elif status_lower == 'doubtful':
                return "🟠"  # ~25% chance to play
            elif status_lower == 'questionable':
                return "🟡"  # ~50% chance to play
            elif status_lower == 'probable':
                return "🟢"  # ~75% chance to play
            return "⚪"
        
        # Home team injuries
        analysis += f"**{home_team}:**\n"
        if home_injuries:
            for inj in home_injuries[:6]:  # Show top 6
                emoji = get_status_emoji(inj['status'])
                analysis += f"- {emoji} **{inj['player_name']}** - {inj['status']}"
                if inj['return_date'] and inj['return_date'] != 'Unknown':
                    analysis += f" (return: {inj['return_date']})"
                analysis += "\n"
        else:
            analysis += "- ✅ No injuries reported\n"
        
        analysis += f"\n**{visitor_team}:**\n"
        if visitor_injuries:
            for inj in visitor_injuries[:6]:
                emoji = get_status_emoji(inj['status'])
                analysis += f"- {emoji} **{inj['player_name']}** - {inj['status']}"
                if inj['return_date'] and inj['return_date'] != 'Unknown':
                    analysis += f" (return: {inj['return_date']})"
                analysis += "\n"
        else:
            analysis += "- ✅ No injuries reported\n"
        
        # Injury impact analysis - count by severity
        def count_injuries_by_severity(injuries_list):
            out = len([i for i in injuries_list if i['status'].lower() in ['out', 'out for season']])
            doubtful = len([i for i in injuries_list if i['status'].lower() == 'doubtful'])
            questionable = len([i for i in injuries_list if i['status'].lower() == 'questionable'])
            return out, doubtful, questionable
        
        home_out, home_doubt, home_quest = count_injuries_by_severity(home_injuries)
        visitor_out, visitor_doubt, visitor_quest = count_injuries_by_severity(visitor_injuries)
        
        # Calculate "impact score" (out=3, doubtful=2, questionable=1)
        home_impact = home_out * 3 + home_doubt * 2 + home_quest
        visitor_impact = visitor_out * 3 + visitor_doubt * 2 + visitor_quest
        
        if home_impact > visitor_impact + 3:
            analysis += f"\n⚠️ **Injury Advantage: {visitor_team}**\n"
            analysis += f"- {home_team}: {home_out} Out, {home_doubt} Doubtful, {home_quest} Questionable\n"
            analysis += f"- {visitor_team}: {visitor_out} Out, {visitor_doubt} Doubtful, {visitor_quest} Questionable\n"
            analysis += "*Consider adjusting confidence toward {visitor_team}*\n"
        elif visitor_impact > home_impact + 3:
            analysis += f"\n⚠️ **Injury Advantage: {home_team}**\n"
            analysis += f"- {home_team}: {home_out} Out, {home_doubt} Doubtful, {home_quest} Questionable\n"
            analysis += f"- {visitor_team}: {visitor_out} Out, {visitor_doubt} Doubtful, {visitor_quest} Questionable\n"
            analysis += f"*Consider adjusting confidence toward {home_team}*\n"
        else:
            analysis += "\n✅ **Injury Situation**: Relatively balanced - no major advantage either way.\n"
    
    # Model info
    analysis += "\n#### 🤖 Model Info:\n"
    analysis += f"4-model voting ensemble (Random Forest, Gradient Boosting, Extra Trees, Ridge/Logistic) trained on {3235} historical games.\n"
    analysis += "\n⚠️ **Note**: The model does NOT factor in current injuries. Use the injury report above to adjust your confidence in the predictions.\n"
    
    # Statistical summary
    analysis += "\n#### 📈 Prediction Summary:\n"
    predicted_spread = prediction['predicted_spread']
    analysis += f"- **Predicted Spread**: {format_spread(-abs(predicted_spread))} ({predicted_winner})\n"
    analysis += f"- **Predicted Total**: {prediction['predicted_total']:.1f}\n"
    analysis += f"- **Win Confidence**: {get_confidence_level(winner_prob)[0]} ({winner_prob:.1%})\n"
    
    return analysis


def display_game_predictions(games_df, predictions_df, odds_df, all_features, box_scores_df=None):
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
    
    # Combine data
    combined = games_df.copy()
    combined = combined.join(predictions_df)
    
    # Sort by confidence (highest home_win_probability or visitor_win_probability)
    combined['max_win_prob'] = combined[['home_win_probability', 'visitor_win_probability']].max(axis=1)
    combined = combined.sort_values('max_win_prob', ascending=False)
    
    # Display each game
    for idx, game in combined.iterrows():
        # Get specific box scores for this game
        game_box_scores = None
        if box_scores_df is not None and not box_scores_df.empty:
            game_box_scores = box_scores_df[box_scores_df['game_id'] == game['id']]
            
        display_game_card(game, odds_df, all_features.loc[idx] if idx in all_features.index else {}, injuries, game_box_scores)
        st.markdown("---")


def display_game_card(game, odds_df, features, injuries=None, box_scores=None):
    """Display a single game prediction card with Vegas odds comparison"""
    
    home_id = game['home_team_id']
    visitor_id = game['visitor_team_id']
    
    home_name = team_logos.get_team_name(home_id)
    visitor_name = team_logos.get_team_name(visitor_id)
    
    home_logo = team_logos.get_team_logo(home_id)
    visitor_logo = team_logos.get_team_logo(visitor_id)
    
    home_prob = game['home_win_probability']
    visitor_prob = game['visitor_win_probability']
    
    # Determine favorite
    is_home_favorite = home_prob > visitor_prob
    favorite_name = home_name if is_home_favorite else visitor_name
    favorite_prob = max(home_prob, visitor_prob)
    
    # Get Vegas odds
    game_id = game.get('id')
    vegas_odds = get_consensus_odds(odds_df, game_id) if game_id else None
    
    # Container
    st.markdown('<div class="game-container">', unsafe_allow_html=True)
    
    # Team matchup section
    col1, col2, col3 = st.columns([1, 0.3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="team-section">
            <div class="team-label">👥 Away</div>
            <img src="{visitor_logo}" class="team-logo">
            <div class="team-name">{visitor_name}</div>
            <div class="win-probability {'prob-red' if visitor_prob < 0.5 else 'prob-green'}">{visitor_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get game status and scores
        status = game.get('status', '')
        home_score = game.get('home_team_score', 0)
        visitor_score = game.get('visitor_team_score', 0)
        period = game.get('period', 0)
        time_str = game.get('time', '')
        
        # Determine center display
        center_content = "vs"
        center_color = "#999"
        center_size = "1.2rem"
        
        # Show score if score exists OR game is active/final
        # Note: Sometimes API returns 0-0 for very early game, but usually updates quickly.
        # We check period > 0 to imply game started.
        has_score = (home_score > 0 or visitor_score > 0)
        is_started = (period > 0) or (status in ['Final', 'In Progress'])
        
        if has_score or is_started:
            center_content = f"{visitor_score} - {home_score}"
            center_color = "#000000"  # Black text
            center_size = "1.8rem"
            
        # Game time / Status display
        time_display = ""
        if status == 'Final':
            time_display = f"<div style='font-size: 0.85rem; color: #000; font-weight: 600;'>FINAL</div>"
        elif time_str:
            # e.g. "Q3 11:07"
            time_display = f"<div style='font-size: 0.85rem; color: #d63384; font-weight: 600;'>{time_str}</div>"
        
        st.markdown(f"""
        <div style="text-align: center; padding-top: 70px;">
            <div style="font-size: {center_size}; font-weight: 800; color: {center_color}; margin-bottom: 0.2rem;">
                {center_content}
            </div>
            {time_display}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="team-section">
            <div class="team-label">🏠 Home</div>
            <img src="{home_logo}" class="team-logo">
            <div class="team-name">{home_name}</div>
            <div class="win-probability {'prob-red' if home_prob < 0.5 else 'prob-green'}">{home_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Best Bet Banner Logic
    # 1. Calculate Spread Edge
    spread_pick = home_name if is_home_favorite else visitor_name
    spread_value = -abs(game['predicted_spread'])
    
    best_bet_str = f"{spread_pick} {format_spread(spread_value)}"
    best_edge = 0.0
    best_conf_text = "LOW" # Default
    
    # Spread Edge Calculation
    if vegas_odds and vegas_odds['has_odds']:
        raw_spread_diff = game['predicted_spread'] + vegas_odds['spread_home']
        spread_edge = abs(raw_spread_diff)
        
        # Determine pick side
        if raw_spread_diff > 0:
            s_pick = home_name
            s_line = vegas_odds['spread_home']
        else:
            s_pick = visitor_name
            s_line = vegas_odds['spread_away']
            
        best_bet_str = f"{s_pick} {format_spread(s_line)}"
        best_edge = spread_edge
    
    # 2. Calculate Total Edge
    total_val = game['predicted_total']
    if vegas_odds and vegas_odds['has_odds']:
        vegas_total = vegas_odds['total']
        total_diff = total_val - vegas_total
        total_edge = abs(total_diff)
        
        # If Total Edge is better, switch Best Bet
        if total_edge > best_edge:
            pick_side = "OVER" if total_diff > 0 else "UNDER"
            best_bet_str = f"{pick_side} {vegas_total:.1f}"
            best_edge = total_edge
            best_conf_text = "LOW" # Will be updated below
            
    # 3. Determine Confidence Text based on BEST EDGE
    if best_edge >= 5:
        best_conf_text = "HIGH (LOCK)"
    elif best_edge >= 3:
        best_conf_text = "HIGH (VALUE)"
    else:
        # Fallback to model probability if edge is small or using spread
        base_conf, _ = get_confidence_level(favorite_prob)
        if best_edge < 1:
             best_conf_text = base_conf

    st.markdown(f"""
    <div class="best-bet-banner">
        <span class="best-bet-title">🔥 Best Bet: {best_bet_str}</span> - {best_conf_text} confidence pick based on elite ensemble model consensus.
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction cards with Vegas comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ml_pick = favorite_name
        ml_prob_display = f"{favorite_prob:.0%}"
        
        # Show Vegas comparison if available
        vegas_ml_text = ""
        ml_lock = ""
        if vegas_odds and vegas_odds['has_odds']:
            vegas_ml = vegas_odds['moneyline_home'] if is_home_favorite else vegas_odds['moneyline_away']
            vegas_prob = vegas_odds['implied_home_prob'] if is_home_favorite else vegas_odds['implied_away_prob']
            edge = calculate_edge(favorite_prob, vegas_prob)
            
            edge_color = "#28a745" if edge > 3 else "#ffc107" if edge > 0 else "#dc3545"
            
            # Lock indicator for moneyline
            if edge >= 10:
                ml_lock = '<div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">🔒 LOCK</div>'
            elif edge >= 5:
                ml_lock = '<div style="background: #28a745; color: #fff; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">💎 VALUE</div>'
            
            vegas_ml_text = f'<div style="font-size: 0.95rem; color: #666; margin-top: 0.4rem;">Vegas: {format_american_odds(vegas_ml)} ({vegas_prob:.0%})</div><div style="font-size: 0.95rem; color: {edge_color}; font-weight: 700;">Edge: {edge:+.1f}%</div>{ml_lock}'
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-title"><span class="pred-icon">💰</span>Moneyline</div>
            <div class="pred-team">{ml_pick}</div>
            <div class="pred-value"><span class="checkmark">✅</span> {ml_prob_display}</div>
            {vegas_ml_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate spreads for BOTH teams
        favorite_spread = -abs(game['predicted_spread'])  # Favorite gives points (negative)
        underdog_spread = abs(game['predicted_spread'])   # Underdog gets points (positive)
        
        underdog_name = visitor_name if is_home_favorite else home_name
        
        # Show Vegas comparison
        vegas_spread_text = ""
        spread_lock = ""
        if vegas_odds and vegas_odds['has_odds']:
            vegas_spread = vegas_odds['spread_home'] if is_home_favorite else vegas_odds['spread_away']
            spread_diff = abs(spread_value - vegas_spread)
            
            diff_color = "#28a745" if spread_diff > 2 else "#ffc107" if spread_diff > 1 else "#666"
            
            # Lock indicator for spread (significant difference from Vegas line)
            if spread_diff >= 5:
                spread_lock = '<div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">🔒 LOCK</div>'
            elif spread_diff >= 3:
                spread_lock = '<div style="background: #28a745; color: #fff; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">💎 VALUE</div>'
            
            # Vegas spreads for both teams
            vegas_fav_spread = vegas_odds['spread_home'] if is_home_favorite else vegas_odds['spread_away']
            vegas_dog_spread = vegas_odds['spread_away'] if is_home_favorite else vegas_odds['spread_home']
            
            # Smart Recommendation Logic
            model_spread_home = game['predicted_spread']
            vegas_spread_home = vegas_odds['spread_home']
            # Calculate Cover Margin: Predicted Margin + Spread
            # If > 0, Home Covers. If < 0, Visitor Covers.
            raw_diff = model_spread_home + vegas_spread_home
            
            rec_text = ""
            rec_color = "#666"
            
            # Threshold for recommendation (1 point difference)
            if abs(raw_diff) >= 1.0:
                if raw_diff > 0:
                    # Model thinks Home is better than Vegas implies -> Bet Home
                    pick_team = home_name
                    pick_line = format_spread(vegas_odds['spread_home'])
                    rec_color = "#28a745" if raw_diff >= 3 else "#ffc107"
                else:
                    # Model thinks Home is worse -> Bet Visitor
                    pick_team = visitor_name
                    pick_line = format_spread(vegas_odds['spread_away'])
                    rec_color = "#28a745" if raw_diff <= -3 else "#ffc107"
                
                rec_text = f"💎 PICK: {pick_team} {pick_line}"
            else:
                rec_text = "No Value"
            
            # Lock indicator logic (keep same thresholds relative to absolute diff)
            spread_diff = abs(raw_diff)
            if spread_diff >= 5:
                spread_lock = '<div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">🔒 LOCK</div>'
            elif spread_diff >= 3:
                spread_lock = '<div style="background: #28a745; color: #fff; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">💎 VALUE</div>'
            
            vegas_spread_text = f'<div style="font-size: 0.95rem; color: #666; margin-top: 0.6rem; border-top: 1px solid #ddd; padding-top: 0.5rem;"><div style="font-weight: 700; margin-bottom: 0.3rem;">Vegas Lines:</div><div style="padding: 0.15rem 0;">{spread_pick}: {format_spread(vegas_fav_spread)}</div><div style="padding: 0.15rem 0;">{underdog_name}: {format_spread(vegas_dog_spread)}</div></div><div style="font-size: 0.95rem; color: {rec_color}; font-weight: 700; margin-top: 0.3rem;">{rec_text} (Edge: {spread_diff:.1f})</div>{spread_lock}'
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-title"><span class="pred-icon">📊</span>Spread</div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; font-size: 1.15rem;">
                <span style="font-weight: 600;">{spread_pick}</span>
                <span style="color: #dc3545; font-weight: 700; font-size: 1.3rem;">{format_spread(favorite_spread)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; font-size: 1.15rem; border-top: 1px solid #ddd;">
                <span style="font-weight: 600;">{underdog_name}</span>
                <span style="color: #28a745; font-weight: 700; font-size: 1.3rem;">{format_spread(underdog_spread)}</span>
            </div>
            {vegas_spread_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total = game['predicted_total']
        
        # Compare with Vegas total
        vegas_total_text = ""
        over_under = "PUSH"
        total_lock = ""
        
        if vegas_odds and vegas_odds['has_odds']:
            vegas_total = vegas_odds['total']
            total_diff = total - vegas_total
            
            # Tighter threshold: 0.5 points for PUSH
            if total_diff > 0.5:
                over_under = "OVER"
                ou_color = "#28a745"
            elif total_diff < -0.5:
                over_under = "UNDER"
                ou_color = "#28a745"
            else:
                over_under = "PUSH"
                ou_color = "#ffc107"
            
            # Lock indicator for totals
            if abs(total_diff) >= 6:
                total_lock = '<div style="background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">🔒 LOCK</div>'
            elif abs(total_diff) >= 4:
                total_lock = '<div style="background: #28a745; color: #fff; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-top: 0.3rem; text-align: center;">💎 VALUE</div>'
            
            # Smart Recommendation Logic for Totals
            if abs(total_diff) >= 1.0:
                pick_side = "OVER" if total_diff > 0 else "UNDER"
                pick_line = f"{vegas_total:.1f}"
                rec_color = "#28a745" if abs(total_diff) >= 3 else "#ffc107"
                rec_text = f"💎 PICK: {pick_side} {pick_line}"
            else:
                rec_text = "No Value"
                rec_color = "#666"

            diff_color = "#28a745" if abs(total_diff) > 3 else "#ffc107" if abs(total_diff) > 1 else "#666"
            
            # Display Pick + Edge
            vegas_total_text = f'<div style="font-size: 0.95rem; color: #666; margin-top: 0.4rem;">Vegas: {vegas_total:.1f}</div><div style="font-size: 0.95rem; color: {rec_color}; font-weight: 700;">{rec_text} (Edge: {abs(total_diff):.1f})</div>{total_lock}'
        else:
            # No Vegas odds, use simple threshold
            over_under = "UNDER" if total < 220 else "OVER"
            ou_color = "#666"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-title"><span class="pred-icon">🎯</span>Total</div>
            <div class="pred-team" style="color: {ou_color};">{over_under} {total:.1f}</div>
            <div class="pred-value"><span class="checkmark">✅</span> {total:.1f}</div>
            {vegas_total_text}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Box Scores if available
    if box_scores is not None and not box_scores.empty:
        with st.expander("📊 Box Scores"):
            # Split by team
            home_scores = box_scores[box_scores['team_id'] == home_id].sort_values('pts', ascending=False)
            visitor_scores = box_scores[box_scores['team_id'] == visitor_id].sort_values('pts', ascending=False)
            
            # Simple column config for display
            cols = ['first_name', 'last_name', 'pts', 'reb', 'ast', 'stl', 'blk', 'turnover', 'min']
            
            st.markdown(f"**{visitor_name}**")
            st.dataframe(
                visitor_scores[cols].rename(columns={
                    'first_name': 'First', 'last_name': 'Last', 
                    'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 
                    'stl': 'STL', 'blk': 'BLK', 'turnover': 'TO', 'min': 'MIN'
                }),
                hide_index=True,
                # use_container_width=True (deprecated)
            )
            
            st.markdown(f"**{home_name}**")
            st.dataframe(
                home_scores[cols].rename(columns={
                    'first_name': 'First', 'last_name': 'Last', 
                    'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 
                    'stl': 'STL', 'blk': 'BLK', 'turnover': 'TO', 'min': 'MIN'
                }),
                hide_index=True,
                # use_container_width=True (deprecated)
            )
            
    # Elite Analysis Expander
    with st.expander("🔍 Elite Analysis - Detailed Reasoning"):
        prediction_dict = {
            'home_win_probability': home_prob,
            'predicted_spread': game['predicted_spread'],
            'predicted_total': game['predicted_total']
        }
        analysis = generate_elite_analysis(game, prediction_dict, features, vegas_odds, injuries)
        st.markdown(analysis)


def main():
    """Main Streamlit app"""
    
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
        
        # Model info
        st.subheader("🤖 Model Info")
        model = load_model()
        
        if model and model.training_info:
            st.metric("Test Accuracy", f"{model.training_info['metrics']['winner_test_accuracy']:.1%}")
            st.metric("Score MAE", f"{model.training_info['metrics']['home_score_test_mae']:.1f} pts")
            st.caption(f"Trained: {model.training_info.get('trained_at', 'Unknown')[:10]}")
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
                    
                    with st.expander("📊 View Training Chart"):
                        st.line_chart(df_history.set_index('Training #')[['Total Games']])
                        st.caption("Model learns from more games with each training")
            else:
                st.info("📊 Train the model multiple times to track improvement")
        except Exception as e:
            st.caption("Training history will appear after first retrain")
        
        st.markdown("---")
        
        # Retrain button
        st.subheader("🔄 Model Updates")
        if st.button("Retrain Model", help="Fetch latest data and retrain the model"):
            with st.spinner("Retraining model with latest data... This may take 10-30 minutes..."):
                import train_model
                train_model.main()
                # Clear ALL caches to force reload
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Model retrained successfully!")
                st.info("🔄 Reloading app with new model...")
                st.balloons()
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
            st.success(f"Fetched box scores for {len(box_scores_df['game_id'].unique())} games")
        
        # Generate predictions
        data_mgr = DataManager()
        feature_eng = FeatureEngineer()
        
        # Get historical data for context
        current_season = target_date.year if target_date.month >= 10 else target_date.year - 1
        
        historical_data = data_mgr.get_complete_training_data([current_season - 1, current_season])
        
        # Build features for each game (Vegas odds shown in UI only, not model features)
        all_features = []
        for _, game in games_df.iterrows():
            try:
                features = feature_eng.build_features_for_game(
                    game.to_dict(),
                    historical_data,
                    current_season
                )
                if features is not None:
                    all_features.append(features)
            except Exception as e:
                st.error(f"Error building features for game {game.get('id')}: {str(e)}")
                continue
        
        if not all_features:
            st.error("Could not generate features for any games. Please check data availability.")
            return
        
        features_df = pd.DataFrame(all_features)
        
        # Make predictions
        predictions_df = model.predict(features_df)
        
        # Display predictions
        display_game_predictions(games_df, predictions_df, odds_df, features_df, box_scores_df)


if __name__ == "__main__":
    main()
