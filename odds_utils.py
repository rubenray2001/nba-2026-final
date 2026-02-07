"""
Vegas Odds Utilities
Parse and aggregate betting odds from multiple vendors
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def get_consensus_odds(odds_df: pd.DataFrame, game_id: int, preferred_vendors: list = None) -> Dict:
    """
    Get consensus odds for a game from multiple vendors
    
    Args:
        odds_df: DataFrame with odds data
        game_id: Game ID to get odds for
        preferred_vendors: List of preferred vendors (e.g., ['draftkings', 'fanduel'])
    
    Returns:
        Dict with consensus odds values
    """
    if odds_df.empty:
        return get_default_odds()
    
    # Filter to this game (use .copy() to avoid SettingWithCopyWarning)
    game_odds = odds_df[odds_df['game_id'] == game_id].copy()
    
    if game_odds.empty:
        return get_default_odds()
    
    # If preferred vendors specified, try to use those first
    if preferred_vendors:
        preferred_odds = game_odds[game_odds['vendor'].isin(preferred_vendors)]
        if not preferred_odds.empty:
            game_odds = preferred_odds.copy()
    
    # Convert to numeric, handling any string values
    numeric_cols = ['spread_home_value', 'spread_away_value', 'total_value', 
                   'moneyline_home_odds', 'moneyline_away_odds']
    
    for col in numeric_cols:
        if col in game_odds.columns:
            # Clean string values (remove quotes if present)
            if game_odds[col].dtype == object:
                game_odds[col] = game_odds[col].astype(str).str.replace('"', '').str.replace("'", "")
            game_odds.loc[:, col] = pd.to_numeric(game_odds[col], errors='coerce')
    
    # Filter out garbage data before computing consensus
    # BallDontLie API sometimes returns junk values (-10000 ML, +38.5 spread)
    clean_odds = game_odds.copy()
    clean_odds.loc[clean_odds['moneyline_home_odds'].abs() > 5000, 'moneyline_home_odds'] = np.nan
    clean_odds.loc[clean_odds['moneyline_away_odds'].abs() > 5000, 'moneyline_away_odds'] = np.nan
    clean_odds.loc[clean_odds['spread_home_value'].abs() > 25, 'spread_home_value'] = np.nan
    clean_odds.loc[clean_odds['spread_away_value'].abs() > 25, 'spread_away_value'] = np.nan
    clean_odds.loc[(clean_odds['total_value'] < 170) | (clean_odds['total_value'] > 280), 'total_value'] = np.nan
    
    # Calculate consensus (median across vendors, ignoring garbage)
    consensus = {
        'spread_home': clean_odds['spread_home_value'].median(),
        'spread_away': clean_odds['spread_away_value'].median(),
        'total': clean_odds['total_value'].median(),
        'moneyline_home': clean_odds['moneyline_home_odds'].median(),
        'moneyline_away': clean_odds['moneyline_away_odds'].median(),
        'num_vendors': len(game_odds),
        'has_odds': True
    }
    
    # If all values were garbage, mark as no odds
    if pd.isna(consensus['spread_home']) and pd.isna(consensus['moneyline_home']):
        return get_default_odds()
    
    # Calculate implied probabilities from moneyline
    consensus['implied_home_prob'] = moneyline_to_probability(consensus['moneyline_home'])
    consensus['implied_away_prob'] = moneyline_to_probability(consensus['moneyline_away'])
    
    return consensus


def get_default_odds() -> Dict:
    """Return default odds when no data available"""
    return {
        'spread_home': 0.0,
        'spread_away': 0.0,
        'total': 220.0,  # League average
        'moneyline_home': -110,
        'moneyline_away': -110,
        'implied_home_prob': 0.5,
        'implied_away_prob': 0.5,
        'num_vendors': 0,
        'has_odds': False
    }


def moneyline_to_probability(moneyline: float) -> float:
    """
    Convert moneyline odds to implied probability
    
    Args:
        moneyline: American odds (e.g., -150, +130)
    
    Returns:
        Probability between 0 and 1
    """
    if pd.isna(moneyline) or moneyline == 0:
        return 0.5
    
    # Filter out garbage values from API (e.g., -199900)
    if abs(moneyline) > 5000:
        return 0.5
    
    if moneyline < 0:
        # Favorite
        prob = abs(moneyline) / (abs(moneyline) + 100)
    else:
        # Underdog
        prob = 100 / (moneyline + 100)
    
    # Clamp to realistic range
    return max(0.05, min(0.95, prob))


def probability_to_moneyline(probability: float) -> int:
    """
    Convert probability to moneyline odds
    
    Args:
        probability: Win probability (0 to 1)
    
    Returns:
        American moneyline odds
    """
    # Clamp to avoid division by zero at extremes
    probability = max(0.01, min(0.99, probability))
    
    if probability >= 0.5:
        # Favorite
        ml = -(probability / (1 - probability)) * 100
    else:
        # Underdog
        ml = ((1 - probability) / probability) * 100
    
    return int(ml)


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """
    Calculate betting edge (model vs market)
    
    Args:
        model_prob: Model's predicted probability
        implied_prob: Market's implied probability
    
    Returns:
        Edge percentage (positive = value bet)
    """
    return (model_prob - implied_prob) * 100


def get_odds_features(odds_df: pd.DataFrame, game_id: int) -> Dict:
    """
    Extract odds-based features for model training
    
    Args:
        odds_df: DataFrame with odds data
        game_id: Game ID
    
    Returns:
        Dict with odds features
    """
    consensus = get_consensus_odds(odds_df, game_id)
    
    features = {
        'vegas_spread_home': consensus['spread_home'],
        'vegas_total': consensus['total'],
        'vegas_implied_home_prob': consensus['implied_home_prob'],
        'vegas_implied_away_prob': consensus['implied_away_prob'],
        'vegas_has_odds': 1 if consensus['has_odds'] else 0
    }
    
    return features


def format_american_odds(odds: float) -> str:
    """Format odds as American style (+150 or -150)"""
    if pd.isna(odds) or odds is None:
        return "N/A"
    if odds > 0:
        return f"+{int(odds)}"
    return f"{int(odds)}"


def get_best_bet_recommendation(model_spread: float, vegas_spread: float, 
                                model_prob: float, vegas_prob: float) -> Dict:
    """
    Determine best bet based on model vs Vegas comparison
    
    Args:
        model_spread: Model's predicted spread
        vegas_spread: Vegas spread
        model_prob: Model's win probability
        vegas_prob: Vegas implied probability
    
    Returns:
        Dict with bet recommendation
    """
    spread_diff = abs(model_spread - vegas_spread)
    prob_edge = (model_prob - vegas_prob) * 100
    
    recommendation = {
        'spread_edge': spread_diff,
        'probability_edge': prob_edge,
        'has_value': False,
        'bet_type': None,
        'confidence': 'LOW'
    }
    
    # Determine if there's betting value
    if prob_edge >= 5:  # Model thinks team has 5%+ better chance than market
        recommendation['has_value'] = True
        recommendation['bet_type'] = 'MONEYLINE'
        
        if prob_edge >= 10:
            recommendation['confidence'] = 'HIGH'
        elif prob_edge >= 7:
            recommendation['confidence'] = 'MEDIUM'
    
    if spread_diff >= 3:  # Model spread differs by 3+ points
        recommendation['has_value'] = True
        recommendation['bet_type'] = 'SPREAD'
        
        if spread_diff >= 5:
            recommendation['confidence'] = 'HIGH'
        elif spread_diff >= 4:
            recommendation['confidence'] = 'MEDIUM'
    
    return recommendation
