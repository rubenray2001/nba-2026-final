"""
Vegas Odds Utilities
Parse and aggregate betting odds from multiple vendors
Identical to NBA model
"""
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Optional


def get_consensus_odds(odds_df: pd.DataFrame, game_id: int, preferred_vendors: list = None) -> Dict:
    """Get consensus odds for a game from multiple vendors"""
    if odds_df.empty:
        return get_default_odds()
    
    game_odds = odds_df[odds_df['game_id'] == game_id].copy()
    
    if game_odds.empty:
        # Try string comparison
        game_odds = odds_df[odds_df['game_id'].astype(str) == str(game_id)].copy()
    
    if game_odds.empty:
        return get_default_odds()
    
    if preferred_vendors:
        preferred_odds = game_odds[game_odds['vendor'].isin(preferred_vendors)]
        if not preferred_odds.empty:
            game_odds = preferred_odds.copy()
    
    numeric_cols = ['spread_home_value', 'spread_away_value', 'total_value',
                   'moneyline_home_odds', 'moneyline_away_odds']
    
    for col in numeric_cols:
        if col in game_odds.columns:
            if game_odds[col].dtype == object:
                game_odds[col] = game_odds[col].astype(str).str.replace('"', '').str.replace("'", "")
            game_odds.loc[:, col] = pd.to_numeric(game_odds[col], errors='coerce')
    
    clean_odds = game_odds.copy()
    clean_odds.loc[clean_odds['moneyline_home_odds'].abs() > 5000, 'moneyline_home_odds'] = np.nan
    clean_odds.loc[clean_odds['moneyline_away_odds'].abs() > 5000, 'moneyline_away_odds'] = np.nan
    # College spreads can be larger than NBA (30+ point favorites)
    clean_odds.loc[clean_odds['spread_home_value'].abs() > 40, 'spread_home_value'] = np.nan
    clean_odds.loc[clean_odds['spread_away_value'].abs() > 40, 'spread_away_value'] = np.nan
    # College totals typically 110-180
    clean_odds.loc[(clean_odds['total_value'] < 90) | (clean_odds['total_value'] > 220), 'total_value'] = np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        consensus = {
            'spread_home': clean_odds['spread_home_value'].median(),
            'spread_away': clean_odds['spread_away_value'].median(),
            'total': clean_odds['total_value'].median(),
            'moneyline_home': clean_odds['moneyline_home_odds'].median(),
            'moneyline_away': clean_odds['moneyline_away_odds'].median(),
            'num_vendors': len(game_odds),
            'has_odds': True
        }
    
    if pd.isna(consensus['spread_home']) and pd.isna(consensus['moneyline_home']):
        return get_default_odds()
    
    consensus['implied_home_prob'] = moneyline_to_probability(consensus['moneyline_home'])
    consensus['implied_away_prob'] = moneyline_to_probability(consensus['moneyline_away'])
    
    return consensus


def get_default_odds() -> Dict:
    """Return default odds when no data available"""
    return {
        'spread_home': 0.0,
        'spread_away': 0.0,
        'total': 140.0,  # College avg (men's ~144, women's ~134)
        'moneyline_home': -110,
        'moneyline_away': -110,
        'implied_home_prob': 0.5,
        'implied_away_prob': 0.5,
        'num_vendors': 0,
        'has_odds': False
    }


def moneyline_to_probability(moneyline: float) -> float:
    """Convert moneyline odds to implied probability"""
    if pd.isna(moneyline) or moneyline == 0:
        return 0.5
    if abs(moneyline) > 5000:
        return 0.5
    if moneyline < 0:
        prob = abs(moneyline) / (abs(moneyline) + 100)
    else:
        prob = 100 / (moneyline + 100)
    return max(0.05, min(0.95, prob))


def probability_to_moneyline(probability: float) -> int:
    """Convert probability to moneyline odds"""
    probability = max(0.01, min(0.99, probability))
    if probability >= 0.5:
        ml = -(probability / (1 - probability)) * 100
    else:
        ml = ((1 - probability) / probability) * 100
    return int(ml)


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """Calculate betting edge (model vs market)"""
    return (model_prob - implied_prob) * 100


def format_american_odds(odds: float) -> str:
    """Format odds as American style (+150 or -150)"""
    if pd.isna(odds) or odds is None:
        return "N/A"
    if odds > 0:
        return f"+{int(odds)}"
    return f"{int(odds)}"
