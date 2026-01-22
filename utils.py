"""
Utility functions for the NBA prediction model
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def get_date_range(start_date: str, end_date: str) -> list:
    """Generate list of dates between start and end"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def calculate_accuracy_metrics(predictions: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    """Calculate various accuracy metrics"""
    metrics = {}
    
    # Winner accuracy
    if 'predicted_winner' in predictions.columns and 'actual_winner' in actuals.columns:
        correct = (predictions['predicted_winner'] == actuals['actual_winner']).sum()
        total = len(predictions)
        metrics['winner_accuracy'] = correct / total if total > 0 else 0
    
    # Score MAE
    if 'predicted_home_score' in predictions.columns and 'actual_home_score' in actuals.columns:
        mae_home = np.mean(np.abs(predictions['predicted_home_score'] - actuals['actual_home_score']))
        mae_visitor = np.mean(np.abs(predictions['predicted_visitor_score'] - actuals['actual_visitor_score']))
        metrics['score_mae'] = (mae_home + mae_visitor) / 2
    
    # Spread accuracy
    if 'predicted_spread' in predictions.columns and 'actual_spread' in actuals.columns:
        mae_spread = np.mean(np.abs(predictions['predicted_spread'] - actuals['actual_spread']))
        metrics['spread_mae'] = mae_spread
    
    return metrics


def format_record(wins: int, losses: int) -> str:
    """Format team record as W-L"""
    return f"{wins}-{losses}"


def calculate_roi(predictions: pd.DataFrame, actuals: pd.DataFrame, bet_amount: float = 100) -> float:
    """Calculate ROI for predictions"""
    # Simplified ROI calculation
    # In reality, this would need odds data
    
    total_bet = len(predictions) * bet_amount
    winnings = 0
    
    for idx, pred in predictions.iterrows():
        if idx in actuals.index:
            actual = actuals.loc[idx]
            
            # Check if prediction was correct
            if pred.get('predicted_winner') == actual.get('actual_winner'):
                # Simplified: assume -110 odds
                winnings += bet_amount * 0.91  # Win
            else:
                winnings -= bet_amount  # Loss
    
    roi = ((winnings - total_bet) / total_bet) * 100 if total_bet > 0 else 0
    return roi


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that dataframe has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return False
    return True


def safe_division(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is 0"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def get_season_from_date(date_str: str) -> int:
    """Get NBA season year from a date string"""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # NBA season starts in October
    if date.month >= 10:
        return date.year
    else:
        return date.year - 1


def exponential_decay_weights(n: int, decay_rate: float = 0.95) -> np.ndarray:
    """Generate exponential decay weights for time series"""
    weights = np.array([decay_rate ** i for i in range(n)])
    return weights / weights.sum()  # Normalize


def safe_get(dictionary: dict, key: str, default=0):
    """Safely get value from dictionary"""
    return dictionary.get(key, default)
