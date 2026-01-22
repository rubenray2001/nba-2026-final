import sys
import os
import json
from api_client import BallDontLieClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_historical_scores():
    try:
        client = BallDontLieClient()
        logger.info("Fetching games for season 2015...")
        
        # Fetch just one page of games from 2015
        games = client.get_games(seasons=[2015], per_page=5)
        
        if not games:
            logger.error("No games returned!")
            return
            
        logger.info(f"Fetched {len(games)} games.")
        
        for i, game in enumerate(games):
            print(f"\n--- Game {i+1} ---")
            print(f"Date: {game.get('date')}")
            print(f"Status: {game.get('status')}")
            print(f"Home: {game.get('home_team', {}).get('full_name')} - Score: {game.get('home_team_score')}")
            print(f"Visitor: {game.get('visitor_team', {}).get('full_name')} - Score: {game.get('visitor_team_score')}")
            print("Raw Score Data:")
            print(f"  home_team_score: {game.get('home_team_score')} (Type: {type(game.get('home_team_score'))})")
            print(f"  visitor_team_score: {game.get('visitor_team_score')} (Type: {type(game.get('visitor_team_score'))})")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_historical_scores()
