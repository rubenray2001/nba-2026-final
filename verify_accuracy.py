import pandas as pd
from features_enhanced import EnhancedFeatureEngineer
import config

def verify_all_features():
    print("\n[VERIFYING FULL FEATURE ENGINE ACCURACY...]")
    fe = EnhancedFeatureEngineer()
    
    # ---------------------------------------------------------
    # 1. SETUP MOCK DATA
    # ---------------------------------------------------------
    current_date = '2024-03-01'
    home_id = 1610612760 # OKC
    visitor_id = 1610612738 # Celtics
    
    # Mock Target Game
    game = {
        'game_id': 'verify_target',
        'home_team_id': home_id,
        'visitor_team_id': visitor_id,
        'date': current_date,
        'season': 2023,
        'home_team_score': 0, 'visitor_team_score': 0
    }
    
    # Mock History (5 games for OKC: 4 Wins, 1 Loss) -> Expect 80% Win Rate
    # Dates: Feb 20, 22, 24, 26, 28
    history_rows = []
    for i in range(5):
        date = f'2024-02-2{i}'
        won = 1 if i < 4 else 0 # 4 wins, 1 loss
        history_rows.append({
            'game_id': f'hist_{i}',
            'date': date,
            'home_team_id': home_id,
            'visitor_team_id': 999, # Some other team
            'home_team_score': 110 if won else 90,
            'visitor_team_score': 100 if won else 115,
            'season': 2023,
            'status': 'Final'
        })
        
    # Mock H2H History (OKC beat BOS once before)
    history_rows.append({
        'game_id': 'h2h_1',
        'date': '2024-01-01',
        'home_team_id': home_id,
        'visitor_team_id': visitor_id,
        'home_team_score': 120,
        'visitor_team_score': 110,
        'season': 2023,
        'status': 'Final'
    })
    
    historical_df = pd.DataFrame(history_rows)
    
    # Mock Vegas Odds
    odds_data = [{
        'game_id': 'verify_target',
        'spread_home_value': -5.5,
        'total_value': 230.5,
        'moneyline_home_odds': -200,
        'moneyline_away_odds': 170
    }]
    odds_df = pd.DataFrame(odds_data)
    
    # Mock Injuries (SGA Out)
    injuries = [{
        'player': {'first_name': 'Shai', 'last_name': 'Gilgeous-Alexander', 'team_id': home_id, 'position': 'G'},
        'status': 'Out', 'description': 'Rest'
    }]

    # ---------------------------------------------------------
    # 2. RUN BUILDER
    # ---------------------------------------------------------
    print("Running feature builder with full context...")
    features = fe.build_features_for_game(
        game, 
        historical_data={'games': historical_df}, 
        current_season=2023,
        injuries=injuries,
        odds_df=odds_df,
        player_stats={} 
    )
    
    # ---------------------------------------------------------
    # 3. VERIFY RESULTS
    # ---------------------------------------------------------
    print("\n[RESULTS]:")
    
    # A. Rolling Stats (Expect 0.80)
    # Note: Logic depends on 'home_win_pct_last5' matching the mock history
    # The base FeatureEngineer calculates this from the `games` df provided.
    # It might filter by date < current_date.
    
    win_pct = features.get('home_win_pct_last5', 0)
    print(f"Rolling Win % (Last 5): {win_pct:.2f} (Expected: 0.80)")
    
    # B. Vegas Odds
    spread = features.get('vegas_spread_home', 0)
    print(f"Vegas Spread:           {spread} (Expected: -5.5)")
    
    # C. H2H
    h2h_wins = features.get('h2h_home_wins', 0)
    print(f"H2H Home Wins:          {h2h_wins} (Expected: 1)")
    
    # D. Injury
    impact = features.get('home_injury_impact', 0)
    print(f"Injury Impact:          {impact} (Expected: 31.5)")

    # Final Check
    checks = [
        abs(win_pct - 0.80) < 0.01,
        abs(spread + 5.5) < 0.1,
        h2h_wins == 1,
        abs(impact - 31.5) < 1.0
    ]
    
    if all(checks):
        print("\n[ALL ACCURACY CHECKS PASSED]: Rolling stats, Vegas, H2H, and Injury logic are correct.")
    else:
        print("\n[ACCURACY CHECK FAILED]: One or more features incorrect.")

if __name__ == "__main__":
    verify_all_features()
