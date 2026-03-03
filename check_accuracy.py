"""Check tonight's predictions"""
import json

d = json.load(open('data/predictions.json'))
games = d['games']

# Find Feb 12 predictions (today)
today_games = {k: v for k, v in games.items() if v.get('predicted_at', '').startswith('2026-02-12')}

if not today_games:
    # Check Feb 11 predictions without results (might be tonight's games)
    pending = {k: v for k, v in games.items() if v.get('result') is None}
    print(f"No Feb 12 predictions found. {len(pending)} pending predictions:")
    for k, v in pending.items():
        conf = v.get('confidence', 0)
        tier = "LOCK" if conf >= 0.70 else "HIGH" if conf >= 0.65 else "GOOD" if conf >= 0.60 else "STD" if conf >= 0.55 else "LOW"
        ml_pick = v.get('betting_ml_pick', '-')
        ml_conf = v.get('betting_ml_conf', 0)
        print(f"  {v.get('visitor_team','?')} @ {v.get('home_team','?')}")
        print(f"    Winner Pick: {v.get('predicted_winner','?')} ({conf:.0%}) [{tier}]")
        print(f"    ML Pick: {ml_pick} ({ml_conf:.0%})")
        print(f"    Spread Pick: {v.get('betting_spread_pick','-')} | O/U Pick: {v.get('betting_total_pick','-')}")
        print()
else:
    print(f"Found {len(today_games)} predictions for today:")
    for k, v in sorted(today_games.items(), key=lambda x: x[1].get('confidence', 0), reverse=True):
        conf = v.get('confidence', 0)
        tier = "LOCK" if conf >= 0.70 else "HIGH" if conf >= 0.65 else "GOOD" if conf >= 0.60 else "STD" if conf >= 0.55 else "LOW"
        ml_pick = v.get('betting_ml_pick', '-')
        ml_conf = v.get('betting_ml_conf', 0)
        print(f"  {v.get('visitor_team','?')} @ {v.get('home_team','?')}")
        print(f"    Winner Pick: {v.get('predicted_winner','?')} ({conf:.0%}) [{tier}]")
        print(f"    ML Pick: {ml_pick} ({ml_conf:.0%})")
        print(f"    Spread Pick: {v.get('betting_spread_pick','-')} | O/U Pick: {v.get('betting_total_pick','-')}")
        print()
