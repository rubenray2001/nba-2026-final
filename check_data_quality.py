
import pandas as pd
try:
    df = pd.read_csv("data/training_data.csv")
    print(f"Loaded {len(df)} rows.")
    
    # Check date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        is_sorted = df['date'].is_monotonic_increasing
        print(f"Is data sorted by date? {is_sorted}")
        if not is_sorted:
            print("Data is NOT sorted by date! This invalidates the time-series split.")
            # Show where it breaks
            print(df[['date', 'game_id']].head(10))
            print("...")
            print(df[['date', 'game_id']].tail(10))
    else:
        print("No 'date' column found!")

    # Check class balance
    if 'home_won' in df.columns:
        print("\nClass balance:")
        print(df['home_won'].value_counts(normalize=True))
        
    # Check for NaN
    print(f"\nNaN values:\n{df.isnull().sum().sum()}")
    
except Exception as e:
    print(f"Error: {e}")
