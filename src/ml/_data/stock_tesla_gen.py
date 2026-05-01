import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_tesla_trading_data(n_years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * n_years)
    dates = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
    
    # 1. Base Price Simulation (Modeling Tesla's growth and volatility)
    n = len(dates)
    returns = np.random.normal(0.001, 0.03, n) # High volatility typical for TSLA
    price = 50 * np.exp(np.cumsum(returns))
    
    # 2. Market Context (SPY Simulation)
    spy_returns = np.random.normal(0.0005, 0.01, n)
    spy_price = 300 * np.exp(np.cumsum(spy_returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': price,
        'SPY_Close': spy_price
    })
    
    # 3. Corporate Actions: Splits
    # Tesla Split 1: 5-for-1 on 2020-08-31
    # Tesla Split 2: 3-for-1 on 2022-08-25
    df['Split_Factor'] = 1.0
    df.loc[df['Date'] == '2020-08-31', 'Split_Factor'] = 5.0
    df.loc[df['Date'] == '2022-08-25', 'Split_Factor'] = 3.0
    
    # Generate OHLC
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.01, n))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.015, n)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.015, n)))
    df['Volume'] = (np.random.randint(10, 100, n) * 1000000).astype(float)
    
    # 4. Technical Indicators
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    
    # Relative Performance
    df['Relative_Strength_SPY'] = df['Close'] / df['SPY_Close']
    
    # 5. Data Complexities (Injecting "Dirt")
    # Missing values (Simulated market outages)
    df.iloc[random.sample(range(n), 20), [1, 2, 3, 4]] = np.nan
    
    # Extreme Volatility spikes
    spike_idx = random.sample(range(n), 10)
    df.loc[df.index[spike_idx], 'High'] *= 1.5
    df.loc[df.index[spike_idx], 'Low'] *= 0.5
    
    # Inconsistent Date formats
    df['Date_Str'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if np.random.random() > 0.1 else str(int(x.timestamp())))
    
    df.to_csv('stock_tesla.csv', index=False)
    print(f"Generated {len(df)} trading days in stock_tesla.csv")

import random
if __name__ == "__main__":
    generate_tesla_trading_data(5)
