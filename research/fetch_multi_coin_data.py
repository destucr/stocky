"""
Fetch historical OHLC data from Binance API for multiple coins.
This will give us more training data for better model accuracy.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# Binance public API (no key needed for OHLC)
BASE_URL = "https://api.binance.com/api/v3/klines"

# Coins to fetch
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

# Timeframes
INTERVALS = {
    "1h": 3600000,   # 1 hour in ms
    "5m": 300000     # 5 min in ms (optional, lots of data)
}

DATA_DIR = "research/data"

def fetch_klines(symbol, interval, start_time, end_time, limit=1000):
    """Fetch klines from Binance API"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return []

def fetch_all_history(symbol, interval="1h", days_back=365*3):
    """Fetch complete history for a symbol"""
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    interval_ms = INTERVALS.get(interval, 3600000)
    
    print(f"Fetching {symbol} {interval} data from {days_back} days ago...")
    
    while current_start < end_time:
        klines = fetch_klines(symbol, interval, current_start, end_time)
        
        if not klines:
            break
            
        all_data.extend(klines)
        
        # Move to next batch
        current_start = klines[-1][0] + interval_ms
        
        # Progress
        progress = (current_start - start_time) / (end_time - start_time) * 100
        print(f"  {symbol}: {len(all_data)} candles ({progress:.1f}%)", end="\r")
        
        # Rate limit
        time.sleep(0.1)
    
    print(f"  {symbol}: {len(all_data)} candles (100%)    ")
    return all_data

def klines_to_dataframe(klines):
    """Convert Binance klines to DataFrame"""
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                'taker_buy_base', 'taker_buy_quote']:
        df[col] = pd.to_numeric(df[col])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    return df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for symbol in SYMBOLS:
        # Fetch 1h data (3 years)
        klines = fetch_all_history(symbol, interval="1h", days_back=365*3)
        
        if klines:
            df = klines_to_dataframe(klines)
            filename = f"{DATA_DIR}/{symbol.lower()}_1h.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Saved {filename} ({len(df)} rows)")
        
        time.sleep(1)  # Be nice to API
    
    # Combine all into one training file
    print("\nCombining all data...")
    all_dfs = []
    
    for symbol in SYMBOLS:
        filename = f"{DATA_DIR}/{symbol.lower()}_1h.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['symbol'] = symbol
            all_dfs.append(df)
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(f"{DATA_DIR}/all_coins_1h.csv", index=False)
        print(f"✅ Saved combined data: {len(combined)} total rows")

if __name__ == "__main__":
    main()
