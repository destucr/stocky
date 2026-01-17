#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def generate_mock_data(symbol, interval, limit=10):
    """Test mock data generation"""
    print(f"🔄 Generating mock data for {symbol} (API unavailable)")
    
    # Base prices for different symbols (realistic as of 2024)
    base_prices = {
        'BTCUSDT': 42000,
        'ETHUSDT': 2500,
        'SOLUSDT': 90,
        'BNBUSDT': 280,
        'ADAUSDT': 0.45,
        'XRPUSDT': 0.55
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generate time series (1 hour intervals)
    end_time = datetime.now(timezone.utc)
    times = [end_time - timedelta(hours=i) for i in range(limit, 0, -1)]
    
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(times):
        # Random walk with realistic crypto volatility (2-5% hourly moves)
        volatility = 0.02 + (np.random.random() * 0.03)  # 2-5% volatility
        price_change = np.random.normal(0, volatility)
        
        # Calculate OHLC
        open_price = current_price
        close_price = open_price * (1 + price_change)
        
        # High/Low with some randomness
        high_low_spread = abs(price_change) + np.random.random() * 0.01
        high_price = max(open_price, close_price) * (1 + high_low_spread)
        low_price = min(open_price, close_price) * (1 - high_low_spread)
        
        # Volume (realistic for crypto)
        base_volume = 1000000 if 'BTC' in symbol else 500000
        volume = base_volume * (0.5 + np.random.random() * 1.5)
        
        # Create kline data format
        open_time_ms = int(timestamp.timestamp() * 1000)
        close_time_ms = open_time_ms + (60 * 60 * 1000)  # 1 hour later
        
        kline = [
            open_time_ms,           # Open time
            f"{open_price:.8f}",    # Open
            f"{high_price:.8f}",    # High  
            f"{low_price:.8f}",     # Low
            f"{close_price:.8f}",   # Close
            f"{volume:.8f}",        # Volume
            close_time_ms,          # Close time
            "0",                    # Quote asset volume
            100 + int(np.random.random() * 200),  # Number of trades
            "0",                    # Taker buy base asset volume
            "0",                    # Taker buy quote asset volume
            "0"                     # Ignore
        ]
        data.append(kline)
        current_price = close_price
    
    # Parse into DataFrame (same format as real API)
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(data)
    df.columns = columns
    
    print("BEFORE conversion:")
    print(f"  close column type: {df['close'].dtype}")
    print(f"  close sample: {df['close'].head(3).tolist()}")
    
    # Convert to numeric types (critical for calculations)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("AFTER conversion:")
    print(f"  close column type: {df['close'].dtype}")
    print(f"  close sample: {df['close'].head(3).tolist()}")
        
    # Convert timestamp
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Test calculation that was failing
    try:
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        print("✅ Returns calculation successful")
        print(f"  Returns sample: {df['returns'].head(3).tolist()}")
    except Exception as e:
        print(f"❌ Returns calculation failed: {e}")
    
    return df

if __name__ == "__main__":
    df = generate_mock_data('BTCUSDT', '1h', 10)
    print(f"\nFinal DataFrame shape: {df.shape}")
    print(df.head())