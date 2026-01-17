import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# Configuration
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
START_DATE = '2023-01-01'  # Start date for data collection
OUTPUT_DIR = 'research/data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'btc_usdt_1h.csv')

def fetch_binance_klines(symbol, interval, start_time, end_time=None, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    klines = []
    
    current_time = start_time
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_time,
            'limit': limit
        }
        if end_time:
            params['endTime'] = end_time
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"No more data received at timestamp {current_time}.")
                break
            
            klines.extend(data)
            
            # Update current_time to the close time of the last candle + 1ms
            last_close_time = data[-1][6]
            current_time = last_close_time + 1
            
            # Progress update
            last_date = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
            print(f"Fetched {len(data)} candles. Last date: {last_date}")
            
            # Check if we reached the current time (approx)
            if data[-1][0] >= int(time.time() * 1000):
                break
                
            # Respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5) # Backoff on error
            continue

    return klines

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    print(f"Starting download for {SYMBOL} {INTERVAL} from {START_DATE}...")
    raw_data = fetch_binance_klines(SYMBOL, INTERVAL, start_ts)
    
    print(f"Total candles fetched: {len(raw_data)}")
    
    # Process into DataFrame
    # Binance columns: Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Volume, Number of Trades, Taker Buy Base Asset Volume, Taker Buy Quote Asset Volume, Ignore
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
