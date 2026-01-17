import time
import json
import joblib
import pandas as pd
import numpy as np
import requests
import os
import argparse
import psycopg2
from datetime import datetime, timezone, timedelta
import threading

# Paths
MODEL_PATH = os.getenv('MODEL_PATH', 'research/model.joblib')
META_PATH = os.getenv('META_PATH', 'research/model_metadata.json')

# DB Config
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASSWORD', 'password')
DB_NAME = os.getenv('DB_NAME', 'stocky')

# Backend Config - Use this for live data from Finnhub
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8080')

# Trading Configuration - Use Finnhub symbol format
TRADING_SYMBOLS = ['BINANCE:BTCUSDT', 'BINANCE:ETHUSDT', 'BINANCE:SOLUSDT', 'BINANCE:BNBUSDT', 'BINANCE:ADAUSDT', 'BINANCE:XRPUSDT']
INTERVAL = 86400  # 86400 seconds = 1 day candles (daily timeframe for day trading)
REAL_TIME_CHECK_INTERVAL = 3600  # Check every hour (daily signals don't need frequent checks)
ANALYSIS_COOLDOWN = 3600  # Analyze every hour (re-evaluate as daily candle develops)

# Signal thresholds - RSI rules for daily timeframe
RSI_OVERSOLD = 30     # Buy when RSI drops below this
RSI_OVERBOUGHT = 70   # Sell when RSI rises above this
RSI_EXTREME_LOW = 25  # Strong buy signal (daily RSI rarely goes this low)
RSI_EXTREME_HIGH = 75 # Strong sell signal (daily RSI rarely goes this high)

# Target/Stop for daily signals (larger moves expected on daily timeframe)
TARGET_PROFIT = 0.03   # 3% profit target
STOP_LOSS = 0.02       # 2% stop loss

# Legacy compatibility
FETCH_SYMBOL = 'BINANCE:BTCUSDT'  # Default symbol
DB_SYMBOL = 'BINANCE:BTCUSDT'  # Default DB mapping

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def broadcast_signal(payload):
    try:
        url = f"{BACKEND_URL}/api/internal/broadcast_signal"
        resp = requests.post(url, json=payload, timeout=2)
        if resp.status_code == 200:
            print("Signal broadcasted successfully.")
        else:
            print(f"Broadcast failed: {resp.status_code}")
    except Exception as e:
        print(f"Broadcast error: {e}")

def save_signal(conn, symbol, sig_type, price, target, stop, prob, timestamp, confidence="medium", rsi=50, trend="NEUTRAL", momentum=0, reason="", expires=None):
    if not conn:
        return
    try:
        cur = conn.cursor()
        query = """
            INSERT INTO signals (symbol, signal_type, price, target, stop, probability, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (symbol, sig_type, price, target, stop, prob, timestamp))
        conn.commit()
        cur.close()
        print(f"💾 Saved Signal: {sig_type} @ ${price:.2f} (Prob: {prob:.3f}, Confidence: {confidence})")
        
        # Enhanced broadcast with ALL actionable details for frontend
        payload = {
            "symbol": symbol,
            "type": sig_type,
            "price": price,
            "target": target,
            "stop": stop,
            "probability": prob,
            "confidence": confidence,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "interval": INTERVAL,
            "analysis_type": "real_time",
            # New enhanced fields for actionable UI
            "rsi": rsi,
            "trend": trend,
            "momentum": momentum,
            "reason": reason,
            "expires": expires if expires else ""
        }
        broadcast_signal(payload)
        
    except Exception as e:
        print(f"❌ Failed to save signal: {e}")
        conn.rollback()

def get_symbol_mapping(symbol):
    """Map Binance symbols to internal database symbols"""
    mapping = {
        'BTCUSDT': 'BINANCE:BTCUSDT',
        'ETHUSDT': 'BINANCE:ETHUSDT', 
        'SOLUSDT': 'BINANCE:SOLUSDT',
        'BNBUSDT': 'BINANCE:BNBUSDT',
        'ADAUSDT': 'BINANCE:ADAUSDT',
        'XRPUSDT': 'BINANCE:XRPUSDT'
    }
    return mapping.get(symbol, f'BINANCE:{symbol}')

# Price bounds for sanity checking (min, max)
PRICE_BOUNDS = {
    'BTCUSDT': (50000, 200000),
    'ETHUSDT': (1500, 10000),
    'SOLUSDT': (50, 500),
    'BNBUSDT': (200, 1500),
    'ADAUSDT': (0.20, 3.00),
    'XRPUSDT': (0.30, 5.00)
}

def validate_price(symbol, price):
    """Check if price is within reasonable bounds for the symbol."""
    bounds = PRICE_BOUNDS.get(symbol, (0, float('inf')))
    return bounds[0] <= price <= bounds[1]

def use_historical_data(symbol, limit=100):
    """
    Use historical CSV data as fallback when API is unavailable.
    Samples from different market conditions in historical data.
    """
    # Only have BTC historical data
    if symbol != 'BTCUSDT':
        print(f"📊 No historical data for {symbol}, using mock data")
        return generate_mock_data(symbol, '1h', limit)
    
    csv_path = '/app/data/btc_usdt_1h.csv'
    if not os.path.exists(csv_path):
        csv_path = 'research/data/btc_usdt_1h.csv'
    
    if not os.path.exists(csv_path):
        print(f"📊 Historical CSV not found, using mock data for {symbol}")
        return generate_mock_data(symbol, '1h', limit)
    
    try:
        full_df = pd.read_csv(csv_path)
        full_df['open_time'] = pd.to_datetime(full_df['open_time'])
        
        # Calculate returns for each window to find bullish/bearish periods
        full_df['returns'] = full_df['close'].pct_change()
        full_df['rolling_return'] = full_df['returns'].rolling(limit).sum()
        
        # Sample based on current hour (changes every hour for variety)
        current_hour = datetime.now(timezone.utc).hour
        np.random.seed(current_hour + datetime.now(timezone.utc).day * 24)
        
        # Find periods with positive momentum (bullish) or negative (bearish)
        # Randomly pick bullish or bearish based on time
        pick_bullish = np.random.random() > 0.4  # 60% chance of bullish sample
        
        valid_starts = []
        for i in range(len(full_df) - limit):
            window_return = full_df['close'].iloc[i + limit - 1] / full_df['close'].iloc[i] - 1
            if pick_bullish and window_return > 0.01:  # >1% gain
                valid_starts.append(i)
            elif not pick_bullish and window_return < -0.01:  # >1% loss
                valid_starts.append(i)
        
        if not valid_starts:
            # Fall back to random sample
            start_idx = np.random.randint(0, len(full_df) - limit)
        else:
            start_idx = np.random.choice(valid_starts)
        
        df = full_df.iloc[start_idx:start_idx + limit].copy()
        
        # Adjust timestamps to simulate "recent" data
        latest_time = datetime.now(timezone.utc)
        time_offsets = [(limit - i - 1) for i in range(len(df))]
        df['open_time'] = [latest_time - timedelta(hours=offset) for offset in time_offsets]
        
        # Scale prices to current market level
        last_historical_price = df['close'].iloc[-1]
        target_price = 95000  # Current approx BTC price
        scale_factor = target_price / last_historical_price
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * scale_factor
        
        # Calculate the trend of this sample
        sample_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        trend_type = "BULLISH" if sample_return > 0 else "BEARISH"
        
        print(f"📊 Using historical data for {symbol} ({trend_type} sample: {sample_return:+.2f}%)")
        print(f"    Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return generate_mock_data(symbol, '1h', limit)

def generate_mock_data(symbol, interval, limit=100):
    """
    Generates realistic mock trading data with trending behavior.
    Simulates real market conditions including trends and momentum.
    """
    print(f"🔄 Generating mock data for {symbol} (API unavailable)")
    
    # Base prices for different symbols (Updated for Jan 2026)
    base_prices = {
        'BTCUSDT': 95000,
        'ETHUSDT': 3300,
        'SOLUSDT': 190,
        'BNBUSDT': 650,
        'ADAUSDT': 0.85,
        'XRPUSDT': 2.20
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Set random seed based on current hour for consistency within the hour
    end_time = datetime.now(timezone.utc)
    np.random.seed(int(end_time.timestamp() / 3600) % 10000)
    
    # Determine market regime (trending or ranging)
    # Changes every few hours
    regime_seed = int(end_time.timestamp() / (3600 * 4)) % 100
    np.random.seed(regime_seed)
    
    # 60% chance of bullish trend, 25% bearish, 15% ranging
    regime_roll = np.random.random()
    if regime_roll < 0.60:
        trend_bias = 0.0003  # Slight upward drift per candle
        regime = "BULLISH"
    elif regime_roll < 0.85:
        trend_bias = -0.0002  # Slight downward drift
        regime = "BEARISH"
    else:
        trend_bias = 0.0
        regime = "RANGING"
    
    # Reset seed for actual data generation
    np.random.seed(int(end_time.timestamp() / 3600) % 10000 + hash(symbol) % 1000)
    
    # Realistic hourly volatility
    hourly_volatility = 0.004  # 0.4% hourly volatility
    mean_reversion_speed = 0.05
    
    # Generate time series
    times = [end_time - timedelta(hours=i) for i in range(limit, 0, -1)]
    
    data = []
    current_price = base_price * (0.97 + np.random.random() * 0.06)  # Start within ±3% of base
    
    for i, timestamp in enumerate(times):
        # Trend component
        trend_component = trend_bias
        
        # Add momentum (recent moves tend to continue slightly)
        if i > 5:
            recent_return = (current_price - float(data[i-5][4].replace(',', ''))) / float(data[i-5][4].replace(',', ''))
            trend_component += recent_return * 0.1
        
        # Mean reversion component (pulls back to base)
        mean_reversion = mean_reversion_speed * (base_price - current_price) / base_price
        
        # Random component
        random_shock = np.random.normal(0, hourly_volatility)
        
        # Combined price change
        price_change = trend_component + mean_reversion * 0.3 + random_shock
        
        # Clamp to prevent extreme moves
        price_change = np.clip(price_change, -0.015, 0.015)
        
        # Calculate OHLC
        open_price = current_price
        close_price = open_price * (1 + price_change)
        
        # Ensure price stays within bounds
        close_price = np.clip(close_price, base_price * 0.85, base_price * 1.15)
        
        # High/Low with realistic intrabar movement
        intrabar_vol = 0.002 + np.random.random() * 0.002
        if close_price > open_price:  # Green candle
            high_price = close_price * (1 + intrabar_vol * 0.5)
            low_price = open_price * (1 - intrabar_vol)
        else:  # Red candle
            high_price = open_price * (1 + intrabar_vol)
            low_price = close_price * (1 - intrabar_vol * 0.5)
        
        # Volume
        base_volume = 1000000 if 'BTC' in symbol else 500000
        volume = base_volume * (0.5 + np.random.random() * 1.5)
        
        # Create kline
        open_time_ms = int(timestamp.timestamp() * 1000)
        close_time_ms = open_time_ms + (60 * 60 * 1000)
        
        kline = [
            open_time_ms,
            f"{open_price:.8f}",
            f"{high_price:.8f}",
            f"{low_price:.8f}",
            f"{close_price:.8f}",
            f"{volume:.8f}",
            close_time_ms,
            "0", 100 + int(np.random.random() * 200), "0", "0", "0"
        ]
        data.append(kline)
        current_price = close_price
    
    # Parse into DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(data)
    df.columns = columns
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Calculate sample trend
    sample_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    print(f"✅ Generated {len(df)} mock candles for {symbol} ({regime})")
    print(f"    Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"    Current price: ${df['close'].iloc[-1]:.2f} ({sample_return:+.2f}%)")
    return df

def fetch_from_backend(symbol, interval=60, limit=120):
    """
    Fetch OHLC data from the Go backend which gets live data from Finnhub.
    """
    try:
        url = f"{BACKEND_URL}/api/history"
        params = {
            'symbol': symbol,
            'interval': interval,  # seconds (60 = 1 minute)
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or len(data) == 0:
            print(f"⚠️ No data returned from backend for {symbol}")
            return None
        
        # Backend returns array of OHLC objects
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns to match expected format
        if 'time' in df.columns:
            df['open_time'] = pd.to_datetime(df['time'])
        elif 'timestamp' in df.columns:
            df['open_time'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing volume with 0
        if 'volume' not in df.columns:
            df['volume'] = 1000000  # Default volume for crypto
        
        # Sort by time
        df = df.sort_values('open_time').reset_index(drop=True)
        
        current_price = df['close'].iloc[-1]
        print(f"✅ Fetched {len(df)} candles from backend for {symbol}")
        print(f"    Latest price: ${current_price:.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Backend fetch failed for {symbol}: {e}")
        return None

def fetch_latest_price(symbol):
    """
    Fetch the latest price from the backend's /api/prices endpoint.
    """
    try:
        url = f"{BACKEND_URL}/api/prices"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        prices = response.json()
        
        if symbol in prices:
            return float(prices[symbol])
        
        # Try without BINANCE: prefix
        short_symbol = symbol.replace('BINANCE:', '')
        if short_symbol in prices:
            return float(prices[short_symbol])
            
        return None
    except Exception as e:
        print(f"❌ Failed to fetch latest price: {e}")
        return None

def fetch_recent_data(symbol, interval=86400, limit=50):
    """
    Fetches the last N daily candles from the backend.
    Returns None if not enough data (no fallback to mock data).
    """
    # First try the backend
    df = fetch_from_backend(symbol, interval, limit)
    
    if df is not None and len(df) >= 20:
        # Sanity check on prices
        current_price = df['close'].iloc[-1]
        # Extract base symbol for price validation
        base_symbol = symbol.replace('BINANCE:', '')
        if validate_price(base_symbol, current_price):
            return df
        else:
            print(f"⚠️ Price sanity check failed for {symbol}: ${current_price:.2f}")
            return None
    
    # Not enough real data - don't use mock data (leads to false signals)
    if df is not None:
        print(f"⏳ Waiting for more data: {symbol} has {len(df)} daily candles (need 20+)")
    else:
        print(f"⏳ No data yet for {symbol}")
    return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def compute_enhanced_features(df):
    """
    Enhanced feature engineering with additional technical indicators for better signals.
    """
    df = df.copy()
    
    # Debug data types before calculations
    print(f"    DataFrame dtypes before features: close={df['close'].dtype}, volume={df['volume'].dtype}")
    print(f"    Sample close values: {df['close'].head(2).tolist()}")
    
    try:
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
    except Exception as e:
        print(f"❌ Error in returns calculation: {e}")
        print(f"    close column type: {df['close'].dtype}")
        print(f"    close sample: {df['close'].head(3).tolist()}")
        raise
    
    # Momentum
    df['rsi'] = compute_rsi(df['close'])
    df['rsi_ma'] = df['rsi'].rolling(10).mean()
    
    # Volatility
    df['atr'] = compute_atr(df)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['rel_vol'] = df['volume'] / df['vol_ma']
    
    # Time
    df['hour'] = df['open_time'].dt.hour
    df['dow'] = df['open_time'].dt.dayofweek
    
    # Lags
    for i in [1, 3, 6, 12]:
        df[f'ret_lag_{i}'] = df['returns'].shift(i)
    
    return df

def compute_features(df):
    """
    Original feature engineering for backward compatibility.
    """
    return compute_enhanced_features(df)

def detect_trend(df):
    """
    Detect current market trend based on recent price action.
    Returns: 'UPTREND', 'DOWNTREND', or 'NEUTRAL'
    """
    if len(df) < 20:
        return 'NEUTRAL', 0.0
    
    # Calculate short-term and medium-term momentum
    close_prices = df['close'].values
    
    # Last 6 candles momentum (short-term)
    short_momentum = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) >= 6 else 0
    
    # Last 12 candles momentum (medium-term)
    medium_momentum = (close_prices[-1] - close_prices[-12]) / close_prices[-12] if len(close_prices) >= 12 else 0
    
    # Count green vs red candles in last 10
    green_count = 0
    for i in range(-10, 0):
        if i < -len(df):
            continue
        if df['close'].iloc[i] > df['open'].iloc[i]:
            green_count += 1
    
    green_ratio = green_count / 10.0
    
    # Determine trend
    momentum_score = (short_momentum * 0.6 + medium_momentum * 0.4) * 100  # Convert to percentage
    
    if short_momentum > 0.005 and green_ratio >= 0.6:
        return 'UPTREND', momentum_score
    elif short_momentum < -0.005 and green_ratio <= 0.4:
        return 'DOWNTREND', momentum_score
    else:
        return 'NEUTRAL', momentum_score

def generate_signal_recommendation(prob, price, meta, symbol, df=None):
    """
    Generate BUY/SELL/HOLD using simple RSI rules (more reliable than ML).
    
    Strategy:
    - BUY when RSI < 30 (oversold) AND price showing recovery
    - SELL when RSI > 70 (overbought) AND price showing weakness
    - HOLD otherwise
    """
    signal_timestamp = datetime.now(timezone.utc)
    
    # Get RSI and trend from data
    rsi = 50  # Default neutral
    trend = 'NEUTRAL'
    momentum_score = 0.0
    
    if df is not None and len(df) > 0:
        trend, momentum_score = detect_trend(df)
        
        # Get current RSI from last row
        if 'rsi' in df.columns:
            rsi = float(df['rsi'].iloc[-1])
    
    # Simple RSI-based rules
    sig_type = "HOLD"
    target_price = 0.0
    stop_price = 0.0
    confidence = "LOW"
    action_msg = "⏸️ HOLD"
    signal_reason = ""
    
    # BUY CONDITIONS: RSI oversold + uptrend confirmation
    if rsi < RSI_EXTREME_LOW:
        sig_type = "BUY"
        confidence = "HIGH"
        signal_reason = f"RSI extremely oversold ({rsi:.1f})"
        action_msg = f"🟢 STRONG BUY"
    elif rsi < RSI_OVERSOLD and trend in ['UPTREND', 'NEUTRAL']:
        sig_type = "BUY"
        confidence = "MEDIUM" if trend == 'UPTREND' else "LOW"
        signal_reason = f"RSI oversold ({rsi:.1f}) + {trend}"
        action_msg = f"🟢 BUY"
    
    # SELL CONDITIONS: RSI overbought + downtrend confirmation  
    elif rsi > RSI_EXTREME_HIGH:
        sig_type = "SELL"
        confidence = "HIGH"
        signal_reason = f"RSI extremely overbought ({rsi:.1f})"
        action_msg = f"🔴 STRONG SELL"
    elif rsi > RSI_OVERBOUGHT and trend in ['DOWNTREND', 'NEUTRAL']:
        sig_type = "SELL"
        confidence = "MEDIUM" if trend == 'DOWNTREND' else "LOW"
        signal_reason = f"RSI overbought ({rsi:.1f}) + {trend}"
        action_msg = f"🔴 SELL"
    else:
        signal_reason = f"RSI neutral ({rsi:.1f})"
        action_msg = f"⏸️ HOLD/WAIT"
    
    # Calculate target and stop prices
    if sig_type == "BUY":
        target_price = price * (1 + TARGET_PROFIT)
        stop_price = price * (1 - STOP_LOSS)
    elif sig_type == "SELL":
        target_price = price * (1 - TARGET_PROFIT)
        stop_price = price * (1 + STOP_LOSS)
    
    # Calculate signal validity and next check
    signal_valid_hours = 24  # Daily signals valid for 24 hours
    signal_expires = signal_timestamp + timedelta(hours=signal_valid_hours)
    next_analysis = signal_timestamp + timedelta(seconds=ANALYSIS_COOLDOWN)
    
    # Print clear, actionable output
    print(f"\n{'='*65}")
    print(f"  {symbol}")
    print(f"{'='*65}")
    print(f"  💰 CURRENT PRICE: ${price:,.2f}")
    print(f"  📊 RSI: {rsi:.1f} | Trend: {trend} ({momentum_score:+.2f}%)")
    print(f"{'='*65}")
    
    if sig_type == "BUY":
        print(f"  🟢 ACTION: BUY NOW")
        print(f"  📍 Entry Price: ${price:,.2f}")
        print(f"  🎯 Take Profit: ${target_price:,.2f} (+{TARGET_PROFIT*100:.1f}%)")
        print(f"  🛑 Stop Loss:   ${stop_price:,.2f} (-{STOP_LOSS*100:.1f}%)")
        print(f"  ⏱️  Signal Valid Until: {signal_expires.strftime('%H:%M UTC')}")
        print(f"  📈 Confidence: {confidence}")
        print(f"  💡 Reason: {signal_reason}")
        print(f"{'='*65}")
        print(f"  📋 INSTRUCTIONS:")
        print(f"     1. Place BUY order at ${price:,.2f}")
        print(f"     2. Set take-profit at ${target_price:,.2f}")
        print(f"     3. Set stop-loss at ${stop_price:,.2f}")
        print(f"     4. If neither hit in 24hrs, re-evaluate")
        
    elif sig_type == "SELL":
        print(f"  🔴 ACTION: SELL/SHORT NOW")
        print(f"  📍 Entry Price: ${price:,.2f}")
        print(f"  🎯 Take Profit: ${target_price:,.2f} (-{TARGET_PROFIT*100:.1f}%)")
        print(f"  🛑 Stop Loss:   ${stop_price:,.2f} (+{STOP_LOSS*100:.1f}%)")
        print(f"  ⏱️  Signal Valid Until: {signal_expires.strftime('%H:%M UTC')}")
        print(f"  📈 Confidence: {confidence}")
        print(f"  💡 Reason: {signal_reason}")
        print(f"{'='*65}")
        print(f"  📋 INSTRUCTIONS:")
        print(f"     1. Place SELL/SHORT order at ${price:,.2f}")
        print(f"     2. Set take-profit at ${target_price:,.2f}")
        print(f"     3. Set stop-loss at ${stop_price:,.2f}")
        print(f"     4. If neither hit in 24hrs, re-evaluate")
        
    else:  # HOLD
        # Calculate how far RSI is from triggering a signal
        if rsi < 50:
            rsi_to_buy = RSI_OVERSOLD - rsi
            print(f"  ⏸️  ACTION: WAIT - No clear signal")
            print(f"  💡 RSI at {rsi:.1f} - needs to drop {rsi_to_buy:.1f} more for BUY signal")
        else:
            rsi_to_sell = rsi - RSI_OVERBOUGHT
            if rsi_to_sell < 0:
                print(f"  ⏸️  ACTION: WAIT - No clear signal")
                print(f"  💡 RSI at {rsi:.1f} - needs to rise {-rsi_to_sell:.1f} more for SELL signal")
            else:
                print(f"  ⏸️  ACTION: WAIT - No clear signal")
                print(f"  💡 RSI at {rsi:.1f} - in neutral zone (30-70)")
    
    print(f"{'='*65}")
    print(f"  ⏰ Next Analysis: {next_analysis.strftime('%H:%M:%S UTC')} ({ANALYSIS_COOLDOWN}s)")
    print(f"{'='*65}\n")
    
    return {
        'type': sig_type,
        'price': price,
        'target': target_price,
        'stop': stop_price,
        'probability': prob,
        'confidence': confidence,
        'timestamp': signal_timestamp,
        'expires': signal_expires.isoformat(),
        'trend': trend,
        'momentum': momentum_score,
        'rsi': rsi,
        'reason': signal_reason
    }

def analyze_symbol_realtime(symbol, model, meta, conn):
    """
    Perform daily timeframe analysis for a single symbol.
    Returns None if not enough data - signals are only generated with real data.
    """
    try:
        # Fetch recent daily candles from backend (50 days for good RSI calculation)
        df = fetch_recent_data(symbol, INTERVAL, limit=50)
        
        if df is None:
            # Message already printed by fetch_recent_data
            return None
            
        if df.empty or len(df) < 20:  # Need enough history for features
            print(f"⏳ {symbol}: Only {len(df)} daily candles (need 20+ for features)")
            return None
        
        # Try to get latest real-time price from backend
        latest_price = fetch_latest_price(symbol)
        
        # Compute features
        full_df = compute_features(df)
        
        if len(full_df) < 20:
            print(f"⚠️  Not enough data after feature computation for {symbol}")
            return None
        
        # Use most recent row for analysis
        signal_row = full_df.iloc[-1]
        
        # Use real-time price if available, otherwise use candle close
        if latest_price is not None:
            close_price = latest_price
            print(f"📡 Using real-time price: ${close_price:.2f}")
        else:
            close_price = float(signal_row['close'])
        
        # Prepare features for model
        try:
            input_vector = signal_row[meta['features']].values.reshape(1, -1)
        except KeyError as e:
            print(f"❌ Missing features for {symbol}: {e}")
            print(f"   Available: {list(signal_row.index)}")
            print(f"   Required: {meta['features']}")
            return None
            
        # Get model prediction
        prob = float(model.predict_proba(input_vector)[0][1])
        
        # Generate recommendation with trend data
        signal = generate_signal_recommendation(prob, close_price, meta, symbol, df=full_df)
        
        # Save to database and broadcast - use the symbol directly (already in BINANCE:BTCUSDT format)
        if conn:
            save_signal(
                conn, symbol, signal['type'], signal['price'],
                signal['target'], signal['stop'], signal['probability'],
                signal['timestamp'], signal['confidence'],
                rsi=signal.get('rsi', 50),
                trend=signal.get('trend', 'NEUTRAL'),
                momentum=signal.get('momentum', 0),
                reason=signal.get('reason', ''),
                expires=signal.get('expires', '')
            )
        
        return signal
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_real_time_monitoring():
    """
    Continuous real-time monitoring and analysis using live Finnhub data.
    """
    print("🚀 Starting Daily Trading Signal System")
    print(f"📊 Monitoring symbols: {', '.join(TRADING_SYMBOLS)}")
    print(f"🌐 Data source: {BACKEND_URL} (Finnhub)")
    print(f"⏱️  Candle interval: Daily (1D)")
    print(f"🔄 Signal check every {ANALYSIS_COOLDOWN//3600} hour(s)")
    print(f"📈 Strategy: Daily RSI (Buy < {RSI_OVERSOLD}, Sell > {RSI_OVERBOUGHT})")
    print(f"🎯 Targets: +{TARGET_PROFIT*100:.0f}% TP / -{STOP_LOSS*100:.0f}% SL")
    
    # Load model and metadata
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        print("❌ Model or metadata not found. Run training first.")
        return

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    
    print(f"✅ Model loaded with threshold: {meta['threshold']}")
    print(f"📈 Target profit: {meta['target_profit']*100}% | Stop loss: {meta['stop_loss']*100}%")
    
    # Database connection with retry
    conn = None
    for i in range(5):
        conn = get_db_connection()
        if conn:
            break
        print(f"⏳ DB connection attempt {i+1}/5 failed, retrying in 5s...")
        time.sleep(5)
    
    if not conn:
        print("❌ Could not connect to database, running without persistence.")
    
    # Track last analysis times to avoid duplicates
    last_analysis = {}
    
    while True:
        try:
            current_time = datetime.now(timezone.utc)
            print(f"\n🔍 [{current_time.strftime('%H:%M:%S')}] Real-time market scan...")
            
            # Analyze each symbol
            for symbol in TRADING_SYMBOLS:
                try:
                    # Check if we need to analyze this symbol
                    last_time = last_analysis.get(symbol, datetime.min.replace(tzinfo=timezone.utc))
                    time_diff = (current_time - last_time).total_seconds()
                    
                    # Analyze every ANALYSIS_COOLDOWN seconds (5 min default)
                    if time_diff >= ANALYSIS_COOLDOWN or symbol not in last_analysis:
                        signal = analyze_symbol_realtime(symbol, model, meta, conn)
                        if signal:
                            last_analysis[symbol] = current_time
                    else:
                        remaining = ANALYSIS_COOLDOWN - time_diff
                        print(f"⏳ {symbol}: Next analysis in {remaining:.0f}s")
                        
                except Exception as e:
                    print(f"❌ Error with {symbol}: {e}")
                    continue
            
            # Wait before next scan
            print(f"💤 Sleeping for {REAL_TIME_CHECK_INTERVAL} seconds...")
            time.sleep(REAL_TIME_CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down real-time monitoring...")
            break
        except Exception as e:
            print(f"❌ Unexpected error in main loop: {e}")
            time.sleep(10)  # Wait before retrying
    
    if conn:
        conn.close()
    print("👋 Real-time monitoring stopped.")

def run_agent(once=False):
    """
    Legacy single-symbol analysis function for backward compatibility.
    """
    if once:
        # Run single analysis
        symbol = FETCH_SYMBOL
        db_symbol = get_symbol_mapping(symbol)
        
        if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
            print("❌ Model or metadata not found.")
            return

        model = joblib.load(MODEL_PATH)
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        
        conn = get_db_connection()
        signal = analyze_symbol_realtime(symbol, model, meta, conn)
        
        if conn:
            conn.close()
        return
    
    # Run continuous real-time monitoring for all symbols
    run_real_time_monitoring()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    run_agent(once=args.once)
