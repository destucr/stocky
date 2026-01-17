import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, classification_report
import joblib

# Configuration
DATA_PATH = 'research/data/btc_usdt_1h.csv'
FEE = 0.0006
SLIPPAGE = 0.0001
TOTAL_COST = FEE + SLIPPAGE

# Strategy Params
TARGET_PROFIT = 0.0150  # 1.5% Target
STOP_LOSS = 0.0100      # 1.0% Stop
HORIZON = 24            # 24 hours

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

def apply_triple_barrier_labeling(df, target=TARGET_PROFIT, stop=STOP_LOSS, horizon=HORIZON):
    labels = []
    
    # Pre-calculate rolling max high and min low for speed? 
    # Actually, iterative is clearer for the triple barrier logic on a dataframe this size (~300k rows)
    # But slow in pure python. Let's try to vectorize or use rolling.
    
    # We want to know if within [t+1, t+horizon], Price hits Target BEFORE Stop. 
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon) 
    
    # Rolling Max High in next horizon
    future_highs = df['high'].rolling(window=indexer).max()
    # Rolling Min Low in next horizon
    future_lows = df['low'].rolling(window=indexer).min()
    
    # This is rough because it doesn't tell us WHICH happened first.
    # For a strict label, we need the timestamp.
    # But for a "Research Agent" MVP, let's refine:
    # Label 1 if (High > Entry*(1+Target)) AND (Low > Entry*(1-Stop)) -> This assumes High happens first or Low never triggers. 
    # Not perfect. 
    
    # Correct Vectorized Approach:
    # It's hard to vectorize "which came first".
    # Let's stick to a simpler horizon target for the classifier:
    # Target: Return of the Close price 'horizon' steps ahead? No. 
    
    # Let's iterate. 300k rows takes ~10-20 seconds in simple python loop.
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    n = len(df)
    y = np.zeros(n)
    
    print("Generating labels (Triple Barrier)...")
    for i in range(n - horizon):
        entry = closes[i]
        upper = entry * (1 + target)
        lower = entry * (1 - stop)
        
        label = 0 # Default: No trade / Time expiry / Stop loss
        
        for k in range(1, horizon + 1):
            curr_high = highs[i+k]
            curr_low = lows[i+k]
            
            if curr_low < lower:
                label = 0 # Stopped out
                break
            if curr_high > upper:
                label = 1 # Target hit
                break
        
        y[i] = label
        
    return pd.Series(y, index=df.index)

def prepare_features(df):
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
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
        
    # Drop NaNs
    df.dropna(inplace=True)
    return df

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['open_time'] = pd.to_datetime(df['open_time'])
    
    print("Preparing features...")
    df = prepare_features(df)
    
    print("Generating Labels...")
    df['label'] = apply_triple_barrier_labeling(df)
    
    # Remove the last 'horizon' rows where label is invalid
    df = df.iloc[:-HORIZON]
    
    # Class balance check
    positives = df['label'].sum()
    print(f"Positive Labels: {positives} ({positives/len(df):.2%})")
    
    # Split
    split_idx = int(len(df) * 0.8) # 80% Train, 20% Test
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = [c for c in df.columns if c not in ['open_time', 'close_time', 'label', 'ignore', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'returns']]
    
    print(f"Features: {feature_cols}")
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    print("Training Gradient Boosting...")
    # HistGradientBoosting is fast and handles large datasets well
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.1, 
        max_depth=5, 
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    probs = model.predict_proba(X_test)[:, 1]
    
    # Real PnL Backtest
    # We need to simulate the actual outcome for every 'Buy' signal in the test set
    
    # Re-index X_test to match df indices to access Price data
    # (X_test is a slice, so it keeps original index)
    test_indices = X_test.index
    
    print("\n--- Detailed Backtest Analysis ---")
    
    # Pre-calculate outcomes for the test set to speed up threshold scanning
    # For each index in test set, determine the PnL if we bought
    
    outcomes = []
    
    # We need access to the full arrays for high/low lookup
    all_highs = df['high'].values
    all_lows = df['low'].values
    all_closes = df['close'].values
    
    # Map df index to integer location
    # df.index is RangeIndex or Int64Index? 
    # It was reset? No, read_csv creates default RangeIndex 0..N
    # But we dropped rows. So index is not contiguous 0..N? 
    # Let's check df.index. It seems to be 0..N-Horizon.
    
    # To be safe, we use get_indexer
    # indices = df.index.get_indexer(test_indices) # This returns integer positions in 'df'
    # Wait, 'df' here is the filtered one (iloc[:-HORIZON]). 
    # But high/low arrays need to go BEYOND the test set end for the lookahead.
    # We need the original full dataframe for price lookups?
    # Yes.
    
    # Reload full data for lookup to ensure we have the 'future' candles for the very last test samples
    full_df = pd.read_csv(DATA_PATH)
    full_highs = full_df['high'].values
    full_lows = full_df['low'].values
    full_closes = full_df['close'].values
    
    test_start_idx = test_indices[0]
    test_end_idx = test_indices[-1]
    
    print("Calculating realized outcomes for all test samples...")
    realized_pnl = np.zeros(len(test_indices))
    
    for i, idx in enumerate(test_indices):
        entry_price = full_closes[idx]
        target_price = entry_price * (1 + TARGET_PROFIT)
        stop_price = entry_price * (1 - STOP_LOSS)
        
        exit_pnl = 0
        
        # Check horizon
        for k in range(1, HORIZON + 1):
            if idx + k >= len(full_closes):
                break
                
            curr_high = full_highs[idx+k]
            curr_low = full_lows[idx+k]
            curr_close = full_closes[idx+k]
            
            # Check Stop first (Conservative)
            if curr_low <= stop_price:
                # Stopped out
                exit_pnl = -STOP_LOSS - (2 * TOTAL_COST) # Assuming percentage loss fixed
                # Or more precisely: (stop_price - entry_price)/entry_price - Costs
                break
                
            # Check Target
            if curr_high >= target_price:
                exit_pnl = TARGET_PROFIT - (2 * TOTAL_COST)
                break
                
            # If Time Expiry
            if k == HORIZON:
                raw_return = (curr_close - entry_price) / entry_price
                exit_pnl = raw_return - (2 * TOTAL_COST)
        
        realized_pnl[i] = exit_pnl

    # Now scan thresholds
    for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        mask = probs > t
        trades = mask.sum()
        
        if trades == 0:
            continue
            
        total_pnl = realized_pnl[mask].sum()
        avg_pnl = realized_pnl[mask].mean()
        win_rate = (realized_pnl[mask] > 0).mean()
        
        print(f"Thresh {t}: Trades={trades}, WinRate={win_rate:.1%}, Total PnL={total_pnl:.4f}, Avg PnL per Trade={avg_pnl:.4%}")
        
        if avg_pnl > 0:
            best_pnl = avg_pnl
            best_thresh = t

    print(f"\nBest Threshold: {best_thresh}")
    if best_pnl > 0:
        print("DECISION: Strategy Feasible (Positive Expectancy).")
        
        # Save trades for the best threshold
        mask = probs > best_thresh
        trade_indices = test_indices[mask]
        trade_pnl = realized_pnl[mask]
        
        # Get timestamps
        trade_times = full_df.loc[trade_indices, 'open_time']
        
        trades_df = pd.DataFrame({
            'entry_time': trade_times,
            'pnl': trade_pnl
        })
        trades_df.to_csv('research/model_trades.csv', index=False)
        print("Trades saved to research/model_trades.csv")
    else:
        print("DECISION: Strategy Unprofitable. Abstain.")

if __name__ == "__main__":
    main()
