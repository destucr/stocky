import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

# Configuration
DATA_PATH = 'research/data/btc_usdt_1h.csv'
FEE = 0.0006
SLIPPAGE = 0.0001
TOTAL_COST = FEE + SLIPPAGE
TARGET_PROFIT = 0.0150
STOP_LOSS = 0.0100
HORIZON = 24
THRESHOLD = 0.7

def apply_triple_barrier_labeling(df, target=TARGET_PROFIT, stop=STOP_LOSS, horizon=HORIZON):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    y = np.zeros(n)
    
    for i in range(n - horizon):
        entry = closes[i]
        upper = entry * (1 + target)
        lower = entry * (1 - stop)
        label = 0
        for k in range(1, horizon + 1):
            if lows[i+k] < lower:
                label = 0
                break
            if highs[i+k] > upper:
                label = 1
                break
        y[i] = label
    return pd.Series(y, index=df.index)

def prepare_features(df):
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi'] = 100 - (100 / (1 + (df['returns'].mask(df['returns']<0,0).rolling(14).mean() / df['returns'].mask(df['returns']>0,0).abs().rolling(14).mean())))
    df['rsi_ma'] = df['rsi'].rolling(10).mean()
    
    # ATR
    h_l = df['high'] - df['low']
    h_c = (df['high'] - df['close'].shift()).abs()
    l_c = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['rel_vol'] = df['volume'] / df['vol_ma']
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    df['dow'] = pd.to_datetime(df['open_time']).dt.dayofweek
    for i in [1, 3, 6, 12]:
        df[f'ret_lag_{i}'] = df['returns'].shift(i)
    df.dropna(inplace=True)
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    full_df = df.copy() # Keep raw for price lookup
    
    df = prepare_features(df)
    df['label'] = apply_triple_barrier_labeling(df)
    df = df.iloc[:-HORIZON]
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = [c for c in df.columns if c not in ['open_time', 'close_time', 'label', 'ignore', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'returns']]
    
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, class_weight='balanced', random_state=42)
    model.fit(train_df[feature_cols], train_df['label'])
    
    probs = model.predict_proba(test_df[feature_cols])[:, 1]
    
    # Extract Trades
    mask = probs > THRESHOLD
    test_indices = test_df.index
    trade_indices = test_indices[mask]
    
    print(f"Analyzing Threshold {THRESHOLD} with {len(trade_indices)} trades...")
    
    pnl = []
    dates = []
    
    full_closes = full_df['close'].values
    full_highs = full_df['high'].values
    full_lows = full_df['low'].values
    
    for idx in trade_indices:
        entry_price = full_closes[idx]
        target_price = entry_price * (1 + TARGET_PROFIT)
        stop_price = entry_price * (1 - STOP_LOSS)
        
        trade_pnl = 0
        for k in range(1, HORIZON + 1):
            curr_high = full_highs[idx+k]
            curr_low = full_lows[idx+k]
            curr_close = full_closes[idx+k]
            
            if curr_low <= stop_price:
                trade_pnl = -STOP_LOSS - (2 * TOTAL_COST)
                break
            if curr_high >= target_price:
                trade_pnl = TARGET_PROFIT - (2 * TOTAL_COST)
                break
            if k == HORIZON:
                trade_pnl = (curr_close - entry_price)/entry_price - (2 * TOTAL_COST)
        
        pnl.append(trade_pnl)
        dates.append(full_df.loc[idx, 'open_time'])
        
    results = pd.DataFrame({'date': dates, 'pnl': pnl})
    results['cum_pnl'] = results['pnl'].cumsum()
    
    print(results.describe())
    
    # Save for plotting or inspection
    results.to_csv('research/equity_curve.csv', index=False)
    print("Saved to research/equity_curve.csv")

if __name__ == "__main__":
    main()
