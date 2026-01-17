import pandas as pd
import numpy as np
import joblib
import json
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

# Configuration
DATA_PATH = 'research/data/btc_usdt_1h.csv'
MODEL_PATH = 'research/model.joblib'
META_PATH = 'research/model_metadata.json'

print(f"Scikit-learn version: {sklearn.__version__}")

TARGET_PROFIT = 0.0150
STOP_LOSS = 0.0100
HORIZON = 24

def apply_triple_barrier_labeling(df, target=TARGET_PROFIT, stop=STOP_LOSS, horizon=HORIZON):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    y = np.zeros(n)
    
    print("Generating labels...")
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
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    df['dow'] = pd.to_datetime(df['open_time']).dt.dayofweek
    
    # Lags
    for i in [1, 3, 6, 12]:
        df[f'ret_lag_{i}'] = df['returns'].shift(i)
        
    # Drop NaNs
    df.dropna(inplace=True)
    return df

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preparing features...")
    df = prepare_features(df)
    
    print("Labeling Data...")
    df['label'] = apply_triple_barrier_labeling(df)
    
    # Drop the last 'horizon' rows where labels are invalid
    valid_df = df.iloc[:-HORIZON].copy()
    
    feature_cols = [c for c in df.columns if c not in ['open_time', 'close_time', 'label', 'ignore', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'returns']]
    
    print(f"Training on {len(valid_df)} samples with features: {feature_cols}")
    
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.1, 
        max_depth=5, 
        class_weight='balanced',
        random_state=42
    )
    model.fit(valid_df[feature_cols], valid_df['label'])
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    
    # Save Metadata (Features, Thresholds)
    metadata = {
        'features': feature_cols,
        'threshold': 0.85, # Updated for higher accuracy
        'target_profit': TARGET_PROFIT,
        'stop_loss': STOP_LOSS,
        'horizon': HORIZON
    }
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}")

if __name__ == "__main__":
    main()
