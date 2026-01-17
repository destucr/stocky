"""
Train XGBoost on 5-minute BTC data (320K samples)
More data = better model
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_PATH = 'research/data/btc_usdt_5m.csv'
MODEL_PATH = 'research/model.joblib'
META_PATH = 'research/model_metadata.json'

# Adjusted for 5-min timeframe
TARGET_PROFIT = 0.008  # 0.8% profit (smaller for 5min)
STOP_LOSS = 0.005      # 0.5% stop
HORIZON = 36           # 36 candles = 3 hours lookahead

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

def apply_triple_barrier_labeling(df, target, stop, horizon):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    y = np.zeros(n)
    
    for i in range(n - horizon):
        entry = closes[i]
        upper = entry * (1 + target)
        lower = entry * (1 - stop)
        
        for k in range(1, horizon + 1):
            if lows[i+k] < lower:
                y[i] = 0
                break
            if highs[i+k] > upper:
                y[i] = 1
                break
    
    return pd.Series(y, index=df.index)

def prepare_features(df):
    df = df.copy()
    
    # Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # RSI
    df['rsi'] = compute_rsi(df['close'], 14)
    df['rsi_ma'] = df['rsi'].rolling(10).mean()
    
    # ATR
    df['atr'] = compute_atr(df, 14)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['rel_vol'] = df['volume'] / df['vol_ma']
    
    # Time
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    df['dow'] = pd.to_datetime(df['open_time']).dt.dayofweek
    
    # Lagged returns
    for i in [1, 3, 6, 12]:
        df[f'ret_lag_{i}'] = df['returns'].shift(i)
    
    df.dropna(inplace=True)
    return df

def main():
    print("=" * 60)
    print("XGBoost Training on 5-Minute Data (320K samples)")
    print("=" * 60)
    
    print("\n[1/5] Loading 5-min data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} candles")
    
    print("\n[2/5] Preparing features...")
    df = prepare_features(df)
    
    feature_cols = ['rsi', 'rsi_ma', 'atr', 'atr_pct', 'vol_ma', 'rel_vol', 
                    'hour', 'dow', 'ret_lag_1', 'ret_lag_3', 'ret_lag_6', 'ret_lag_12']
    
    print(f"  Features: {feature_cols}")
    
    print("\n[3/5] Labeling...")
    df['label'] = apply_triple_barrier_labeling(df, TARGET_PROFIT, STOP_LOSS, HORIZON)
    df = df.iloc[:-HORIZON].copy()
    
    buy_ratio = df['label'].mean()
    print(f"  Class balance: {buy_ratio*100:.1f}% buys")
    
    print("\n[4/5] Training XGBoost...")
    
    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train = df.iloc[:split_idx][feature_cols]
    y_train = df.iloc[:split_idx]['label']
    X_test = df.iloc[split_idx:][feature_cols]
    y_test = df.iloc[split_idx:]['label']
    
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=(1 - buy_ratio) / buy_ratio,
        early_stopping_rounds=30,
        eval_metric='auc',
        verbosity=0
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")
    
    print("\n[5/5] Evaluating...")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Find best threshold
    best_thresh, best_precision = 0.5, 0
    for t in np.arange(0.5, 0.9, 0.05):
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() > 0:
            prec = precision_score(y_test, y_pred)
            if prec > best_precision:
                best_precision = prec
                best_thresh = t
    
    y_pred = (y_proba >= best_thresh).astype(int)
    
    print(f"\n  Optimal Threshold: {best_thresh:.2f}")
    print(f"  Test AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
    
    # Profit simulation
    wins = ((y_pred == 1) & (y_test == 1)).sum()
    losses = ((y_pred == 1) & (y_test == 0)).sum()
    total = wins + losses
    
    if total > 0:
        win_rate = wins / total
        expected_pnl = win_rate * TARGET_PROFIT - (1 - win_rate) * STOP_LOSS
        breakeven = STOP_LOSS / (TARGET_PROFIT + STOP_LOSS)
        
        print(f"\n  Trades: {total:,}")
        print(f"  Win Rate: {win_rate*100:.1f}% (need {breakeven*100:.1f}% to break even)")
        print(f"  Expected PnL/trade: {expected_pnl*100:+.3f}%")
        
        if expected_pnl > 0:
            print("  ✅ PROFITABLE!")
        else:
            print("  ⚠️ Not profitable yet")
    
    # Save
    joblib.dump(model, MODEL_PATH)
    
    metadata = {
        'features': feature_cols,
        'threshold': best_thresh,
        'target_profit': TARGET_PROFIT,
        'stop_loss': STOP_LOSS,
        'horizon': HORIZON,
        'model_type': 'XGBoost-5min',
        'timeframe': '5m'
    }
    
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
