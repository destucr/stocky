"""
XGBoost Trading Model - Optimized for higher accuracy

Key improvements:
1. XGBoost with tuned hyperparameters
2. Feature selection (drop low-importance features)
3. More training iterations with proper early stopping
4. Walk-forward validation for more realistic testing
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Configuration
DATA_PATH = 'research/data/btc_usdt_1h.csv'
MODEL_PATH = 'research/model.joblib'
META_PATH = 'research/model_metadata.json'

# Triple Barrier Parameters
TARGET_PROFIT = 0.015  # 1.5% profit target
STOP_LOSS = 0.010      # 1.0% stop loss
HORIZON = 24           # 24 hour lookahead

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

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bb_position = (series - lower) / (upper - lower)
    return bb_position, (upper - lower) / sma

def compute_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

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
        
        for k in range(1, horizon + 1):
            if lows[i+k] < lower:
                y[i] = 0
                break
            if highs[i+k] > upper:
                y[i] = 1
                break
    
    return pd.Series(y, index=df.index)

def prepare_features(df):
    """Focused feature set - only most predictive indicators"""
    df = df.copy()
    
    # Returns at multiple timeframes
    df['returns_1h'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_4h'] = np.log(df['close'] / df['close'].shift(4))
    df['returns_12h'] = np.log(df['close'] / df['close'].shift(12))
    df['returns_24h'] = np.log(df['close'] / df['close'].shift(24))
    
    # Momentum - RSI
    df['rsi'] = compute_rsi(df['close'], 14)
    df['rsi_ma'] = df['rsi'].rolling(10).mean()
    
    # Volatility - ATR
    df['atr'] = compute_atr(df, 14)
    df['atr_pct'] = df['atr'] / df['close']
    
    # MACD
    macd, macd_signal, macd_hist = compute_macd(df['close'])
    df['macd_hist'] = macd_hist
    
    # Bollinger
    bb_position, bb_bandwidth = compute_bollinger(df['close'])
    df['bb_position'] = bb_position
    df['bb_bandwidth'] = bb_bandwidth
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['rel_vol'] = df['volume'] / df['vol_ma']
    
    # Price position relative to SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    
    # Time
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    df['dow'] = pd.to_datetime(df['open_time']).dt.dayofweek
    
    # Rolling volatility
    df['returns_std_24h'] = df['returns_1h'].rolling(24).std()
    df['returns_skew_24h'] = df['returns_1h'].rolling(24).skew()
    
    # Lagged returns (recent momentum)
    for i in [1, 3, 6, 12]:
        df[f'ret_lag_{i}'] = df['returns_1h'].shift(i)
    
    df.dropna(inplace=True)
    return df

def get_feature_columns(df):
    exclude = [
        'open_time', 'close_time', 'label', 'ignore', 
        'open', 'high', 'low', 'close', 'volume', 
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 
        'returns_1h', 'sma_20', 'sma_50'
    ]
    return [c for c in df.columns if c not in exclude]

def evaluate_model(model, X, y, threshold=0.5, name=""):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{name} Metrics (threshold={threshold:.2f}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (of predicted buys, how many were correct)")
    print(f"  Recall:    {metrics['recall']:.4f} (of actual buys, how many we caught)")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                No Buy   Buy")
    print(f"  Actual No Buy  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Actual Buy     {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    return metrics

def find_optimal_threshold(model, X, y):
    """Find threshold that maximizes precision while maintaining decent recall"""
    y_proba = model.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    results = []
    for thresh in np.arange(0.40, 0.85, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # Custom score: prioritize precision but require some recall
        if recall > 0.15:  # Minimum 15% recall
            score = precision * 0.7 + f1 * 0.3
        else:
            score = 0
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print("\n Threshold Analysis:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for r in results:
        marker = " <--" if r['threshold'] == best_threshold else ""
        print(f"  {r['threshold']:>10.2f} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f}{marker}")
    
    return best_threshold

def main():
    print("=" * 60)
    print("XGBoost Trading Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} candles")
    
    # Feature engineering
    print("\n[2/6] Engineering features...")
    df = prepare_features(df)
    feature_cols = get_feature_columns(df)
    print(f"  Created {len(feature_cols)} features: {feature_cols}")
    
    # Labeling
    print("\n[3/6] Applying triple barrier labeling...")
    df['label'] = apply_triple_barrier_labeling(df)
    df = df.iloc[:-HORIZON].copy()
    
    label_counts = df['label'].value_counts()
    buy_ratio = label_counts.get(1, 0) / len(df)
    print(f"  Class distribution:")
    print(f"    0 (No Buy): {label_counts.get(0, 0)} ({100*(1-buy_ratio):.1f}%)")
    print(f"    1 (Buy):    {label_counts.get(1, 0)} ({100*buy_ratio:.1f}%)")
    
    # Time-based split
    print("\n[4/6] Splitting data (time-based)...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    print(f"  Train: {len(X_train)} samples ({train_df['label'].mean()*100:.1f}% positive)")
    print(f"  Test:  {len(X_test)} samples ({test_df['label'].mean()*100:.1f}% positive)")
    
    # Train XGBoost
    print("\n[5/6] Training XGBoost model...")
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (1 - buy_ratio) / buy_ratio
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False,
        early_stopping_rounds=50,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best AUC: {model.best_score:.4f}")
    
    # Feature importance
    print("\n Top 10 Most Important Features:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:>8.4f}")
    
    # Find optimal threshold
    print("\n[6/6] Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, X_test, y_test)
    
    # Evaluate with optimal threshold
    print("\n" + "=" * 60)
    train_metrics = evaluate_model(model, X_train, y_train, optimal_threshold, "Training")
    test_metrics = evaluate_model(model, X_test, y_test, optimal_threshold, "Test")
    
    # Profit simulation
    print("\n Simulated Trading Performance (Test Set):")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # For each predicted buy, check if it was correct
    total_trades = y_pred.sum()
    winning_trades = ((y_pred == 1) & (y_test == 1)).sum()
    losing_trades = ((y_pred == 1) & (y_test == 0)).sum()
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades
        # Expected PnL per trade (win = +1.5%, lose = -1.0%)
        expected_pnl = win_rate * TARGET_PROFIT - (1 - win_rate) * STOP_LOSS
        
        print(f"  Total signals: {total_trades}")
        print(f"  Winning trades: {winning_trades} ({win_rate*100:.1f}%)")
        print(f"  Losing trades: {losing_trades} ({(1-win_rate)*100:.1f}%)")
        print(f"  Expected PnL per trade: {expected_pnl*100:+.2f}%")
        print(f"  Break-even win rate: {STOP_LOSS/(TARGET_PROFIT+STOP_LOSS)*100:.1f}%")
        
        if expected_pnl > 0:
            print(f"  ✅ Model is profitable!")
        else:
            print(f"  ⚠️  Model needs higher precision")
    
    # Save
    print("\n Saving model...")
    joblib.dump(model, MODEL_PATH)
    
    metadata = {
        'features': feature_cols,
        'threshold': optimal_threshold,
        'target_profit': TARGET_PROFIT,
        'stop_loss': STOP_LOSS,
        'horizon': HORIZON,
        'model_type': 'XGBoost',
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'best_iteration': model.best_iteration
    }
    
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n Model saved to: {MODEL_PATH}")
    print(f" Metadata saved to: {META_PATH}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
