import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Configuration
DATA_PATH = 'research/data/btc_usdt_1h.csv'
FEE = 0.0006  # 0.06% Taker fee
SLIPPAGE = 0.0001 # 0.01% Slippage assumption (conservative)
TOTAL_COST = FEE + SLIPPAGE # Per side

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Features
    for i in range(1, 6):
        df[f'ret_lag_{i}'] = df['returns'].shift(i)
        
    df['rsi'] = compute_rsi(df['close'])
    df['volatility'] = df['returns'].rolling(20).std()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['rel_vol'] = df['volume'] / df['vol_ma']
    
    # Target: Next period return sign
    # 1 if return > 0, 0 otherwise
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    df.dropna(inplace=True)
    return df

def run_backtest(df, predictions, probabilities):
    # Vectorized backtest
    # Strategy: 
    # Long if Prob(1) > 0.55
    # Short if Prob(1) < 0.45
    # Cash otherwise
    
    signals = pd.Series(0, index=df.index)
    signals[probabilities > 0.55] = 1
    signals[probabilities < 0.45] = -1
    
    # Calculate PnL
    # Return of the asset * position of previous candle
    # We enter at Open of t+1 (which is approx Close of t), so we capture Return of t+1
    asset_returns = df['returns'].shift(-1) # This aligns signals[t] with return[t+1]
    
    gross_pnl = signals * asset_returns
    
    # Cost calculation: 
    # We pay fees when we CHANGE position.
    # Current Position - Previous Position != 0
    trades = signals.diff().abs()
    costs = trades * TOTAL_COST
    
    net_pnl = gross_pnl - costs
    
    return net_pnl, signals

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)
    
    # Split data: First 70% Train, Next 30% Test (Simple split for baseline)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = [c for c in df.columns if c not in ['open_time', 'close_time', 'target', 'ignore', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'returns']]
    
    print(f"Features used: {feature_cols}")
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print("Training Logistic Regression...")
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    
    # Backtest
    net_returns, signals = run_backtest(test_df, preds, probs)
    
    total_return = net_returns.sum()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(288 * 365) # Annualized
    
    print("\n--- Backtest Results (Fee Aware) ---")
    print(f"Total Net Log Return: {total_return:.4f}")
    print(f"Annualized Sharpe: {sharpe:.4f}")
    print(f"Trade Count: {signals.diff().abs().sum()}")
    print(f"Win Rate (Gross): {len(net_returns[net_returns > 0]) / len(net_returns[net_returns != 0]):.4f}")
    
    # Comparison
    buy_hold_return = test_df['returns'].sum()
    print(f"Buy & Hold Return: {buy_hold_return:.4f}")

    if total_return > 0 and total_return > buy_hold_return:
        print("\nDECISION: Strategy shows promise (Baseline beats B&H).")
    elif total_return > 0:
        print("\nDECISION: Strategy profitable but underperforms B&H.")
    else:
        print("\nDECISION: Strategy unprofitable after fees.")

if __name__ == "__main__":
    main()
