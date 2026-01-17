import pandas as pd
import numpy as np

DATA_PATH = 'research/data/btc_usdt_5m.csv'
FEE = 0.0006
SLIPPAGE = 0.0001
TOTAL_COST = FEE + SLIPPAGE # Per side

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    df = pd.read_csv(DATA_PATH)
    df['rsi'] = compute_rsi(df['close'])
    
    # Simple Strategy:
    # Buy if RSI < 30
    # Sell (Exit) if RSI > 50 or Stop Loss (-1%) or Take Profit (+1%)
    # This is a "Long Only" test for mean reversion
    
    in_position = False
    entry_price = 0
    trades = []
    
    closes = df['close'].values
    rsis = df['rsi'].values
    
    for i in range(15, len(df)):
        price = closes[i]
        rsi = rsis[i]
        
        if not in_position:
            if rsi < 30:
                in_position = True
                entry_price = price
        else:
            # Exit Logic
            # 1. RSI Reversion
            if rsi > 50:
                pnl = (price - entry_price) / entry_price - (2 * TOTAL_COST)
                trades.append(pnl)
                in_position = False
            # 2. Stop Loss (Fixed 1%)
            elif price < entry_price * 0.99:
                pnl = (price - entry_price) / entry_price - (2 * TOTAL_COST)
                trades.append(pnl)
                in_position = False
    
    trades = np.array(trades)
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {(trades > 0).mean():.2%}")
    print(f"Avg PnL: {trades.mean():.4%}")
    print(f"Total PnL: {trades.sum():.4f}")
    
    if trades.mean() > 0:
        print("DECISION: Mean Reversion has potential.")
    else:
        print("DECISION: Simple Mean Reversion fails after fees.")

if __name__ == "__main__":
    main()
