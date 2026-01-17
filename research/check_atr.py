import pandas as pd
import numpy as np

df = pd.read_csv('research/data/btc_usdt_1h.csv')
high_low = df['high'] - df['low']
high_close = (df['high'] - df['close'].shift()).abs()
low_close = (df['low'] - df['close'].shift()).abs()
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = ranges.max(axis=1)
atr = true_range.rolling(24).mean()
atr_pct = atr / df['close']

print(f"Median ATR % (24h): {atr_pct.median():.4%}")
print(f"Mean ATR % (24h): {atr_pct.mean():.4%}")
