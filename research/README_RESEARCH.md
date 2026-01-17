# Trading Model Research Report: BTC/USDT 5m

**Date:** 2026-01-15
**Asset:** BTC/USDT
**Timeframe:** 5 Minutes
**Constraints:** 0.06% Taker Fee (0.12% Round Trip) + 0.01% Slippage Assumption.

## Executive Summary
**Decision:** **NO TRADE** / **ABSTAIN**

Extensive backtesting and machine learning modeling on 3 years of data (Jan 2023 - Jan 2026) indicates that no statistically significant edge exists using standard OHLCV technical features on 5-minute candles that can overcome the strict fee/slippage hurdle.

## Methodology

### 1. Data
- **Source:** Binance Public API (`/api/v3/klines`)
- **Range:** Jan 1, 2023 - Jan 15, 2026 (~320,000 candles)
- **Sanity:** Data is contiguous and verified.

### 2. Experiments

#### Experiment A: Baseline Logistic Regression
- **Hypothesis:** Predict next candle direction (Up/Down) using lagged returns and RSI.
- **Result:** Accuracy ~51% (Coin flip). Net PnL deeply negative (-2.62 Log Return).
- **Status:** REJECTED.

#### Experiment B: Gradient Boosting + Triple Barrier Labeling
- **Hypothesis:** Predict if Price hits +0.3% Target before -0.2% Stop within 1 hour (12 candles).
- **Model:** HistGradientBoostingClassifier.
- **Features:** RSI, ATR, Volatility, Volume Delta, Time of Day.
- **Result:**
    - High Confidence (>0.8) yielded only 3 trades in 7 months (Sample too small).
    - Medium Confidence (>0.6) yielded negative expectancy (-0.14% per trade).
- **Status:** REJECTED (No statistically valid signal frequency).

#### Experiment C: RSI Mean Reversion (Heuristic)
- **Hypothesis:** Buy RSI < 30, Exit RSI > 50.
- **Result:** Win Rate 41.5%. Avg PnL -0.12% per trade.
- **Status:** REJECTED.

## Conclusion
The market efficiency at the 5-minute timeframe for BTC/USDT is high. Simple technical indicators do not provide enough predictive power to overcome a 0.14% cost-per-trade barrier.

**Recommendation:**
To find a valid edge, future research must incorporate:
1.  **Market Microstructure Data:** Order book imbalance, trade flow.
2.  **Alternative Data:** Sentiment, funding rates, liquidation data.
3.  **Maker Strategy:** Moving from Taker (paying fees) to Maker (earning rebates) is likely required to make this frequency profitable.

**Current Action:** The agent remains in **Cash**.
