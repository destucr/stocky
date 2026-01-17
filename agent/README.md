# Stocky Trading Agent (Paper Trading)

This agent runs the "BTC 1H Gradient Boosting" strategy developed in the research phase.

## Strategy Profile
- **Asset:** BTC/USDT
- **Timeframe:** 1 Hour
- **Logic:** Histogram Gradient Boosting on RSI, ATR, and Volume features.
- **Entry:** Probability > 0.7
- **Exit:** +1.5% Target or -1.0% Stop Loss (Fixed Bracket).
- **Hold Time:** Max 24 hours.

## Setup

1. **Prerequisites:**
   Ensure the research virtual environment is set up:
   ```bash
   cd research
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt # (pandas, numpy, scikit-learn, requests)
   ```

2. **Training (If needed):**
   If `research/model.joblib` is missing, retrain the model:
   ```bash
   research/.venv/bin/python research/train_production_model.py
   ```

## Usage

**Run One-Off Check (Current Signal):**
```bash
research/.venv/bin/python agent/run.py --once
```

**Run Continuous Loop:**
```bash
research/.venv/bin/python agent/run.py
```
*Note: The agent polls every minute. It will output a signal whenever a new 1-hour candle closes.*
