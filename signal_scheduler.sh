#!/bin/bash

# Scheduled Signal Generator for Stocky Trading System
# Generates realistic trading signals every 30 minutes during market hours

LOG_FILE="/tmp/stocky_signals.log"
DEMO_SCRIPT="/Users/destucr/Desktop/Stocky/demo_signals.py"

echo "$(date): Starting Stocky Signal Generator" >> "$LOG_FILE"

# Function to generate signals
generate_signals() {
    echo "$(date): Generating new trading signals..." >> "$LOG_FILE"
    cd /Users/destucr/Desktop/Stocky
    python "$DEMO_SCRIPT" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "$(date): ✅ Signals generated successfully" >> "$LOG_FILE"
    else
        echo "$(date): ❌ Signal generation failed" >> "$LOG_FILE"
    fi
}

# Run initial signal generation
generate_signals

# Set up continuous generation (every 30 minutes)
while true; do
    sleep 1800  # 30 minutes
    generate_signals
done