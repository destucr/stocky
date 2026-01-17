#!/usr/bin/env python3
"""
Demo script to showcase the enhanced real-time trading signals
"""

import json
import requests
import time
from datetime import datetime, timezone
import random

# Backend URL
BACKEND_URL = 'http://localhost:8080'

def simulate_signal_generation():
    """
    Simulate the enhanced AI agent generating realistic BUY/SELL/HOLD signals
    """
    symbols = ['BINANCE:BTCUSDT', 'BINANCE:ETHUSDT', 'BINANCE:SOLUSDT']
    
    for i, symbol in enumerate(symbols):
        # Simulate different signal types
        if i == 0:  # BUY signal
            signal_type = "BUY"
            probability = 0.78  # High confidence BUY
            price = 97500.0
            target = price * 1.015  # 1.5% target
            stop = price * 0.99     # 1% stop
            confidence = "HIGH"
        elif i == 1:  # SELL signal
            signal_type = "SELL" 
            probability = 0.23  # High confidence SELL
            price = 3420.0
            target = price * 0.985  # 1.5% target (lower for short)
            stop = price * 1.01     # 1% stop (higher for short)
            confidence = "HIGH"
        else:  # HOLD signal
            signal_type = "HOLD"
            probability = 0.52  # Neutral probability
            price = 220.0
            target = 0.0
            stop = 0.0
            confidence = "LOW"
        
        # Create enhanced signal payload
        payload = {
            "symbol": symbol,
            "type": signal_type,
            "price": price,
            "target": target,
            "stop": stop,
            "probability": probability,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interval": "1h",
            "analysis_type": "real_time"
        }
        
        print(f"🚀 Sending {signal_type} signal for {symbol.replace('BINANCE:', '')}:")
        print(f"   💰 Price: ${price:,.2f}")
        print(f"   🎯 Probability: {probability:.3f} ({confidence} confidence)")
        if signal_type != "HOLD":
            print(f"   📈 Target: ${target:,.2f}")
            print(f"   🛑 Stop: ${stop:,.2f}")
        print()
        
        # Send to backend
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/internal/broadcast_signal",
                json=payload,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ Signal sent successfully for {symbol}")
            else:
                print(f"❌ Failed to send signal: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending signal: {e}")
        
        time.sleep(2)  # Small delay between signals

def main():
    print("🎯 Real-Time Trading Signal Demonstration")
    print("=" * 50)
    print("📊 Enhanced AI Agent with BUY/SELL/HOLD Recommendations")
    print("⏱️  Optimized for 1-hour interval trading")
    print("🔄 Real-time confidence levels and multi-symbol analysis")
    print()
    
    # Check if backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/api/prices", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and accessible")
        else:
            print("⚠️  Backend responded but may have issues")
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        print("Make sure the backend is running on localhost:8080")
        return
    
    print("\n🎬 Starting signal demonstration...")
    print("Open your frontend to see the enhanced SignalWidget in action!")
    print("URL: http://localhost:3000")
    print()
    
    # Generate demo signals
    simulate_signal_generation()
    
    print("✨ Demo completed! Check your frontend for the enhanced signals.")
    print("The new features include:")
    print("  • 🟢 BUY signals with bullish momentum")
    print("  • 🔴 SELL signals with bearish momentum") 
    print("  • ⏸️  HOLD signals when market is uncertain")
    print("  • 📊 Confidence levels (HIGH/MEDIUM/LOW)")
    print("  • 📈 Real-time probability display")
    print("  • ⏱️  1-hour trading optimized")

if __name__ == "__main__":
    main()