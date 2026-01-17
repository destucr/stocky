#!/usr/bin/env python3

import time
import json
import pandas as pd
import numpy as np
import requests
import os
import psycopg2
from datetime import datetime, timezone
import threading

# DB Config
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASSWORD', 'password')
DB_NAME = os.getenv('DB_NAME', 'stocky')

# Backend Config
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8080')

# Trading symbols
TRADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def broadcast_signal(payload):
    try:
        url = f"{BACKEND_URL}/api/internal/broadcast_signal"
        resp = requests.post(url, json=payload, timeout=2)
        if resp.status_code == 200:
            print("✅ Signal broadcasted successfully.")
        else:
            print(f"❌ Broadcast failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Broadcast error: {e}")

def save_signal(conn, symbol, sig_type, price, target, stop, prob, timestamp, confidence="medium"):
    if not conn:
        return
    try:
        cur = conn.cursor()
        query = """
            INSERT INTO signals (symbol, signal_type, price, target, stop, probability, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (symbol, sig_type, price, target, stop, prob, timestamp))
        conn.commit()
        cur.close()
        print(f"💾 Saved Signal: {sig_type} @ ${price:.2f} (Prob: {prob:.3f}, Confidence: {confidence})")
        
        # Enhanced broadcast with more details
        payload = {
            "symbol": symbol,
            "type": sig_type,
            "price": price,
            "target": target,
            "stop": stop,
            "probability": prob,
            "confidence": confidence,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            "interval": "1h",
            "analysis_type": "demo_mode"
        }
        broadcast_signal(payload)
        
    except Exception as e:
        print(f"❌ Failed to save signal: {e}")
        if conn:
            conn.rollback()

def get_symbol_mapping(symbol):
    """Map Binance symbols to internal database symbols"""
    mapping = {
        'BTCUSDT': 'BINANCE:BTCUSDT',
        'ETHUSDT': 'BINANCE:ETHUSDT', 
        'SOLUSDT': 'BINANCE:SOLUSDT',
        'BNBUSDT': 'BINANCE:BNBUSDT',
        'ADAUSDT': 'BINANCE:ADAUSDT',
        'XRPUSDT': 'BINANCE:XRPUSDT'
    }
    return mapping.get(symbol, f'BINANCE:{symbol}')

def generate_demo_signals():
    """Generate realistic demo trading signals"""
    print("🚀 Starting Demo Signal Generator")
    print(f"📊 Symbols: {', '.join(TRADING_SYMBOLS)}")
    print("🎭 Mode: DEMO (realistic mock signals)\n")
    
    # Base prices for realistic signals
    base_prices = {
        'BTCUSDT': 42000,
        'ETHUSDT': 2500,
        'SOLUSDT': 90,
        'BNBUSDT': 280,
        'ADAUSDT': 0.45,
        'XRPUSDT': 0.55
    }
    
    conn = get_db_connection()
    signal_count = 0
    
    while True:
        try:
            print(f"\n🔍 [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Generating demo signals...")
            
            for symbol in TRADING_SYMBOLS:
                # Generate realistic price variation
                base_price = base_prices[symbol]
                price_variation = 1 + (np.random.random() - 0.5) * 0.1  # ±5% variation
                current_price = base_price * price_variation
                
                # Generate signal probabilities with realistic patterns
                # Create some clustering around signal thresholds for realism
                rand = np.random.random()
                if rand < 0.3:  # 30% BUY signals
                    prob = 0.7 + np.random.random() * 0.25  # 0.7-0.95
                    sig_type = "BUY"
                    confidence = "HIGH" if prob > 0.8 else "MEDIUM"
                    target = current_price * 1.015  # 1.5% target
                    stop = current_price * 0.99     # 1% stop
                    
                elif rand < 0.5:  # 20% SELL signals  
                    prob = 0.05 + np.random.random() * 0.25  # 0.05-0.30
                    sig_type = "SELL"
                    confidence = "HIGH" if prob < 0.2 else "MEDIUM"
                    target = current_price * 0.985  # 1.5% target down
                    stop = current_price * 1.01     # 1% stop up
                    
                else:  # 50% HOLD signals
                    prob = 0.3 + np.random.random() * 0.4  # 0.3-0.7
                    sig_type = "HOLD"
                    confidence = "LOW"
                    target = 0.0
                    stop = 0.0
                
                # Generate fresh timestamp
                signal_timestamp = datetime.now(timezone.utc)
                
                # Map symbol for database
                db_symbol = get_symbol_mapping(symbol)
                
                # Save and broadcast
                save_signal(conn, db_symbol, sig_type, current_price, target, stop, prob, signal_timestamp, confidence)
                signal_count += 1
                
                # Add small delay between symbols
                time.sleep(0.5)
            
            print(f"✅ Generated {len(TRADING_SYMBOLS)} signals (Total: {signal_count})")
            print("💤 Sleeping for 60 seconds until next batch...")
            time.sleep(60)  # Generate new signals every minute
            
        except KeyboardInterrupt:
            print("\n👋 Demo signal generator stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in demo generator: {e}")
            time.sleep(10)  # Brief pause on error
    
    if conn:
        conn.close()

if __name__ == "__main__":
    generate_demo_signals()