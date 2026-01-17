import psycopg2
import os
import requests
import json
from datetime import datetime, timezone

# DB Config (matching docker-compose external port)
DB_HOST = 'localhost'
DB_PORT = '5433' 
DB_USER = 'postgres'
DB_PASS = 'password'
DB_NAME = 'stocky'

def main():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME
        )
        cur = conn.cursor()
        
        # Insert a fake BUY signal for BTCUSDT
        timestamp = datetime.now(timezone.utc)
        symbol = 'BINANCE:BTCUSDT'
        price = 96500.0
        
        query = """
            INSERT INTO signals (symbol, signal_type, price, target, stop, probability, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (symbol, 'BUY', price, 98000.0, 95500.0, 0.92, timestamp))
        conn.commit()
        print("Test signal injected into DB.")
        cur.close()
        conn.close()
        
        # Now Broadcast via HTTP to Backend
        payload = {
            "symbol": symbol,
            "type": "BUY",
            "price": price,
            "target": 98000.0,
            "stop": 95500.0,
            "probability": 0.92,
            "timestamp": timestamp.isoformat()
        }
        
        print("Broadcasting to WebSocket...")
        resp = requests.post('http://localhost:8080/api/internal/broadcast_signal', json=payload)
        if resp.status_code == 200:
            print("Broadcast Success!")
        else:
            print(f"Broadcast Failed: {resp.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()