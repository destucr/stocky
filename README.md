# Stocky

Stocky is a high-performance, real-time stock and cryptocurrency tracking dashboard. Built with a focus on ultra-low latency and long-term data persistence, it features a Go backend capable of processing high-frequency trade data and a React frontend that bypasses standard rendering bottlenecks for an "instant" feel.

![Stocky Screenshot](assets/screenshot.png)

## Key Features

- **Ultra-Low Latency:** Optimized data path using zero-copy broadcasting, `TCP_NODELAY`, and WebSocket write-batching.
- **"Binance-Speed" UI:** Bypasses React's Virtual DOM for high-frequency legend updates using direct DOM manipulation.
- **Multi-Chart Visualization:** Switch instantly between **Candlestick**, **Line**, and **Mountain (Area)** views.
- **Infinite Persistence:** Dual-table archiving system that stores raw trades for 24h and permanently archives 1-minute OHLC candles.
- **Gapless History:** Intelligent API that merges archived records with live aggregated trades via SQL UNION for a seamless chart.
- **Smart Logic:** High-performance rolling Moving Averages (MA7, MA25, MA99) and tick-based color coding (Buy/Sell detection).
- **SVG Logos:** High-resolution vector logos sourced from TradingView and Coingecko with graceful error fallbacks.

## Performance Stack

### Backend (Go)
- **Zero-Copy Broadcast:** Raw bytes from Finnhub are sent directly to clients, eliminating JSON overhead in the hot path.
- **DB Worker Pool:** Dedicated goroutine pool for non-blocking database persistence.
- **Optimized Networking:** Nagle's algorithm disabled (`TCP_NODELAY`) for immediate packet transmission.
- **Advanced SQL:** Deterministic OHLC aggregation with bucket-level deduplication.

### Frontend (React + TS)
- **Direct DOM Access:** Uses `Refs` and `useImperativeHandle` to update prices without triggering full component re-renders.
- **Frame-Level Batching:** Aggregates multiple micro-trades per WebSocket frame into a single chart redraw.
- **Logical Range Sync:** Perfectly synchronized Price and Volume charts that maintain horizontal buffers during scrolling.

## Tech Stack

- **Language:** Go 1.24+, TypeScript
- **Database:** PostgreSQL (with `pgx` pool)
- **Real-time:** Gorilla WebSocket
- **Charting:** Lightweight Charts (TradingView)
- **UI:** Material UI (MUI)

## Getting Started

### Prerequisites
- Go 1.24+
- Node.js & npm
- PostgreSQL

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/destucr/stocky.git
   cd stocky
   ```

2. **Backend Setup:**
   - Create a `.env` in `backend/` with `DB_CONN_STRING` and `FINNHUB_API_KEY`.
   - Run: `go run main.go`

3. **Frontend Setup:**
   - In `frontend/web/`: `npm install` then `npm run dev`

## License

This project is licensed under the Apache License 2.0.