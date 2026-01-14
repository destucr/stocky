# Stocky

Stocky is a real-time stock and cryptocurrency tracking application featuring live data visualization with interactive charts. It utilizes a Go backend to fetch data from Finnhub via WebSockets and stores it in PostgreSQL, while the React frontend provides a smooth and responsive user experience using `lightweight-charts`.

![Stocky Screenshot](assets/screenshot.png)

## Features

- **Real-time Data:** Live price updates via WebSockets.
- **Interactive Charts:** High-performance Candlestick and Line charts using `lightweight-charts`.
- **OHLC History:** Historical data aggregation and visualization across various timeframes (1s to 1d).
- **Technical Indicators:** Built-in Moving Averages (MA7, MA25, MA99).
- **Responsive Design:** Optimized for single-screen viewing with a clean, professional low-light UI.
- **Symbol Metadata:** Automatic fetching of logos and names for supported assets.

## Tech Stack

### Backend
- **Go (Golang)**
- **PostgreSQL** (with `pgx` for high-performance interaction)
- **Finnhub API** (Real-time trade data)
- **Gorilla WebSocket** (Bi-directional communication)

### Frontend
- **React** (TypeScript)
- **Vite** (Build tool)
- **Material UI (MUI)** (Component library)
- **Lightweight Charts** (Performance-focused charting library)

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
   - Navigate to the `backend` folder.
   - Create a `.env` file with your `DB_CONN_STRING` and `FINNHUB_API_KEY`.
   - Install dependencies and run:
     ```bash
     go mod download
     go run main.go
     ```

3. **Frontend Setup:**
   - Navigate to `frontend/web`.
   - Install dependencies and start the development server:
     ```bash
     npm install
     npm run dev
     ```

## Project Structure

```
Stocky/
├── backend/             # Go source code
│   ├── internal/        # Core logic (store, websocket, finnhub)
│   └── main.go          # Entry point
└── frontend/
    └── web/             # React source code
        ├── src/
        │   ├── App.tsx  # Main application logic
        │   └── theme.ts # Custom MUI theme
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0.
