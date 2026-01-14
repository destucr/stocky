# Stocky

Real-time stock and crypto tracking dashboard. Go backend with zero-copy WebSocket broadcasting, React frontend with direct DOM manipulation for high-frequency updates.

![Stocky Screenshot](assets/screenshot.png)

## Features

WebSocket streaming from Finnhub with PostgreSQL persistence. Frontend uses direct DOM updates for price changes. Dual storage: 24h raw trades, permanent 1min candles. Candlestick, line, and area charts with MA7/25/99.

## Stack

![Go](https://img.shields.io/badge/Go-1.24+-00ADD8?style=flat&logo=go&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)
![Material UI](https://img.shields.io/badge/Material_UI-0081CB?style=flat&logo=mui&logoColor=white)

## Setup

**Prerequisites:** Go 1.24+, Node.js, PostgreSQL

**1. Clone the repository**
```bash
git clone https://github.com/destucr/stocky.git
cd stocky
```

**2. Configure backend**

Create `backend/.env`:
```
DB_CONN_STRING=postgresql://user:password@localhost:5432/stocky
FINNHUB_API_KEY=your_api_key
```

Get a free API key from [Finnhub](https://finnhub.io/register).

**3. Initialize database**
```bash
psql -U postgres -c "CREATE DATABASE stocky;"
# Run migrations if available
```

**4. Start backend**
```bash
cd backend
go run main.go
```

**5. Start frontend** (new terminal)
```bash
cd frontend/web
npm install
npm run dev
```

Open browser to the URL shown by Vite (typically `localhost:5173`).

## License

Apache License 2.0
