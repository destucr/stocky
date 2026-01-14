package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/destucr/stocky-backend/internal/config"
	"github.com/destucr/stocky-backend/internal/finnhub"
	"github.com/destucr/stocky-backend/internal/store"
	"github.com/destucr/stocky-backend/internal/websocket"
)

func main() {
	// Initialize structured logger with Debug level
	opts := &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, opts))
	slog.SetDefault(logger)

	cfg, err := config.LoadConfig()
	if err != nil {
		slog.Error("Failed to load configuration", "error", err)
		os.Exit(1)
	}

	// Initialize Postgres Store with retries
	var dbStore *store.PostgresStore
	for i := 0; i < 10; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		dbStore, err = store.NewPostgresStore(ctx, cfg.DBConnString)
		cancel()
		if err == nil {
			break
		}
		slog.Warn("Waiting for database to be ready...", "attempt", i+1, "error", err)
		time.Sleep(2 * time.Second)
	}

	if err != nil {
		slog.Error("Failed to initialize postgres store after retries", "error", err)
		os.Exit(1)
	}
	defer dbStore.Close()
	slog.Info("Postgres store initialized")

	// Initialize In-Memory Store (for latest prices)
	memStore := store.NewPriceStore(100)

	hub := websocket.NewHub()
	go hub.Run()

	fc := finnhub.NewFinnhubClient(hub, memStore, dbStore, cfg.FinnhubAPIKey)
	go fc.Connect()

	// WebSocket endpoint
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		websocket.ServeWs(hub, w, r)
	})

	// API Handlers with CORS
	apiHandler := func(h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
			w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
			
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			h(w, r)
		}
	}

	// API: Get latest prices
	http.HandleFunc("/api/prices", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		prices := memStore.GetLatestPrices()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(prices)
	}))

	// API: Get metadata for symbols
	http.HandleFunc("/api/metadata", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		// Basic metadata mapping. In a real app, this could be fetched from Finnhub or a cache.
		metadata := map[string]map[string]string{
			"BINANCE:BTCUSDT": {
				"name": "Bitcoin / Tether",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png",
			},
			"BINANCE:ETHUSDT": {
				"name": "Ethereum / Tether",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png", // Generic crypto for now or Eth logo if known
			},
			"BINANCE:SOLUSDT": {
				"name": "Solana / Tether",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png",
			},
			"OANDA:EUR_USD": {
				"name": "Euro / US Dollar",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png",
			},
			"OANDA:GBP_USD": {
				"name": "British Pound / US Dollar",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png",
			},
			"OANDA:USD_JPY": {
				"name": "US Dollar / Japanese Yen",
				"logo": "https://static.finnhub.io/logo/8746ad10-c033-11ea-8000-000000000000.png",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metadata)
	}))

	// API: Get history for a symbol
	http.HandleFunc("/api/history", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		symbol := r.URL.Query().Get("symbol")
		if symbol == "" {
			http.Error(w, "symbol query parameter is required", http.StatusBadRequest)
			return
		}

		intervalStr := r.URL.Query().Get("interval")
		limitStr := r.URL.Query().Get("limit")
		limit := 300
		if limitStr != "" {
			fmt.Sscanf(limitStr, "%d", &limit)
		}
		if limit <= 0 || limit > 2000 {
			limit = 500
		}

		var history interface{}
		var err error

		if intervalStr != "" {
			var interval int
			fmt.Sscanf(intervalStr, "%d", &interval)
			if interval <= 0 {
				interval = 60
			}
			history, err = dbStore.GetOHLCHistory(context.Background(), symbol, interval, limit)
		} else {
			history, err = dbStore.GetHistory(context.Background(), symbol, limit)
		}

		if err != nil {
			slog.Error("Failed to get history from DB", "error", err, "symbol", symbol)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(history)
	}))

	slog.Info("Starting server", "port", cfg.Port)
	if err := http.ListenAndServe(":"+cfg.Port, nil); err != nil {
		slog.Error("Server failed to start", "error", err)
		os.Exit(1)
	}
}
