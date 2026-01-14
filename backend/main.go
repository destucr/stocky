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

	// Initialize Router
	mux := http.NewServeMux()

	// WebSocket endpoint
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		websocket.ServeWs(hub, w, r)
	})

	// API: Get latest prices
	mux.HandleFunc("/api/prices", func(w http.ResponseWriter, r *http.Request) {
		prices := memStore.GetLatestPrices()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(prices)
	})

			// API: Get metadata for symbols
			mux.HandleFunc("/api/metadata", func(w http.ResponseWriter, r *http.Request) {
				metadata := map[string]map[string]string{
					"BINANCE:BTCUSDT": {
						"name": "Bitcoin / Tether",
						"logo": "https://assets.coingecko.com/coins/images/1/small/bitcoin.png",
					},
					"BINANCE:ETHUSDT": {
						"name": "Ethereum / Tether",
						"logo": "https://assets.coingecko.com/coins/images/279/small/ethereum.png",
					},
					"BINANCE:SOLUSDT": {
						"name": "Solana / Tether",
						"logo": "https://assets.coingecko.com/coins/images/4128/small/solana.png",
					},
					"BINANCE:BNBUSDT": {
						"name": "BNB / Tether",
						"logo": "https://assets.coingecko.com/coins/images/825/small/bnb.png",
					},
					"BINANCE:ADAUSDT": {
						"name": "Cardano / Tether",
						"logo": "https://assets.coingecko.com/coins/images/975/small/cardano.png",
					},
					"BINANCE:XRPUSDT": {
						"name": "XRP / Tether",
						"logo": "https://assets.coingecko.com/coins/images/44/small/xrp.png",
					},
					"BINANCE:DOTUSDT": {
						"name": "Polkadot / Tether",
						"logo": "https://assets.coingecko.com/coins/images/12171/small/polkadot.png",
					},
					"AAPL": {
						"name": "Apple Inc.",
						"logo": "https://logo.clearbit.com/apple.com",
					},
					"MSFT": {
						"name": "Microsoft Corporation",
						"logo": "https://logo.clearbit.com/microsoft.com",
					},
					"GOOGL": {
						"name": "Alphabet Inc.",
						"logo": "https://logo.clearbit.com/google.com",
					},
					"AMZN": {
						"name": "Amazon.com, Inc.",
						"logo": "https://logo.clearbit.com/amazon.com",
					},
					"TSLA": {
						"name": "Tesla, Inc.",
						"logo": "https://logo.clearbit.com/tesla.com",
					},
					"NVDA": {
						"name": "NVIDIA Corporation",
						"logo": "https://logo.clearbit.com/nvidia.com",
					},
					"META": {
						"name": "Meta Platforms, Inc.",
						"logo": "https://logo.clearbit.com/meta.com",
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(metadata)
			})
	// API: Get history for a symbol
	mux.HandleFunc("/api/history", func(w http.ResponseWriter, r *http.Request) {
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
	})

	// Global CORS and Logging Middleware
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		mux.ServeHTTP(w, r)
	})

	slog.Info("Starting server", "port", cfg.Port)
	if err := http.ListenAndServe(":"+cfg.Port, handler); err != nil {
		slog.Error("Server failed to start", "error", err)
		os.Exit(1)
	}
}
