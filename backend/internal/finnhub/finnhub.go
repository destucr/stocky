package finnhub

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	finnhub "github.com/Finnhub-Stock-API/finnhub-go/v2"
	"github.com/destucr/stocky-backend/internal/store"
	internalws "github.com/destucr/stocky-backend/internal/websocket"
	"github.com/gorilla/websocket"
)

type TradeData struct {
	Price  float64 `json:"p"`
	Symbol string  `json:"s"`
	Time   int64   `json:"t"`
	Volume float64 `json:"v"`
}

type FinnhubMessage struct {
	Type      string      `json:"type"`
	Data      []TradeData `json:"data"`
	Msg       string      `json:"msg"`
	FetchedAt int64       `json:"fetched_at"` // Added for latency tracking
}

type FinnhubClient struct {
	Hub      *internalws.Hub
	MemStore *store.PriceStore
	DBStore  *store.PostgresStore
	Token    string
	conn     *websocket.Conn
	sdk      *finnhub.DefaultApiService
}

func NewFinnhubClient(hub *internalws.Hub, mem *store.PriceStore, db *store.PostgresStore, token string) *FinnhubClient {
	cfg := finnhub.NewConfiguration()
	cfg.AddDefaultHeader("X-Finnhub-Token", token)
	sdkClient := finnhub.NewAPIClient(cfg).DefaultApi

	return &FinnhubClient{
		Hub:      hub,
		MemStore: mem,
		DBStore:  db,
		Token:    token,
		sdk:      sdkClient,
	}
}

func (fc *FinnhubClient) Connect() {
	url := fmt.Sprintf("wss://ws.finnhub.io?token=%s", fc.Token)

	for {
		slog.Info("Connecting to Finnhub WebSocket...")
		conn, _, err := websocket.DefaultDialer.Dial(url, nil)
		if err != nil {
			slog.Error("Failed to connect to Finnhub", "error", err)
			time.Sleep(5 * time.Second)
			continue
		}
		fc.conn = conn
		slog.Info("Connected to Finnhub WebSocket")

		symbols := fc.fetchAllAvailableSymbols()
		fc.subscribe(symbols)
		fc.listen()

		slog.Warn("Finnhub connection lost, retrying in 5 seconds...")
		time.Sleep(5 * time.Second)
	}
}

func (fc *FinnhubClient) fetchAllAvailableSymbols() []string {
	var allSymbols []string
	
	hotSymbols := []string{
		"BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:SOLUSDT", 
		"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
	}
	allSymbols = append(allSymbols, hotSymbols...)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	var mu sync.Mutex
	
	wg.Add(2)
	go func() {
		defer wg.Done()
		res, _, err := fc.sdk.CryptoSymbols(ctx).Exchange("BINANCE").Execute()
		if err == nil {
			mu.Lock()
			for _, s := range res {
				if s.Symbol != nil {
					allSymbols = append(allSymbols, *s.Symbol)
				}
			}
			mu.Unlock()
		}
	}()

	go func() {
		defer wg.Done()
		res, _, err := fc.sdk.StockSymbols(ctx).Exchange("US").Execute()
		if err == nil {
			mu.Lock()
			for _, s := range res {
				if s.Symbol != nil {
					allSymbols = append(allSymbols, *s.Symbol)
				}
			}
			mu.Unlock()
		}
	}()

	wg.Wait()

	unique := make([]string, 0)
	seen := make(map[string]bool)
	for _, s := range allSymbols {
		if !seen[s] {
			seen[s] = true
			unique = append(unique, s)
		}
	}

	const maxAllowed = 50
	if len(unique) > maxAllowed {
		return unique[:maxAllowed]
	}
	return unique
}

func (fc *FinnhubClient) subscribe(symbols []string) {
	slog.Info("Starting subscriptions", "count", len(symbols))
	for i, s := range symbols {
		msg, _ := json.Marshal(map[string]interface{}{"type": "subscribe", "symbol": s})
		if err := fc.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			slog.Error("Failed to subscribe to symbol", "symbol", s, "error", err)
			return
		}
		if i%10 == 0 && i > 0 {
			time.Sleep(50 * time.Millisecond)
		}
	}
	slog.Info("Finished processing subscriptions")
}

func (fc *FinnhubClient) listen() {
	defer fc.conn.Close()
	ctx := context.Background()
	for {
		_, message, err := fc.conn.ReadMessage()
		if err != nil {
			slog.Error("Error reading from Finnhub WebSocket", "error", err)
			return
		}

		// Direct broadcast of raw bytes to minimize marshaling latency
		// We use a select with default to avoid blocking if the hub is busy
		select {
		case fc.Hub.Broadcast <- message:
		default:
			slog.Warn("Broadcast channel full, skipping message for realtime")
		}

		// Process for storage/memstore without blocking broadcast
		reader := strings.NewReader(string(message))
		decoder := json.NewDecoder(reader)
		for decoder.More() {
			var fMsg FinnhubMessage
			if err := decoder.Decode(&fMsg); err == nil {
				if fMsg.Type == "trade" {
					for _, trade := range fMsg.Data {
						fc.MemStore.UpdatePrice(trade.Symbol, trade.Price, trade.Time)
						fc.DBStore.SaveTrade(ctx, trade.Symbol, trade.Price, trade.Volume, trade.Time)
					}
				}
			}
		}
	}
}