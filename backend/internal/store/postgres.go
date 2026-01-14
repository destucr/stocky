package store

import (
	"context"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type PostgresStore struct {
	pool *pgxpool.Pool
}

func NewPostgresStore(ctx context.Context, connString string) (*PostgresStore, error) {
	pool, err := pgxpool.New(ctx, connString)
	if err != nil {
		return nil, err
	}

	// Verify connection
	if err := pool.Ping(ctx); err != nil {
		return nil, err
	}

	s := &PostgresStore{pool: pool}
	if err := s.InitSchema(ctx); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *PostgresStore) InitSchema(ctx context.Context) error {
	query := `
	CREATE TABLE IF NOT EXISTS trades (
		id SERIAL PRIMARY KEY,
		symbol TEXT NOT NULL,
		price DOUBLE PRECISION NOT NULL,
		volume DOUBLE PRECISION NOT NULL,
		timestamp TIMESTAMPTZ NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades (symbol, timestamp DESC);

	CREATE TABLE IF NOT EXISTS latest_prices (
		symbol TEXT PRIMARY KEY,
		price DOUBLE PRECISION NOT NULL,
		timestamp TIMESTAMPTZ NOT NULL
	);
	`
	_, err := s.pool.Exec(ctx, query)
	return err
}

func (s *PostgresStore) SaveTrade(ctx context.Context, symbol string, price, volume float64, ts int64) {
	t := time.UnixMilli(ts)
	
	// 1. Insert into history table
	queryTrade := `INSERT INTO trades (symbol, price, volume, timestamp) VALUES ($1, $2, $3, $4)`
	_, err := s.pool.Exec(ctx, queryTrade, symbol, price, volume, t)
	if err != nil {
		slog.Error("Failed to save trade to postgres", "error", err, "symbol", symbol)
	}

	// 2. Upsert into latest prices table
	queryLatest := `
		INSERT INTO latest_prices (symbol, price, timestamp) 
		VALUES ($1, $2, $3)
		ON CONFLICT (symbol) DO UPDATE 
		SET price = EXCLUDED.price, timestamp = EXCLUDED.timestamp`
	_, err = s.pool.Exec(ctx, queryLatest, symbol, price, t)
	if err != nil {
		slog.Error("Failed to update latest price in postgres", "error", err, "symbol", symbol)
	}
}

func (s *PostgresStore) GetHistory(ctx context.Context, symbol string, limit int) ([]PricePoint, error) {
	query := `
		SELECT price, timestamp 
		FROM trades 
		WHERE symbol = $1 
		ORDER BY timestamp DESC 
		LIMIT $2`
	
	rows, err := s.pool.Query(ctx, query, symbol, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var history []PricePoint
	for rows.Next() {
		var p PricePoint
		if err := rows.Scan(&p.Price, &p.Timestamp); err != nil {
			return nil, err
		}
		history = append(history, p)
	}
	
	// Reverse to get chronological order for charts
	for i, j := 0, len(history)-1; i < j; i, j = i+1, j-1 {
		history[i], history[j] = history[j], history[i]
	}

	return history, nil
}

func (s *PostgresStore) GetOHLCHistory(ctx context.Context, symbol string, intervalSeconds int, limit int) ([]Candle, error) {
	// Simple OHLC aggregation using array_agg for first/last
	query := `
		SELECT 
			(to_timestamp(floor(extract(epoch from timestamp) / $1) * $1) AT TIME ZONE 'UTC')::timestamptz AS bucket,
			COALESCE((array_agg(price ORDER BY timestamp ASC, id ASC))[1], 0.0)::DOUBLE PRECISION AS open,
			COALESCE(max(price), 0.0)::DOUBLE PRECISION AS high,
			COALESCE(min(price), 0.0)::DOUBLE PRECISION AS low,
			COALESCE((array_agg(price ORDER BY timestamp DESC, id DESC))[1], 0.0)::DOUBLE PRECISION AS close,
			COALESCE(sum(volume), 0.0)::DOUBLE PRECISION AS volume
		FROM trades 
		WHERE symbol = $2 
		GROUP BY bucket 
		ORDER BY bucket DESC 
		LIMIT $3`

	rows, err := s.pool.Query(ctx, query, intervalSeconds, symbol, limit)
	if err != nil {
		slog.Error("Failed to execute OHLC history query", "error", err, "symbol", symbol, "interval", intervalSeconds)
		return nil, err
	}
	if rows == nil {
		return []Candle{}, nil
	}
	defer rows.Close()

	history := make([]Candle, 0)
	for rows.Next() {
		var c Candle
		if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume); err != nil {
			return nil, err
		}
		history = append(history, c)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Reverse to get chronological order
	for i, j := 0, len(history)-1; i < j; i, j = i+1, j-1 {
		history[i], history[j] = history[j], history[i]
	}

	return history, nil
}

func (s *PostgresStore) Close() {
	s.pool.Close()
}
