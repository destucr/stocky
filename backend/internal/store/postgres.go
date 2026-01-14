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

	CREATE TABLE IF NOT EXISTS persistent_candles (
		symbol TEXT NOT NULL,
		interval_secs INTEGER NOT NULL,
		open DOUBLE PRECISION NOT NULL,
		high DOUBLE PRECISION NOT NULL,
		low DOUBLE PRECISION NOT NULL,
		close DOUBLE PRECISION NOT NULL,
		volume DOUBLE PRECISION NOT NULL,
		timestamp TIMESTAMPTZ NOT NULL,
		PRIMARY KEY (symbol, interval_secs, timestamp)
	);
	CREATE INDEX IF NOT EXISTS idx_candles_lookup ON persistent_candles (symbol, interval_secs, timestamp DESC);
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
	// 1. First, try to get as much as possible from persistent_candles
	queryPersistent := `
		SELECT timestamp, open, high, low, close, volume
		FROM persistent_candles
		WHERE symbol = $1 AND interval_secs = $2
		ORDER BY timestamp DESC
		LIMIT $3`
	
	rows, err := s.pool.Query(ctx, queryPersistent, symbol, intervalSeconds, limit)
	if err != nil {
		return nil, err
	}
	
	history := make([]Candle, 0)
	lastPersistentTs := time.Time{}
	for rows.Next() {
		var c Candle
		if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume); err != nil {
			rows.Close()
			return nil, err
		}
		history = append(history, c)
		if lastPersistentTs.IsZero() {
			lastPersistentTs = c.Time
		}
	}
	rows.Close()

	// 2. If we need more data or the latest data (which isn't archived yet), 
	// fetch from the raw trades table
	if len(history) < limit {
		remaining := limit - len(history)
		queryTrades := `
			SELECT 
				(to_timestamp(floor(extract(epoch from timestamp) / $1) * $1) AT TIME ZONE 'UTC')::timestamptz AS bucket,
				COALESCE((array_agg(price ORDER BY timestamp ASC, id ASC))[1], 0.0)::DOUBLE PRECISION AS open,
				COALESCE(max(price), 0.0)::DOUBLE PRECISION AS high,
				COALESCE(min(price), 0.0)::DOUBLE PRECISION AS low,
				COALESCE((array_agg(price ORDER BY timestamp DESC, id DESC))[1], 0.0)::DOUBLE PRECISION AS close,
				COALESCE(sum(volume), 0.0)::DOUBLE PRECISION AS volume
			FROM trades 
			WHERE symbol = $2 AND timestamp > $3
			GROUP BY bucket 
			ORDER BY bucket DESC 
			LIMIT $4`

		rows, err = s.pool.Query(ctx, queryTrades, intervalSeconds, symbol, lastPersistentTs, remaining)
		if err != nil {
			return nil, err
		}
		defer rows.Close()

		for rows.Next() {
			var c Candle
			if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume); err != nil {
				return nil, err
			}
			history = append(history, c)
		}
	}

	// Reverse to get chronological order
	for i, j := 0, len(history)-1; i < j; i, j = i+1, j-1 {
		history[i], history[j] = history[j], history[i]
	}

	return history, nil
}

func (s *PostgresStore) CleanupOldTrades(ctx context.Context, retention time.Duration) (int64, error) {
	threshold := time.Now().Add(-retention)
	query := `DELETE FROM trades WHERE timestamp < $1`
	
	result, err := s.pool.Exec(ctx, query, threshold)
	if err != nil {
		return 0, err
	}
	
	return result.RowsAffected(), nil
}

func (s *PostgresStore) ArchiveTradesToCandles(ctx context.Context, intervalSeconds int) error {
	// Aggregate trades from the last hour into persistent candles
	// We use ON CONFLICT to avoid duplicates if the task runs multiple times
	query := `
		INSERT INTO persistent_candles (symbol, interval_secs, open, high, low, close, volume, timestamp)
		SELECT 
			symbol,
			$1 as interval_secs,
			COALESCE((array_agg(price ORDER BY timestamp ASC, id ASC))[1], 0.0) as open,
			COALESCE(max(price), 0.0) as high,
			COALESCE(min(price), 0.0) as low,
			COALESCE((array_agg(price ORDER BY timestamp DESC, id DESC))[1], 0.0) as close,
			COALESCE(sum(volume), 0.0) as volume,
			(to_timestamp(floor(extract(epoch from timestamp) / $1) * $1) AT TIME ZONE 'UTC')::timestamptz as bucket
		FROM trades
		WHERE timestamp >= (NOW() - INTERVAL '2 hours')
		GROUP BY symbol, bucket
		ON CONFLICT (symbol, interval_secs, timestamp) DO UPDATE 
		SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
		    close = EXCLUDED.close, volume = EXCLUDED.volume`
	
	_, err := s.pool.Exec(ctx, query, intervalSeconds)
	return err
}

func (s *PostgresStore) Close() {
	s.pool.Close()
}
