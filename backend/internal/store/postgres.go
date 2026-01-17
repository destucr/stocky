package store

import (
	"context"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type PostgresStore struct {
	pool    *pgxpool.Pool
	tradeCh chan tradeJob
}

type tradeJob struct {
	symbol string
	price  float64
	volume float64
	ts     int64
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

	s := &PostgresStore{
		pool:    pool,
		tradeCh: make(chan tradeJob, 10000), // Large buffer for trade bursts
	}
	if err := s.InitSchema(ctx); err != nil {
		return nil, err
	}

	// Start DB worker pool (4 workers is usually enough for most DBs)
	for i := 0; i < 4; i++ {
		go s.tradeWorker()
	}

	return s, nil
}

func (s *PostgresStore) tradeWorker() {
	ctx := context.Background()
	for job := range s.tradeCh {
		t := time.UnixMilli(job.ts)
		
		// 1. Insert into history table
		queryTrade := `INSERT INTO trades (symbol, price, volume, timestamp) VALUES ($1, $2, $3, $4)`
		_, err := s.pool.Exec(ctx, queryTrade, job.symbol, job.price, job.volume, t)
		if err != nil {
			slog.Error("Failed to save trade to postgres", "error", err, "symbol", job.symbol)
		}

		// 2. Upsert into latest prices table
		queryLatest := `
			INSERT INTO latest_prices (symbol, price, timestamp) 
			VALUES ($1, $2, $3)
			ON CONFLICT (symbol) DO UPDATE 
			SET price = EXCLUDED.price, timestamp = EXCLUDED.timestamp`
		_, err = s.pool.Exec(ctx, queryLatest, job.symbol, job.price, t)
		if err != nil {
			slog.Error("Failed to update latest price in postgres", "error", err, "symbol", job.symbol)
		}
	}
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

	CREATE TABLE IF NOT EXISTS signals (
		id SERIAL PRIMARY KEY,
		symbol TEXT NOT NULL,
		signal_type TEXT NOT NULL,
		price DOUBLE PRECISION NOT NULL,
		target DOUBLE PRECISION NOT NULL,
		stop DOUBLE PRECISION NOT NULL,
		probability DOUBLE PRECISION NOT NULL,
		timestamp TIMESTAMPTZ NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_signals_lookup ON signals (symbol, timestamp DESC);
	`
	_, err := s.pool.Exec(ctx, query)
	return err
}

func (s *PostgresStore) SaveTrade(ctx context.Context, symbol string, price, volume float64, ts int64) {
	select {
	case s.tradeCh <- tradeJob{symbol, price, volume, ts}:
	default:
		slog.Warn("DB trade channel full, dropping trade for persistence", "symbol", symbol)
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
	// This query combines archived persistent candles with real-time aggregated trades.
	// It uses a UNION to ensure we always get the absolute latest data from the trades table,
	// even if it hasn't been archived yet.
	query := `
		WITH live_data AS (
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
		),
		archived_data AS (
			SELECT 
				timestamp as bucket, open, high, low, close, volume
			FROM persistent_candles
			WHERE symbol = $2 AND interval_secs = $1
		)
		SELECT bucket, open, high, low, close, volume FROM (
			SELECT * FROM live_data
			UNION ALL
			SELECT * FROM archived_data
		) combined
		ORDER BY bucket DESC
		LIMIT $3`

	rows, err := s.pool.Query(ctx, query, intervalSeconds, symbol, limit)
	if err != nil {
		slog.Error("Failed to fetch combined OHLC history", "error", err, "symbol", symbol)
		return nil, err
	}
	defer rows.Close()

	history := make([]Candle, 0)
	seenBuckets := make(map[int64]bool)

	for rows.Next() {
		var c Candle
		if err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume); err != nil {
			return nil, err
		}
		
		// Deduplicate: If a bucket exists in both tables, the live_data (newer aggregation) wins
		ts := c.Time.Unix()
		if !seenBuckets[ts] {
			history = append(history, c)
			seenBuckets[ts] = true
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
	// Archive everything currently in the trades table. 
	// ON CONFLICT ensures we don't create duplicates and always have the latest aggregate.
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
		GROUP BY symbol, bucket
		ON CONFLICT (symbol, interval_secs, timestamp) DO UPDATE 
		SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
		    close = EXCLUDED.close, volume = EXCLUDED.volume`
	
	_, err := s.pool.Exec(ctx, query, intervalSeconds)
	return err
}

type Signal struct {
	ID          int64     `json:"id"`
	Symbol      string    `json:"symbol"`
	Type        string    `json:"type"`
	Price       float64   `json:"price"`
	Target      float64   `json:"target"`
	Stop        float64   `json:"stop"`
	Probability float64   `json:"probability"`
	Timestamp   time.Time `json:"timestamp"`
}

func (s *PostgresStore) GetLatestSignal(ctx context.Context, symbol string) (*Signal, error) {
	query := `
		SELECT id, symbol, signal_type, price, target, stop, probability, timestamp
		FROM signals
		WHERE symbol = $1
		ORDER BY timestamp DESC
		LIMIT 1`
	
	var sig Signal
	err := s.pool.QueryRow(ctx, query, symbol).Scan(
		&sig.ID, &sig.Symbol, &sig.Type, &sig.Price, &sig.Target, &sig.Stop, &sig.Probability, &sig.Timestamp,
	)
	if err != nil {
		return nil, err
	}
	return &sig, nil
}

func (s *PostgresStore) Close() {
	s.pool.Close()
}
