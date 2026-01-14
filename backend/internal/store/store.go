package store

import (
	"sync"
	"time"
)

type PricePoint struct {
	Price     float64   `json:"price"`
	Timestamp time.Time `json:"timestamp"`
}

type Candle struct {
	Time   time.Time `json:"time"`
	Open   float64   `json:"open"`
	High   float64   `json:"high"`
	Low    float64   `json:"low"`
	Close  float64   `json:"close"`
	Volume float64   `json:"volume"`
}

type PriceStore struct {
	mu           sync.RWMutex
	latestPrices map[string]float64
	history      map[string][]PricePoint
	maxHistory   int
}

func NewPriceStore(maxHistory int) *PriceStore {
	return &PriceStore{
		latestPrices: make(map[string]float64),
		history:      make(map[string][]PricePoint),
		maxHistory:   maxHistory,
	}
}

func (s *PriceStore) UpdatePrice(symbol string, price float64, ts int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Update latest price
	s.latestPrices[symbol] = price

	// Convert unix ms to time.Time
	t := time.UnixMilli(ts)

	// Update history
	point := PricePoint{Price: price, Timestamp: t}
	s.history[symbol] = append(s.history[symbol], point)

	// Keep history within limits
	if len(s.history[symbol]) > s.maxHistory {
		s.history[symbol] = s.history[symbol][1:]
	}
}

func (s *PriceStore) GetLatestPrices() map[string]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	// Return a copy to avoid race conditions
	copy := make(map[string]float64)
	for k, v := range s.latestPrices {
		copy[k] = v
	}
	return copy
}

func (s *PriceStore) GetHistory(symbol string) []PricePoint {
	s.mu.RLock()
	defer s.mu.RUnlock()

	h, ok := s.history[symbol]
	if !ok {
		return []PricePoint{}
	}
	
	// Return a copy
	copy := make([]PricePoint, len(h))
	for i, v := range h {
		copy[i] = v
	}
	return copy
}
