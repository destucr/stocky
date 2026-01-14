package websocket

import (
	"log/slog"
)

// Hub maintains the set of active clients and broadcasts messages to the clients.
type Hub struct {
	// Registered clients.
	clients map[*Client]bool

	// Inbound messages from Finnhub.
	Broadcast chan []byte

	// Register requests from the clients.
	register chan *Client

	// Unregister requests from clients.
	unregister chan *Client
}

func NewHub() *Hub {
	return &Hub{
		Broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		clients:    make(map[*Client]bool),
	}
}

func (h *Hub) Run() {
	slog.Info("WebSocket Hub started")
	for {
		select {
		case client := <-h.register:
			h.clients[client] = true
			slog.Info("Client registered", "total_clients", len(h.clients))
		case client := <-h.unregister:
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
				slog.Info("Client unregistered", "total_clients", len(h.clients))
			}
		case message := <-h.Broadcast:
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					slog.Warn("Client message buffer full, dropping client")
					close(client.send)
					delete(h.clients, client)
				}
			}
		}
	}
}