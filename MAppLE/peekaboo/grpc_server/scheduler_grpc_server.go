package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"

	"google.golang.org/grpc"
	"peekaboo/mpquic_SAC/mpquicpb" // Import generated protobuf code
	"peekaboo/mpquic_SAC/policy"
)

// MPQUICServer implements the gRPC service defined in the .proto file
type MPQUICServer struct {
	mpquicpb.UnimplementedMPQUICServiceServer
	mu          sync.RWMutex
	networkState *mpquicpb.AggDelivState
}

// NewMPQUICServer initializes the gRPC server
func NewMPQUICServer() *MPQUICServer {
	return &MPQUICServer{
		networkState: &mpquicpb.AggDelivState{}, // Initialize with empty state
	}
}

// GetNetworkState handles requests from SACAgent to fetch the current network state
func (s *MPQUICServer) GetNetworkState(ctx context.Context, req *mpquicpb.Empty) (*mpquicpb.AggDelivState, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.networkState == nil {
		return nil, fmt.Errorf("network state not initialized")
	}

	log.Println("Sending current network state to SACAgent")
	return s.networkState, nil
}

// SendPolicy updates the current policy probabilities
func (s *MPQUICServer) SendPolicy(ctx context.Context, req *mpquicpb.Policy) (*mpquicpb.Empty, error) {
	if len(req.PathProbabilities) == 0 {
		return &mpquicpb.Empty{}, fmt.Errorf("received empty policy from SACAgent")
	}

	// Update the global policy
	policy.UpdatePolicy(req.PathProbabilities)

	log.Printf("Received new policy: %v\n", req.PathProbabilities)
	return &mpquicpb.Empty{}, nil
}

// UpdateNetworkState updates the current network state (called from QUIC scheduler)
func (s *MPQUICServer) UpdateNetworkState(state *mpquicpb.AggDelivState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.networkState = state
}

// StartGRPCServer starts the gRPC server to listen for SACAgent communication
func StartGRPCServer(server *MPQUICServer, port string) {
	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", port, err)
	}

	grpcServer := grpc.NewServer()
	mpquicpb.RegisterMPQUICServiceServer(grpcServer, server)

	log.Printf("gRPC server listening on port %s\n", port)
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Failed to serve gRPC server: %v", err)
	}
}

// Example integration with QUIC scheduler
func main() {
	// Initialize the gRPC server
	server := NewMPQUICServer()

	// Start the gRPC server in a separate goroutine
	go StartGRPCServer(server, "50051")

	// Simulate periodic updates to the network state (from the QUIC scheduler)
	go func() {
		for {
			state := &mpquicpb.AggDelivState{
				ResolutionType: "1080p",
				Bitrate:        4500,
				PathsInfo: []*mpquicpb.PathInfo{
					{
						PathId:                "0",
						Active:                true,
						RTT:                   30.5,
						HistoricalThroughput:  50.0,
						PacketLoss:            0.01,
						Bandwidth:             100.0,
						Cwnd:                  30.0,
						Jitter:                5.0,
					},
					{
						PathId:                "1",
						Active:                true,
						RTT:                   40.2,
						HistoricalThroughput:  45.0,
						PacketLoss:            0.02,
						Bandwidth:             80.0,
						Cwnd:                  25.0,
						Jitter:                7.0,
					},
				},
			}

			// Update the network state
			server.UpdateNetworkState(state)
			log.Println("Updated network state")
			time.Sleep(5 * time.Second)
		}
	}()

	// Simulate the QUIC scheduler using the current policy
	for {
		action := GetAction()
		log.Printf("Scheduler selected path: %d\n", action)
		time.Sleep(1 * time.Second) // Simulate packet scheduling
}
