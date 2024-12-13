package quic

import (
	"context"
	"fmt"
	"sync"
)

// Define a simple in-memory policy storage
var currentPolicy = struct {
	sync.RWMutex
	PathProbs []float64
}{}

// gRPC service struct
type MPQUICServer struct {
	UnimplementedMPQUICServiceServer
}

// gRPC SelectPath method
func (s *MPQUICServer) SelectPath(ctx context.Context, req *AggDelivState) (*PathInfo, error) {
	fmt.Println("Received state:", req)
	selectedPath := req.GetPathsInfo()[0] // Example: select the first path
	return selectedPath, nil
}

// gRPC SendPolicy method
func (s *MPQUICServer) SendPolicy(ctx context.Context, req *Policy) (*Empty, error) {
	if len(req.PathProbabilities) == 0 {
		return &Empty{}, fmt.Errorf("empty policy received")
	}

	currentPolicy.Lock()
	defer currentPolicy.Unlock()
	currentPolicy.PathProbs = req.PathProbabilities
	fmt.Println("Updated policy:", req.PathProbabilities)

	return &Empty{}, nil
}

func GetAction() int {
	currentPolicy.RLock()
	defer currentPolicy.RUnlock()

	if len(currentPolicy.PathProbs) == 0 {
		return 0
	}

	// Random sampling based on probabilities
	cumulativeProbs := make([]float64, len(currentPolicy.PathProbs))
	cumulativeProbs[0] = currentPolicy.PathProbs[0]
	for i := 1; i < len(currentPolicy.PathProbs); i++ {
		cumulativeProbs[i] = cumulativeProbs[i-1] + currentPolicy.PathProbs[i]
	}

	r := rand.Float64()
	for i, val := range cumulativeProbs {
		if val >= r {
			return i
		}
	}
	return len(currentPolicy.PathProbs) - 1
}
