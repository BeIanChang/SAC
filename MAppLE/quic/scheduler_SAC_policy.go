package quic

import {
	"sync"
	"math/rand"
}

type SACPolicy struct {
	mu sync.RWMutex,
	PathProbs []float64
}

var currentPolicy = &SACPolicy{};

func UpdatePolicy(newProbs []float64) {
	currentPolicy.mu.Lock()
	defer currentPolicy.mu.Unlock()
	currentPolicy.PathProbs = newProbs
}

func GetAction() int {
	currentPolicy.mu.Lock()
	defer currentPolicy.mu.Unlock()

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

