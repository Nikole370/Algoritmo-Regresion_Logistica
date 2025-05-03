package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func predict(X []float64, weights []float64) float64 {
	var z float64
	for i := 0; i < len(X); i++ {
		z += X[i] * weights[i]
	}
	return sigmoid(z)
}

// ❌ Esta versión contiene una sección crítica: varias goroutines modifican `weights` directamente
func trainWithRaceCondition(X [][]float64, y []float64, learningRate float64, iterations int, numGoroutines int) []float64 {
	n := len(X)
	features := len(X[0])
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		var wg sync.WaitGroup
		chunkSize := n / numGoroutines

		for g := 0; g < numGoroutines; g++ {
			start := g * chunkSize
			end := start + chunkSize
			if g == numGoroutines-1 {
				end = n
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					pred := predict(X[i], weights)
					error := pred - y[i]
					for j := 0; j < features; j++ {
						// ❌ Sección crítica: modificación concurrente sin sincronización
						weights[j] -= learningRate * error * X[i][j] / float64(n)
					}
				}
			}(start, end)
		}
		wg.Wait()
	}
	return weights
}

func main() {
	rand.Seed(time.Now().UnixNano())

	X := [][]float64{
		{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},
	}
	y := []float64{0, 0, 0, 1, 1, 1}

	weights := trainWithRaceCondition(X, y, 0.1, 1000, 3)
	fmt.Println("Pesos entrenados (con error de concurrencia):", weights)

	prob := predict([]float64{1, 6.5}, weights)
	fmt.Printf("Probabilidad estimada: %.4f\n", prob)
}
