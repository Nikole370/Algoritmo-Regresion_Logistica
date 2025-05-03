package main

import (
	"fmt"
	"math"
	"math/rand"
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

func computeGradients(start, end int, X [][]float64, y []float64, weights []float64, gradientsChan chan []float64) {
	features := len(weights)
	gradients := make([]float64, features)

	for i := start; i < end; i++ {
		pred := predict(X[i], weights)
		error := pred - y[i]
		for j := 0; j < features; j++ {
			gradients[j] += error * X[i][j]
		}
	}
	gradientsChan <- gradients // ✅ Enviar el gradiente parcial de forma segura
}

func trainConcurrent(X [][]float64, y []float64, learningRate float64, iterations int, numGoroutines int) []float64 {
	features := len(X[0])
	n := len(X)
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		gradientsChan := make(chan []float64, numGoroutines)
		chunkSize := n / numGoroutines

		// Lanzar goroutines
		for g := 0; g < numGoroutines; g++ {
			start := g * chunkSize
			end := start + chunkSize
			if g == numGoroutines-1 {
				end = n // último grupo toma lo que sobre
			}
			// ✅ Cada goroutine calcula gradiente local
			go computeGradients(start, end, X, y, weights, gradientsChan)
		}

		// Sumar gradientes
		totalGradients := make([]float64, features)
		for g := 0; g < numGoroutines; g++ {
			partial := <-gradientsChan
			for j := 0; j < features; j++ {
				totalGradients[j] += partial[j] // ✅ Agregado en main thread
			}
		}

		// Actualizar pesos
		for j := 0; j < features; j++ {
			weights[j] -= learningRate * totalGradients[j] / float64(n)
		}
	}
	return weights
}
func main() {
	rand.Seed(time.Now().UnixNano())

	// Datos
	n := 1000000
	X := make([][]float64, n)
	y := make([]float64, n)

	for i := 0; i < n; i++ {
		x1 := rand.Float64() * 10 // primera característica
		x2 := rand.Float64() * 10 // segunda característica
		label := 0.0
		if x1+x2 > 10 { // regla simple para clasificar
			label = 1.0
		}
		X[i] = []float64{1, x1, x2} // agregamos el bias (1)
		y[i] = label
	}

	learningRate := 0.1
	iterations := 1000
	numGoroutines := 3
	start := time.Now()
	weights := trainConcurrent(X, y, learningRate, iterations, numGoroutines)

	fmt.Println("Pesos entrenados (concurrente):", weights)

	prob := predict([]float64{1, 6.5}, weights)
	fmt.Printf("Probabilidad de clase 1: %.4f\n", prob)
	elapsed := time.Since(start)
	fmt.Println("Tiempo de ejecución:", elapsed)
}
