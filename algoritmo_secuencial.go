package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

//	implementación secuencial de la regresión logística
//

// Función sigmoide
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Función para calcular la predicción
func predict(X []float64, weights []float64) float64 {
	var z float64
	for i := 0; i < len(X); i++ {
		z += X[i] * weights[i]
	}
	return sigmoid(z)
}

// Entrenamiento con gradiente descendente
func train(X [][]float64, y []float64, learningRate float64, iterations int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([]float64, features)

		for i := 0; i < len(X); i++ {
			pred := predict(X[i], weights)
			error := pred - y[i]

			for j := 0; j < features; j++ {
				gradients[j] += error * X[i][j]
			}
		}

		// Actualizar pesos
		for j := 0; j < features; j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
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
	start := time.Now()
	weights := train(X, y, learningRate, iterations)

	fmt.Println("Pesos entrenados:", weights)

	// Prueba de predicción
	nuevaMuestra := []float64{1, 6.5}
	prob := predict(nuevaMuestra, weights)
	fmt.Printf("Probabilidad de clase 1: %.4f\n", prob)
	elapsed := time.Since(start)
	fmt.Println("Tiempo de ejecución:", elapsed)
}
