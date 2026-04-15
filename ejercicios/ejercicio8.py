"""
Ejercicio 8: Perceptrón multicapa para XOR usando Simulated Annealing
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class SimpleNeuralNetwork:
    """Red neuronal simple con pesos ajustables"""
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.flatten_weights = None
        
        # Inicializar pesos
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        activation = X
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)
        
        # Capa de salida
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z)
        return output
    
    def compute_error(self, X, y):
        """Error cuadrático medio"""
        predictions = self.forward(X)
        return np.mean((predictions - y) ** 2)
    
    def get_weights_flat(self):
        """Retorna todos los pesos en un vector"""
        weights_list = []
        for w in self.weights:
            weights_list.append(w.flatten())
        for b in self.biases:
            weights_list.append(b.flatten())
        return np.concatenate(weights_list)
    
    def set_weights_flat(self, flat_weights):
        """Establece pesos desde un vector"""
        idx = 0
        for i in range(len(self.weights)):
            size = self.weights[i].size
            self.weights[i] = flat_weights[idx:idx+size].reshape(self.weights[i].shape)
            idx += size
        for i in range(len(self.biases)):
            size = self.biases[i].size
            self.biases[i] = flat_weights[idx:idx+size].reshape(self.biases[i].shape)
            idx += size


def simulated_annealing(network, X, y, max_iterations=10000, initial_temp=1.0, 
                        cooling_rate=0.995, min_temp=1e-5):
    """
    Simulated Annealing para entrenar la red
    
    Args:
        network: Red neuronal a entrenar
        X: Datos de entrada
        y: Datos de salida
        max_iterations: Número máximo de iteraciones
        initial_temp: Temperatura inicial
        cooling_rate: Factor de enfriamiento
        min_temp: Temperatura mínima
    """
    errors = []
    current_weights = network.get_weights_flat()
    current_error = network.compute_error(X, y)
    errors.append(current_error)
    
    best_weights = current_weights.copy()
    best_error = current_error
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generar nueva solución vecina
        neighbor_weights = current_weights + np.random.normal(0, temperature, current_weights.shape)
        network.set_weights_flat(neighbor_weights)
        neighbor_error = network.compute_error(X, y)
        
        # Criterio de aceptación (Metropolis)
        delta_error = neighbor_error - current_error
        if delta_error < 0 or np.random.random() < np.exp(-delta_error / temperature):
            current_weights = neighbor_weights.copy()
            current_error = neighbor_error
        else:
            network.set_weights_flat(current_weights)
        
        # Actualizar mejor solución encontrada
        if current_error < best_error:
            best_error = current_error
            best_weights = current_weights.copy()
        
        errors.append(best_error)
        
        # Enfriar
        temperature *= cooling_rate
        temperature = max(temperature, min_temp)
        
        if (iteration + 1) % 1000 == 0:
            print(f"Iteración {iteration + 1}/{max_iterations}, Error: {best_error:.6f}, Temp: {temperature:.6f}")
    
    # Establecer mejor solución encontrada
    network.set_weights_flat(best_weights)
    return errors


def ejercicio8():
    """Ejercicio 8: XOR con Simulated Annealing"""
    
    print("=" * 60)
    print("EJERCICIO 8: XOR con Simulated Annealing")
    print("=" * 60)
    
    # Crear conjunto de datos XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Crear y entrenar red
    network = SimpleNeuralNetwork([2, 8, 1])
    
    print("Iniciando entrenamiento con Simulated Annealing...")
    errors = simulated_annealing(network, X, y, max_iterations=20000, 
                                initial_temp=1.0, cooling_rate=0.9995)
    
    # Predicciones finales
    predictions = network.forward(X)
    print("\n--- Resultados finales ---")
    print(f"Entrada -> Predicción -> Esperado")
    for i in range(len(X)):
        print(f"{X[i]} -> {predictions[i][0]:.4f} -> {y[i][0]}")
    
    # Graficar evolución del error
    plt.figure(figsize=(12, 6))
    plt.plot(errors)
    plt.xlabel('Iteración')
    plt.ylabel('Error (MSE)')
    plt.title('Evolución del error - Simulated Annealing (XOR)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ejercicio8()
