"""
Ejercicio 9: Perceptrón multicapa para XOR usando Algoritmo Genético
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class GeneticNeuralNetwork:
    """Red neuronal con parámetros optimizables por algoritmo genético"""
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
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
        
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid(z)
        return output
    
    def compute_fitness(self, X, y):
        """Calcula el fitness (inverso del error)"""
        predictions = self.forward(X)
        mse = np.mean((predictions - y) ** 2)
        fitness = 1.0 / (1.0 + mse)  # Fitness entre 0 y 1
        return fitness
    
    def get_genome(self):
        """Retorna el genoma (todos los pesos)"""
        genome = []
        for w in self.weights:
            genome.append(w.flatten())
        for b in self.biases:
            genome.append(b.flatten())
        return np.concatenate(genome)
    
    def set_genome(self, genome):
        """Establece el genoma"""
        idx = 0
        for i in range(len(self.weights)):
            size = self.weights[i].size
            self.weights[i] = genome[idx:idx+size].reshape(self.weights[i].shape)
            idx += size
        for i in range(len(self.biases)):
            size = self.biases[i].size
            self.biases[i] = genome[idx:idx+size].reshape(self.biases[i].shape)
            idx += size
    
    def copy(self):
        """Crea una copia de la red"""
        new_network = GeneticNeuralNetwork(self.layer_sizes)
        new_network.set_genome(self.get_genome().copy())
        return new_network


def crossover(parent1: GeneticNeuralNetwork, parent2: GeneticNeuralNetwork, 
              crossover_prob: float = 0.7) -> GeneticNeuralNetwork:
    """Operador de crossover"""
    child = parent1.copy()
    genome1 = parent1.get_genome()
    genome2 = parent2.get_genome()
    
    child_genome = np.where(np.random.random(genome1.shape) < crossover_prob, 
                            genome1, genome2)
    child.set_genome(child_genome)
    return child


def mutate(network: GeneticNeuralNetwork, mutation_rate: float = 0.1, 
           mutation_strength: float = 0.5) -> None:
    """Operador de mutación"""
    genome = network.get_genome()
    mutation_mask = np.random.random(genome.shape) < mutation_rate
    mutations = np.random.normal(0, mutation_strength, genome.shape)
    genome[mutation_mask] += mutations[mutation_mask]
    network.set_genome(genome)


def genetic_algorithm(X, y, population_size: int = 50, generations: int = 200,
                     crossover_prob: float = 0.7, mutation_rate: float = 0.1,
                     elitism_rate: float = 0.1):
    """
    Algoritmo genético para entrenar la red
    
    Args:
        X: Datos de entrada
        y: Datos de salida
        population_size: Tamaño de la población
        generations: Número de generaciones
        crossover_prob: Probabilidad de crossover
        mutation_rate: Tasa de mutación
        elitism_rate: Porcentaje de élite a conservar
    """
    # Crear población inicial
    population = [GeneticNeuralNetwork([2, 8, 1]) for _ in range(population_size)]
    
    fitness_history = []
    best_fitness_history = []
    
    n_elite = max(1, int(population_size * elitism_rate))
    
    for generation in range(generations):
        # Calcular fitness
        fitnesses = [ind.compute_fitness(X, y) for ind in population]
        
        avg_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        fitness_history.append(avg_fitness)
        best_fitness_history.append(max_fitness)
        
        if (generation + 1) % 20 == 0:
            print(f"Generación {generation + 1}/{generations}, "
                  f"Fitness promedio: {avg_fitness:.6f}, "
                  f"Mejor: {max_fitness:.6f}")
        
        # Selección por torneo
        selected = []
        for _ in range(population_size - n_elite):
            # Torneo de 3
            tournament_indices = np.random.choice(population_size, 3, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
        
        # Élite
        elite_indices = np.argsort(fitnesses)[-n_elite:]
        elite = [population[i].copy() for i in elite_indices]
        
        # Reproducción
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = selected[np.random.randint(len(selected))]
            parent2 = selected[np.random.randint(len(selected))]
            
            child = crossover(parent1, parent2, crossover_prob)
            mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Mejor individuo final
    final_fitnesses = [ind.compute_fitness(X, y) for ind in population]
    best_idx = np.argmax(final_fitnesses)
    best_network = population[best_idx]
    
    return best_network, fitness_history, best_fitness_history


def ejercicio9a():
    """Ejercicio 9a: XOR con Algoritmo Genético"""
    
    print("=" * 60)
    print("EJERCICIO 9a: XOR con Algoritmo Genético")
    print("=" * 60)
    
    # Crear conjunto de datos XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("Iniciando entrenamiento con Algoritmo Genético...")
    best_network, avg_fitness, max_fitness = genetic_algorithm(
        X, y, 
        population_size=100, 
        generations=300,
        crossover_prob=0.8,
        mutation_rate=0.15
    )
    
    # Predicciones finales
    predictions = best_network.forward(X)
    print("\n--- Resultados finales ---")
    print(f"Entrada -> Predicción -> Esperado")
    for i in range(len(X)):
        print(f"{X[i]} -> {predictions[i][0]:.4f} -> {y[i][0]}")
    
    # Graficar fitness
    plt.figure(figsize=(12, 6))
    plt.plot(avg_fitness, label='Fitness promedio', alpha=0.7)
    plt.plot(max_fitness, label='Mejor fitness', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del fitness - Algoritmo Genético (XOR)')
    plt.legend()
    plt.grid(True)
    plt.show()


def ejercicio9b():
    """Ejercicio 9b: Impacto de parámetros genéticos"""
    
    print("\n" + "=" * 60)
    print("EJERCICIO 9b: Impacto de parámetros genéticos")
    print("=" * 60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Estudiar diferentes configuraciones
    configs = [
        {"mutation_rate": 0.05, "crossover_prob": 0.8, "pop_size": 50, 
         "label": "Mut=0.05, Cross=0.8, Pop=50"},
        {"mutation_rate": 0.2, "crossover_prob": 0.8, "pop_size": 50, 
         "label": "Mut=0.2, Cross=0.8, Pop=50"},
        {"mutation_rate": 0.1, "crossover_prob": 0.5, "pop_size": 50, 
         "label": "Mut=0.1, Cross=0.5, Pop=50"},
        {"mutation_rate": 0.1, "crossover_prob": 0.9, "pop_size": 50, 
         "label": "Mut=0.1, Cross=0.9, Pop=50"},
        {"mutation_rate": 0.1, "crossover_prob": 0.8, "pop_size": 30, 
         "label": "Mut=0.1, Cross=0.8, Pop=30"},
        {"mutation_rate": 0.1, "crossover_prob": 0.8, "pop_size": 100, 
         "label": "Mut=0.1, Cross=0.8, Pop=100"},
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        print(f"\nProbando: {config['label']}")
        _, avg_fitness, max_fitness = genetic_algorithm(
            X, y,
            population_size=config["pop_size"],
            generations=200,
            crossover_prob=config["crossover_prob"],
            mutation_rate=config["mutation_rate"]
        )
        
        ax = axes[idx]
        ax.plot(max_fitness, linewidth=2)
        ax.set_xlabel('Generación')
        ax.set_ylabel('Mejor Fitness')
        ax.set_title(config["label"])
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio9a()
    ejercicio9b()
