"""
EJEMPLOS DE USO - Fragmentos de código para cada ejercicio
"""

# ============================================================================
# EJERCICIO 1: Perceptrón Simple
# ============================================================================

from utils.perceptron_simple import PerceptronSimple, create_logic_dataset
import matplotlib.pyplot as plt

# Crear dataset AND con 2 entradas
X, y = create_logic_dataset('AND', 2)

# Crear y entrenar perceptrón
perceptron = PerceptronSimple(learning_rate=0.1, max_epochs=100)
perceptron.fit(X, y)

# Realizar predicciones
predictions = perceptron.predict(X)

# Visualizar resultados
from utils.perceptron_simple import plot_decision_boundary_2d
plot_decision_boundary_2d(perceptron, X, y, "AND - Recta Discriminadora")
perceptron.plot_training_error()


# ============================================================================
# EJERCICIO 3: Perceptrón Multicapa para XOR
# ============================================================================

from utils.perceptron_multicapa import PerceptronMulticapa, create_xor_dataset
import numpy as np

# Crear dataset XOR
X, y = create_xor_dataset(2)

# Crear red neuronal
# [inputs] -> [hidden1] -> [hidden2] -> [output]
red = PerceptronMulticapa([2, 4, 1], learning_rate=0.5, activation='sigmoid')

# Entrenar
red.fit(X, y, epochs=500, batch_size=len(X))

# Predicciones
predictions = red.predict(X)

# Visualizar
red.plot_training_history()


# ============================================================================
# EJERCICIO 4: Red para función sin(x)+cos(y)+z
# ============================================================================

from utils.perceptron_multicapa import PerceptronMulticapa
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de entrenamiento
np.random.seed(42)
n_samples = 500

x = np.random.uniform(0, 2*np.pi, n_samples)
y = np.random.uniform(0, 2*np.pi, n_samples)
z = np.random.uniform(-1, 1, n_samples)

# Función objetivo
f = np.sin(x) + np.cos(y) + z

# Preparar datos
X = np.column_stack([x, y, z])
y_out = f.reshape(-1, 1)

# Normalizar
X = (X - X.mean(axis=0)) / X.std(axis=0)
y_out = (y_out - y_out.mean()) / y_out.std()

# Crear red
red = PerceptronMulticapa([3, 16, 8, 1], learning_rate=0.05)
red.fit(X, y_out, epochs=500, batch_size=32)

# Visualizar
red.plot_training_history()


# ============================================================================
# EJERCICIO 8: XOR con Simulated Annealing
# ============================================================================

from ejercicios.ejercicio8 import SimpleNeuralNetwork, simulated_annealing
import numpy as np
import matplotlib.pyplot as plt

# Dataset XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Crear red
network = SimpleNeuralNetwork([2, 8, 1])

# Entrenar con Simulated Annealing
errors = simulated_annealing(network, X, y, max_iterations=10000)

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(errors)
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Simulated Annealing - XOR')
plt.show()


# ============================================================================
# EJERCICIO 9: XOR con Algoritmo Genético
# ============================================================================

from ejercicios.ejercicio9 import genetic_algorithm, GeneticNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Dataset XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Entrenar con AG
best_network, avg_fitness, max_fitness = genetic_algorithm(
    X, y,
    population_size=100,
    generations=300,
    crossover_prob=0.8,
    mutation_rate=0.15
)

# Visualizar
plt.figure(figsize=(10, 6))
plt.plot(max_fitness)
plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.title('Algoritmo Genético - XOR')
plt.show()


# ============================================================================
# COMPARACIÓN: Diferentes optimizadores para XOR
# ============================================================================

import numpy as np
import time

# Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

results = {}

# 1. Backpropagation (más rápido)
print("Entrenando con Backpropagation...")
from utils.perceptron_multicapa import PerceptronMulticapa
start = time.time()
red = PerceptronMulticapa([2, 8, 1], learning_rate=0.5)
red.fit(X.astype(np.float32), y.astype(np.float32), epochs=500, batch_size=4)
bp_time = time.time() - start
bp_error = red.training_errors[-1]
results['Backpropagation'] = {'tiempo': bp_time, 'error': bp_error}

# 2. Simulated Annealing
print("Entrenando con Simulated Annealing...")
from ejercicios.ejercicio8 import SimpleNeuralNetwork, simulated_annealing
start = time.time()
network = SimpleNeuralNetwork([2, 8, 1])
errors = simulated_annealing(network, X, y, max_iterations=5000)
sa_time = time.time() - start
sa_error = errors[-1]
results['Simulated Annealing'] = {'tiempo': sa_time, 'error': sa_error}

# 3. Algoritmo Genético
print("Entrenando con Algoritmo Genético...")
from ejercicios.ejercicio9 import genetic_algorithm
start = time.time()
best_net, _, _ = genetic_algorithm(X, y, population_size=50, generations=100)
ag_time = time.time() - start
ag_fitness = best_net.compute_fitness(X, y)
ag_error = 1.0 / ag_fitness - 1.0
results['Algoritmo Genético'] = {'tiempo': ag_time, 'error': ag_error}

# Resultados
print("\n" + "="*60)
print("COMPARACIÓN DE OPTIMIZADORES PARA XOR")
print("="*60)
for method, data in results.items():
    print(f"{method}:")
    print(f"  Tiempo: {data['tiempo']:.3f}s")
    print(f"  Error: {data['error']:.6f}")
    print()
