"""
Ejercicio 2: Capacidad del perceptrón en función del número de patrones
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
from utils.perceptron_simple import PerceptronSimple


def ejercicio2():
    """Analizar cómo varía la capacidad del perceptrón con el número de patrones"""
    
    print("=" * 60)
    print("EJERCICIO 2: Capacidad del Perceptrón")
    print("=" * 60)
    
    n_inputs = 2
    n_trials = 10  # Número de experimentos
    pattern_counts = np.arange(5, 50, 5)  # Números de patrones a probar
    success_rates = []
    convergence_epochs = []
    
    for n_patterns in pattern_counts:
        successes = 0
        epochs_needed = []
        
        for trial in range(n_trials):
            # Generar datos aleatorios
            X = np.random.randint(0, 2, size=(n_patterns, n_inputs))
            y = np.random.randint(0, 2, size=n_patterns)
            
            perceptron = PerceptronSimple(learning_rate=0.1, max_epochs=500)
            
            # Entrenar
            initial_errors = n_patterns
            perceptron.fit(X, y)
            final_errors = perceptron.training_errors[-1]
            
            # Considerar éxito si converge
            if final_errors == 0:
                successes += 1
                epochs_needed.append(len(perceptron.training_errors))
        
        success_rate = successes / n_trials
        success_rates.append(success_rate)
        
        avg_epochs = np.mean(epochs_needed) if epochs_needed else 500
        convergence_epochs.append(avg_epochs)
        
        print(f"Patrones: {n_patterns}, Tasa de éxito: {success_rate*100:.1f}%, Épocas promedio: {avg_epochs:.1f}")
    
    # Graficar resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Tasa de éxito vs patrones
    ax1.plot(pattern_counts, success_rates, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de patrones enseñados')
    ax1.set_ylabel('Tasa de convergencia')
    ax1.set_title('Capacidad del Perceptrón')
    ax1.grid(True)
    
    # Épocas de convergencia vs patrones
    ax2.plot(pattern_counts, convergence_epochs, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de patrones enseñados')
    ax2.set_ylabel('Épocas de convergencia (promedio)')
    ax2.set_title('Convergencia vs Número de Patrones')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio2()
