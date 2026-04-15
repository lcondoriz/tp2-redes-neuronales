"""
Ejercicio 4: Red Backpropagation para aprender f(x,y,z)=sin(x)+cos(y)+z
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
from utils.perceptron_multicapa import PerceptronMulticapa


def generate_dataset(n_samples: int = 1000, seed: int = None):
    """Genera conjunto de datos para f(x,y,z)=sin(x)+cos(y)+z"""
    if seed is not None:
        np.random.seed(seed)
    
    # x, y ∈ [0, 2π]
    # z ∈ [-1, 1]
    x = np.random.uniform(0, 2*np.pi, n_samples)
    y = np.random.uniform(0, 2*np.pi, n_samples)
    z = np.random.uniform(-1, 1, n_samples)
    
    # Función objetivo
    f = np.sin(x) + np.cos(y) + z
    
    # Normalizar entrada y salida
    X = np.column_stack([x, y, z])
    y_out = f.reshape(-1, 1)
    
    # Normalizar
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y_out.mean(), y_out.std()
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y_out - y_mean) / y_std
    
    return X_norm, y_norm, (X_mean, X_std, y_mean, y_std)


def ejercicio4a():
    """Ejercicio 4a: Entrenar red para f(x,y,z)=sin(x)+cos(y)+z"""
    
    print("=" * 60)
    print("EJERCICIO 4a: Red Backpropagation para sin(x)+cos(y)+z")
    print("=" * 60)
    
    # Generar datos
    X_train, y_train, params = generate_dataset(n_samples=500, seed=42)
    X_val, y_val, _ = generate_dataset(n_samples=100, seed=43)
    
    # Normalizar validación con parámetros de entrenamiento
    X_mean, X_std, y_mean, y_std = params
    X_val = (X_val - X_mean) / X_std
    y_val = (y_val - y_mean) / y_std
    
    # Crear y entrenar red
    red = PerceptronMulticapa([3, 16, 8, 1], learning_rate=0.05, activation='relu')
    red.fit(X_train, y_train, epochs=500, batch_size=32, X_val=X_val, y_val=y_val)
    
    # Graficar evolución del error
    red.plot_training_history()


def ejercicio4b():
    """Ejercicio 4b: Estudiar impacto del tamaño de minibatch"""
    
    print("\n" + "=" * 60)
    print("EJERCICIO 4b: Impacto del tamaño de minibatch")
    print("=" * 60)
    
    X_train, y_train, params = generate_dataset(n_samples=40, seed=42)
    X_val, y_val, _ = generate_dataset(n_samples=20, seed=43)
    
    X_mean, X_std, y_mean, y_std = params
    X_val = (X_val - X_mean) / X_std
    y_val = (y_val - y_mean) / y_std
    
    batch_sizes = [40, 20, 1]  # Batch completo, mitad, tamaño 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, batch_size in enumerate(batch_sizes):
        print(f"\nEntrenando con batch_size = {batch_size}")
        red = PerceptronMulticapa([3, 30, 1], learning_rate=0.1, activation='sigmoid')
        red.fit(X_train, y_train, epochs=200, batch_size=batch_size, X_val=X_val, y_val=y_val)
        
        ax = axes[idx]
        ax.plot(red.training_errors, label='Error de entrenamiento')
        ax.plot(red.validation_errors, label='Error de validación')
        ax.set_xlabel('Época')
        ax.set_ylabel('Error (MSE)')
        ax.set_title(f'Batch size = {batch_size}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio4a()
    ejercicio4b()
