"""
Ejercicio 3: Perceptrón Multicapa para XOR (2 y 4 entradas)
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

from utils.perceptron_multicapa import PerceptronMulticapa, create_xor_dataset
import numpy as np


def ejercicio3():
    """Implementar perceptrón multicapa para XOR con Backpropagation"""
    
    print("=" * 60)
    print("EJERCICIO 3: Perceptrón Multicapa - XOR (Backpropagation)")
    print("=" * 60)
    
    # XOR con 2 entradas
    print("\n--- XOR con 2 entradas ---")
    X_2, y_2 = create_xor_dataset(2)
    
    # Crear red: 2 inputs -> 4 hidden -> 1 output
    red = PerceptronMulticapa([2, 4, 1], learning_rate=0.5, activation='sigmoid')
    red.fit(X_2, y_2, epochs=1000, batch_size=len(X_2))
    
    predictions = red.predict(X_2)
    predictions_binary = (predictions > 0.5).astype(int)
    
    print(f"Predicciones: {predictions.flatten()}")
    print(f"Binarias:     {predictions_binary.flatten()}")
    print(f"Esperado:     {y_2.flatten()}")
    print(f"Precisión: {np.sum(predictions_binary == y_2) / len(y_2) * 100:.2f}%")
    
    red.plot_training_history()
    
    # XOR con 4 entradas
    print("\n--- XOR con 4 entradas ---")
    X_4, y_4 = create_xor_dataset(4)
    
    # Crear red: 4 inputs -> 8 hidden -> 1 output
    red = PerceptronMulticapa([4, 8, 1], learning_rate=0.5, activation='sigmoid')
    red.fit(X_4, y_4, epochs=1000, batch_size=len(X_4))
    
    predictions = red.predict(X_4)
    predictions_binary = (predictions > 0.5).astype(int)
    
    print(f"Precisión: {np.sum(predictions_binary == y_4) / len(y_4) * 100:.2f}%")
    
    red.plot_training_history()


if __name__ == "__main__":
    ejercicio3()
