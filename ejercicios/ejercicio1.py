"""
Ejercicio 1: Perceptrón simple para AND y OR (2 y 4 entradas)
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

from utils.perceptron_simple import PerceptronSimple, create_logic_dataset, plot_decision_boundary_2d
import matplotlib.pyplot as plt


def ejercicio1():
    """Implementar perceptrón simple para AND y OR"""
    
    print("=" * 60)
    print("EJERCICIO 1: Perceptrón Simple (AND y OR)")
    print("=" * 60)
    
    # AND y OR con 2 entradas
    for function in ['AND', 'OR']:
        print(f"\n--- {function} con 2 entradas ---")
        X_2, y_2 = create_logic_dataset(function, 2)
        
        perceptron = PerceptronSimple(learning_rate=0.1, max_epochs=100)
        perceptron.fit(X_2, y_2)
        
        print(f"Predicciones: {perceptron.predict(X_2)}")
        print(f"Esperado:     {y_2}")
        
        # Graficar recta discriminadora
        plot_decision_boundary_2d(perceptron, X_2, y_2, f"Recta discriminadora - {function} (2 entradas)")
        
        # Graficar evolución del error
        perceptron.plot_training_error()
    
    # AND y OR con 4 entradas
    for function in ['AND', 'OR']:
        print(f"\n--- {function} con 4 entradas ---")
        X_4, y_4 = create_logic_dataset(function, 4)
        
        perceptron = PerceptronSimple(learning_rate=0.1, max_epochs=100)
        perceptron.fit(X_4, y_4)
        
        print(f"Precisión: {sum(perceptron.predict(X_4) == y_4) / len(y_4) * 100:.2f}%")
        
        # Graficar evolución del error
        perceptron.plot_training_error()


if __name__ == "__main__":
    ejercicio1()
