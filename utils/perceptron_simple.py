"""
Implementación del Perceptrón Simple
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class PerceptronSimple:
    """Perceptrón simple para aprender funciones lógicas"""
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 1000, threshold: float = 0.5):
        """
        Args:
            learning_rate: Tasa de aprendizaje
            max_epochs: Número máximo de épocas
            threshold: Umbral de activación
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.training_errors = []
    
    def activation_function(self, x: float) -> int:
        """Función de activación paso (escalón)"""
        return 1 if x >= self.threshold else 0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicción sobre una muestra o conjunto de muestras"""
        z = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(val) for val in z])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el perceptrón
        
        Args:
            X: Conjunto de entrenamiento (N, n_features)
            y: Etiquetas (N,)
        """
        n_samples, n_features = X.shape
        
        # Inicialización de pesos y bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.training_errors = []
        
        # Entrenamiento
        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation_function(z)
                error = y[i] - y_pred
                
                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
            
            self.training_errors.append(errors)
            
            if errors == 0:
                print(f"Convergencia alcanzada en época {epoch}")
                break
        
        print(f"Entrenamiento completado. Errores finales: {errors}")
    
    def plot_training_error(self) -> None:
        """Grafica la evolución del error durante el entrenamiento"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_errors)
        plt.xlabel('Época')
        plt.ylabel('Número de errores')
        plt.title('Evolución del error durante el entrenamiento')
        plt.grid(True)
        plt.show()


def plot_decision_boundary_2d(perceptron: PerceptronSimple, X: np.ndarray, y: np.ndarray, 
                              title: str = "Recta Discriminadora") -> None:
    """
    Grafica la recta discriminadora y los puntos de entrada para el caso 2D
    
    Args:
        perceptron: Perceptrón entrenado
        X: Datos de entrada (N, 2)
        y: Etiquetas (N,)
        title: Título del gráfico
    """
    plt.figure(figsize=(10, 8))
    
    # Graficar puntos
    points_0 = X[y == 0]
    points_1 = X[y == 1]
    
    plt.scatter(points_0[:, 0], points_0[:, 1], c='red', label='Clase 0', marker='o', s=100)
    plt.scatter(points_1[:, 0], points_1[:, 1], c='blue', label='Clase 1', marker='s', s=100)
    
    # Graficar recta discriminadora
    # w1*x1 + w2*x2 + b = 0.5
    if abs(perceptron.weights[1]) > 1e-10:
        x1 = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
        x2 = (perceptron.threshold - perceptron.bias - perceptron.weights[0] * x1) / perceptron.weights[1]
        plt.plot(x1, x2, 'g--', label='Frontera de decisión', linewidth=2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def create_logic_dataset(function: str, n_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea conjuntos de datos para funciones lógicas
    
    Args:
        function: 'AND' u 'OR'
        n_inputs: 2 o 4
    
    Returns:
        X: Matriz de entrada (2^n_inputs, n_inputs)
        y: Vector de salida (2^n_inputs,)
    """
    n_samples = 2 ** n_inputs
    X = np.array([list(map(int, format(i, f'0{n_inputs}b'))) for i in range(n_samples)])
    
    if function == 'AND':
        y = np.array([1 if np.all(x == 1) else 0 for x in X])
    elif function == 'OR':
        y = np.array([1 if np.any(x == 1) else 0 for x in X])
    else:
        raise ValueError("function debe ser 'AND' u 'OR'")
    
    return X, y
