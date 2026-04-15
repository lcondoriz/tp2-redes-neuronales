"""
Implementación del Perceptrón Multicapa con Backpropagation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class PerceptronMulticapa:
    """Perceptrón multicapa con Backpropagation"""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1, 
                 activation: str = 'sigmoid', output_activation: str = 'sigmoid'):
        """
        Args:
            layer_sizes: Lista con el número de neuronas en cada capa
                         [n_inputs, n_hidden1, ..., n_outputs]
            learning_rate: Tasa de aprendizaje
            activation: Función de activación para capas ocultas ('sigmoid', 'relu', 'tanh')
            output_activation: Función de activación para capa de salida
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = len(layer_sizes)
        
        # Inicializar pesos y bias
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.training_errors = []
        self.validation_errors = []
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, output: np.ndarray) -> np.ndarray:
        """Derivada de la función sigmoide"""
        return output * (1 - output)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Función ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ReLU"""
        return (x > 0).astype(float)
    
    def tanh_activation(self, x: np.ndarray) -> np.ndarray:
        """Función tanh"""
        return np.tanh(x)
    
    def tanh_derivative(self, output: np.ndarray) -> np.ndarray:
        """Derivada de tanh"""
        return 1 - output ** 2
    
    def get_activation(self, activation_name: str):
        """Retorna la función de activación y su derivada"""
        if activation_name == 'sigmoid':
            return self.sigmoid, self.sigmoid_derivative
        elif activation_name == 'relu':
            return self.relu, self.relu_derivative
        elif activation_name == 'tanh':
            return self.tanh_activation, self.tanh_derivative
        else:
            raise ValueError(f"Activación desconocida: {activation_name}")
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Pase hacia adelante
        
        Returns:
            activations: Lista de activaciones en cada capa
            z_values: Lista de valores pre-activación
        """
        activations = [X]
        z_values = []
        
        for layer in range(self.n_layers - 1):
            z = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]
            z_values.append(z)
            
            # Usar activación según la capa
            if layer == self.n_layers - 2:  # Capa de salida
                act_fn, _ = self.get_activation(self.output_activation)
            else:  # Capas ocultas
                act_fn, _ = self.get_activation(self.activation)
            
            activation = act_fn(z)
            activations.append(activation)
        
        return activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                 z_values: List[np.ndarray]) -> None:
        """
        Pase hacia atrás (Backpropagation)
        """
        m = X.shape[0]  # Número de muestras
        
        # Inicializar gradientes
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Calcular error en la capa de salida
        _, output_act_deriv = self.get_activation(self.output_activation)
        delta = (activations[-1] - y) * output_act_deriv(activations[-1])
        
        # Backpropagation
        for layer in range(self.n_layers - 2, -1, -1):
            weight_gradients[layer] = np.dot(activations[layer].T, delta) / m
            bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True) / m
            
            if layer > 0:
                delta = np.dot(delta, self.weights[layer].T)
                _, act_deriv = self.get_activation(self.activation)
                delta *= act_deriv(activations[layer])
        
        # Actualizar pesos y bias
        for layer in range(self.n_layers - 1):
            self.weights[layer] -= self.learning_rate * weight_gradients[layer]
            self.biases[layer] -= self.learning_rate * bias_gradients[layer]
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            batch_size: int = None, X_val: np.ndarray = None, 
            y_val: np.ndarray = None) -> None:
        """
        Entrena la red neuronal
        
        Args:
            X: Datos de entrenamiento
            y: Etiquetas de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño del minibatch (None para batch completo)
            X_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        self.training_errors = []
        self.validation_errors = []
        
        for epoch in range(epochs):
            # Entrenamiento en minibatches
            indices = np.random.permutation(X.shape[0])
            
            for i in range(0, X.shape[0], batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                activations, z_values = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations, z_values)
            
            # Calcular error de entrenamiento
            activations, _ = self.forward(X)
            train_error = np.mean((activations[-1] - y) ** 2)
            self.training_errors.append(train_error)
            
            # Calcular error de validación si se proporciona
            if X_val is not None and y_val is not None:
                activations_val, _ = self.forward(X_val)
                val_error = np.mean((activations_val[-1] - y_val) ** 2)
                self.validation_errors.append(val_error)
            
            if (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs}, Error: {train_error:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        activations, _ = self.forward(X)
        return activations[-1]
    
    def plot_training_history(self) -> None:
        """Grafica la evolución del error durante el entrenamiento"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.training_errors, label='Error de entrenamiento')
        if self.validation_errors:
            plt.plot(self.validation_errors, label='Error de validación')
        
        plt.xlabel('Época')
        plt.ylabel('Error (MSE)')
        plt.title('Evolución del error durante el entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.show()


def create_xor_dataset(n_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea conjunto de datos para la función XOR
    
    Args:
        n_inputs: 2 o 4
    
    Returns:
        X: Matriz de entrada
        y: Vector de salida
    """
    n_samples = 2 ** n_inputs
    X = np.array([list(map(int, format(i, f'0{n_inputs}b'))) for i in range(n_samples)])
    
    # XOR: 1 si el número de 1s es impar, 0 si es par
    y = np.array([np.sum(x) % 2 for x in X]).reshape(-1, 1)
    
    return X.astype(np.float32), y.astype(np.float32)
