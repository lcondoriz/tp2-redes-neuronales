"""
Ejercicio 5: Máquina Restringida de Boltzmann con MNIST
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class RestrictedBoltzmannMachine:
    """Máquina Restringida de Boltzmann (RBM)"""
    
    def __init__(self, n_visible: int, n_hidden: int, learning_rate: float = 0.01):
        """
        Args:
            n_visible: Número de unidades visibles
            n_hidden: Número de unidades ocultas
            learning_rate: Tasa de aprendizaje
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Inicializar pesos
        self.weights = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
        
        self.reconstruction_errors = []
    
    def sigmoid(self, x):
        """Función sigmoide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_hidden(self, visible):
        """Muestrea unidades ocultas dadas las visibles"""
        activation = np.dot(visible, self.weights) + self.hidden_bias
        probabilities = self.sigmoid(activation)
        samples = (probabilities > np.random.random(probabilities.shape)).astype(float)
        return probabilities, samples
    
    def sample_visible(self, hidden):
        """Muestrea unidades visibles dadas las ocultas"""
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        probabilities = self.sigmoid(activation)
        samples = (probabilities > np.random.random(probabilities.shape)).astype(float)
        return probabilities, samples
    
    def gibbs_sampling(self, visible, steps=1):
        """Muestreo de Gibbs"""
        h_probs, h_samples = self.sample_hidden(visible)
        
        for _ in range(steps):
            v_probs, v_samples = self.sample_visible(h_samples)
            h_probs, h_samples = self.sample_hidden(v_samples)
        
        return v_probs, v_samples, h_probs, h_samples
    
    def train(self, X, epochs: int = 100, batch_size: int = 32):
        """
        Entrena la RBM usando Contrastive Divergence
        
        Args:
            X: Datos de entrenamiento (N, n_visible)
            epochs: Número de épocas
            batch_size: Tamaño del minibatch
        """
        self.reconstruction_errors = []
        n_batches = len(X) // batch_size
        
        for epoch in range(epochs):
            epoch_error = 0
            
            # Mezclar datos
            indices = np.random.permutation(len(X))
            
            for batch_idx in range(n_batches):
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch = X[batch_indices]
                
                # Fase positiva (datos)
                h_probs_pos, h_samples_pos = self.sample_hidden(batch)
                
                # Fase negativa (reconstrucción)
                v_probs_neg, v_samples_neg, h_probs_neg, h_samples_neg = \
                    self.gibbs_sampling(batch, steps=1)
                
                # Calcular error de reconstrucción
                error = np.mean((batch - v_probs_neg) ** 2)
                epoch_error += error
                
                # Actualizar pesos
                pos_gradient = np.dot(batch.T, h_probs_pos)
                neg_gradient = np.dot(v_probs_neg.T, h_probs_neg)
                
                self.weights += self.learning_rate * (pos_gradient - neg_gradient) / batch_size
                self.visible_bias += self.learning_rate * np.mean(batch - v_probs_neg, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(h_probs_pos - h_probs_neg, axis=0)
            
            avg_error = epoch_error / n_batches
            self.reconstruction_errors.append(avg_error)
            
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs}, Error de reconstrucción: {avg_error:.6f}")
    
    def reconstruct(self, X, steps=1):
        """Reconstruye los datos"""
        _, _, _, h_samples = self.gibbs_sampling(X, steps=steps)
        v_probs, _ = self.sample_visible(h_samples)
        return v_probs


def ejercicio5():
    """Ejercicio 5: RBM con MNIST"""
    
    print("=" * 60)
    print("EJERCICIO 5: Máquina Restringida de Boltzmann - MNIST")
    print("=" * 60)
    
    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        print("Requiere: pip install torchvision torch")
        return
    
    # Cargar MNIST
    print("Cargando dataset MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Usar solo primeras 5000 muestras para facilitar entrenamiento
    X = torch.cat([mnist_train[i][0].view(-1) for i in range(min(5000, len(mnist_train)))])
    X = X.numpy().reshape(min(5000, len(mnist_train)), -1)
    
    # Usar solo primeras 100 muestras para demostración rápida
    X_demo = X[:1000]
    
    print(f"Datos: {X_demo.shape}")
    
    # Crear y entrenar RBM
    rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=256, learning_rate=0.01)
    rbm.train(X_demo, epochs=50, batch_size=32)
    
    # Graficar error de reconstrucción
    plt.figure(figsize=(10, 6))
    plt.plot(rbm.reconstruction_errors)
    plt.xlabel('Época')
    plt.ylabel('Error de reconstrucción (MSE)')
    plt.title('RBM - MNIST: Error de reconstrucción')
    plt.grid(True)
    plt.show()
    
    # Mostrar ejemplos de reconstrucción para cada dígito
    print("\nReconstruyendo ejemplos de cada dígito...")
    
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    
    for digit in range(5):
        # Encontrar una imagen de este dígito
        digit_indices = np.where(np.array([mnist_train[i][1] for i in range(len(X_demo))]) == digit)[0]
        if len(digit_indices) > 0:
            idx = digit_indices[0]
            original = X_demo[idx]
            reconstructed = rbm.reconstruct(original.reshape(1, -1), steps=1)[0]
            
            # Mostrar original y reconstruida
            axes[digit, 0].imshow(original.reshape(28, 28), cmap='gray')
            axes[digit, 0].set_title(f'Original {digit}')
            axes[digit, 0].axis('off')
            
            axes[digit, 1].imshow(reconstructed.reshape(28, 28), cmap='gray')
            axes[digit, 1].set_title(f'Reconstruida {digit}')
            axes[digit, 1].axis('off')
            
            # Mostrar múltiples reconstrucciones
            for recon_idx in range(2):
                reconstructed = rbm.reconstruct(original.reshape(1, -1), steps=10)[0]
                axes[digit, 2 + recon_idx].imshow(reconstructed.reshape(28, 28), cmap='gray')
                axes[digit, 2 + recon_idx].set_title(f'Recon {recon_idx+1}')
                axes[digit, 2 + recon_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio5()
