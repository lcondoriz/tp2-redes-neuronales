"""
Ejercicio 7: Autoencoder para MNIST
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt
import time


def ejercicio7():
    """Ejercicio 7: Autoencoder para MNIST"""
    
    print("=" * 60)
    print("EJERCICIO 7: Autoencoder - MNIST")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("Requiere: pip install torch torchvision scikit-learn")
        return
    
    # Preparar datos
    print("Cargando MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Definir Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 32)  # Representación latente de baja dimensionalidad
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x.view(-1, 784))
            decoded = self.decoder(encoded)
            return decoded, encoded
    
    # Entrenar Autoencoder
    print("\n--- Entrenando Autoencoder ---")
    autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    start_time = time.time()
    
    for epoch in range(20):
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(images)
            loss = criterion(reconstructed, images.view(-1, 784))
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Época {epoch + 1}/20, Loss: {loss.item():.6f}")
    
    autoencoder_time = time.time() - start_time
    print(f"Tiempo de entrenamiento: {autoencoder_time:.2f}s")
    
    # Extraer representaciones latentes
    print("\n--- Extrayendo representaciones latentes ---")
    autoencoder.eval()
    
    X_train_latent = []
    y_train = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            _, encoded = autoencoder(images)
            X_train_latent.append(encoded.cpu().numpy())
            y_train.append(labels.numpy())
    
    X_train_latent = np.vstack(X_train_latent)
    y_train = np.concatenate(y_train)
    
    X_test_latent = []
    y_test = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, encoded = autoencoder(images)
            X_test_latent.append(encoded.cpu().numpy())
            y_test.append(labels.numpy())
    
    X_test_latent = np.vstack(X_test_latent)
    y_test = np.concatenate(y_test)
    
    # Entrenar clasificador sobre representación latente
    print("\n--- Entrenando clasificador sobre representación latente ---")
    start_time = time.time()
    
    clf_latent = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf_latent.fit(X_train_latent, y_train)
    
    latent_train_time = time.time() - start_time
    accuracy_latent = accuracy_score(y_test, clf_latent.predict(X_test_latent))
    
    print(f"Precisión (representación latente): {accuracy_latent*100:.2f}%")
    print(f"Tiempo de entrenamiento: {latent_train_time:.2f}s")
    
    # Entrenar clasificador sobre imágenes originales
    print("\n--- Entrenando clasificador sobre imágenes originales ---")
    X_train_orig = []
    y_train_orig = []
    for images, labels in train_loader:
        X_train_orig.append(images.numpy().reshape(images.shape[0], -1))
        y_train_orig.append(labels.numpy())
    
    X_train_orig = np.vstack(X_train_orig)
    y_train_orig = np.concatenate(y_train_orig)
    
    X_test_orig = []
    y_test_orig = []
    for images, labels in test_loader:
        X_test_orig.append(images.numpy().reshape(images.shape[0], -1))
        y_test_orig.append(labels.numpy())
    
    X_test_orig = np.vstack(X_test_orig)
    y_test_orig = np.concatenate(y_test_orig)
    
    start_time = time.time()
    
    clf_orig = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf_orig.fit(X_train_orig, y_train_orig)
    
    orig_train_time = time.time() - start_time
    accuracy_orig = accuracy_score(y_test_orig, clf_orig.predict(X_test_orig))
    
    print(f"Precisión (imágenes originales): {accuracy_orig*100:.2f}%")
    print(f"Tiempo de entrenamiento: {orig_train_time:.2f}s")
    
    # Comparación
    print("\n--- COMPARACIÓN ---")
    print(f"Con autoencoder (32D): {accuracy_latent*100:.2f}% en {latent_train_time:.2f}s")
    print(f"Sin autoencoder (784D): {accuracy_orig*100:.2f}% en {orig_train_time:.2f}s")
    print(f"Tiempo ahorrado: {(orig_train_time - latent_train_time):.2f}s")
    
    # Visualizar reconstrucciones
    print("\n--- Visualizando reconstrucciones ---")
    autoencoder.eval()
    
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device)
    reconstructed, _ = autoencoder(sample_images)
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        # Originales
        axes[0, i].imshow(sample_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstruidas
        axes[1, i].imshow(reconstructed[i].detach().cpu().reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstruida')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio7()
