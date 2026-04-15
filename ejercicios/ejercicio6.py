"""
Ejercicio 6: Red Convolucional para MNIST
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

import numpy as np
import matplotlib.pyplot as plt


def ejercicio6():
    """Ejercicio 6: Red Convolucional y Perceptrón Multicapa en MNIST"""
    
    print("=" * 60)
    print("EJERCICIO 6: Red Convolucional vs Perceptrón Multicapa - MNIST")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import time
    except ImportError:
        print("Requiere: pip install torch torchvision")
        return
    
    # Preparar datos
    print("Cargando MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Ejercicio 6a: Red Convolucional pequeña
    print("\n--- Red Convolucional pequeña ---")
    
    class SmallCNN(nn.Module):
        def __init__(self):
            super(SmallCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(8 * 14 * 14, 32)
            self.fc2 = nn.Linear(32, 10)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(-1, 8 * 14 * 14)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model_cnn = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
    
    print(f"Parámetros: {sum(p.numel() for p in model_cnn.parameters())}")
    
    # Entrenar CNN
    epochs = 5
    start_time = time.time()
    
    for epoch in range(epochs):
        model_cnn.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    cnn_train_time = time.time() - start_time
    
    # Evaluar CNN
    model_cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_cnn(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    cnn_accuracy = 100 * correct / total
    print(f"Precisión CNN: {cnn_accuracy:.2f}%")
    print(f"Tiempo de entrenamiento: {cnn_train_time:.2f}s")
    
    # Ejercicio 6b: Perceptrón Multicapa pequeño
    print("\n--- Perceptrón Multicapa pequeño ---")
    
    class SmallMLP(nn.Module):
        def __init__(self):
            super(SmallMLP, self).__init__()
            self.fc1 = nn.Linear(784, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model_mlp = SmallMLP().to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)
    
    print(f"Parámetros: {sum(p.numel() for p in model_mlp.parameters())}")
    
    # Entrenar MLP
    start_time = time.time()
    
    for epoch in range(epochs):
        model_mlp.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_mlp(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    mlp_train_time = time.time() - start_time
    
    # Evaluar MLP
    model_mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_mlp(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    mlp_accuracy = 100 * correct / total
    print(f"Precisión MLP: {mlp_accuracy:.2f}%")
    print(f"Tiempo de entrenamiento: {mlp_train_time:.2f}s")
    
    # Comparar resultados
    print("\n--- COMPARACIÓN ---")
    print(f"CNN: {cnn_accuracy:.2f}% precisión, {sum(p.numel() for p in model_cnn.parameters())} parámetros")
    print(f"MLP: {mlp_accuracy:.2f}% precisión, {sum(p.numel() for p in model_mlp.parameters())} parámetros")


if __name__ == "__main__":
    import torch
    ejercicio6()
