# Trabajo Práctico 2: Redes Neuronales

Implementación completa de los 9 ejercicios del curso de Redes Neuronales, cubriendo desde perceptrones simples hasta algoritmos avanzados como máquinas de Boltzmann, redes convolucionales, autoencoders, simulated annealing y algoritmos genéticos.

## Estructura del Proyecto

```
tp2-redes-neuronales/
├── README.md                    # Documentación
├── requirements.txt             # Dependencias
├── main.py                      # Menú interactivo
├── utils/
│   ├── perceptron_simple.py    # Perceptrón simple
│   └── perceptron_multicapa.py  # Perceptrón multicapa + Backpropagation
├── ejercicios/
│   ├── ejercicio1.py           # Perceptrón simple (AND/OR)
│   ├── ejercicio2.py           # Capacidad del perceptrón
│   ├── ejercicio3.py           # Perceptrón multicapa (XOR)
│   ├── ejercicio4.py           # Red para sin(x)+cos(y)+z
│   ├── ejercicio5.py           # Máquina Restringida de Boltzmann
│   ├── ejercicio6.py           # Red Convolucional
│   ├── ejercicio7.py           # Autoencoder
│   ├── ejercicio8.py           # Simulated Annealing
│   └── ejercicio9.py           # Algoritmo Genético
└── data/                       # Datasets (MNIST)
```

## Instalación

### 1. Clonar el repositorio

```bash
cd /home/lu/Escritorio/tp2-redes-neuronales
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Menú interactivo

```bash
python main.py
```

### Ejecutar ejercicios específicos

```bash
python ejercicios/ejercicio1.py   # Perceptrón Simple
python ejercicios/ejercicio3.py   # XOR con Backpropagation
python ejercicios/ejercicio9.py   # XOR con Algoritmo Genético
```

## Ejercicios Implementados

### ✅ Ejercicio 1: Perceptrón Simple
Aprender funciones lógicas AND y OR (2 y 4 entradas)
- Visualización de recta discriminadora
- Gráficos de evolución del error

### ✅ Ejercicio 2: Capacidad del Perceptrón
Analizar variación con número de patrones
- Tasa de convergencia vs patrones
- Épocas de convergencia promedio

### ✅ Ejercicio 3: Perceptrón Multicapa (XOR)
Aprender XOR de 2 y 4 entradas con Backpropagation
- Arquitectura: [2,4,1] y [4,8,1]

### ✅ Ejercicio 4: Red Backpropagation
Aprender f(x,y,z) = sin(x) + cos(y) + z
- Parte 4a: Entrenamiento y validación
- Parte 4b: Impacto del minibatch (40, 20, 1)

### ✅ Ejercicio 5: Máquina Restringida de Boltzmann
Entrenar RBM con MNIST
- Contrastive Divergence
- Reconstrucción de dígitos

### ✅ Ejercicio 6: Red Convolucional
Clasificación MNIST (≥90%)
- CNN vs MLP más pequeños

### ✅ Ejercicio 7: Autoencoder
Representaciones de baja dimensionalidad (MNIST)
- Comparación: con/sin autoencoder
- Precisión vs tiempo de entrenamiento

### ✅ Ejercicio 8: Simulated Annealing
Resolver XOR con Simulated Annealing
- Temperatura inicial: 1.0
- Factor enfriamiento: 0.995

### ✅ Ejercicio 9: Algoritmo Genético
Resolver XOR y análisis de parámetros
- Parte 9a: Algoritmo genético básico
- Parte 9b: Impacto de mutación, crossover y tamaño población

## Dependencias

```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
torch>=1.10.0
torchvision>=0.11.0
```

## Notas Importantes

- MNIST se descarga automáticamente en `./data`
- PyTorch usa GPU si está disponible
- Los gráficos se abren en ventanas interactivas
- Algunos ejercicios tardan varios minutos

## Resultados Esperados

- Ej. 1: AND/OR convergen rápidamente
- Ej. 2: Tasa de convergencia disminuye con más patrones
- Ej. 3: XOR aprende correctamente
- Ej. 4: Red aproxima función; batch_size=1 más variabilidad
- Ej. 5: RBM reconstruye dígitos
- Ej. 6: CNN ~2-5K parámetros, MLP ~50-100K
- Ej. 7: Autoencoder acelera sin perder precisión
- Ej. 8: Simulated Annealing converge (más lento)
- Ej. 9: AG resuelve XOR; tamaño población > tasa mutación
