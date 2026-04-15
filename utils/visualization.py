"""
Utilidades de visualización y plotting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """Grafica matriz de confusión"""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    
    # Etiquetas
    classes = np.unique(y_true)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Textos
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
    
    ax.set_ylabel('Etiqueta verdadera')
    ax.set_xlabel('Etiqueta predicha')
    ax.set_title(title)
    plt.colorbar(im)
    plt.show()


def plot_decision_regions(X, y, classifier, title="Regiones de Decisión", 
                          resolution=0.02, alpha=0.4):
    """Grafica regiones de decisión para clasificadores 2D"""
    
    # Definir límites
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Crear malla
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predicciones
    if hasattr(classifier, 'predict'):
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = (classifier(np.c_[xx.ravel(), yy.ravel()]) > 0.5).astype(int)
    
    Z = Z.reshape(xx.shape)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu_r, alpha=alpha)
    
    # Puntos de datos
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu_r, 
                         edgecolor='k', s=50)
    
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    plt.show()


def plot_learning_curves(train_scores, val_scores, title="Curvas de Aprendizaje"):
    """Grafica curvas de aprendizaje"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(train_scores))
    
    ax.plot(x, train_scores, 'b-', label='Entrenamiento', linewidth=2)
    ax.plot(x, val_scores, 'r-', label='Validación', linewidth=2)
    
    ax.fill_between(x, train_scores, val_scores, alpha=0.1)
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()


def plot_network_weights(network, layer_idx=0):
    """Visualiza pesos de una capa como imágenes (útil para capas convolucionales)"""
    
    weights = network.weights[layer_idx]
    
    if weights.ndim == 2:
        # Capas completamente conectadas
        if weights.shape[0] <= 784:  # Probablemente 28x28
            size = int(np.sqrt(weights.shape[0]))
            if size * size == weights.shape[0]:
                weights = weights.reshape(size, size, -1)
    
    if weights.ndim == 4:
        # Capas convolucionales
        n_filters = weights.shape[3]
        n_cols = int(np.sqrt(n_filters))
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(n_filters):
            w = weights[:, :, 0, i]
            axes[i].imshow(w, cmap='gray')
            axes[i].axis('off')
        
        plt.suptitle(f'Filtros de la capa {layer_idx}')
        plt.show()


def plot_distribution(data, bins=50, title="Distribución", xlabel="Valor"):
    """Grafica distribución de datos"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frecuencia')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Agregar estadísticas
    mean = np.mean(data)
    std = np.std(data)
    ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Media: {mean:.2f}')
    ax.axvline(mean + std, color='g', linestyle='--', linewidth=2, label=f'±Std: {std:.2f}')
    ax.axvline(mean - std, color='g', linestyle='--', linewidth=2)
    
    ax.legend()
    plt.show()


def animate_training(errors, title="Evolución del Error"):
    """Crea animación del entrenamiento (requiere matplotlib con soporte de animación)"""
    
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], lw=2)
        ax.set_ylim(0, max(errors) * 1.1)
        ax.set_xlim(0, len(errors))
        ax.set_xlabel('Época')
        ax.set_ylabel('Error')
        ax.set_title(title)
        ax.grid(True)
        
        def animate(frame):
            line.set_data(range(frame), errors[:frame])
            return line,
        
        anim = FuncAnimation(fig, animate, frames=len(errors), interval=50, blit=True)
        plt.show()
    
    except ImportError:
        print("Para animaciones, instale: pip install matplotlib")
        plot_distribution(errors)


def summarize_model(network):
    """Imprime resumen de la arquitectura de la red"""
    
    print("\n" + "="*60)
    print("RESUMEN DE LA RED")
    print("="*60)
    
    print(f"\nArquitectura: {' -> '.join(map(str, network.layer_sizes))}")
    
    total_params = 0
    for i, w in enumerate(network.weights):
        params = w.size + network.biases[i].size
        total_params += params
        print(f"Capa {i+1}: {w.shape[0]} x {w.shape[1]} + {network.biases[i].size} bias = {params} parámetros")
    
    print(f"\nTotal de parámetros: {total_params}")
    print("="*60 + "\n")
