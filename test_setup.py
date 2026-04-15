"""
Script de prueba rápida de los componentes principales
Verifica que las instalaciones básicas funcionen correctamente
"""
import sys
import os

def test_imports():
    """Verifica que todos los módulos se puedan importar"""
    
    print("Verificando importaciones...")
    print("-" * 60)
    
    try:
        import numpy
        print("✓ NumPy")
    except ImportError:
        print("✗ NumPy - NO INSTALADO")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError:
        print("✗ Matplotlib - NO INSTALADO")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        print("✗ scikit-learn - NO INSTALADO")
        return False
    
    try:
        import scipy
        print("✓ SciPy")
    except ImportError:
        print("✗ SciPy - NO INSTALADO")
        return False
    
    print("\nDependencias opcionales:")
    print("-" * 60)
    
    try:
        import torch
        print("✓ PyTorch (para MNIST/GPU)")
    except ImportError:
        print("✗ PyTorch - NO INSTALADO (opcional)")
    
    try:
        import torchvision
        print("✓ Torchvision")
    except ImportError:
        print("✗ Torchvision - NO INSTALADO (opcional)")
    
    try:
        import tensorflow
        print("✓ TensorFlow")
    except ImportError:
        print("✗ TensorFlow - NO INSTALADO (opcional)")
    
    return True


def test_utils():
    """Prueba que los módulos de utilidades funcionen"""
    
    print("\n\nProbando módulos de utilidades...")
    print("-" * 60)
    
    try:
        from utils.perceptron_simple import PerceptronSimple, create_logic_dataset
        print("✓ utils.perceptron_simple")
        
        # Prueba rápida
        X, y = create_logic_dataset('AND', 2)
        print(f"  - Conjunto AND 2D creado: {X.shape}")
        
    except Exception as e:
        print(f"✗ utils.perceptron_simple - ERROR: {e}")
        return False
    
    try:
        from utils.perceptron_multicapa import PerceptronMulticapa, create_xor_dataset
        print("✓ utils.perceptron_multicapa")
        
        # Prueba rápida
        X, y = create_xor_dataset(2)
        print(f"  - Conjunto XOR 2D creado: {X.shape}")
        
    except Exception as e:
        print(f"✗ utils.perceptron_multicapa - ERROR: {e}")
        return False
    
    try:
        from utils.visualization import summarize_model
        print("✓ utils.visualization")
    except Exception as e:
        print(f"✗ utils.visualization - ERROR: {e}")
        return False
    
    return True


def test_quick_training():
    """Realiza entrenamiento rápido para verificar funcionamiento"""
    
    print("\n\nProbando entrenamiento rápido...")
    print("-" * 60)
    
    try:
        import numpy as np
        from utils.perceptron_simple import PerceptronSimple, create_logic_dataset
        
        print("Entrenando perceptrón simple para AND...")
        X, y = create_logic_dataset('AND', 2)
        perceptron = PerceptronSimple(learning_rate=0.1, max_epochs=100)
        perceptron.fit(X, y)
        
        predictions = perceptron.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        print(f"  - Precisión: {accuracy*100:.1f}%")
        
        if accuracy == 1.0:
            print("✓ Perceptrón simple funcionando correctamente")
        else:
            print("⚠ Perceptrón simple entrenó pero con baja precisión")
        
    except Exception as e:
        print(f"✗ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Ejecuta todas las pruebas"""
    
    print("=" * 60)
    print("VERIFICACIÓN DE INSTALACIÓN - TP2 REDES NEURONALES")
    print("=" * 60)
    
    if not test_imports():
        print("\n⚠ Faltan dependencias importantes. Por favor ejecute:")
        print("   pip install -r requirements.txt")
        return False
    
    if not test_utils():
        print("\n⚠ Error en módulos de utilidades")
        return False
    
    if not test_quick_training():
        print("\n⚠ Error en entrenamiento")
        return False
    
    print("\n" + "=" * 60)
    print("✓ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
    print("=" * 60)
    print("\nEjecute: python main.py")
    print("Para ver el menú interactivo de ejercicios")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
