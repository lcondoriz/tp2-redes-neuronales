#!/bin/bash
# Script de instalación y configuración
# Ejecutar: bash setup.sh

echo "========================================"
echo "INSTALACIÓN - TP2 REDES NEURONALES"
echo "========================================"
echo ""

# Crear directorio de datos si no existe
echo "Creando directorio de datos..."
mkdir -p data

# Instalar dependencias
echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Mostrar versiones instaladas
echo ""
echo "Versiones instaladas:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"

# Intentar importar PyTorch
echo ""
echo "Verificando dependencias opcionales..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch no instalado (opcional)"
python -c "import torchvision; print('✓ Torchvision instalado')" 2>/dev/null || echo "✗ Torchvision no instalado (opcional)"

# Ejecutar pruebas
echo ""
echo "Ejecutando pruebas de configuración..."
python test_setup.py

echo ""
echo "========================================"
echo "¡Instalación completada!"
echo "========================================"
echo ""
echo "Para comenzar, ejecute:"
echo "  python main.py"
echo ""
echo "O pruebe un ejercicio específico:"
echo "  python ejercicios/ejercicio1.py"
echo ""
