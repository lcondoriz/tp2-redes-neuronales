"""
Utils - Módulo de utilidades para redes neuronales
"""

from .perceptron_simple import (
    PerceptronSimple,
    plot_decision_boundary_2d,
    create_logic_dataset
)

from .perceptron_multicapa import (
    PerceptronMulticapa,
    create_xor_dataset
)

from .visualization import (
    plot_confusion_matrix,
    plot_decision_regions,
    plot_learning_curves,
    summarize_model
)

__all__ = [
    'PerceptronSimple',
    'PerceptronMulticapa',
    'plot_decision_boundary_2d',
    'create_logic_dataset',
    'create_xor_dataset',
    'plot_confusion_matrix',
    'plot_decision_regions',
    'plot_learning_curves',
    'summarize_model'
]
