"""
Data loader module for memory_pair experiments.

This module provides unified data loaders for MNIST, CIFAR10, and Covtype datasets
with fallback simulation capabilities.
"""

from .mnist import get_rotating_mnist_stream
from .cifar10 import get_cifar10_stream
from .covtype import get_covtype_stream
from .streams import make_stream

__version__ = "1.0.0"
__all__ = [
    'get_rotating_mnist_stream',
    'get_cifar10_stream', 
    'get_covtype_stream',
    'make_stream'
]