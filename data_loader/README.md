# Data Loader Module

A unified data loading module for memory_pair experiments that provides consistent interfaces for MNIST, CIFAR10, and Covtype datasets with robust fallback simulation capabilities.

## Dataset Loaders

| Dataset Key | Loader Function | Description |
|-------------|-----------------|-------------|
| `rotmnist` | `get_rotating_mnist_stream` | MNIST digits (28×28 grayscale) with rotation-based drift |
| `cifar10` | `get_cifar10_stream` | CIFAR10 images (32×32×3 RGB) with rotation-based drift |
| `covtype` | `get_covtype_stream` | Forest Cover Type (54D tabular) with mean-shift drift |

## Usage Example

```python
from data_loader import get_rotating_mnist_stream

# Get IID stream
stream = get_rotating_mnist_stream(mode="iid", batch_size=32, seed=42)
for x_batch, y_batch in stream:
    # Process batch
    print(f"Batch shape: {x_batch.shape}, Labels: {y_batch.shape}")
    break
```

## Reproducibility Test

Run the sanity check to verify deterministic behavior:

```bash
python sanity_check.py
```

## Features

- **Fail-safe simulation**: Automatically falls back to deterministic simulation if real datasets cannot be downloaded
- **Consistent interfaces**: All loaders follow the same API pattern
- **Drift simulation**: Supports IID, drift, and adversarial modes
- **Reproducibility**: Deterministic behavior with seed control
- **Minimal dependencies**: Only requires numpy, optional dependencies for enhanced functionality