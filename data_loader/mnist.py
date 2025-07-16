"""
MNIST data loader with fallback simulation.
"""
import os
import numpy as np
from .utils import set_global_seed, ensure_cache_dir, download_with_progress
from .streams import make_stream


def download_rotating_mnist(data_dir=None, split="train"):
    """
    Download MNIST dataset using torchvision.
    Falls back to simulation if torchvision is not available or download fails.
    
    Args:
        data_dir: Directory to store data (default: ~/.cache/memory_pair_data)
        split: "train" or "test"
        
    Returns:
        (X, y): Tuple of (data, labels) as numpy arrays
    """
    if data_dir is None:
        data_dir = ensure_cache_dir()
    
    try:
        # Try to use torchvision
        import torchvision
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        
        # Download MNIST
        is_train = (split == "train")
        dataset = datasets.MNIST(
            root=data_dir,
            train=is_train,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Convert to numpy arrays
        X = []
        y = []
        for data, target in dataset:
            # Convert tensor to numpy and scale to 0-255 uint8
            img = (data.numpy() * 255).astype(np.uint8)
            X.append(img)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Remove channel dimension if it exists (torchvision adds channel dim)
        if len(X.shape) == 4 and X.shape[1] == 1:
            X = X.squeeze(1)
        
        print(f"Successfully loaded MNIST {split} data: {X.shape}")
        return X, y
        
    except (ImportError, Exception) as e:
        print(f"Failed to load MNIST with torchvision: {e}")
        print("Falling back to simulated MNIST data...")
        
        # Fallback to simulation
        n_samples = 60000 if split == "train" else 10000
        return _simulate_mnist(n=n_samples, seed=42)


def get_rotating_mnist_stream(mode, batch_size=1, seed=42):
    """
    Get MNIST data stream with specified mode.
    
    Args:
        mode: "iid", "drift", or "adv"
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        
    Returns:
        Generator yielding (x_batch, y_batch)
    """
    set_global_seed(seed)
    
    # Load training data
    X, y = download_rotating_mnist(split="train")
    
    # Define drift function for MNIST (rotate images)
    def mnist_drift_fn(X_batch, drift_step):
        """Rotate MNIST images by 15 degrees per drift step."""
        angle = 15 * drift_step
        return _rotate_mnist_images(X_batch, angle)
    
    # Create stream
    return make_stream(X, y, mode, drift_fn=mnist_drift_fn, batch_size=batch_size, seed=seed)


def _simulate_mnist(n=70000, seed=42):
    """
    Simulate MNIST-like data with deterministic random generation.
    
    Args:
        n: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        (X, y): Tuple of (data, labels) as numpy arrays
        X shape: (n, 28, 28) uint8
        y shape: (n,) int64 with 10 classes
    """
    set_global_seed(seed)
    
    # Generate random images with some structure
    X = np.random.randint(0, 256, size=(n, 28, 28), dtype=np.uint8)
    
    # Add some simple patterns to make it more MNIST-like
    for i in range(n):
        # Add some vertical/horizontal lines randomly
        if np.random.random() < 0.3:
            # Vertical line
            col = np.random.randint(8, 20)
            X[i, 8:20, col] = 255
        if np.random.random() < 0.3:
            # Horizontal line
            row = np.random.randint(8, 20)
            X[i, row, 8:20] = 255
        
        # Add some rectangular shapes
        if np.random.random() < 0.2:
            r1, r2 = sorted(np.random.randint(5, 23, 2))
            c1, c2 = sorted(np.random.randint(5, 23, 2))
            X[i, r1:r2, c1:c2] = np.random.randint(128, 256)
    
    # Generate labels (0-9)
    y = np.random.randint(0, 10, size=n, dtype=np.int64)
    
    print(f"Generated simulated MNIST data: {X.shape}, {y.shape}")
    return X, y


def _rotate_mnist_images(X, angle):
    """Rotate MNIST images by given angle in degrees."""
    try:
        from scipy.ndimage import rotate
        X_rotated = np.zeros_like(X)
        for i in range(len(X)):
            X_rotated[i] = rotate(X[i], angle, reshape=False, mode='constant')
        return X_rotated
    except ImportError:
        # Fallback: add some noise as transformation
        noise = np.random.normal(0, 5, X.shape)
        return np.clip(X + noise, 0, 255).astype(X.dtype)