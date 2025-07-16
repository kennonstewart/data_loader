"""
CIFAR10 data loader with fallback simulation.
"""
import os
import numpy as np
from .utils import set_global_seed, ensure_cache_dir, download_with_progress
from .streams import make_stream


def download_cifar10(data_dir=None, split="train"):
    """
    Download CIFAR10 dataset using torchvision.
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
        
        # Download CIFAR10
        is_train = (split == "train")
        dataset = datasets.CIFAR10(
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
            # torchvision returns (C, H, W), we want (H, W, C)
            img = (data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            X.append(img)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Successfully loaded CIFAR10 {split} data: {X.shape}")
        return X, y
        
    except (ImportError, Exception) as e:
        print(f"Failed to load CIFAR10 with torchvision: {e}")
        print("Falling back to simulated CIFAR10 data...")
        
        # Fallback to simulation
        n_samples = 50000 if split == "train" else 10000
        return _simulate_cifar10(n=n_samples, seed=42)


def get_cifar10_stream(mode, batch_size=1, seed=42):
    """
    Get CIFAR10 data stream with specified mode.
    
    Args:
        mode: "iid", "drift", or "adv"
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        
    Returns:
        Generator yielding (x_batch, y_batch)
    """
    set_global_seed(seed)
    
    # Load training data
    X, y = download_cifar10(split="train")
    
    # Define drift function for CIFAR10 (rotate images)
    def cifar10_drift_fn(X_batch, drift_step):
        """Rotate CIFAR10 images by 15 degrees per drift step."""
        angle = 15 * drift_step
        return _rotate_cifar10_images(X_batch, angle)
    
    # Create stream
    return make_stream(X, y, mode, drift_fn=cifar10_drift_fn, batch_size=batch_size, seed=seed)


def _simulate_cifar10(n=60000, seed=42):
    """
    Simulate CIFAR10-like data with deterministic random generation.
    
    Args:
        n: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        (X, y): Tuple of (data, labels) as numpy arrays
        X shape: (n, 32, 32, 3) uint8
        y shape: (n,) int64 with 10 classes
    """
    set_global_seed(seed)
    
    # Generate random color images
    X = np.random.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    
    # Add some simple patterns to make it more CIFAR10-like
    for i in range(n):
        # Add some colored blocks
        if np.random.random() < 0.4:
            r1, r2 = sorted(np.random.randint(0, 32, 2))
            c1, c2 = sorted(np.random.randint(0, 32, 2))
            color = np.random.randint(0, 256, 3)
            X[i, r1:r2, c1:c2] = color
        
        # Add some gradients
        if np.random.random() < 0.3:
            channel = np.random.randint(0, 3)
            gradient = np.linspace(0, 255, 32).astype(np.uint8)
            if np.random.random() < 0.5:
                # Horizontal gradient
                X[i, :, :, channel] = gradient[np.newaxis, :]
            else:
                # Vertical gradient
                X[i, :, :, channel] = gradient[:, np.newaxis]
        
        # Add some random noise overlay
        if np.random.random() < 0.2:
            noise = np.random.randint(-30, 30, size=(32, 32, 3))
            X[i] = np.clip(X[i].astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    # Generate labels (0-9)
    y = np.random.randint(0, 10, size=n, dtype=np.int64)
    
    print(f"Generated simulated CIFAR10 data: {X.shape}, {y.shape}")
    return X, y


def _rotate_cifar10_images(X, angle):
    """Rotate CIFAR10 images by given angle in degrees."""
    try:
        from scipy.ndimage import rotate
        X_rotated = np.zeros_like(X)
        for i in range(len(X)):
            for c in range(3):  # RGB channels
                X_rotated[i, :, :, c] = rotate(X[i, :, :, c], angle, reshape=False, mode='constant')
        return X_rotated
    except ImportError:
        # Fallback: add some noise as transformation
        noise = np.random.normal(0, 5, X.shape)
        return np.clip(X + noise, 0, 255).astype(X.dtype)