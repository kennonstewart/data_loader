"""
Covtype (Forest Cover Type) data loader with fallback simulation.
"""
import os
import numpy as np
from .utils import set_global_seed, ensure_cache_dir, download_with_progress
from .streams import make_stream


def download_covtype(data_dir=None):
    """
    Download Covtype dataset using scikit-learn.
    Falls back to simulation if scikit-learn is not available or download fails.
    
    Args:
        data_dir: Directory to store data (default: ~/.cache/memory_pair_data)
        
    Returns:
        (X, y): Tuple of (data, labels) as numpy arrays
    """
    if data_dir is None:
        data_dir = ensure_cache_dir()
    
    try:
        # Try to use scikit-learn
        from sklearn.datasets import fetch_covtype
        
        # Download Covtype dataset
        data = fetch_covtype(data_home=data_dir, download_if_missing=True)
        
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64) - 1  # Convert to 0-based indexing
        
        print(f"Successfully loaded Covtype data: {X.shape}, {y.shape}")
        return X, y
        
    except (ImportError, Exception) as e:
        print(f"Failed to load Covtype with scikit-learn: {e}")
        print("Falling back to simulated Covtype data...")
        
        # Fallback to simulation
        return _simulate_covtype(n=581012, d=54, seed=42)


def get_covtype_stream(mode, batch_size=1, seed=42):
    """
    Get Covtype data stream with specified mode.
    
    Args:
        mode: "iid", "drift", or "adv"
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility
        
    Returns:
        Generator yielding (x_batch, y_batch)
    """
    set_global_seed(seed)
    
    # Load data
    X, y = download_covtype()
    
    # Define drift function for Covtype (shift tabular mean)
    def covtype_drift_fn(X_batch, drift_step):
        """Shift Covtype features by 0.1 * drift_step."""
        shift = 0.1 * drift_step
        return X_batch + shift
    
    # Create stream
    return make_stream(X, y, mode, drift_fn=covtype_drift_fn, batch_size=batch_size, seed=seed)


def _simulate_covtype(n=581012, d=54, seed=42):
    """
    Simulate Covtype-like data with deterministic random generation.
    
    Args:
        n: Number of samples to generate
        d: Number of features (dimensions)
        seed: Random seed for reproducibility
        
    Returns:
        (X, y): Tuple of (data, labels) as numpy arrays
        X shape: (n, d) float32
        y shape: (n,) int64 with 7 classes
    """
    set_global_seed(seed)
    
    # Generate random tabular data
    X = np.random.randn(n, d).astype(np.float32)
    
    # Add some structure to make it more realistic
    # First 10 features: elevation-related (positive values)
    X[:, :10] = np.abs(X[:, :10]) * 100 + 1000
    
    # Next 4 features: aspects (normalized to 0-360 range)
    X[:, 10:14] = (X[:, 10:14] % 360).astype(np.float32)
    
    # Next 10 features: slopes (0-90 range)
    X[:, 14:24] = np.abs(X[:, 14:24]) * 30
    
    # Next 10 features: distances (positive values)
    X[:, 24:34] = np.abs(X[:, 24:34]) * 1000
    
    # Remaining features: binary/categorical (0 or 1)
    X[:, 34:] = (X[:, 34:] > 0).astype(np.float32)
    
    # Generate labels (0-6) based on some features
    # Create clusters based on elevation and slope
    elevation_cluster = (X[:, 0] > 1500).astype(int)
    slope_cluster = (X[:, 14] > 15).astype(int)
    aspect_cluster = (X[:, 10] > 180).astype(int)
    
    y = (elevation_cluster * 2 + slope_cluster * 2 + aspect_cluster + 
         np.random.randint(0, 2, n)) % 7
    y = y.astype(np.int64)
    
    print(f"Generated simulated Covtype data: {X.shape}, {y.shape}")
    return X, y