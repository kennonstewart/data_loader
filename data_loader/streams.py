"""
Stream generation utilities for data_loader module.
"""
import numpy as np
from .utils import set_global_seed


def make_stream(X, y, mode, drift_fn=None, adv_fn=None, batch_size=1, seed=42):
    """
    Generator that yields (x_t, y_t) one at a time.
    
    Args:
        X: Input data array (n_samples, ...)
        y: Target labels array (n_samples,)
        mode: Stream mode - "iid", "drift", or "adv"
        drift_fn: Function to apply drift transformation every 1000 steps
        adv_fn: Function to apply adversarial permutation every 500 steps
        batch_size: Number of samples per batch (default 1)
        seed: Random seed for reproducibility
    
    Yields:
        (x_batch, y_batch): Batch of samples and labels
    """
    set_global_seed(seed)
    n_samples = len(X)
    
    if mode == "iid":
        # Independent and identically distributed sampling
        step = 0
        while True:
            indices = np.random.choice(n_samples, batch_size, replace=True)
            yield X[indices], y[indices]
            step += 1
            
    elif mode == "drift":
        # Apply drift transformation every 1000 steps
        step = 0
        X_current = X.copy()
        
        while True:
            # Apply drift transformation every 1000 steps
            if step > 0 and step % 1000 == 0:
                if drift_fn is not None:
                    X_current = drift_fn(X_current, step // 1000)
                else:
                    X_current = _default_drift_fn(X_current, step // 1000)
            
            indices = np.random.choice(n_samples, batch_size, replace=True)
            yield X_current[indices], y[indices]
            step += 1
            
    elif mode == "adv":
        # Adversarial permutation every 500 steps
        step = 0
        
        while True:
            # Apply adversarial permutation every 500 steps
            if step > 0 and step % 500 == 0:
                if adv_fn is not None:
                    perm_indices = adv_fn(n_samples, seed + step // 500)
                else:
                    perm_indices = _default_adv_fn(n_samples, seed + step // 500)
                X_perm = X[perm_indices]
                y_perm = y[perm_indices]
            else:
                X_perm = X
                y_perm = y
            
            indices = np.random.choice(n_samples, batch_size, replace=True)
            yield X_perm[indices], y_perm[indices]
            step += 1
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'iid', 'drift', or 'adv'")


def _default_drift_fn(X, drift_step):
    """Default drift function that rotates images or shifts tabular mean."""
    if len(X.shape) == 4:  # Image data (N, H, W, C)
        # Rotate images by 15 degrees * drift_step
        angle = 15 * drift_step
        return _rotate_images(X, angle)
    elif len(X.shape) == 3:  # Image data (N, H, W)
        # Rotate images by 15 degrees * drift_step
        angle = 15 * drift_step
        return _rotate_images(X, angle)
    else:  # Tabular data
        # Shift mean by 0.1 * drift_step
        shift = 0.1 * drift_step
        return X + shift


def _default_adv_fn(n_samples, seed):
    """Default adversarial function that permutes indices."""
    np.random.seed(seed)
    return np.random.permutation(n_samples)


def _rotate_images(X, angle):
    """Rotate images by given angle in degrees."""
    # Simple rotation implementation using scipy if available
    try:
        from scipy.ndimage import rotate
        X_rotated = np.zeros_like(X)
        for i in range(len(X)):
            if len(X.shape) == 4:  # (N, H, W, C)
                for c in range(X.shape[3]):
                    X_rotated[i, :, :, c] = rotate(X[i, :, :, c], angle, reshape=False, mode='constant')
            else:  # (N, H, W)
                X_rotated[i] = rotate(X[i], angle, reshape=False, mode='constant')
        return X_rotated
    except ImportError:
        # Fallback: just add some noise as a simple transformation
        noise = np.random.normal(0, 0.01, X.shape)
        return np.clip(X + noise, 0, 255).astype(X.dtype)