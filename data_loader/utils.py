"""
Utility functions for data_loader module.
"""
import os
import random
import hashlib
import numpy as np
from urllib.request import urlretrieve
from urllib.error import URLError


def set_global_seed(seed=42):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def download_with_progress(url, target_path):
    """Download file from URL with progress indication."""
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading... {percent}%", end="", flush=True)
        
        urlretrieve(url, target_path, reporthook=progress_hook)
        print(f"\nDownload completed: {target_path}")
        return True
    except (URLError, OSError) as e:
        print(f"\nDownload failed: {e}")
        return False


def ensure_cache_dir():
    """Ensure cache directory exists and return its path."""
    cache_dir = os.path.expanduser("~/.cache/memory_pair_data")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def compute_sha256(data):
    """Compute SHA256 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha256(data).hexdigest()