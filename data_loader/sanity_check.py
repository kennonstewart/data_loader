#!/usr/bin/env python3
"""
Sanity check script for data_loader module.

Usage:
    python sanity_check.py --dataset rotmnist --mode drift --T 5000
"""

import argparse
import hashlib
import numpy as np
import sys
import os

# Add parent directory to path to import data_loader package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data_loader import get_rotating_mnist_stream, get_cifar10_stream, get_covtype_stream


def compute_stream_hash(stream_data):
    """Compute SHA256 hash of stream data for reproducibility check."""
    # Convert to bytes and hash
    data_bytes = np.array(stream_data).tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Sanity check for data_loader module')
    parser.add_argument('--dataset', choices=['rotmnist', 'cifar10', 'covtype'], 
                        default='rotmnist', help='Dataset to test')
    parser.add_argument('--mode', choices=['iid', 'drift', 'adv'], 
                        default='drift', help='Stream mode')
    parser.add_argument('--T', type=int, default=5000, 
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Testing {args.dataset} dataset in {args.mode} mode...")
    print(f"Generating {args.T} samples with seed {args.seed}")
    print("-" * 50)
    
    # Get the appropriate stream
    if args.dataset == 'rotmnist':
        stream = get_rotating_mnist_stream(args.mode, batch_size=1, seed=args.seed)
    elif args.dataset == 'cifar10':
        stream = get_cifar10_stream(args.mode, batch_size=1, seed=args.seed)
    elif args.dataset == 'covtype':
        stream = get_covtype_stream(args.mode, batch_size=1, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Collect samples and show first 5
    samples = []
    labels = []
    
    print("First 5 samples:")
    for i, (x_batch, y_batch) in enumerate(stream):
        if i >= args.T:
            break
            
        x = x_batch[0]  # Get first sample from batch
        y = y_batch[0]  # Get first label from batch
        
        samples.append(x)
        labels.append(y)
        
        if i < 5:
            print(f"Sample {i+1}:")
            print(f"  Shape: {x.shape}")
            print(f"  Label: {y}")
            print(f"  Data type: {x.dtype}")
            
            # Show some statistics
            if len(x.shape) > 1:  # Image data
                print(f"  Min/Max: {x.min():.2f}/{x.max():.2f}")
                print(f"  Mean: {x.mean():.2f}")
            else:  # Tabular data
                print(f"  First 5 features: {x[:5]}")
            print()
    
    # Compute hash for reproducibility
    combined_data = []
    for i in range(min(100, len(samples))):  # Use first 100 samples for hash
        combined_data.extend(samples[i].flatten())
        combined_data.append(labels[i])
    
    stream_hash = compute_stream_hash(combined_data)
    
    print(f"Generated {len(samples)} samples")
    print(f"SHA256 hash of first 100 samples: {stream_hash}")
    print(f"Label distribution: {np.bincount(labels[:100])}")
    
    # Test reproducibility
    print("\nTesting reproducibility...")
    if args.dataset == 'rotmnist':
        stream2 = get_rotating_mnist_stream(args.mode, batch_size=1, seed=args.seed)
    elif args.dataset == 'cifar10':
        stream2 = get_cifar10_stream(args.mode, batch_size=1, seed=args.seed)
    elif args.dataset == 'covtype':
        stream2 = get_covtype_stream(args.mode, batch_size=1, seed=args.seed)
    
    # Get first sample from second stream
    x2, y2 = next(stream2)
    x2, y2 = x2[0], y2[0]
    
    # Compare with first sample from first stream
    if np.array_equal(samples[0], x2) and labels[0] == y2:
        print("✓ Reproducibility test PASSED")
    else:
        print("✗ Reproducibility test FAILED")
        print(f"  First sample differs: {np.array_equal(samples[0], x2)}")
        print(f"  First label differs: {labels[0] == y2}")
    
    print("\nSanity check completed!")


if __name__ == "__main__":
    main()