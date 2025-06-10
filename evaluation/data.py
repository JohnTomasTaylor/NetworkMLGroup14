# evaluation/data.py

"""
Data handling utilities for EEG ensemble evaluation pipeline.
Handles dataset, dataloader, and adjacency construction for graph models.
Colab- and cross-platform-ready, supports both train and test features.
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# --- Dynamic import ---
# Try code/GCN_LSTM_Pro, fallback to root if needed
try:
    sys.path.append('/content/networkML/code/GCN_LSTM_Pro')
    from dataset_multiwin import EEGMultiWinDataset, collate_fn_multiscale
except ImportError:
    from dataset_multiwin import EEGMultiWinDataset, collate_fn_multiscale

from evaluation.config import (
    FEATURE_ROOT, NUM_CHANNELS, BATCH_SIZE, NUM_WORKERS, DEVICE, EDGE_INDEX_PATH
)

def get_dataloader(meta_path, batch_size=None, shuffle=False, num_workers=None, feature_root=None):
    """
    Returns a DataLoader for EEG multi-scale features.
    Args:
        meta_path: Path to .parquet or .csv containing segments (should have correct columns).
        batch_size: Batch size (int, defaults to BATCH_SIZE from config).
        shuffle: Whether to shuffle dataset (bool).
        num_workers: Number of worker threads for DataLoader.
        feature_root: Optional override for features root directory.
    Returns:
        torch.utils.data.DataLoader
    """
    meta_path = Path(meta_path).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file {meta_path} does not exist.")
    if str(meta_path).endswith('.parquet'):
        df = pd.read_parquet(meta_path)
    elif str(meta_path).endswith('.csv'):
        df = pd.read_csv(meta_path)
    else:
        raise ValueError(f"Unsupported file extension for metadata file: {meta_path}")
    root = feature_root if feature_root is not None else FEATURE_ROOT
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Feature directory {root} does not exist.")
    dataset = EEGMultiWinDataset(df, feature_dirs_root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size or BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn_multiscale,
        num_workers=num_workers if num_workers is not None else NUM_WORKERS
    )

def load_fixed_adj_norm(edge_index_path=None, num_channels=None, device=None):
    """
    Loads and returns a normalized adjacency matrix for GCN models.
    Args:
        edge_index_path: Optional path to .pt file with edge_index.
        num_channels: Number of EEG channels.
        device: torch.device to load tensor to.
    Returns:
        adj_norm: torch.Tensor [num_channels, num_channels]
    """
    edge_index_path = Path(edge_index_path or EDGE_INDEX_PATH).resolve()
    if not edge_index_path.exists():
        raise FileNotFoundError(f"Edge index file {edge_index_path} does not exist.")
    C = num_channels or NUM_CHANNELS
    dev = device or DEVICE
    edge_index = torch.load(edge_index_path, map_location=dev)
    A_phys = torch.zeros((C, C), device=dev)
    row, col = edge_index
    A_phys[row, col] = 1.0
    A_phys[col, row] = 1.0
    eps = 1e-6
    A_phys = A_phys + torch.eye(C, device=dev) * eps
    deg = A_phys.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ A_phys @ D_inv_sqrt  # [C, C]
    return adj_norm
