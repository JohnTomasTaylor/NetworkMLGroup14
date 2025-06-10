# evaluation/model_loader.py

"""
Model construction and loading utilities for EEG ensemble pipeline.
Supports MultiScale_GCN_LSTM architecture (Zimu's code).
"""

import torch
from pathlib import Path
import sys

from evaluation.config import IN_FEATS, NUM_CHANNELS, DEVICE

# --- Dynamic import ---
try:
    sys.path.append('/content/networkML/code/GCN_LSTM_Pro')
    from advance_model import MultiScale_GCN_LSTM
except ImportError:
    from advance_model import MultiScale_GCN_LSTM

def build_model(
    fixed_adj_norm,
    in_feats=None,
    num_classes=2,
    share_weights=False,
    gcn_hidden=128,
    lstm_hidden=128,
    D=128,
    device=None,
    verbose=False,
    **kwargs
):
    """
    Creates an instance of the MultiScale_GCN_LSTM model.
    Args:
        fixed_adj_norm: torch.Tensor, normalized adjacency matrix (e.g., from data.load_fixed_adj_norm)
        in_feats: dict, with keys 'F05', 'F1', 'F2' for input features per timescale
        num_classes: int, number of output classes (default 2)
        share_weights: bool, share branch weights across scales (default False)
        gcn_hidden, lstm_hidden, D: model hyperparameters (int)
        device: torch.device (optional; defaults to config.DEVICE)
        verbose: bool, print model summary on creation
        kwargs: for future extension
    Returns:
        torch.nn.Module, ready for use
    """
    if in_feats is None:
        in_feats = IN_FEATS
    dev = device or DEVICE
    model = MultiScale_GCN_LSTM(
        in_feat_05=in_feats['F05'],
        in_feat_1=in_feats['F1'],
        in_feat_2=in_feats['F2'],
        gcn_hidden=gcn_hidden,
        lstm_hidden=lstm_hidden,
        D=D,
        num_classes=num_classes,
        share_weights=share_weights,
        fixed_adj_norm=fixed_adj_norm,
        num_channels=NUM_CHANNELS
    ).to(dev)
    if verbose:
        print(model)
    return model

def load_model(
    model_path,
    fixed_adj_norm,
    in_feats=None,
    strict=True,
    eval_mode=True,
    device=None,
    verbose=False,
    **kwargs
):
    """
    Loads model weights from disk and returns ready-to-use instance.
    Args:
        model_path: str or Path, path to .pth weights file
        fixed_adj_norm: torch.Tensor, normalized adjacency matrix
        in_feats: dict, input feature config (or None to use defaults)
        strict: bool, strict state dict matching
        eval_mode: bool, if True set model.eval(), else model.train()
        device: torch.device (optional; defaults to config.DEVICE)
        verbose: bool, print success messages and model summary
        kwargs: additional model build arguments
    Returns:
        torch.nn.Module
    Raises:
        FileNotFoundError if weights file not found
        RuntimeError for state dict mismatch
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights file {model_path} not found.")
    model = build_model(fixed_adj_norm, in_feats=in_feats, device=device, verbose=verbose, **kwargs)
    state = torch.load(model_path, map_location=device or DEVICE)
    try:
        model.load_state_dict(state, strict=strict)
    except Exception as e:
        print(f"Error loading state dict from {model_path}: {e}")
        raise
    if eval_mode:
        model.eval()
    else:
        model.train()
    if verbose:
        print(f"Loaded model from {model_path} | eval_mode={eval_mode}")
    return model
