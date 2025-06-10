# evaluation/ensemble.py

"""
Ensemble strategies for EEG pipeline.
Supports mean, weighted, and majority-vote ensembles.
"""

import numpy as np

def simple_average_ensemble(prob_arrays):
    """
    Mean-ensembles model probability arrays (e.g., from N models).
    Args:
        prob_arrays: List of np.ndarray or list, each shape [N,]
            (N = number of samples, len = number of models)
    Returns:
        mean_probs: np.ndarray, shape [N,]
    """
    if not prob_arrays:
        raise ValueError("prob_arrays list is empty!")
    stacked = np.stack(prob_arrays, axis=0)
    return np.mean(stacked, axis=0)

def weighted_average_ensemble(prob_arrays, weights):
    """
    Weighted mean-ensembles model probability arrays.
    Args:
        prob_arrays: List of np.ndarray [N,]
        weights: List/np.ndarray of weights, shape [num_models,]
    Returns:
        weighted_probs: np.ndarray [N,]
    """
    if not prob_arrays:
        raise ValueError("prob_arrays list is empty!")
    stacked = np.stack(prob_arrays, axis=0)
    weights = np.asarray(weights)
    if len(weights) != stacked.shape[0]:
        raise ValueError(f"weights length {len(weights)} does not match num_models {stacked.shape[0]}")
    weights = weights.reshape(-1, 1)
    return np.sum(stacked * weights, axis=0) / np.sum(weights)

def majority_vote_ensemble(pred_arrays):
    """
    Majority-vote ensembles model predictions (0/1) from multiple models.
    Args:
        pred_arrays: List of np.ndarray [N,] (N = number of samples)
    Returns:
        voted_preds: np.ndarray [N,], dtype int (0/1)
    Notes:
        If tie, rounds up (e.g., 0.5 -> 1).
    """
    if not pred_arrays:
        raise ValueError("pred_arrays list is empty!")
    stacked = np.stack(pred_arrays, axis=0)
    voted = np.round(np.mean(stacked, axis=0))
    return voted.astype(int)
