# evaluation/threshold.py

"""
Threshold optimization utilities for EEG ensemble pipeline.
Supports macro-F1 and other metric maximization, and robust application to probabilities.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def search_best_threshold(
    probs, labels, metric='macro-f1', thresholds=None, verbose=False
):
    """
    Finds the threshold (between 0 and 1) that maximizes the chosen metric
    (usually macro-F1) on validation/out-of-fold predictions.
    Args:
        probs: np.ndarray/list, shape (N,), positive-class probabilities
        labels: np.ndarray/list, shape (N,), ground truth (0/1)
        metric: str, 'macro-f1' (default), 'accuracy', 'recall', 'precision'
        thresholds: iterable, search grid; if None, uses np.arange(0.05, 0.95, 0.01)
        verbose: bool, print all intermediate scores
    Returns:
        best_thr: float, threshold with highest metric
        best_score: float, value of best metric
        metrics_dict: dict {threshold: score}
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have same length.")
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)
    metrics_dict = {}
    best_thr, best_score = 0.5, -np.inf  # Allow for negative metrics
    for t in thresholds:
        preds = (probs > t).astype(int)
        if metric == 'macro-f1':
            score = f1_score(labels, preds, average='macro')
        elif metric == 'accuracy':
            score = (preds == labels).mean()
        elif metric == 'recall':
            score = recall_score(labels, preds, average='macro')
        elif metric == 'precision':
            score = precision_score(labels, preds, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        metrics_dict[t] = score
        if score > best_score:
            best_thr, best_score = t, score
        if verbose:
            print(f"Threshold {t:.2f}: {metric} = {score:.4f}")
    return best_thr, best_score, metrics_dict

def apply_threshold(probs, threshold):
    """
    Converts probabilities to binary predictions using a threshold.
    Args:
        probs: np.ndarray/list, shape (N,)
        threshold: float, value between 0 and 1
    Returns:
        preds: np.ndarray (N,), dtype int (0/1)
    """
    return (np.asarray(probs) > threshold).astype(int)
