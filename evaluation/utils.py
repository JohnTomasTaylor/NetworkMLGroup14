# evaluation/utils.py

"""
Utility functions for EEG ensemble evaluation pipeline.
Includes: reproducibility, metric computation, and formatted logging.
"""

import numpy as np
import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def set_seed(seed=1206):
    """
    Sets all random seeds for reproducibility (Python, NumPy, PyTorch).
    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def macro_f1(preds, labels):
    """
    Returns macro F1-score, scaled to 0–100.
    Args:
        preds: np.ndarray/list/tensor, shape (N,)
        labels: np.ndarray/list/tensor, shape (N,)
    Returns:
        float: macro F1 (0-100)
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return f1_score(labels, preds, average='macro') * 100

def macro_precision(preds, labels):
    """
    Returns macro precision, scaled to 0–100.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return precision_score(labels, preds, average='macro') * 100

def macro_recall(preds, labels):
    """
    Returns macro recall, scaled to 0–100.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return recall_score(labels, preds, average='macro') * 100

def accuracy(preds, labels):
    """
    Returns accuracy, scaled to 0–100.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return (preds == labels).mean() * 100

def print_metrics(prefix, preds, labels):
    """
    Prints accuracy, precision, recall, and macro F1, and returns as dict.
    Args:
        prefix: str, prefix for the log line (e.g. 'Fold 1')
        preds, labels: array-like (N,)
    Returns:
        dict with acc, prec, rec, f1 (all 0-100)
    """
    f1 = macro_f1(preds, labels)
    p = macro_precision(preds, labels)
    r = macro_recall(preds, labels)
    acc = accuracy(preds, labels)
    print(f"{prefix} | Acc: {acc:.2f}% | Prec: {p:.2f}% | Rec: {r:.2f}% | Macro-F1: {f1:.2f}%")
    return dict(acc=acc, prec=p, rec=r, f1=f1)

def get_confusion(preds, labels, print_matrix=False):
    """
    Returns (and optionally prints) the confusion matrix.
    Args:
        preds: array-like (N,)
        labels: array-like (N,)
        print_matrix: bool, if True prints the confusion matrix.
    Returns:
        np.ndarray: confusion matrix (shape [n_classes, n_classes])
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = confusion_matrix(labels, preds)
    if print_matrix:
        print("Confusion Matrix (rows: true, cols: pred):\n", cm)
    return cm
