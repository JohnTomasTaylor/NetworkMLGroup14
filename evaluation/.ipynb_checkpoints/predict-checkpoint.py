# evaluation/predict.py

"""
Inference utilities for EEG ensemble evaluation pipeline.
Supports single-model and multi-model (ensemble) probability prediction.
"""

import torch
import numpy as np
from tqdm import tqdm

def predict_probs(model, dataloader, device=None, return_ids=True, verbose=True):
    """
    Predicts positive-class probabilities for all samples in dataloader using model.
    Args:
        model: torch.nn.Module (should already be eval() mode)
        dataloader: torch.utils.data.DataLoader (test or validation)
        device: torch.device (if None, uses model's device)
        return_ids: bool, whether to also return sample IDs
        verbose: bool, show tqdm progress bar
    Returns:
        probs: np.ndarray, shape [N,], positive class probabilities
        ids: list of sample ids/filenames (if return_ids is True)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_probs = []
    all_ids = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Inference", disable=not verbose)
        for batch in iterator:
            x05 = batch["feat_05"].to(device)
            x1  = batch["feat_1"].to(device)
            x2  = batch["feat_2"].to(device)
            out = model(x05, x1, x2)  # [B, 2]
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            if return_ids:
                # Handles both filename and id field
                if "filename" in batch:
                    all_ids.extend(batch["filename"])
                elif "id" in batch:
                    all_ids.extend(batch["id"])
                else:
                    raise ValueError("No 'filename' or 'id' in batch. "
                                     "Adjust loader/collate_fn or update here.")
    all_probs = np.concatenate(all_probs, axis=0)
    if return_ids:
        return all_probs, all_ids
    else:
        return all_probs

def ensemble_predict(models, dataloader, device=None, return_ids=True, verbose=True):
    """
    Computes ensemble (mean) probabilities over all models.
    Args:
        models: list of torch.nn.Module, each loaded and eval()-mode
        dataloader: DataLoader, as for single prediction
        device: torch.device (applies to all models)
        return_ids: bool, whether to return sample IDs
        verbose: bool, print progress
    Returns:
        mean_probs: np.ndarray [N,]
        ids: list of ids/filenames
    """
    all_model_probs = []
    for i, model in enumerate(models):
        if verbose:
            print(f"Running model {i+1}/{len(models)} for ensemble...")
        probs, ids = predict_probs(model, dataloader, device=device, return_ids=True, verbose=verbose)
        all_model_probs.append(probs)
    mean_probs = np.mean(np.stack(all_model_probs, axis=0), axis=0)
    return mean_probs, ids
