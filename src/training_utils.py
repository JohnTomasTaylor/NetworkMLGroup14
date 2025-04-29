import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import wandb
from seiz_eeg.dataset import EEGDataset


from src.models.lstms import SimpleLSTM

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1)

# check
def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module,
    device: torch.device
) -> Tuple[List[int], List[float], float]:
    """
    Trains the model for one epoch.
    Returns the labels, logits, and the average loss.
    """
    model.train()
    all_labels = []
    all_logits = []
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.float().to(device), labels.unsqueeze(1).float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().flatten().tolist())
        all_logits.extend(outputs.detach().cpu().flatten().tolist())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    train_metrics = compute_metrics(all_labels, all_logits)
    
    return epoch_loss, train_metrics

# check
def validate(dataloader: DataLoader, 
             model: nn.Module, 
             criterion: nn.Module,
             device: torch.device) -> Tuple[List[int], List[float], float]:
    """
    Evaluates the model on a validation set.
    Returns the labels, logits, and the loss.
    """
    model.eval()
    all_labels = []
    all_logits = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.float().to(device), labels.unsqueeze(1).float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().flatten().tolist())
            all_logits.extend(outputs.detach().cpu().flatten().tolist())

    val_loss = running_loss / len(dataloader.dataset)
    val_metrics = compute_metrics(all_labels, all_logits)

    return val_loss, val_metrics


def compute_metrics(labels: List[int], logits: List[float], is_binary: bool = True) -> Dict[str, float]:
    """
    Calculates metrics from labels and logits
    """
    if is_binary:
        preds = (np.array(logits) > 0.0).astype(int)
    else:
        preds = np.argmax(np.array(logits), axis=1)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
    }
    
    return metrics


sweep_config = {
    'method': 'bayes',  # ou 'grid', 'random'
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'value': 5
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1,
            'distribution': 'log_uniform'
        },
        'batch_size': {
            'values': [1, 2, 4]
        },
        'hidden_dim': {
            'values': [2, 4, 8]
        },
        'optimizer': {
            'values': ["adam", "sgd"]
        }
    }
}


def create_train_fn(build_dataset_fn, build_network_fn, build_optimizer_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(config=None):
        # Initialize a new wandb run
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config

            tr_loader, val_loader = build_dataset_fn(config.batch_size)
            network = build_network_fn(config)
            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = build_optimizer_fn(network, config.optimizer, config.learning_rate)

            for epoch in range(config.epochs):
                train_loss, train_metrics = train_epoch(tr_loader, network, optimizer, criterion, device)
                val_loss, val_metrics = validate(val_loader, network, criterion, device)

                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
    return train

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer