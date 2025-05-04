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
import tqdm
import wandb
from seiz_eeg.dataset import EEGDataset
import yaml


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

# TODO: Add wandb config file to add more options for training, clip_gradients, ...
def train_epoch(
    dataloader: DataLoader, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module,
    device: torch.device
) -> Tuple[List[int], List[float], float]:
    """
    Trains the model for one epoch using the provided dataloader, optimizer, and loss criterion.
    Args:
        dataloader (DataLoader): DataLoader providing the training data in batches.
        model (nn.Module): The neural network model to be trained.
        optimizer (optim.Optimizer): Optimizer used to update the model parameters.
        criterion (nn.Module): Loss function used to compute the training loss.
        device (torch.device): Device on which the computation will be performed (e.g., 'cpu' or 'cuda').
    Returns:
        avg_loss (float): The average loss for the epoch.
        
        train_metrics (dict): A dictionary containing computed training metrics (e.g., accuracy, precision, recall).
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

def validate(dataloader: DataLoader, 
            
             model: nn.Module, 
             criterion: nn.Module,
             device: torch.device) -> Tuple[List[int], List[float], float]:
    """
    Validates a given model on a provided dataloader using the specified loss criterion.
    Args:
        dataloader (DataLoader): The DataLoader providing the validation dataset.
        model (nn.Module): The neural network model to validate.
        criterion (nn.Module): The loss function used to compute the validation loss.
        device (torch.device): The device (CPU or GPU) to perform computations on.
    Returns:
        val_loss (float): The average validation loss over the dataset.

        val_metrics (Dict[str, float]): A dictionary containing computed validation 
        metrics such as accuracy, precision, recall, etc.
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
    Computes evaluation metrics based on the provided labels and logits.

    Metrics calculated:
    - Accuracy: The ratio of correctly predicted instances to the total instances.
    - Precision: The weighted precision score, accounting for class imbalance.
    - Macro-F1: The F1 score averaged across all classes, treating each class equally.
    - Recall: The weighted recall score, accounting for class imbalance.

    Args:
        labels (List[int]): Ground truth labels.
        logits (List[float]): Predicted logits or probabilities.
        is_binary (bool): Whether the task is binary classification. Defaults to True.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
    """
    if is_binary:
        preds = (np.array(logits) > 0.0).astype(int)
    else:
        preds = np.argmax(np.array(logits), axis=1)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0.0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0.0),
    }
    
    return metrics


def create_train_fn(
        model_class,
        build_dataloaders_fn, 
        build_optimizer_fn,
        train_epoch_fn,
        validate_fn,
        checkpoint_freq=None,
    ):
    """
    Generates a `train` function for wandb integration, handling dataset setup, model 
    initialization, training, validation, and logging. 

    Args:
        model_class (Type[nn.Module]): The actual model class to be instantiated.
            This should be a class reference (e.g., MyModel), not a string.
        build_dataloaders_fn (callable): A function that takes a `config` dictionary and 
            returns a tuple of PyTorch DataLoaders (train_loader, val_loader)
        build_optimizer_fn (callable): A function that takes a model, an optimizer name, 
            and a learning rate, and returns an optimizer instance.
        train_epoch (callable): A function that takes (train_loader, model, optimizer, 
            criterion, device) and performs one training epoch, returning 
            (train_loss, train_metrics).
        validate (callable): A function that takes (val_loader, model, criterion, device) 
            and performs validation, returning (val_loss, val_metrics).
        checkpoint_freq (int): The number of epochs between each checkpoint. 
            If None only final weights of the run are saved

    Returns:
        callable: A `train` function for wandb sweeps, logging metrics per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(config=None):
        # Initialize a new wandb run
        with wandb.init(config=config) as run:
            # Transform wandb config to dict so access always uses dict syntax
            config = dict(wandb.config)

            tr_loader, val_loader = build_dataloaders_fn(config)
            model = build_model(model_class, config).to(device)
            # TODO: Add criterion as a param
            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = build_optimizer_fn(model, config["optimizer"], config["learning_rate"])

            # Prepare the checkpoin directories
            create_checkpoint_dirs(run.id, config, model, checkpoints=bool(checkpoint_freq))

            for epoch in tqdm.tqdm(range(config["epochs"])):
                train_loss, train_metrics = train_epoch_fn(tr_loader, model, optimizer, criterion, device)
                val_loss, val_metrics = validate_fn(val_loader, model, criterion, device)

                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })

                if checkpoint_freq and (epoch + 1) % checkpoint_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, get_checkpoint_model_path(run.id, epoch))


            # Save final version of model and optimizer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, get_final_model_path(run.id))


    return train

def get_final_model_path(run_id, base_path="model_weights"):
    return Path(base_path) / run_id / "final_checkpoint.pt"

def get_checkpoint_model_path(run_id, epoch, base_path="model_weights"):
    return Path(base_path) / run_id / "checkpoints" / f"epoch_{epoch + 1}.pt"

def create_checkpoint_dirs(run_id, config, model, base_path="model_weights", checkpoints=True):
    """
    Creates the following dir structure
    /base_path
        /run_id
            /checkpoints
            config (the saved hyperparamters configs of the current run) is an instance of wandb.config
    """
    run_dir = Path(base_path) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if checkpoints:
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
    # Save the config as a YAML file with model class name
    config['model_class'] = str(model.__class__.__name__)
    print(config)
    print(type(config))
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def load_checkpoint(checkpoint_path, run_config_path, model_class):
    """
    Loads a checkpoint and its configuration to CPU.

    Args:
        checkpoint_path (str or Path): Path to the saved model checkpoint (.pt file).
        run_config_path (str or Path): Path to the YAML configuration file for the run.
        model_class (class name): The model class that needs to be loaded ex: SimpleLSTM

    Returns:
        model (nn.Module): The model with loaded state_dict.
        optimizer (torch.optim.Optimizer): The optimizer with loaded state_dict.
        config (dict): The configuration dictionary.
    """
    # Load config
    with open(run_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dynamically load model class based on saved config
    model = build_model(model_class, config)
    optimizer = build_optimizer(model, config["optimizer"], config["learning_rate"])

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, config


def build_model(model_class, config: dict):
    """
    Instantiates a model from the given model class using parameters specified in the config dictionary.
    The model parameters are dynamically inferred from the model_class signature
    Args:
        model_class (Type[nn.Module]): The actual model class to be instantiated.
            This should be a class reference (e.g., MyModel), not a string.
        config (dict): A dictionary containing configuration parameters.
            Only keys that match parameter names in the model_class constructor will be used.
    Returns:
        An instantiated model object of the specified model_class.
    """
    
    import inspect
    constructor_params = inspect.signature(model_class).parameters
    model_kwargs = {k: config[k] for k in constructor_params if k in config}
    return model_class(**model_kwargs)


def build_optimizer(model, optimizer, learning_rate):
    """
    Builds the optimizer for the run according to the wandb configs
    """
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    return optimizer

def create_build_dataloaders(dataset_tr, dataset_val):
    """
    Creates a function to build training and validation dataloaders. 
    Useful if loading datasets is slow, so you load them before a sweep.

    Args:
        dataset_tr (Dataset): The training dataset.
        dataset_val (Dataset): The validation dataset.
    Returns:
        function: A function that takes a configuration object with a batch_size attribute 
                  and returns the training and validation DataLoader instances.
    """

    def build_dataset(config):
        loader_tr = DataLoader(dataset_tr, batch_size=config["batch_size"], shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)
        return loader_tr, loader_val
    
    return build_dataset


def run_sweep(
    project_name: str,
    config_path: str,
    seed: int,
    sweep_run_count: int,
    checkpoint_freq: int,
    model_class,
    build_dataloaders_fn,
    build_optimizer_fn,
    train_epoch_fn=train_epoch,
    validate_fn=validate,
):
    """
    Set up and execute a hyperparameter sweep using Weights & Biases.
    
    Args:
        project_name: Name of the WandB project to log results to
        config_path: Path to the YAML configuration file containing sweep parameters
        seed: Random seed for reproducibility
        sweep_run_count: Number of runs to execute during the sweep
        model_class (Type[nn.Module]): The actual model class to be instantiated.
            This should be a class reference (e.g., MyModel), not a string.
        build_dataloaders_fn: Function to construct training and validation data loaders
        build_optimizer_fn: Function to construct the optimizer
        train_epoch_fn: Function to execute training for a single epoch
        validate_fn: Function to validate model performance
    
    Returns:
        The sweep ID generated by WandB
    """
    with open(config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
    
    seed_everything(seed)
    
    # Create the training function with the provided components
    train_fn = create_train_fn(
        model_class=model_class,
        build_dataloaders_fn=build_dataloaders_fn,
        build_optimizer_fn=build_optimizer_fn,
        train_epoch_fn=train_epoch_fn,
        validate_fn=validate_fn,
        checkpoint_freq=checkpoint_freq
    )
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train_fn, count=sweep_run_count)
    return sweep_id
