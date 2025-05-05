# EEG Seizure Detection - Model Sweep Framework

## Setup
```bash
pip install -r requirements.txt
pip install git+https://github.com/LTS4/seizure_eeg.git
```

## Quick Start

```python
# 1. Preload datasets (preload if loading is slow)
dataset_tr, dataset_val = get_train_val_dataset(
    tr_ratio_min=0.8,
    tr_ratio_max=0.85,
    seed=seed,
    signal_transform=fft_filtering,
    label_transform=None,
    prefetch=True,
    resample_label=False
)

# 2. Create dataloader function
build_dataloaders_fn = create_build_dataloaders(dataset_tr, dataset_val)

# 3. Run the sweep
sweep_id = run_sweep(
    project_name=wandb_project_name,        # W&B project name for logging
    config_path=config_file_path,           # Path to YAML config file
    seed=seed,                              # Random seed for reproducibility
    sweep_run_count=count,                  # Number of runs to execute
    checkpoint_freq=checkpoint_freq,        # Save frequency (epochs), None = final only
    model_class=SimpleLSTM,                 # Your model class 
    build_dataloaders_fn=build_dataloaders_fn,  # Creates train/val dataloaders
    build_optimizer_fn=build_optimizer,     # Configures optimizer from config
    train_epoch_fn=train_epoch,             # Handles training epoch
    validate_fn=validate                    # Performs validation
)
```

## Implementation Steps

1. **Define your model**: Create a model class with clear initialization parameters

2. **Configure datasets**: Prepare your dataset using provided functions or customize your own.
You can customize the `build_dataloaders_fn` to handle additional parameters from your config file (e.g., batch size, split ratio, feature selection).

3. **Define the optimizer** (optional): Map configuration parameters to optimizer settings

4. **Customize training** (optional): Override default training/validation functions for special cases (I think GNNs might need a different input)

5. **Create sweep configuration**: Define parameters in YAML matching your model's `__init__` parameters

## W&B Configuration Format

### Config File Structure
```yaml
method: random  # Search strategy (random, grid, bayes)
metric:
    name: val_f1  # Metric to optimize
    goal: maximize  # Optimization direction

parameters:
    # Fixed parameters
    epochs:
        value: 150
        
    # Discrete search spaces
    hidden_dim:
        values: [64, 128, 256, 512]
    num_layers:
        values: [1, 2, 3]
        
    # Continuous parameters
    learning_rate:
        value: 0.001  # Fixed value
        # min: 0.0005  # Or define a range
        # max: 0.05    
        # distribution: log_uniform_values
```

### Runtime Configuration
During execution, W&B flattens this structure into a simple dictionary:

```python
{
    "epochs": 150,
    "hidden_dim": 64,  # Selected from range
    "num_layers": 2,   # Selected from range
    "learning_rate": 0.001,
    # ...other parameters
}
```

Important:

- Parameter names must match your model's __init__ parameters exactly
- ALL parameters in your model's __init__ must be present in the config file (even with fixed values). Missing parameters will cause errors during model instantiation

## Project Structure

```
/configs         # YAML configuration files
    config.yaml
/scitas          # Task scripts
    task.run
/src             # Source code
    /model       # Model definitions
    ...          # Training and data loading utilities
/model_weights   # Saved model checkpoints
    /wandb_run_id        # Example: model_weights/1b25n5jl
        /checkpoints     # If checkpoint_freq was specified
            epoch_10.pt
            epoch_20.pt
        final_checkpoint.pt
        config.yaml
```

## Running on Izar (SCITAS)

To run your sweeps on the SCITAS Izar cluster:

1. Check the example files in the `/scitas` directory
2. Submit your job:
   ```bash
   sbatch scitas/your_scitas_run_file
   ```

3. Monitor job status:
   ```bash
   Squeue
   ```

4. For debugging, use interactive mode:
   ```bash
   Sinteract -g gpu:1 -c10 -t 2:0:0 -m 32G
   ```
   This requests 1 GPU, 10 CPU cores, 2 hours of runtime, and 32GB of memory.

## Working with Checkpoints

Models are saved based on the `checkpoint_freq` parameter:
- If specified: saves at specified epoch intervals in `/checkpoints` subfolder
- If `None`: saves only the final model as `final_checkpoint.pt`

### Loading a Model
```python
from src.training_utils import load_checkpoint

model, optimizer, config = load_checkpoint(
    checkpoint_path,    # Path to .pt file
    run_config_path,    # Path to config.yaml
    model_class         # Your model class
)
```

## Examples
See `test_sweep.py` and `sweep_transformer.py` for complete working examples.
