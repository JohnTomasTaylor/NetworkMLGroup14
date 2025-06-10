# evaluation/config.py

"""
Configuration and global constants for EEG ensemble evaluation pipeline.
Modify this file for path, device, and experiment settings.
"""

import os
from pathlib import Path
import torch

# ==== Data paths (edit as needed, or use env vars for cloud/Colab) ====
DATA_ROOT = Path(os.environ.get("EEG_DATA_ROOT", "D:/Documents/nml_data/data/train_dataset")).resolve()
META_PATH = DATA_ROOT / "train_segments.parquet"
FEATURE_ROOT = DATA_ROOT / "features_multiwin"
EDGE_INDEX_PATH = Path(os.environ.get("EEG_EDGE_INDEX_PATH", "D:/Documents/nml_data/edge_index/edge_index_knn.pt")).resolve()

# For test/inference
TEST_META_PATH = Path(os.environ.get("EEG_TEST_META_PATH", "D:/Documents/nml_data/data/test_segments.csv")).resolve()

# ==== Model/Experiment paths ====
MODEL_DIR = Path(os.environ.get("EEG_MODEL_DIR", "models")).resolve()
ENSEMBLE_MODEL_PATHS = [
    MODEL_DIR / f"best_model_fold{i}.pth" for i in range(5)
]

LOG_DIR = Path(os.environ.get("EEG_LOG_DIR", "logs")).resolve()
RESULTS_DIR = Path(os.environ.get("EEG_RESULTS_DIR", "results")).resolve()
EXPERIMENT_NAME = os.environ.get("EEG_EXPERIMENT_NAME", "default_experiment")

# ==== Feature/Model dimensions ====
IN_FEATS = {"F05": 145, "F1": 208, "F2": 210}
NUM_CHANNELS = 19

# ==== Hyperparameters ====
BATCH_SIZE = 64
NUM_WORKERS = 4

# ==== Device ====
# You can force CPU by setting EEG_FORCE_CPU=1 as an env var.
if os.environ.get("EEG_FORCE_CPU", "0") == "1":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Misc ====
SEED = 1206  # For reproducibility
