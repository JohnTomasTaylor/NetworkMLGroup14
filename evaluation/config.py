# evaluation/config.py

"""
Configuration and global constants for EEG ensemble evaluation pipeline.
Modify this file for path, device, and experiment settings.
"""

import os
from pathlib import Path
import torch

# ==== Data paths (Colab-READY) ====
DATA_ROOT = Path("/content/networkML/train_dataset")
META_PATH = DATA_ROOT / "train_segments.parquet"
FEATURE_ROOT = DATA_ROOT / "train_dataset/features_multiwin"

# For test/inference
TEST_ROOT = Path("/content/networkML/test_dataset")
TEST_META_PATH = TEST_ROOT / "test_filenames.csv"
TEST_FEATURE_ROOT = TEST_ROOT / "test_dataset/features_multiwin"

# Edge index (for fixed graph, if needed)
EDGE_INDEX_PATH = Path("/content/networkML/code/GCN_LSTM_Pro/edge_index_knn.pt")

# ==== Model paths ====
MODEL_DIR = Path("/content/networkML/code/GCN_LSTM_Pro")
# List your models explicitly if not using 5 folds
ENSEMBLE_MODEL_PATHS = [
    MODEL_DIR / "best_model_Pro.pth",
    MODEL_DIR / "best_model_Pro2.pth"
    # Add more as needed
]

LOG_DIR = Path("/content/networkML/logs")
RESULTS_DIR = Path("/content/networkML/results")
EXPERIMENT_NAME = "colab_experiment"

# ==== Feature/Model dimensions ====
IN_FEATS = {"F05": 145, "F1": 208, "F2": 210}
NUM_CHANNELS = 19

# ==== Hyperparameters ====
BATCH_SIZE = 64
NUM_WORKERS = 4

# ==== Device ====
if os.environ.get("EEG_FORCE_CPU", "0") == "1":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Misc ====
SEED = 1206  # For reproducibility
