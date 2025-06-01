# File: dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch

class EEGFeatureDataset(Dataset):
    def __init__(self, metadata_path=None, feature_dir=None, dataframe=None, transform=None, mean_path=None, std_path=None):
        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True)
        else:
            self.df = pd.read_parquet(metadata_path)
        self.feature_dir = Path(feature_dir)

        assert mean_path is not None and std_path is not None, \
            "mean_path and std_path must be provided for normalization"
        self.mean = np.load(mean_path)   # shape: (393,)
        self.std  = np.load(std_path)    # shape: (393,)
        self.has_label = "label" in self.df.columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self.feature_dir / row["filename"]
        x = np.load(file_path)           # shape: [T, C, F]
        x = x.transpose(1, 0, 2)         # shape: [C, T, F]

        if self.transform:
            x = self.transform(x)

        # Normalize using training set statistics (train_mean, train_std)
        x = (x - self.mean) / self.std

        x = torch.tensor(x, dtype=torch.float32)

        if self.has_label:
            y = torch.tensor(row["label"], dtype=torch.long)
            return x, y
        else:
            # During testing: also return filename for later identification
            return x, row["filename"]
