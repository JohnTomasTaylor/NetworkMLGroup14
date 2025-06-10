import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class EEGMultiWinDataset(Dataset):
    def __init__(self, segment_df, feature_dirs_root):
        """
        Args:
            segment_df (pd.DataFrame): Two possible formats:
                1) Training/Validation: contains columns like "patient", "session", "segment", "label"
                2) Inference: contains only the "filename" column (e.g., "p001_s002_seg00.npy")
            feature_dirs_root (str or Path): Path to the root of the features_multiwin directory.
                The expected directory structure is:
                    ├── 0.5s/
                    ├── 1s/
                    ├── 2s/
                    ├── 0.5s_mean.npy
                    ├── 0.5s_std.npy
                    ├── 1s_mean.npy
                    ├── 1s_std.npy
                    ├── 2s_mean.npy
                    └── 2s_std.npy
        """
        self.df = segment_df.reset_index(drop=True)
        feature_dirs_root = Path(feature_dirs_root)

        # Set up directories for each time window scale
        self.feature_dirs = {
            "0.5s": feature_dirs_root / "0.5s",
            "1s" : feature_dirs_root / "1s",
            "2s" : feature_dirs_root / "2s"
        }

        # Load normalization stats (mean and std) for each scale
        self.norm_stats = {}
        for tag in ["0.5s", "1s", "2s"]:
            mean_path = feature_dirs_root / f"{tag}_mean.npy"
            std_path  = feature_dirs_root / f"{tag}_std.npy"
            self.norm_stats[tag] = {
                "mean": np.load(mean_path),  # shape: [C, F]
                "std" : np.load(std_path)    # shape: [C, F]
            }

        # If no "patient" column, treat this as inference mode
        self.is_inference = "patient" not in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if not self.is_inference:
            # —— Training/Validation mode ——
            patient = row["patient"]
            session = row["session"]
            segment = int(row["segment"])
            label   = int(row["label"])
            fname   = f"{patient}_{session}_seg{segment:02d}.npy"
        else:
            # —— Inference mode: filename only, parse patient/session/segment, label is dummy ——
            fname = row["filename"]
            base = fname.replace(".npy", "")
            parts = base.split("_")
            patient = parts[0]
            session = parts[1]
            segment = int(parts[-1].replace("seg", ""))
            label = -1  # No ground truth label during inference

        feats = {}
        for tag in ["0.5s", "1s", "2s"]:
            fp = self.feature_dirs[tag] / fname
            arr = np.load(fp)               # shape: [T, C, F]
            arr = arr.transpose(1, 0, 2)    # → shape: [C, T, F]

            mean = self.norm_stats[tag]["mean"]  # [C, F]
            std  = self.norm_stats[tag]["std"]   # [C, F]
            arr = (arr - mean[:, None, :]) / (std[:, None, :] + 1e-6)

            feats[f"feat_{tag}"] = torch.from_numpy(arr).float()

        sample = {
            "feat_05": feats["feat_0.5s"],  # [C, T_05, F_05]
            "feat_1" : feats["feat_1s"],   # [C, T_1,  F_1 ]
            "feat_2" : feats["feat_2s"],   # [C, T_2,  F_2 ]
            "label"  : label               # 0/1 for training, -1 for inference
        }

        # Add filename to sample if in inference mode
        if self.is_inference:
            sample["filename"] = fname

        return sample


def collate_fn_multiscale(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of sample dictionaries. Each dict contains:
            - "feat_05": Tensor of shape [C, T_05, F_05]
            - "feat_1" : Tensor of shape [C, T_1,  F_1 ]
            - "feat_2" : Tensor of shape [C, T_2,  F_2 ]
            - "label"  : int
            - [Optional] "filename": str (only in inference mode)

    Returns:
        A dictionary with stacked tensors:
            - "feat_05": Tensor of shape [B, C, T_05, F_05]
            - "feat_1" : Tensor of shape [B, C, T_1,  F_1 ]
            - "feat_2" : Tensor of shape [B, C, T_2,  F_2 ]
            - "label"  : Tensor of shape [B]
            - "filename" (optional): List of strings, length B
    """
    feats05 = [item["feat_05"] for item in batch]
    feats1  = [item["feat_1"] for item in batch]
    feats2  = [item["feat_2"] for item in batch]
    labels  = [item["label"] for item in batch]

    batch_feats05 = torch.stack(feats05, dim=0)
    batch_feats1  = torch.stack(feats1,  dim=0)
    batch_feats2  = torch.stack(feats2,  dim=0)
    batch_labels  = torch.tensor(labels, dtype=torch.long)

    out = {
        "feat_05": batch_feats05,
        "feat_1" : batch_feats1,
        "feat_2" : batch_feats2,
        "label"  : batch_labels
    }

    # Include filenames if present (in inference mode)
    if "filename" in batch[0]:
        fnames = [item["filename"] for item in batch]
        out["filename"] = fnames

    return out
