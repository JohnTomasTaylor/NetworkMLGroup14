from pathlib import Path
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from seiz_eeg.utils import patient_split, resample_label
import numpy as np
from torch.utils.data import Dataset

data_path = "/home/ogut/data"

DATA_ROOT = Path(data_path)

# Check
def load_clips_csv(train=True, resample_labels=False):
    path = "train/segments.parquet" if train else "test/segments.parquet"
    if train and resample_labels:
        return resample_label(pd.read_parquet(DATA_ROOT / path), label=1)
    return pd.read_parquet(DATA_ROOT / path)

# Check
def split_train_val(clips_df, tr_ratio_min, tr_ratio_max, seed=1, print_res=True):
    """
    Cuts up the dataset into train/val by patients by respecting as much as possible 
    that the ratio of segement present in the train set is between tr_ratio_min, tr_ratio_max
    """
    clips_df = clips_df.reset_index()
    tr_patients = patient_split(clips_df, tr_ratio_min, tr_ratio_max, seed=seed)
    clips_training = clips_df[clips_df['patient'].isin(tr_patients)].set_index(["patient", "session", "segment"])
    clips_val = clips_df[~clips_df['patient'].isin(tr_patients)].set_index(["patient", "session", "segment"])

    if print_res:
        print(f"Length of training data: {len(clips_training)}")
        print(f"Length of validation data: {len(clips_val)}")
        print(f"Ratio training/total: {len(clips_training)/(len(clips_training)+len(clips_val))}")

    return clips_training, clips_val

def load_dataset(
        clips_df, 
        train=True, 
        signal_transform=None, 
        label_transform=None,
        prefetch=True,

    ):
    return EEGDataset(
        clips_df,
        signals_root=DATA_ROOT / "train" if train else DATA_ROOT / "test",
        signal_transform=signal_transform,
        label_transform=label_transform,
        prefetch=prefetch,  # If your compute does not allow it, you can use `prefetch=False`
    )

# TODO
def k_fold_datasets(
    dataset: EEGDataset, 
    fold_idx: int, 
    n_splits: int, 
    batch_size: int, 
    num_workers: int = 4,
    stratified: bool = True,
    seed: int = 42
):
    pass
    
    
# Check
def get_dummy_dataset(nb_samples, signal_transform=None, label_transform=None):
    clips_df = load_clips_csv(train=True, resample_labels=False)
    # clips_df.reset_index(inplace=True)
    print(clips_df.iloc[:nb_samples])
    return load_dataset(
        clips_df.iloc[:nb_samples], 
        train=True, 
        signal_transform=signal_transform, 
        label_transform=label_transform,
        prefetch=True,
    )