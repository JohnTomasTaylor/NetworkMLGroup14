from pathlib import Path
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from seiz_eeg.utils import patient_split, resample_label
import numpy as np
from torch.utils.data import Dataset

from src.constants import DATA_ROOT


def load_clips_csv(train=True, resample_labels=False):
    path = "train/segments.parquet" if train else "test/segments.parquet"
    if train and resample_labels:
        return resample_label(pd.read_parquet(DATA_ROOT / path), label=1)
    return pd.read_parquet(DATA_ROOT / path)


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

def get_train_val_dataset(
    tr_ratio_min, 
    tr_ratio_max, 
    seed=1,
    signal_transform=None,
    label_transform=None,
    prefetch=True,
    resample_label=False
):
    clips_df = load_clips_csv(train=True, resample_labels=resample_label)
    clips_tr, clips_val = split_train_val(clips_df,tr_ratio_min, tr_ratio_max, seed)
    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=DATA_ROOT / "train",
        signal_transform=signal_transform,
        label_transform=label_transform,
        prefetch=prefetch,
    )
    dataset_val = EEGDataset(
        clips_val,
        signals_root=DATA_ROOT / "train",
        signal_transform=signal_transform,
        label_transform=label_transform,
        prefetch=prefetch,
    )

    return dataset_tr, dataset_val
    

def get_dummy_dataset(nb_samples, signal_transform=None, label_transform=None, offset=0):
    clips_df = load_clips_csv(train=True, resample_labels=False)
    return EEGDataset(
        clips_df.iloc[offset:offset + nb_samples],
        signals_root=DATA_ROOT / "train",
        signal_transform=signal_transform,
        label_transform=label_transform,
        prefetch=True,
    )


