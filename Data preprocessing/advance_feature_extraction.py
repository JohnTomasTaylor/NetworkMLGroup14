#!/usr/bin/env python3
"""
advance_feature_extraction.py

Multi-scale EEG feature extraction (0.5s, 1s, 2s windows) with GPU acceleration
for time-domain + PSD and CPU parallel CWT for each scale. 
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from joblib import Parallel, delayed

# ─── Global Configuration ───
fs      = 250               # Sampling rate: 250 Hz
seg_len = fs * 12           # Segment length: 12 seconds = 3000 samples
scales  = np.arange(1, 64)  # CWT scales
bands   = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta' : (13, 30),
    'gamma': (30, 45),
}

# ─── Band-Pass Filter (0.5–30 Hz) ───
bp_sos = signal.butter(4, (0.5, 30), btype='bandpass', output='sos', fs=fs)

def time_filter(x: np.ndarray) -> np.ndarray:
    """
     Applies 0.5–30 Hz band-pass filter along time axis
    """
    return signal.sosfiltfilt(bp_sos, x, axis=0)

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per channel
    """
    mu  = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1e-6
    return (x - mu) / std

def build_band_masks(win_len: int):
    """
    # Build frequency band masks for PSD and CWT
    # - psd_masks: to extract band powers from FFT
    # - cwt_masks: to extract band powers from CWT energy matrix
    """
    # - psd_masks: to extract band powers from FFT
    freqs = np.fft.rfftfreq(win_len, d=1/fs)
    psd_masks = {
        name: ((freqs >= lo) & (freqs <= hi))
        for name, (lo, hi) in bands.items()
    }
    # - cwt_masks: to extract band powers from CWT energy matrix
    cwt_freqs = pywt.scale2frequency('morl', scales) * fs  # [n_scale]
    cwt_masks = {
        name: ((cwt_freqs >= lo) & (cwt_freqs <= hi))
        for name, (lo, hi) in bands.items()
    }
    return psd_masks, cwt_masks

# ─── Time-domain + PSD feature extraction on GPU ───
def compute_features_gpu(windows: torch.Tensor, win_len: int, psd_masks):
    """
    windows: torch.Tensor, shape [n_win, win_len, n_ch]
    Returns:
    - time_feats: [n_win, n_ch, 9] time-domain statistics
    - psd_full:   [n_win, n_ch, ≤128] full PSD spectrum
    - psd_band:   [n_win, n_ch, 5] band-wise PSD energy
    """
    n_win, _, n_ch = windows.shape

    # ——— Time-domain features ———
    xavg =     windows.mean(dim=1)                               # Mean
    xarv =    windows.abs().mean(dim=1)                          # Absolute Mean
    xstd =     windows.std(dim=1)                                # Standard Deviation
    xp_p =     windows.max(dim=1).values - windows.min(dim=1).values  # Peak-to-Peak
    zeros = ((windows[:, :-1, :] * windows[:, 1:, :]) < 0).sum(dim=1) / (win_len - 1)  # Zero-Crossing Rate

    # Normalize to unit standard deviation
    mean_expand = xavg.unsqueeze(1)   # [n_win, 1, n_ch]
    std_expand  = xstd.unsqueeze(1)   # [n_win, 1, n_ch]
    normed = (windows - mean_expand) / (std_expand + 1e-6)

    xskew = normed.pow(3).mean(dim=1) - 0          # skewness
    xkurt = normed.pow(4).mean(dim=1) - 3          # kurtosis
    xrms  = torch.sqrt((windows.pow(2)).mean(dim=1))  # RMS
    xdiff = windows[:, 1:, :] - windows[:, :-1, :]
    ll    = xdiff.abs().sum(dim=1)                  # Line Length

    time_feats = torch.stack([xavg, xarv, xstd, xp_p, zeros, xskew, xkurt, xrms, ll], dim=-1)
    # time_feats: [n_win, n_ch, 9]

    # --- PSD features via GPU FFT ---
    # Reshape to [n_win*n_ch, win_len]
    flat = windows.permute(0, 2, 1).reshape(-1, win_len)  # [n_win*n_ch, win_len]
    freqs_complex = torch.fft.rfft(flat, dim=1)          # [n_win*n_ch, win_len/2+1]
    psd = (freqs_complex.abs() ** 2) / win_len            # [n_win*n_ch, n_freq]
    # Reshape to [n_win, n_ch, n_freq]
    psd = psd.reshape(n_win, n_ch, -1)                   # [n_win, n_ch, n_freq]
    n_freq = psd.shape[2]
    # Truncate to first min(128, n_freq) points
    psd_full = psd[:, :, :min(128, n_freq)]              # [n_win, n_ch, ≤128]

    # Band-wise PSD energy using numpy for boolean masking
    psd_np = psd.cpu().numpy()  # move to CPU for numpy operations
    band_list = []
    for mask in psd_masks.values():
        if mask.any():
            # [n_win, n_ch]
            band_vals = psd_np[:, :, mask].mean(axis=2)
        else:
            band_vals = np.zeros((n_win, n_ch))
        band_list.append(torch.from_numpy(band_vals).to(windows.device))
    psd_band = torch.stack(band_list, dim=-1)  # [n_win, n_ch, 5]

    return time_feats, psd_full, psd_band

# ─── CWT energy computation on CPU ───
def _cwt_energy_single(sig: np.ndarray) -> np.ndarray:
    """
    Applies CWT to a single-channel signal of length win_len.
    Returns summed energy along time axis → [n_scale]
    """
    cwtm, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1/fs)
    return np.sum(np.abs(cwtm) ** 2, axis=1)  # [n_scale]

def compute_cwt_energy_cpu(windows: np.ndarray, win_len: int, n_jobs: int = 1) -> np.ndarray:
    """
    windows: numpy array, shape [n_win, win_len, n_ch]
    return: energy: [n_win, n_ch, n_scale]
    """
    n_win, _, n_ch = windows.shape
    # Reshape to [n_win*n_ch, win_len]
    arr = windows.transpose(0, 2, 1).reshape(-1, win_len)
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            energies = list(exe.map(_cwt_energy_single, arr))
    else:
        energies = list(map(_cwt_energy_single, arr))
    energies = np.stack(energies, axis=0)            # [n_win*n_ch, n_scale]
    return energies.reshape(n_win, n_ch, -1)         # [n_win, n_ch, n_scale]

def compute_cwt_band_feats(energy: np.ndarray, cwt_masks) -> np.ndarray:
    """
    Compute average CWT energy for each frequency band.

    Parameters:
        energy (np.ndarray): Shape [n_win, n_ch, n_scale], the CWT energy matrix.
        cwt_masks (dict): Dictionary of boolean masks for each frequency band.

    Returns:
        np.ndarray: Shape [n_win, n_ch, 5], the average energy in each of the five bands.
    """
    band_feats = []
    for mask in cwt_masks.values():
        if mask.any():
            band_feats.append(energy[:, :, mask].mean(axis=2))
        else:
            band_feats.append(np.zeros((energy.shape[0], energy.shape[1])))
    return np.stack(band_feats, axis=-1)

# ─── Feature extraction at a single window scale ───
def extract_feats_for_window(seg: np.ndarray, win_len: int, step: int,
                             cwt_jobs: int = 1, device: str = 'cuda') -> np.ndarray:
    """
    Performs sliding-window on a single segment [seg_len, n_ch],
    extracts time-domain + PSD on GPU, and CWT on CPU.
    Returns: [n_win, n_ch, F] feature array
    """
    # 1) Build masks
    psd_masks, cwt_masks = build_band_masks(win_len)

    # 2) Generate windows: [n_win, win_len, n_ch]
    windows_np = sliding_window_view(seg, (win_len, seg.shape[1]))[::step, 0, :, :]

    n_win, _, n_ch = windows_np.shape

    # 3) Move windows to GPU
    windows = torch.from_numpy(windows_np).float().to(device)

    # 4) Extract time-domain and PSD features on GPU
    time_feats, psd_full, psd_band = compute_features_gpu(windows, win_len, psd_masks)

    # 5) Extract CWT energy on CPU
    energy = compute_cwt_energy_cpu(windows_np, win_len, n_jobs=cwt_jobs)  # [n_win, n_ch, n_scale]
    cwt_band = compute_cwt_band_feats(energy, cwt_masks)                   # [n_win, n_ch, 5] (numpy)

    # 6) Move CWT results to GPU and concatenate
    energy_t   = torch.from_numpy(energy).float().to(device)   # [n_win, n_ch, n_scale]
    cwt_band_t = torch.from_numpy(cwt_band).float().to(device) # [n_win, n_ch, 5]

    # 7) Concatenate all features: [time (9) + psd_full + psd_band (5) + energy (n_scale) + cwt_band (5)]
    feats = torch.cat([time_feats, psd_full, psd_band, energy_t, cwt_band_t], dim=-1)  # [n_win, n_ch, F]
    return feats.cpu().numpy()

# ─── Multi-scale feature extraction ───
def extract_feats_multiwindow(seg: np.ndarray, cwt_jobs: int = 1, device: str = 'cuda'):
    """
    Extracts features from same segment using three window sizes:
    - 0.5s, 1s, 2s → returns [feat_0.5s, feat_1s, feat_2s]
    Each feat_i has shape [n_win_i, n_ch, F_i]
    """
    configs = [
        (int(0.5 * fs), int(0.25 * fs)),  # 0.5s window, 0.25s step
        (fs, fs),                         # 1s window, 1s step
        (fs * 2, fs)                      # 2s window, 1s step
    ]
    feats_all = []
    for win_len, step in configs:
        feats = extract_feats_for_window(seg, win_len, step, cwt_jobs=cwt_jobs, device=device)
        feats_all.append(feats)
    return feats_all

# ─── Single segment processing function (called in parallel) ───
def process_segment(row, DATA_ROOT, CWT_JOBS, window_tags):
    """
    Given one DataFrame row with index (patient, session, seg_id),
    performs:
    - Read signals.parquet
    - Extract 12s segment
    - Band-pass filter + normalize
    - Multi-window feature extraction
    - Save results to corresponding output folders
    """
    # row.name = (patient, session, seg_id)
    patient, session, seg_id = row.name
    # 1) Read raw EEG signal file
    sig_path = DATA_ROOT / "raw" / row['split'] / row["signals_path"]
    sig = pd.read_parquet(sig_path).values  # [, n_ch]
    start = int(row["start_time"] * fs)
    seg = sig[start : start + seg_len]
    # If too short, skip
    if seg.shape[0] < seg_len:
        return

    # 2) Filter + Normalize
    seg = time_filter(seg)
    seg = normalize(seg)

    # 3) Multi-scale feature extraction
    feats_list = extract_feats_multiwindow(seg, cwt_jobs=CWT_JOBS, device='cuda')
    fname = f"{patient}_{session}_seg{int(seg_id):02d}.npy"

    # 4) Save to folders based on window size
    out_base = DATA_ROOT / f"{row['split']}_dataset" / "features_multiwin"
    for tag, feats in zip(window_tags, feats_list):
        out_dir = out_base / tag
        np.save(out_dir / fname, feats)

# ─── Script Entry Point ───
if __name__ == "__main__":
    DATA_ROOT = Path(r"D:\Documents\nml_data")
    CWT_JOBS = 8                  # Number of threads for CPU-based CWT computation
    window_tags = ["0.5s", "1s", "2s"]

    # 1) Merge train/test splits and label with 'split'
    all_rows = []
    for split in ("train", "test"):
        df = pd.read_parquet(DATA_ROOT / "raw" / split / "segments.parquet")
        df = df.reset_index()  
        df['split'] = split    
        all_rows.append(df)

        # Create output directories for 3 window sizes per split
        out_base = DATA_ROOT / f"{split}_dataset" / "features_multiwin"
        for tag in window_tags:
            (out_base / tag).mkdir(parents=True, exist_ok=True)

    # 2) Concatenate all dataframes and reindex by (patient, session, segment)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.index = list(zip(all_df['patient'], all_df['session'], all_df['segment']))

    # 3) Parallel processing of each segment row
    Parallel(n_jobs=4)(
        delayed(process_segment)(row, DATA_ROOT, CWT_JOBS, window_tags)
        for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Processing")
    )
