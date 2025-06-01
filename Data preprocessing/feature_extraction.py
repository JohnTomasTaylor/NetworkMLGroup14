#!/usr/bin/env python3
"""
Feature Extraction Script (standalone .py)
Optimized sliding-window, PSD, CWT feature extraction with optional multiprocessing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# ── Configuration Section ──
fs      = 250
seg_len = fs * 12    # Segment length in samples (duration depends on dataset)
win_len = fs * 2     # Window length = 2 seconds
step    = fs         # Step size = 1 second
bands   = {
    'delta': (0.5, 4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta' : (13, 30),
    'gamma': (30, 45),
}

# Precompute PSD and CWT frequency bands and corresponding masks
freqs     = np.fft.rfftfreq(win_len, d=1/fs)
psd_masks = {name: ((freqs >= lo) & (freqs <= hi)) for name, (lo, hi) in bands.items()}
scales    = np.arange(1, 128)
cwt_freqs = pywt.scale2frequency('morl', scales) * fs
cwt_masks = {name: ((cwt_freqs >= lo) & (cwt_freqs <= hi)) for name, (lo, hi) in bands.items()}

# Bandpass filter: 0.5–30 Hz
bp_sos = signal.butter(4, (0.5, 30), btype='bandpass', output='sos', fs=fs)
def time_filter(x: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(bp_sos, x, axis=0)

# Z-score normalization
def normalize(seg: np.ndarray) -> np.ndarray:
    mu  = seg.mean(axis=0)
    std = seg.std(axis=0)
    std[std < 1e-6] = 1e-6
    return (seg - mu) / std

# Efficient sliding window generation
def get_windows(seg: np.ndarray) -> np.ndarray:
    view = sliding_window_view(seg, (win_len, seg.shape[1]))
    windows = view[::step, 0, :, :]
    return windows

# Time-domain features
def compute_time_feats(windows: np.ndarray) -> np.ndarray:
    xavg       = windows.mean(axis=1)
    xarv       = np.abs(windows).mean(axis=1)
    xstd       = windows.std(axis=1)
    xp_p       = windows.max(axis=1) - windows.min(axis=1)
    zero_cross = ((windows[:, :-1, :] * windows[:, 1:, :]) < 0).sum(axis=1) / (win_len - 1)
    return np.stack([xavg, xarv, xstd, xp_p, zero_cross], axis=-1)

# PSD features
def compute_psd_feats(windows: np.ndarray):
    X   = np.fft.rfft(windows, axis=1)
    psd = (np.abs(X)**2) / win_len
    full = psd[:, :128, :]
    band = np.stack([psd[:, mask, :].mean(axis=1) for mask in psd_masks.values()], axis=-1)
    return full, band

# CWT energy for a single signal
def _cwt_energy_single(sig: np.ndarray) -> np.ndarray:
    cwtm, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1/fs)
    return np.sum(np.abs(cwtm)**2, axis=1)

# Batched CWT energy computation (can be parallelized)
def compute_cwt_energy(windows: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    n_win, _, n_ch = windows.shape
    arr = windows.transpose(0,2,1).reshape(-1, win_len)
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as exe:
            energies = list(exe.map(_cwt_energy_single, arr))
    else:
        energies = list(map(_cwt_energy_single, arr))
    energies = np.stack(energies, axis=0)
    return energies.reshape(n_win, n_ch, -1)

# CWT band-averaged energy features
def compute_cwt_band_feats(energy: np.ndarray) -> np.ndarray:
    band_feats = []
    for mask in cwt_masks.values():
        if mask.any():
            band_feats.append(energy[:, :, mask].mean(axis=2))
        else:
            band_feats.append(np.zeros((energy.shape[0], energy.shape[1])))
    return np.stack(band_feats, axis=-1)

# Main feature extraction pipeline
def extract_feats_batch(seg: np.ndarray, cwt_jobs: int = 1) -> np.ndarray:
    windows    = get_windows(seg)
    time_feats = compute_time_feats(windows)
    psd_full, psd_band = compute_psd_feats(windows)
    energy     = compute_cwt_energy(windows, n_jobs=cwt_jobs)
    cwt_band   = compute_cwt_band_feats(energy)
    feats = np.concatenate([
        time_feats,
        psd_full.transpose(0,2,1),
        psd_band,
        energy,
        cwt_band
    ], axis=-1)
    return feats

# ── Script Entry Point ──
if __name__ == "__main__":
    DATA_ROOT = Path(r"C:\Users\ROG\Downloads\nml_data\data")
    # Number of parallel jobs (adjust based on CPU core count)
    CWT_JOBS = 16

    for split in ("train", "test"):
        segs    = pd.read_parquet(DATA_ROOT/split/"segments.parquet")
        out_dir = DATA_ROOT/f"{split}_dataset"/"features"
        out_dir.mkdir(parents=True, exist_ok=True)

        for (patient, session, seg_id), row in tqdm(segs.iterrows(), total=len(segs), desc=split):
            sig   = pd.read_parquet(DATA_ROOT/split/row["signals_path"]).values
            start = int(row["start_time"] * fs)
            seg   = sig[start : start + seg_len]
            if seg.shape[0] < seg_len:
                continue

            seg   = time_filter(seg)
            seg   = normalize(seg)
            feats = extract_feats_batch(seg, cwt_jobs=CWT_JOBS)

            fname = f"{patient}_{session}_seg{int(seg_id):02d}.npy"
            np.save(out_dir / fname, feats)
