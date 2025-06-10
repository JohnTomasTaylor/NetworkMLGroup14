# Multi-Scale GCN-LSTM for EEG Seizure Detection

This repository contains our implementation for EEG seizure detection using a graph-based neural model. We explore multi-scale temporal modeling combined with spatial GCN layers and temporal LSTMs. The best model achieves top-tier performance on the TUH Seizure Corpus (TUSZ) Kaggle subset.

---

## 🧠 Project Overview

We propose a multi-branch architecture (`MultiScale_GCN_LSTM`) that:
- Extracts EEG features over 0.5s, 1s, and 2s sliding windows.
- Builds dynamic graphs using Pearson correlation per window.
- Applies GCN + BiLSTM per scale, followed by cross-scale attention.
- Outputs seizure predictions using a final MLP classifier.

---

## 📁 Repository Structure

├── Data preprocessing/
│   ├── feature_extraction.py         # For 2-second fixed-window feature extraction
│   └── advance_feature_extraction.py # For 0.5s / 1s / 2s multi-window extraction
│
├── EEG_nml/
│   └── EEG_nml.ipynb                 # Initial GCN model (basic version, single window)
│
├── GCN_LSTM_Pro/
│   ├── advance_model.py             # Multi-scale model definition
│   ├── advance_train.py             # Training script (base)
│   ├── advance_train_focal_loss.py  # Training script with focal loss
│   ├── advance_model_dropout.py     # Model variant with extra dropout
│   ├── dataset_multiwin.py          # Multi-window dataset class
│   ├── best_model_final.pth         # ✅ Best-performing trained model
│   ├── submission.py                # Kaggle inference + submission generation
│   └── *.pth                        # Additional saved models (Pro, Pro2, etc.)
│
├── GCN_LSTM_V1/
│   └── (legacy code)                # Older version using outdated features
│                                    # Includes model, dataset, and training logic
│
├── evaluation/
│   └── (team-written)               # ⚠️ Scripts written by teammate – unclear purpose
│
├── configs/, scitas/, src/         # Supporting configs, Slurm batch scripts, and common utilities
├── requirements.txt                # Python dependency list
├── README.md                       # You're reading it!

---

## 🚀 How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Feature Extraction

- For **2-second fixed window**:
  ```bash
  python Data\ preprocessing/feature_extraction.py
  ```

- For **multi-scale extraction (0.5s, 1s, 2s)**:
  ```bash
  python Data\ preprocessing/advance_feature_extraction.py
  ```

### 3. Training

- Run base multi-scale model:
  ```bash
  python GCN_LSTM_Pro/advance_train.py
  ```

- Try other variants with focal loss or dropout:
  ```bash
  python GCN_LSTM_Pro/advance_train_focal_loss.py
  ```

### 4. Inference & Submission

Use the best model to generate Kaggle submission:

```bash
python GCN_LSTM_Pro/submission.py
```

Output will be saved to:
```
D:\Documents\nml_data\submission.csv
```

---

## 🏆 Best Model

- File: `GCN_LSTM_Pro/best_model_final.pth`
- Description: Multi-scale model with PCC graphs, BiLSTM, and attention fusion.
- Performance: Achieved top results on private leaderboard (macro-F1 ≈ 0.82 on CV)

---

## ⚠️ Notes

- The `evaluation/` folder contains scripts written by a teammate. Their functionality is currently unclear and not directly integrated into the main pipeline.
- The `GCN_LSTM_V1/` folder contains an outdated codebase using a previous version of the feature extraction pipeline. For best performance, use the `GCN_LSTM_Pro/` version.

---

## 📌 Citation

If you use or adapt this repository, please cite:

> Zimu Zhao, John Taylor, Daniele Belfiore, Ahmed Abdelmalek  
> "Graph-Based Neural Networks for EEG Seizure Detection: A Comparative Analysis"  
> EPFL, 2025

---
