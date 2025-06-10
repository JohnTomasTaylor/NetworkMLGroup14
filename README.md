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

```
Data preprocessing/
├── feature_extraction.py          # 2-second fixed-window feature extraction
├── advance_feature_extraction.py  # Multi-scale feature extraction (0.5s, 1s, 2s)

EEG_nml/
├── EEG_nml.ipynb                  # Initial GCN model prototype (single-window)

GCN_LSTM_Pro/
├── advance_model.py               # Multi-scale model definition
├── advance_train.py               # Base training script
├── advance_train_focal_loss.py    # With focal loss
├── advance_model_dropout.py       # With additional dropout
├── dataset_multiwin.py            # Multi-window dataset loader
├── best_model_final.pth           # ✅ Best-performing model checkpoint
├── submission.py                  # Kaggle inference + CSV generation

GCN_LSTM_V1/
├── ...                            # Legacy version based on earlier feature set

evaluation/
├── evaluation_metrics.py          # Model evaluation scripts
├── kaggle_submission_v2.py        # Alternate submission generator

src/
├── base_lstm_model.py             # Standard LSTM training and model code

configs/, scitas/                  # Configuration files and Slurm scripts
requirements.txt                   # Python dependency list
README.md                          # You're reading it!
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Feature Extraction

- For **2-second fixed windows**:
  ```bash
  python Data\ preprocessing/feature_extraction.py
  ```

- For **multi-scale (0.5s, 1s, 2s) extraction**:
  ```bash
  python Data\ preprocessing/advance_feature_extraction.py
  ```

### 3. Training

- Run base multi-scale training:
  ```bash
  python GCN_LSTM_Pro/advance_train.py
  ```

- Try variants with loss or regularisation tricks:
  ```bash
  python GCN_LSTM_Pro/advance_train_focal_loss.py
  ```

### 4. Inference & Submission

- Use the best model to generate submission CSV:
  ```bash
  python GCN_LSTM_Pro/submission.py
  ```

---

## 🏆 Best Model

- File: `GCN_LSTM_Pro/best_model_final.pth`
- Description: Multi-scale GCN+LSTM with dynamic graphs and attention fusion.
- Performance: Macro-F1 ≈ 0.82 (5-fold CV)

---

## 🔗 Resources

- 📁 [Google Drive with models and features](https://drive.google.com/drive/folders/1RJcT7uc8gai7Kw8nBvR_me4ZkgyybQj4)

---

## ⚠️ Notes

- `evaluation/` includes model evaluation scripts and an alternate submission pipeline.
- `GCN_LSTM_V1/` contains a deprecated version using outdated features. We recommend using the `GCN_LSTM_Pro/` pipeline.
- `src/` contains standard LSTM training code for comparison.

---

## 📌 Citation

If you use or adapt this repository, please cite:

> Zimu Zhao, John Taylor, Daniele Belfiore, Ahmed Abdelmalek  
> "Graph-Based Neural Networks for EEG Seizure Detection: A Comparative Analysis"  
> EPFL, 2025


