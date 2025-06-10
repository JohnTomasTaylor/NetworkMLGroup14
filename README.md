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
├── feature_extraction.py           # 2-second fixed-window feature extraction
├── advance_feature_extraction.py   # Multi-scale feature extraction (0.5s, 1s, 2s)

EEG_nml/
├── EEG_nml.ipynb                   # Initial prototype GCN training notebook

GCN_LSTM_Pro/
├── advance_model.py                # Multi-scale model definition
├── advance_model_dropout.py        # Variant with extra dropout
├── advance_train.py                # Training script
├── advance_train_focal_loss.py     # Variant with focal loss
├── dataset_multiwin.py             # Multi-window dataset class
├── best_model_final.pth            # ✅ Best model checkpoint
├── submission.py                   # Kaggle submission generator

GCN_LSTM_V1/
├── (legacy) Model + dataset code for old feature version

evaluation/
├── config.py, data.py, utils.py    # Evaluation config and helpers
├── evaluate.py, threshold.py       # Metric computation and thresholding
├── predict.py, submission.py       # Alternate inference + submission pipeline
├── ensemble.py, model_loader.py    # Ensembling and model loading logic

src/
├── constants.py                    # Basic constants
├── dataloading.py                 # LSTM-specific dataloader
├── transforms.py, training_utils.py  # Preprocessing and training helpers
├── models/                         # Base LSTM model definitions

configs/, scitas/                   # Config YAMLs, Slurm scripts

requirements.txt                    # Python dependency list
README.md                           # You're reading it!
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

- For **multi-scale (0.5s, 1s, 2s)**:
  ```bash
  python Data\ preprocessing/advance_feature_extraction.py
  ```

### 3. Training

- Run baseline multi-scale model:
  ```bash
  python GCN_LSTM_Pro/advance_train.py
  ```

- Try focal loss or dropout variants:
  ```bash
  python GCN_LSTM_Pro/advance_train_focal_loss.py
  ```

### 4. Inference & Submission

- Use best model to generate submission:
  ```bash
  python GCN_LSTM_Pro/submission.py
  ```

Or use alternate scripts from `evaluation/` for prediction, ensembling, and submission generation.

---

## 🏆 Best Model

- File: `GCN_LSTM_Pro/best_model_final.pth`
- Description: Multi-scale GCN+LSTM with dynamic functional graphs and attention fusion.
- Performance: Macro-F1 ≈ 0.82 on cross-validation

---

## 🔗 Resources

- 📁 [Google Drive – features, models, and outputs](https://drive.google.com/drive/folders/1RJcT7uc8gai7Kw8nBvR_me4ZkgyybQj4)

---

## ⚠️ Notes

- The `evaluation/` directory contains evaluation tools, ensembling logic, and an alternate submission pipeline.
- The `src/` directory contains basic LSTM training code and preprocessing utilities.
- `GCN_LSTM_V1/` is deprecated and kept for reference only.

---

## 📌 Citation

If you use or adapt this repository, please cite:

> Zimu Zhao, John Taylor, Daniele Belfiore, Ahmed Abdelmalek  
> "GCN+LSTM for EEG Seizure Detection"  
> EPFL, 2025

