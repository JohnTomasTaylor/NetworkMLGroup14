import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# ── Import the multi-scale Dataset, collate function, and the model ─────────────────────────
from pathlib import Path
from dataset_multiwin import EEGMultiWinDataset, collate_fn_multiscale
from advance_model import MultiScale_GCN_LSTM  

# ─── Configure paths and hyperparameters ─────────────────────
DATA_ROOT    = r"D:\Documents\nml_data\data\train_dataset"
META_PATH    = r"D:\Documents\nml_data\data\train_dataset\train_segments.parquet"

# Root directory for multi-scale features (should contain subfolders like 0.5s/, 1s/, 2s/, and files like 0.5s_mean.npy)
FEATURE_ROOT = Path(r"D:\Documents\nml_data\train_dataset\features_multiwin")
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-4
warmup_epochs= 3
WEIGHT_DECAY = 1e-3
PATIENCE     = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Training / Evaluation functions (unchanged; only x, y structure differs) ───────
def warmup_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(warmup_epochs)
    else:
        return 1.0

def find_best_threshold(probs, labels):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def evaluate(val_loader, model, device):
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        first_batch = True
        for batch in val_loader:
            # Unpack features from three temporal scales and label from batch
            x05 = batch["feat_05"].to(device)  # [B, C, T05, F05]
            x1  = batch["feat_1"].to(device)   # [B, C, T1,  F1 ]
            x2  = batch["feat_2"].to(device)   # [B, C, T2,  F2 ]
            y   = batch["label"].to(device)    # [B]

            out = model(x05, x1, x2)           # [B, 2]
            probs = torch.softmax(out, dim=1)[:, 1]  # positive class probabilities
            preds = torch.argmax(out, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

            if first_batch:
                unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
                print("Prediction distribution:", dict(zip(unique, counts)))
                first_batch = False

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    best_t, best_f1 = find_best_threshold(all_probs, all_labels)
    final_preds = (all_probs > best_t).astype(int)

    acc  = (final_preds == all_labels).mean() * 100
    prec = precision_score(all_labels, final_preds, average='macro') * 100
    rec  = recall_score(all_labels, final_preds, average='macro') * 100
    f1   = best_f1 * 100
    print(f"⚙️ Best Threshold: {best_t:.2f}")
    return acc, prec, rec, f1, best_t

def train(train_loader, val_loader, model, optimizer, scheduler, fold_idx, device):
    patience_counter = 0
    best_macro_f1   = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]")

        for batch in loop:
            # Unpack three-scale features and labels from batch
            x05 = batch["feat_05"].to(device)
            x1  = batch["feat_1"].to(device)
            x2  = batch["feat_2"].to(device)
            y   = batch["label"].to(device)

            # —— Compute class weights within current batch —— 
            counts = torch.bincount(y, minlength=2).float()
            total  = counts.sum()
            batch_weights = total / (2 * counts)

            optimizer.zero_grad()
            out = model(x05, x1, x2)  # [B, 2]
            loss = F.cross_entropy(out, y, weight=batch_weights.to(device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_acc, val_prec, val_rec, val_f1, val_thr = evaluate(val_loader, model, device)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        scheduler.step(val_f1)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | P: {val_prec:.2f}% | "
              f"R: {val_rec:.2f}% | Macro-F1: {val_f1:.2f}% | Thr: {val_thr:.2f}")
        print("LR:", optimizer.param_groups[0]['lr'])

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pth")
            print("✅ Saved new best model (macro-F1).")
        else:
            patience_counter += 1
            print(f"⏸ No macro-F1 improvement: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

# ─── Start training ────────────────────────────
if __name__ == "__main__":
    print("🧠 Start multi-scale training…")

    # ─── Stratified K-Fold Split ──────────────────
    NUM_FOLDS = 5
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # ─── Load metadata ───────────────────────────
    full_df = pd.read_parquet(META_PATH)
    patient_ids    = full_df['patient'].unique()
    patient_labels = [full_df[full_df['patient']==pid]['label'].max() for pid in patient_ids]

    # ─── Load the fixed physical KNN graph (edge_index) ─────────────────
    fixed_edge_index = torch.load(r"D:\Documents\nml_data\edge_index\edge_index_knn.pt").to(device)  # [2, E_fixed]
    # 1.2 Explicitly set number of EEG channels: C
    C = 19
    # 1.3 Construct a zero matrix [C, C]
    A_phys = torch.zeros((C, C), device=device)
    # 1.4 Fill in 1s based on edge_index; for undirected graph, set both (i,j) and (j,i)
    row, col = fixed_edge_index  # row.shape=[E_fixed], col.shape=[E_fixed]
    A_phys[row, col] = 1.0
    A_phys[col, row] = 1.0
    # 2.1 Compute degree and D^{-1/2} A D^{-1/2} normalization
    eps = 1e-6
    A_phys = A_phys + torch.eye(C, device=device) * eps
    deg = A_phys.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    fixed_adj_norm = D_inv_sqrt @ A_phys @ D_inv_sqrt  # [19, 19]


    for fold, (train_pid_idx, val_pid_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n📂 Fold {fold+1}/{NUM_FOLDS}")
        # Get train/val patient IDs
        train_pids = patient_ids[train_pid_idx]
        val_pids   = patient_ids[val_pid_idx]

        train_df = full_df[full_df['patient'].isin(train_pids)].reset_index(drop=True)
        val_df   = full_df[full_df['patient'].isin(val_pids)].reset_index(drop=True)

        # Oversample positives
        pos_df = train_df[train_df['label']==1]
        neg_df = train_df[train_df['label']==0]
        n_to_add = int(len(neg_df)*0.5) - len(pos_df)
        if n_to_add > 0:
            resampled_pos = pos_df.sample(n=n_to_add, replace=True, random_state=42)
            balanced_train_df = pd.concat([train_df, resampled_pos], ignore_index=True)
        else:
            balanced_train_df = train_df

        print(f"Original positives: {len(pos_df)}, negatives: {len(neg_df)}"
              f" → After sampling: {balanced_train_df['label'].sum()} (target 0.7)")

        # ─── uild multi-scale Dataset and point to each scale's feature folder ──────────────
        train_dataset = EEGMultiWinDataset(
            balanced_train_df,
            feature_dirs_root=FEATURE_ROOT 
        )
        val_dataset   = EEGMultiWinDataset(
            val_df,
            feature_dirs_root=FEATURE_ROOT
        )

        # DataLoader with custom collate_fn_multiscale
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_multiscale,   
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_multiscale,  
            num_workers=4
        )

        # ─── Initialize the multi-scale model and provide in_feat for each scale ──────────────
        F05 = 145
        F1  = 208
        F2  = 210

        model = MultiScale_GCN_LSTM(
            in_feat_05=F05,
            in_feat_1 =F1,
            in_feat_2 =F2,
            gcn_hidden=128,
            lstm_hidden=128,
            D=128,
            num_classes=2,
            share_weights=False,              # Whether to share parameters across three temporal branches
            fixed_adj_norm=fixed_adj_norm  # Use the same fixed physical graph for all branches
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=2e-5,
            verbose=True
        )

        # ─── Start training for this fold ─────────────────
        train(train_loader, val_loader, model, optimizer, scheduler, fold, device)

    # Optionally save final model
    torch.save(model.state_dict(), r"D:\Documents\nml_data\multiscale_gcn_lstm_final.pth")
    print("📦 Final multiscale model saved.")