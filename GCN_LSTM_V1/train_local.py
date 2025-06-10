import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GCN_LSTM
from dataset import EEGFeatureDataset
from tqdm import tqdm
from pathlib import Path

# ─── Configure paths and hyperparameters ─────────────────────
DATA_ROOT = r"D:\Documents\nml_data\data\train_dataset"
META_PATH = r"D:\Documents\nml_data\data\train_dataset\train_segments.parquet"
FEATURE_PATH = r"D:\Documents\nml_data\data\train_dataset\features"

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
warmup_epochs = 3
WEIGHT_DECAY = 1e-3
VAL_RATIO = 0.2
PATIENCE = 8  # Tolerance for early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ─── Training function ────────────────────────────
def warmup_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(warmup_epochs)
    else:
        return 1.0  # Later fix at 1, hand to ReduceLROnPlateau
    
def train(train_loader, val_loader, model, optimizer, scheduler, fold_idx, device):
    patience_counter = 0
    best_macro_f1   = 0.0
    #class_weights = torch.tensor([1, 1], dtype=torch.float32).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]")

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            # —— 1. Count positive/negative samples in this batch ——
            counts = torch.bincount(y, minlength=2).float()  # tensor([neg_cnt, pos_cnt])
            total = counts.sum()
            batch_weights = total / (2 * counts)  # Inverse-frequency weighting          

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y, weight=batch_weights.to(device))
            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Grad Norm: {total_norm:.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping to avoid explosion
            optimizer.step()
            total_loss += loss.item() 
            #scheduler.step()  # Update learning rate
            loop.set_postfix(loss=loss.item())

        
        avg_loss = total_loss / len(train_loader)
        val_acc, val_prec, val_rec, val_f1, val_thr = evaluate(val_loader, model, device)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        scheduler.step(val_f1) # Update LR based on val macro-F1
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

# ─── Evaluation function ────────────────────────────
def find_best_threshold(probs, labels):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1


def evaluate(val_loader, model, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        first_batch = True  # Use to print the prediction of the first batch
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)              # [B, 2]
            #preds = torch.argmax(out, dim=1)  # [B]
            probs = torch.softmax(out, dim=1)[:, 1]       
            preds = torch.argmax(out, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            
            if first_batch:
                unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)
                print("Prediction distribution:", dict(zip(unique, counts)))
                first_batch = False

    # Concatenate into 1D tensors / numpy arrays
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Fix threshold=0.4; can also dynamically find best_t
    #best_t = 0.4
    best_t, best_f1 = find_best_threshold(all_probs, all_labels)
    final_preds = (all_probs > best_t).astype(int)
    #best_f1 = f1_score(all_labels, final_preds, average='macro')

    # Precision, Recall, F1 （Set “1” as positive ）
    acc = (final_preds == all_labels).mean() * 100
    prec = precision_score(all_labels, final_preds, average='macro') * 100
    rec  = recall_score(all_labels, final_preds, average='macro') * 100
    f1   = best_f1 * 100     
    print(f"⚙️  Best Threshold: {best_t:.2f}")
    return acc, prec, rec, f1, best_t


# ─── Start training ────────────────────────────
if __name__ == "__main__":
    print("🧠 Start training...")
     # ─── Stratified K-Fold Split ─────────────────────
    NUM_FOLDS = 5
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    # ─── Load metadata (for patient-level split) ───────────────
    full_df = pd.read_parquet(META_PATH)
    # Whether each patient has at least one positive segment
    patient_ids = full_df['patient'].unique()
    patient_labels = [full_df[full_df['patient'] == pid]['label'].max() for pid in patient_ids]

    for fold, (train_pid_idx, val_pid_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n📂 Fold {fold+1}/{NUM_FOLDS}")
        # Get training and validation patient IDs
        train_pids = patient_ids[train_pid_idx]
        val_pids   = patient_ids[val_pid_idx]

        train_df = full_df[full_df['patient'].isin(train_pids)].reset_index(drop=True)
        val_df   = full_df[full_df['patient'].isin(val_pids)].reset_index(drop=True)
        # Oversample positives
        pos_df = train_df[train_df['label'] == 1]
        neg_df = train_df[train_df['label'] == 0]
        n_to_add = int(len(neg_df) * 0.7) - len(pos_df)

        if n_to_add > 0:
            resampled_pos = pos_df.sample(n=n_to_add, replace=True, random_state=42)
            balanced_train_df = pd.concat([train_df, resampled_pos], ignore_index=True)
        else:
            balanced_train_df = train_df

        print(f"Original positives: {len(pos_df)}, negatives: {len(neg_df)} → After sampling: {balanced_train_df['label'].sum()} (target ratio 0.7)")
        
        # Construct Dataset and Dataloader
        train_dataset = EEGFeatureDataset(dataframe=balanced_train_df, feature_dir=FEATURE_PATH, mean_path=r"D:\Documents\nml_data\data\train_dataset\mean.npy", std_path=r"D:\Documents\nml_data\data\train_dataset\std.npy")
        val_dataset   = EEGFeatureDataset(dataframe=val_df, feature_dir=FEATURE_PATH, mean_path=r"D:\Documents\nml_data\data\train_dataset\mean.npy", std_path=r"D:\Documents\nml_data\data\train_dataset\std.npy")
        train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Initialize model and optimizer
        model = GCN_LSTM(in_feat=393, gcn_hidden=128, lstm_hidden=128, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),  lr=LR, weight_decay=1e-3)
        total_steps = EPOCHS * len(train_loader)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',           # Expect higher val_macro_f1
            factor=0.5,           # Learning rate × 0.7
            patience=2,           # Trigger if no improvement for 2 consecutive epochs
            min_lr=2e-5,       # Minimum learning rate
            verbose=True
        )

        # train
        train(train_loader, val_loader, model, optimizer, scheduler, fold, device)
    torch.save(model.state_dict(), r"D:\Documents\nml_data\gcn_lstm_final.pth")
    print("📦 Final model saved as './gcn_lstm_final.pth'.")


