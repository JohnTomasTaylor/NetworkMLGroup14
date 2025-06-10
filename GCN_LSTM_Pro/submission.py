# kaggle_multiwin.py
import os
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# --------------- ① Import multi-scale dataset and model ---------------
from dataset_multiwin import EEGMultiWinDataset, collate_fn_multiscale
from advance_model import MultiScale_GCN_LSTM  # Your model should be defined in advance_model.py

# —— Script entry point —— #
if __name__ == "__main__":
    # 1. Path settings — modify to match your local paths — #
    DATA_ROOT   = r"D:\Documents\nml_data"
    FEATURE_ROOT = Path(r"D:\Documents\nml_data\test_dataset\features_multiwin")
    
    # 1.1 Feature folders for three time scales
    feature_dirs = {
        "0.5s": os.path.join(DATA_ROOT, "test_dataset", "features_multiwin", "0.5s"),
        "1s" : os.path.join(DATA_ROOT, "test_dataset", "features_multiwin", "1s"),
        "2s" : os.path.join(DATA_ROOT, "test_dataset", "features_multiwin", "2s"),
    }

    # 1.2 Path to the trained multi-scale model checkpoint
    model_path  = r"D:\Documents\nml_data\model7\best_model_fold2.pth"
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Build the test DataFrame — one .npy per EEG segment — #
    feature_dir = r"D:\Documents\nml_data\test_dataset\features_multiwin\0.5s"
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    all_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npy")])
    if len(all_files) == 0:
        raise RuntimeError(f"No .npy files found in {feature_dir}")
    test_df = pd.DataFrame({"filename": all_files})

    # Only "filename" column is needed; other info can be inferred later
    test_df = pd.DataFrame({"filename": all_files})

    # 3. Instantiate Dataset & DataLoader — #
    test_dataset = EEGMultiWinDataset(
        segment_df   = test_df,
        feature_dirs_root = FEATURE_ROOT ,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = 32,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = collate_fn_multiscale
    )

    # 4. Instantiate and load the multi-scale model — #

    # ─── 4.1 Load fixed physical KNN graph (edge_index) ─────────────
    fixed_edge_index = torch.load(r"D:\Documents\nml_data\edge_index\edge_index_knn.pt").to(device)  # [2, E_fixed]

    # 4.2 Set number of channels C
    C = 19

    # 4.3 Build a [C, C] zero adjacency matrix
    A_phys = torch.zeros((C, C), device=device)

    # 4.4 Convert edge_index to adjacency matrix (add symmetric edges if needed)
    row, col = fixed_edge_index  # row.shape=[E], col.shape=[E]
    A_phys[row, col] = 1.0
    A_phys[col, row] = 1.0  # Add reverse edges if the graph is not already symmetric

    # 4.5 Compute normalized adjacency (D^{-1/2} A D^{-1/2})
    eps = 1e-6
    A_phys = A_phys + torch.eye(C, device=device) * eps
    deg = A_phys.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    fixed_adj_norm = D_inv_sqrt @ A_phys @ D_inv_sqrt  # [19, 19]

    # 4.6 Feature dimensions for each time scale
    F05 = 145
    F1  = 208
    F2  = 210

    model = MultiScale_GCN_LSTM(
        in_feat_05   = F05,
        in_feat_1    = F1,
        in_feat_2    = F2,
        gcn_hidden   = 128,
        lstm_hidden  = 128,
        D            = 128,
        num_classes  = 2,
        share_weights= False,
        fixed_adj_norm = fixed_adj_norm,
        num_channels = C
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Inference — #
    all_ids   = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            # Each of these is [B, C, T, F] — multi-scale feature tensors
            x05 = batch["feat_05"].to(device)  # [B, 19, T05, 145]
            x1  = batch["feat_1"].to(device)   # [B, 19, T1, 208]
            x2  = batch["feat_2"].to(device)   # [B, 19, T2, 210]
            fnames = batch["filename"]         # List of B filenames

            logits = model(x05, x1, x2)        # [B, 2]
            probs  = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            threshold = 0.6  # Change this to your best validation threshold
            preds = (probs > threshold).long().cpu().tolist()

            all_ids.extend(fnames)
            all_preds.extend(preds)

    # 6. Extract submission ids from filenames — #
    clean_ids = []
    for fname in all_ids:
        base    = fname.replace(".npy", "")
        seg_num = int(base.split("_")[-1].replace("seg", ""))
        prefix  = base.rsplit("_", 1)[0]
        new_id  = f"{prefix}_{seg_num}"
        clean_ids.append(new_id)

    # 7. Save submission file — #
    submit_df = pd.DataFrame({"id": clean_ids, "label": all_preds})
    save_path = r"D:\Documents\nml_data\submission.csv"
    submit_df.to_csv(save_path, index=False)
    print(f"Saved {save_path} with {len(submit_df)} entries.")
