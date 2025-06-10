# evaluation/evaluate.py

"""
Professional hypertuning orchestration:
Tries a sweep of thresholds (and optionally other params), saves each submission,
logs validation metrics, and ranks the top N by performance.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from evaluation.config import (
    SEED, BATCH_SIZE, MODEL_DIR, ENSEMBLE_MODEL_PATHS,
    META_PATH, TEST_META_PATH, FEATURE_ROOT, LOG_DIR, RESULTS_DIR
)
from evaluation.utils import set_seed, print_metrics, get_confusion
from evaluation.data import get_dataloader, load_fixed_adj_norm
from evaluation.model_loader import load_model
from evaluation.predict import predict_probs, ensemble_predict
from evaluation.threshold import apply_threshold
from evaluation.ensemble import simple_average_ensemble
from evaluation.submission import prepare_submission

def main(
    thresholds=None,
    n_submissions=1000,
    ensemble_paths=ENSEMBLE_MODEL_PATHS,
    validation_metric="macro-f1",
    top_n=10
):
    # ==== 1. Setup ====
    set_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"========== Massive Hypertuning Evaluation Started ({timestamp}) ==========")

    # ==== 2. Data & Models ====
    print("Loading data and fixed adjacency...")
    fixed_adj_norm = load_fixed_adj_norm()
    val_loader = get_dataloader(META_PATH, shuffle=False)
    test_loader = get_dataloader(TEST_META_PATH, shuffle=False)
    val_df = pd.read_parquet(META_PATH) if str(META_PATH).endswith(".parquet") else pd.read_csv(META_PATH)
    val_labels = np.array(val_df["label"])
    val_ids = list(val_df["id"]) if "id" in val_df.columns else None

    print("Loading ensemble models...")
    models = [load_model(mp, fixed_adj_norm) for mp in ensemble_paths]

    # ==== 3. Out-of-Fold Validation and Test Probs (cache for all thresholds) ====
    print("Running OOF (validation) and test set predictions (single run)...")
    val_probs, _ = ensemble_predict(models, val_loader)
    test_probs, test_ids = ensemble_predict(models, test_loader)

    # ==== 4. Threshold Search/Submission Generation ====
    # Use a finer grid or a user-provided set of thresholds
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.95, n_submissions)
    meta_records = []

    sample_submission_path = "sample_submission.csv" if Path("sample_submission.csv").exists() else None

    for k, thr in enumerate(thresholds, 1):
        val_preds = apply_threshold(val_probs, thr)
        test_preds = apply_threshold(test_probs, thr)

        # Compute validation metrics
        f1 = (val_preds == val_labels).mean()  # For debugging only (replace with macro-f1)
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        macro_f1 = f1_score(val_labels, val_preds, average="macro")
        macro_prec = precision_score(val_labels, val_preds, average="macro")
        macro_rec = recall_score(val_labels, val_preds, average="macro")
        acc = accuracy_score(val_labels, val_preds)
        n_pos = (test_preds == 1).sum()
        n_neg = (test_preds == 0).sum()
        label_ratio = n_pos / max(n_neg + n_pos, 1)

        # Save submission
        submission_file = Path(RESULTS_DIR) / f"sample_submission_{k:04d}.csv"
        prepare_submission(
            test_ids, test_preds,
            output_path=submission_file,
            sample_submission_path=sample_submission_path
        )

        meta_records.append({
            "submission_file": str(submission_file),
            "threshold": thr,
            "val_macro_f1": macro_f1,
            "val_macro_prec": macro_prec,
            "val_macro_rec": macro_rec,
            "val_accuracy": acc,
            "n_test_pos": n_pos,
            "n_test_neg": n_neg,
            "test_label_ratio": label_ratio
        })

        print(f"[{k}/{len(thresholds)}] Threshold: {thr:.4f} | "
              f"Val Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f} | Pos ratio: {label_ratio:.4f}")

    # ==== 5. Save meta info and report top-N ====
    meta_df = pd.DataFrame(meta_records)
    meta_csv = Path(RESULTS_DIR) / f"all_submissions_meta_{timestamp}.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"Saved all meta info to {meta_csv}")

    # Sort and report top-N
    top_df = meta_df.sort_values(by="val_macro_f1", ascending=False).head(top_n)
    print("\n========== Top N Submissions by Validation Macro-F1 ==========")
    print(top_df[["submission_file", "threshold", "val_macro_f1", "val_accuracy", "n_test_pos", "n_test_neg"]].to_string(index=False))

    print("\n========== Massive Hypertuning Evaluation Complete ==========")

if __name__ == "__main__":
    main(
        thresholds=np.linspace(0.10, 0.95, 1000),  # or set n_submissions for auto-linspace
        n_submissions=1000,
        top_n=10
    )
