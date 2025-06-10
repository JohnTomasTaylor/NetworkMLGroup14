# evaluation/submission.py

"""
Submission file writer for EEG Kaggle competitions.
Ensures robust ID checking, class balance reporting, and proper formatting.
"""

import pandas as pd
import numpy as np

def prepare_submission(ids, preds, output_path="submission.csv", sample_submission_path=None, return_df=False):
    """
    Prepares and writes a Kaggle submission file, with sanity checks.
    Args:
        ids: List/array of IDs (strings), should match Kaggle test set order.
        preds: Array/list/Series of binary predictions (0/1).
               If floats/probabilities, raises error (threshold first).
        output_path: Destination .csv file.
        sample_submission_path: Optional path to sample_submission.csv (for ID sanity).
        return_df: If True, returns the DataFrame for further inspection.
    Returns:
        True if successful; DataFrame if return_df is True.
    Raises:
        ValueError on duplicate/missing IDs or wrong type.
    """
    ids = list(ids)
    preds = np.asarray(preds)
    
    # Check if preds are float (probs) rather than ints (hard predictions)
    if np.issubdtype(preds.dtype, np.floating):
        raise ValueError(
            "Predictions appear to be probabilities/floats! "
            "Apply a threshold to convert to binary (0/1) before submission."
        )
    if not (np.issubdtype(preds.dtype, np.integer) or set(np.unique(preds)).issubset({0, 1})):
        raise ValueError("Predictions must be binary 0/1 after thresholding.")
    
    # Duplicate/missing IDs check
    if len(set(ids)) != len(ids):
        raise ValueError(f"IDs are not unique! Found {len(ids)-len(set(ids))} duplicates.")
    
    df = pd.DataFrame({"id": ids, "label": preds})
    df["label"] = df["label"].astype(int)

    # Check and reorder by sample_submission, if given
    if sample_submission_path:
        sample_df = pd.read_csv(sample_submission_path)
        sample_ids = list(sample_df["id"])
        if set(ids) != set(sample_ids):
            raise ValueError(
                "IDs in submission do not match sample_submission! "
                "Check for missing or extra samples."
            )
        df = df.set_index("id").reindex(sample_ids).reset_index()

    # Class balance check and warnings
    label_counts = df["label"].value_counts().to_dict()
    print(f"Label counts: {label_counts} | Total: {len(df)}")
    if label_counts.get(1, 0) == 0:
        print("WARNING: No positive (1) predictions! Macro-F1 will be very low.")
    if label_counts.get(0, 0) == 0:
        print("WARNING: No negative (0) predictions! Macro-F1 will be very low.")

    # Write file
    df.to_csv(output_path, index=False)
    print(f"Submission file written: {output_path}")
    if return_df:
        return df
    return True
