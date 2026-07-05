"""
feature_selection.py
---------------------
Selects the top N features by Mutual Information (MI) score and saves
the result as a checkpoint (parquet/csv) to Drive or a local folder.

Note: MI is computed on a 50,000-row sample rather than the full dataset,
for a speed/RAM trade-off.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import (
    DATA_DIR, N_TOP_FEATURES, MI_SAMPLE_SIZE, RANDOM_STATE,
    X_TRAIN_TOP20_FILE, X_TEST_TOP20_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
)
from utils import print_section


def select_top_features_mi(X_train_scaled: pd.DataFrame, y_train: pd.Series,
                             X_test_scaled: pd.DataFrame, y_test: pd.Series = None,
                             n_features: int = N_TOP_FEATURES,
                             sample_size: int = MI_SAMPLE_SIZE,
                             save_checkpoint: bool = True):
    """
    Selects the n_features columns with the highest Mutual Information score.

    Returns: top_features (list), X_train_top (DataFrame), X_test_top (DataFrame), mi_df (DataFrame)
    """
    print_section("MUTUAL INFORMATION — FEATURE SELECTION")

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_train_scaled), size=min(sample_size, len(X_train_scaled)), replace=False)
    X_sample = X_train_scaled.iloc[sample_idx]
    y_sample = y_train.iloc[sample_idx]

    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X_train_scaled.columns, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)
    print(mi_df.head(n_features).to_string())

    top_features = mi_df.head(n_features)["feature"].tolist()
    X_train_top = X_train_scaled[top_features]
    X_test_top = X_test_scaled[top_features]

    if save_checkpoint:
        X_train_top.to_parquet(f"{DATA_DIR}/{X_TRAIN_TOP20_FILE}")
        X_test_top.to_parquet(f"{DATA_DIR}/{X_TEST_TOP20_FILE}")
        y_train.to_csv(f"{DATA_DIR}/{Y_TRAIN_FILE}", index=False)
        if y_test is not None:
            y_test.to_csv(f"{DATA_DIR}/{Y_TEST_FILE}", index=False)
        print("\nCheckpoint saved ✅")

    print("Top-N selected ✅")
    return top_features, X_train_top, X_test_top, mi_df


def load_checkpoint(data_dir: str = DATA_DIR):
    """Loads a previously saved top-feature checkpoint (to resume without recomputing MI)."""
    X_train_top = pd.read_parquet(f"{data_dir}/{X_TRAIN_TOP20_FILE}")
    X_test_top = pd.read_parquet(f"{data_dir}/{X_TEST_TOP20_FILE}")
    y_train = pd.read_csv(f"{data_dir}/{Y_TRAIN_FILE}").squeeze()
    y_test = pd.read_csv(f"{data_dir}/{Y_TEST_FILE}").squeeze()

    print("Checkpoint loaded ✅")
    print("X_train_top:", X_train_top.shape)
    print("X_test_top: ", X_test_top.shape)
    return X_train_top, X_test_top, y_train, y_test
