"""
data_prep.py
------------
Loading, cleaning, dtype optimization, and chronological train/test
splitting for the raw UNSW-NB15 CSV files.

Pipeline stages (matching the notebook's section numbers):
    1. load_raw_data          — merges the 4 raw files + the feature-name CSV
    2. clean_data             — leakage columns, duplicates, missing values
    3. optimize_dtypes        — float64->float32, int64->int32 (for RAM)
    4. validate_data          — range / negative-value / categorical checks
    5. time_based_split       — chronological 80/20 split based on stime
"""

import os
import gc

import pandas as pd

from config import DATA_DIR, RAW_FEATURE_FILE, RAW_DATA_FILES, LEAKAGE_COLS_EARLY, TEST_SIZE
from utils import print_section


def load_raw_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Reads the 4 UNSW-NB15 CSV files + the feature-name file and merges them into one DataFrame."""
    os.chdir(data_dir)

    features = pd.read_csv(RAW_FEATURE_FILE, encoding="latin1")
    col_names = features["Name"].str.strip().tolist()

    dfs = [
        pd.read_csv(f, header=None, encoding="latin1", low_memory=False)
        for f in RAW_DATA_FILES
    ]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = col_names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops leakage columns, fills missing values, removes duplicates,
    standardizes the 'Backdoors' -> 'Backdoor' label, and coerces sport/dsport to numeric.
    """
    df = df.drop(columns=LEAKAGE_COLS_EARLY, errors="ignore")

    df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
    df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})
    df["is_ftp_login"] = df["is_ftp_login"].fillna(0)
    df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].fillna(0)

    df = df.drop_duplicates()

    df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
    df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
    df = df.dropna(subset=["sport", "dsport"])

    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces RAM usage by downcasting float64->float32 and int64->int32."""
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype("int32")
    gc.collect()

    print(f"df ready: {df.shape}")
    print(f"RAM usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Prints range, negative-value, and categorical-consistency checks."""
    print_section("RANGE VALIDATION")
    sport_invalid = df[(df["sport"] < 0) | (df["sport"] > 65535)]
    dsport_invalid = df[(df["dsport"] < 0) | (df["dsport"] > 65535)]
    print(f"sport out-of-range rows: {len(sport_invalid)}")
    print(f"dsport out-of-range rows: {len(dsport_invalid)}")

    print_section("NEGATIVE VALUE CHECK")
    for col in ["dur", "sbytes", "dbytes", "sload", "dload"]:
        neg = df[df[col] < 0]
        print(f"{col} — negative values: {len(neg)} rows")

    print_section("CATEGORICAL CONSISTENCY")
    for col in ["proto", "service", "state"]:
        print(f"\n{col} unique value count: {df[col].nunique()}")
        print(df[col].value_counts().head(8))


def time_based_split(df: pd.DataFrame, drop_cols: list, test_size: float = TEST_SIZE):
    """
    Sorts the DataFrame chronologically by stime and splits the first
    (1-test_size) portion into train, the rest into test. This ensures
    the model never "sees the future" during training — more realistic
    than a random split for a network-traffic time series.
    """
    df_model = df.drop(columns=drop_cols, errors="ignore")

    df_model_sorted = df_model.copy()
    df_model_sorted["stime_orig"] = df["stime"]
    df_model_sorted = df_model_sorted.sort_values("stime_orig")

    split_idx = int(len(df_model_sorted) * (1 - test_size))
    train_df = df_model_sorted.iloc[:split_idx].drop(columns=["stime_orig"])
    test_df = df_model_sorted.iloc[split_idx:].drop(columns=["stime_orig"])

    print(f"Train size: {train_df.shape}")
    print(f"Test size:  {test_df.shape}")
    print("\nTrain attack_cat share:")
    print((train_df["attack_cat"].value_counts(normalize=True) * 100).round(2))
    print("\nTest attack_cat share:")
    print((test_df["attack_cat"].value_counts(normalize=True) * 100).round(2))

    return train_df, test_df
