"""
feature_engineering.py
-----------------------
New feature creation, categorical encoding, and scaling.

Pipeline stages:
    1. add_engineered_features — bytes_per_pkt, port_cat
    2. encode_categoricals     — top-N + 'other' bucketing + one-hot encoding
    3. scale_features          — RobustScaler
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import TOP_PROTO_N, TOP_STATE_N
from utils import port_category, print_section


def add_engineered_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              df_original: pd.DataFrame):
    """
    Creates two new features:
      - bytes_per_pkt: ratio of bytes sent from source to packet count
      - port_cat: well_known/registered/dynamic category of dsport
    (dsport was dropped from the main model DataFrame, so it's pulled back
    from the original df here.)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["bytes_per_pkt"] = train_df["sbytes"] / (train_df["spkts"] + 1)
    test_df["bytes_per_pkt"] = test_df["sbytes"] / (test_df["spkts"] + 1)

    train_df["port_cat"] = df_original.loc[train_df.index, "dsport"].apply(port_category)
    test_df["port_cat"] = df_original.loc[test_df.index, "dsport"].apply(port_category)

    print("bytes_per_pkt created ✅")
    print("port_cat created ✅")
    print(train_df["port_cat"].value_counts())

    return train_df, test_df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          top_proto_n: int = TOP_PROTO_N, top_state_n: int = TOP_STATE_N):
    """
    Limits proto and state to their top-N most frequent values (grouping the
    rest into 'other'), then one-hot encodes all categorical columns.
    Train/test columns are aligned afterward.

    Returns: X_train, y_train, X_test, y_test
    """
    print_section("OPTIMAL ENCODING")

    train_df2 = train_df.copy()
    test_df2 = test_df.copy()

    top_proto = train_df2["proto"].value_counts().head(top_proto_n).index
    train_df2["proto"] = train_df2["proto"].where(train_df2["proto"].isin(top_proto), "other")
    test_df2["proto"] = test_df2["proto"].where(test_df2["proto"].isin(top_proto), "other")

    top_state = train_df2["state"].value_counts().head(top_state_n).index
    train_df2["state"] = train_df2["state"].where(train_df2["state"].isin(top_state), "other")
    test_df2["state"] = test_df2["state"].where(test_df2["state"].isin(top_state), "other")

    train_enc = pd.get_dummies(train_df2, columns=["proto", "service", "state", "port_cat"])
    test_enc = pd.get_dummies(test_df2, columns=["proto", "service", "state", "port_cat"])
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

    X_train = train_enc.drop(columns=["attack_cat", "label"])
    y_train = train_enc["attack_cat"]
    X_test = test_enc.drop(columns=["attack_cat", "label"])
    y_test = test_enc["attack_cat"]

    print(f"Feature count: {X_train.shape[1]}")
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)

    return X_train, y_train, X_test, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scales numeric columns with RobustScaler (robust to outliers)."""
    num_cols = X_train.select_dtypes(include=["float32", "float64", "int32", "int64"]).columns.tolist()

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # ct_ftp_cmd sometimes contains stray strings — force-convert to numeric
    X_train_scaled["ct_ftp_cmd"] = pd.to_numeric(X_train_scaled["ct_ftp_cmd"], errors="coerce").fillna(0)
    X_test_scaled["ct_ftp_cmd"] = pd.to_numeric(X_test_scaled["ct_ftp_cmd"], errors="coerce").fillna(0)

    print("Scaling complete ✅")
    print("X_train_scaled:", X_train_scaled.shape)

    return X_train_scaled, X_test_scaled, scaler
