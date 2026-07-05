"""
run_pipeline.py
----------------
End-to-end execution of the NIDS Lab ML pipeline.
Reproduces every step of the notebook by calling the modules in order.

Usage:
    python run_pipeline.py

Note: if running outside Colab, either edit DATA_DIR in config.py to point
to your data folder (the 4 UNSW-NB15 CSV files), or set the environment
variable instead:
    UNSW_DATA_DIR=/path/to/data python run_pipeline.py
"""

from config import DATA_DIR, LEAKAGE_COLS_MODEL
from data_prep import load_raw_data, clean_data, optimize_dtypes, validate_data, time_based_split
from feature_engineering import add_engineered_features, encode_categoricals, scale_features
from feature_selection import select_top_features_mi
from train import train_random_forest_balanced, train_random_forest_smote, train_xgboost, save_model
from evaluate import evaluate_model, compare_models
from utils import check_ram, free_memory


def main():
    # 1) Load and clean data
    df = load_raw_data(DATA_DIR)
    df = clean_data(df)
    df = optimize_dtypes(df)
    validate_data(df)

    # 2) Chronological train/test split
    train_df, test_df = time_based_split(df, drop_cols=LEAKAGE_COLS_MODEL)

    # 3) Feature engineering
    train_df, test_df = add_engineered_features(train_df, test_df, df_original=df)
    X_train, y_train, X_test, y_test = encode_categoricals(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Free memory — the large raw data is no longer needed
    free_memory(globals(), ["df", "train_df", "test_df"])
    check_ram()

    # 4) Select top-20 features via Mutual Information
    top_features, X_train_top, X_test_top, mi_df = select_top_features_mi(
        X_train_scaled, y_train, X_test_scaled, y_test=y_test
    )

    # 5) Train models
    rf_balanced = train_random_forest_balanced(X_train_top, y_train)
    rf_smote = train_random_forest_smote(X_train_top, y_train)
    xgb_model, label_encoder = train_xgboost(X_train_top, y_train)

    # 6) Evaluate
    res_rf = evaluate_model(rf_balanced, X_test_top, y_test, "Random Forest (balanced)")
    res_smote = evaluate_model(rf_smote, X_test_top, y_test, "Random Forest (SMOTE)")

    y_test_enc = label_encoder.transform(y_test)
    res_xgb = evaluate_model(xgb_model, X_test_top, y_test_enc, "XGBoost", label_encoder=label_encoder)

    compare_models([
        {"Model": "RF (class_weight=balanced)", "Macro F1": round(res_rf["macro_f1"], 4)},
        {"Model": "RF (SMOTE)", "Macro F1": round(res_smote["macro_f1"], 4)},
        {"Model": "XGBoost", "Macro F1": round(res_xgb["macro_f1"], 4)},
    ])

    # 7) Save the winning model (loaded by the PHP dashboard)
    save_model(rf_balanced, f"{DATA_DIR}/models/rf_balanced_final.pkl")


if __name__ == "__main__":
    main()
