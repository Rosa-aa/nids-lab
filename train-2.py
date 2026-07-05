"""
train.py
--------
Trains three models:
    1. Random Forest (class_weight='balanced')
    2. Random Forest + SMOTE/undersampling
    3. XGBoost

Each function returns the fitted model so it can be scored by evaluate.py.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE
from utils import print_section


def train_random_forest_balanced(X_train: pd.DataFrame, y_train: pd.Series,
                                    n_estimators: int = 50, max_depth: int = 15):
    """
    Main model: RF with class_weight='balanced' — accounts for rare
    classes (Analysis, Backdoor, Shellcode, Worms).
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_smote(X_train: pd.DataFrame, y_train: pd.Series,
                                 over_strategy: dict = None, under_strategy: dict = None,
                                 n_estimators: int = 50, max_depth: int = 15):
    """
    Trains RF on data rebalanced with SMOTE (oversampling) + RandomUnderSampler.
    Default strategies match the notebook's Section 7.2.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    print_section("SMOTE + UNDERSAMPLING")

    over_strategy = over_strategy or {
        "Analysis": 5000, "Backdoor": 5000, "Shellcode": 5000,
        "Worms": 3000, "DoS": 8000,
    }
    under_strategy = under_strategy or {"Normal": 50000}

    over = SMOTE(sampling_strategy=over_strategy, random_state=RANDOM_STATE)
    under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)
    pipeline = ImbPipeline([("over", over), ("under", under)])

    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
    print("Before:", y_train.value_counts().to_dict())
    print("After: ", pd.Series(y_train_res).value_counts().to_dict())

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train_res, y_train_res)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                    n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6):
    """
    XGBoost multiclass classifier. XGBoost doesn't accept string labels,
    so a LabelEncoder is required — it's returned alongside the model so
    predictions can be decoded back to class names later.
    """
    from xgboost import XGBClassifier

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train, y_train_enc)
    return model, le


def save_model(model, path: str) -> None:
    """Saves the model to disk with joblib (matches the format the PHP dashboard loads)."""
    joblib.dump(model, path)
    print(f"Model saved: {path}")
