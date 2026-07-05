"""
evaluate.py
-----------
Model evaluation: classification_report, confusion matrix,
model comparison table, and (optional) SHAP explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from utils import print_section


def evaluate_model(model, X_test, y_test, model_name: str = "Model", label_encoder=None):
    """
    Prints a classification_report and returns the macro F1 score.
    Pass label_encoder for encoded-label models like XGBoost.
    """
    y_pred = model.predict(X_test)

    if label_encoder is not None:
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
    else:
        report = classification_report(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n--- {model_name} ---")
    print(report)
    print(f"{model_name} Macro F1: {macro_f1:.4f}")

    return {"model_name": model_name, "macro_f1": macro_f1, "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, labels, title: str = "Confusion Matrix"):
    """Plots the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compare_models(results: list) -> pd.DataFrame:
    """
    results: [{'model_name': ..., 'accuracy': ..., 'macro_f1': ..., 'note': ...}, ...]
    Returns and prints the model comparison table as a DataFrame.
    """
    df = pd.DataFrame(results)
    print_section("MODEL COMPARISON TABLE")
    print(df.to_string(index=False))
    return df


def shap_summary(model, X_sample: pd.DataFrame, class_names=None, max_samples: int = 500):
    """
    Plots a SHAP summary plot for a RandomForest model (on a small sample, for RAM reasons).
    Uses model.classes_ if class_names is not given.
    """
    import shap

    idx = np.random.choice(len(X_sample), size=min(max_samples, len(X_sample)), replace=False)
    X_shap = X_sample.iloc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(shap_values, X_shap, class_names=class_names or model.classes_, show=True)
    return explainer, shap_values


def shap_stratified_sample(X_test: pd.DataFrame, y_test: pd.Series, per_class: int = 50):
    """
    Takes an equal number of samples per class so that SHAP analysis is
    also meaningful for rare classes (Worms, Backdoor, etc.).
    """
    parts = []
    for cls, group in X_test.assign(_label=y_test.values).groupby("_label"):
        parts.append(group.sample(min(len(group), per_class), random_state=42))

    X_strat = pd.concat(parts)
    y_strat = X_strat["_label"].values
    X_strat = X_strat.drop(columns="_label")
    return X_strat, y_strat
