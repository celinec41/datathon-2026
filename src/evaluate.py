# src/evaluate.py
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import numpy as np


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    out = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "report": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }

    # ---- Optional ROC-AUC (multi-class OVR) ----
    # Only compute if:
    # 1) model supports predict_proba
    # 2) y_test has at least 2 classes
    # 3) roc_auc_score succeeds (can fail if shapes/classes mismatch)
    if hasattr(model, "predict_proba"):
        unique_classes = np.unique(y_test)
        if unique_classes.size >= 2:
            try:
                proba = model.predict_proba(X_test)
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                )
            except (ValueError, AttributeError):
                # ValueError: e.g., only one class present, or class mismatch
                # AttributeError: predict_proba not actually usable
                pass

    return out
