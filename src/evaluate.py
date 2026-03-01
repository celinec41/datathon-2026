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
        # both formats are handy
        "report": classification_report(y_test, preds, output_dict=True),
        "report_text": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "preds": preds,  # helpful for debugging/plots
    }

    # ---- Probabilities (if supported) ----
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            out["proba"] = proba
            out["pred_proba_max"] = proba.max(axis=1)
            out["pred_class_from_proba"] = proba.argmax(axis=1)
        except Exception:
            pass

    # ---- Optional ROC-AUC (multi-class OVR) ----
    if hasattr(model, "predict_proba"):
        unique_classes = np.unique(y_test)
        if unique_classes.size >= 2:
            try:
                proba = model.predict_proba(X_test)
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                )
            except (ValueError, AttributeError):
                pass

    return out