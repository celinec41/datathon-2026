# src/preprocess.py
import pandas as pd
from .config import TARGET_COL, FEATURES, CATEGORICAL_COLS


def _impute_and_encode(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Shared preprocessing for BOTH training and inference:
    - numeric: coerce to numeric, fill missing with median (fallback 0 if median is NaN)
    - categorical: fill missing with "Unknown"
    - one-hot encode categoricals (drop_first=False)
    Returns (X_processed, numeric_cols)
    """
    numeric_cols = [c for c in FEATURES if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # numeric -> median (fallback 0)
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        if pd.isna(med):
            med = 0
        X[c] = X[c].fillna(med)

    # categorical -> "Unknown"
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Unknown")

    # one-hot encode categoricals
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X, numeric_cols


def prepare_data(df: pd.DataFrame):
    # Ensure required columns exist
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after load_data/rename: {missing}\n\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    df = df[FEATURES + [TARGET_COL]].copy()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Ensure target is int labels
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()

    X, numeric_cols = _impute_and_encode(X)

    return X, y, numeric_cols


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference preprocessing (no target column required).
    Must match prepare_data() behavior.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURES].copy()
    X, _ = _impute_and_encode(X)
    return X