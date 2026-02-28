import pandas as pd
from .config import TARGET_COL, FEATURES, CATEGORICAL_COLS

def prepare_data(df):
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after load_data/rename: {missing}\n\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    df = df[FEATURES + [TARGET_COL]].copy()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Split
    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()

    # Identify numeric columns
    numeric_cols = [c for c in FEATURES if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # --- Impute missing ---
    # numeric -> median
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # categorical -> "Unknown"
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Unknown")

    # --- One-hot encode categoricals ---
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X, y, numeric_cols


# -------------------------------------------------
# Inference preprocessing (for UI / prediction)
# -------------------------------------------------

def prepare_features(df):
    """
    Apply SAME preprocessing as prepare_data,
    but without using TARGET column.
    Used for UI prediction.
    """

    # ensure feature columns exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    X = df[FEATURES].copy()

    # ---- numeric / categorical split ----
    numeric_cols = [c for c in FEATURES if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # ---- numeric imputation ----
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # ---- categorical imputation ----
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Unknown")

    # ---- one-hot encoding ----
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X
