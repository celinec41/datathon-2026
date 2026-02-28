# src/train.py
from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .config import RANDOM_STATE, TEST_SIZE, DEFAULT_MODEL
from .preprocess import prepare_data


@dataclass
class TrainResult:
    model: Any
    X_test: Any
    y_test: Any


def _make_model(model_name: str):
    model_name = (model_name or DEFAULT_MODEL).lower()

    if model_name == "logreg":
        return LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced"
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1
        )

    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            learning_rate=0.08,
            max_depth=6,
        )

    raise ValueError(f"Unknown model_name='{model_name}'. Use: logreg, rf, hgb")


def train_model(df, model_name: str = DEFAULT_MODEL) -> TrainResult:
    X, y, numeric_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = _make_model(model_name)

    # Only scale for logistic regression
    if (model_name or DEFAULT_MODEL).lower() == "logreg":
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
    else:
        model = clf

    model.fit(X_train, y_train)
    return TrainResult(model=model, X_test=X_test, y_test=y_test)


def train_all_models(df):
    results = {}
    for name in ["logreg", "rf", "hgb"]:
        results[name] = train_model(df, model_name=name)
    return results
