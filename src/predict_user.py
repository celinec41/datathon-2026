# src/predict_user.py
import pandas as pd
from .config import FEATURES, LABEL_MAP
from .preprocess import prepare_features


def _get_clf(model):
    # Pipeline-safe
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


def _label(cls):
    # robust label mapping
    try:
        key = int(cls)
    except Exception:
        key = cls
    return LABEL_MAP.get(key, str(cls))


def predict_user(model, feature_names: list[str], user_input: dict):
    """
    Takes raw UI payload (FEATURES columns), applies SAME preprocessing as training
    (impute + one-hot), aligns columns to training feature_names, then predicts.
    """
    # Ensure all expected raw feature columns exist
    row = {col: user_input.get(col, None) for col in FEATURES}
    user_df = pd.DataFrame([row], columns=FEATURES)

    # Apply inference preprocessing (impute + one-hot on categorical cols)
    X_user = prepare_features(user_df)

    # Align to training dummy columns
    X_user = X_user.reindex(columns=feature_names, fill_value=0)

    clf = _get_clf(model)

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_user)[0]

        # prefer model.classes_ if present, else clf.classes_
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = getattr(clf, "classes_", list(range(len(proba))))

        prob_dict = {_label(cls): float(p) for cls, p in zip(classes, proba)}
        most_likely = max(prob_dict, key=prob_dict.get)

        return {
            "probabilities": prob_dict,
            "most_likely": most_likely,
            "confidence": float(prob_dict[most_likely]),
        }

    # Fallback (no probabilities)
    pred = model.predict(X_user)[0]
    pred_label = _label(pred)
    return {"most_likely": pred_label, "confidence": None, "probabilities": {}}