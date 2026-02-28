# src/predict_user.py
import pandas as pd
from .config import FEATURES, LABEL_MAP


def predict_user(model, user_input: dict):
    # Ensure all required feature columns exist (missing -> None)
    row = {col: user_input.get(col, None) for col in FEATURES}
    user_df = pd.DataFrame([row], columns=FEATURES)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(user_df)[0]
        classes = model.classes_

        prob_dict = {
            LABEL_MAP.get(int(cls), str(cls)): float(prob)
            for cls, prob in zip(classes, probabilities)
        }
        most_likely = max(prob_dict, key=prob_dict.get)

        return {
            "probabilities": prob_dict,
            "most_likely": most_likely,
            "confidence": prob_dict[most_likely]
        }

    # Fallback if model doesn't support proba
    pred = model.predict(user_df)[0]
    pred_label = LABEL_MAP.get(int(pred), str(pred))
    return {"most_likely": pred_label, "confidence": None}
