# from pathlib import Path
# import joblib

# ART_DIR = Path("artifacts")
# ART_DIR.mkdir(exist_ok=True)

# def artifact_path(model_name: str) -> Path:
#     return ART_DIR / f"model_{model_name}.joblib"

# def save_artifact(model_name: str, model, feature_names: list[str]):
#     payload = {"model": model, "feature_names": feature_names}
#     joblib.dump(payload, artifact_path(model_name))

# def load_artifact(model_name: str):
#     p = artifact_path(model_name)
#     if not p.exists():
#         return None
#     return joblib.load(p)
