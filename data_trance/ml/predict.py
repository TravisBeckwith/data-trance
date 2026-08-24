"""
data_trance.ml.predict
========================
Loads the trained recommender model(s) and predicts a transform name from
a feature vector. Both methods return a confidence score; callers should
treat a low-confidence prediction as a signal to fall back to the
rule-based search rather than trusting a shaky guess.
"""

import os

import joblib
import numpy as np

from .features import extract_features, features_to_vector
from .train import MODELS_DIR

_rf_cache = None
_mlp_cache = None
_label_encoder_cache = None
_mlp_scaler_cache = None


def _load_label_encoder():
    global _label_encoder_cache
    if _label_encoder_cache is None:
        path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No trained recommender models found. Run "
                "`python -m data_trance.ml.train` first."
            )
        _label_encoder_cache = joblib.load(path)
    return _label_encoder_cache


def _load_rf():
    global _rf_cache
    if _rf_cache is None:
        path = os.path.join(MODELS_DIR, "rf_recommender.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No trained RandomForest recommender found. Run "
                "`python -m data_trance.ml.train` first."
            )
        _rf_cache = joblib.load(path)
    return _rf_cache


def _load_mlp():
    global _mlp_cache, _mlp_scaler_cache
    if _mlp_cache is None:
        import torch

        from .train import MLPRecommender

        path = os.path.join(MODELS_DIR, "mlp_recommender.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No trained MLP recommender found, or torch isn't installed. "
                "Install the [dl] extra and run `python -m data_trance.ml.train`."
            )
        checkpoint = torch.load(path, weights_only=True)
        model = MLPRecommender(checkpoint["n_features"], checkpoint["n_classes"]).net
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _mlp_cache = model
        _mlp_scaler_cache = joblib.load(os.path.join(MODELS_DIR, "mlp_scaler.joblib"))
    return _mlp_cache, _mlp_scaler_cache


def predict_rf(values: np.ndarray) -> dict:
    """Predict a transform using the RandomForest ('ml' method)."""
    feats = extract_features(values)
    x = features_to_vector(feats).reshape(1, -1)
    clf = _load_rf()
    encoder = _load_label_encoder()

    proba = clf.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    transform = encoder.inverse_transform([idx])[0]
    return {"transform": str(transform), "confidence": float(proba[idx]),
            "method": "ml", "features": feats}


def predict_mlp(values: np.ndarray) -> dict:
    """Predict a transform using the small PyTorch MLP ('dl' method)."""
    import torch

    feats = extract_features(values)
    x = features_to_vector(feats).reshape(1, -1)
    model, scaler = _load_mlp()
    encoder = _load_label_encoder()

    x_scaled = scaler.transform(x)
    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(proba))
    transform = encoder.inverse_transform([idx])[0]
    return {"transform": str(transform), "confidence": float(proba[idx]),
            "method": "dl", "features": feats}


def predict(values: np.ndarray, method: str = "ml") -> dict:
    if method == "ml":
        return predict_rf(values)
    if method == "dl":
        return predict_mlp(values)
    raise ValueError(f"Unknown ML method '{method}'. Use 'ml' or 'dl'.")
