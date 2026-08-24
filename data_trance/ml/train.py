"""
data_trance.ml.train
======================
Trains the ML ("rf", a scikit-learn RandomForest) and DL ("mlp", a small
PyTorch classifier) transform recommenders on synthetic data, and saves
both to data_trance/ml/models/ so they ship with the package and don't
need retraining before first use.

Run with:  python -m data_trance.ml.train
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import FEATURE_NAMES
from .synth import generate_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_rf(X_train, y_train, X_test, y_test, label_encoder) -> dict:
    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  min_samples_leaf=2, random_state=0,
                                  class_weight="balanced")
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    joblib.dump(clf, os.path.join(MODELS_DIR, "rf_recommender.joblib"))
    return {"model": "rf", "train_accuracy": train_acc, "test_accuracy": test_acc}


class MLPRecommender:
    """A small feed-forward classifier: features -> transform name.
    Kept intentionally tiny -- this is a lookup-table-with-generalization
    problem on ~16 numeric features, not an image/language task, so a
    2-hidden-layer MLP is already more capacity than the problem needs."""

    def __init__(self, n_features: int, n_classes: int):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(n_features, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, n_classes),
        )


def train_mlp(X_train, y_train, X_test, y_test, label_encoder,
               epochs: int = 300, lr: float = 3e-3):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"model": "mlp", "skipped": True,
                "reason": "torch is not installed (pip install data-trance[dl])"}

    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # class weights to counter the natural imbalance (e.g. "none" and
    # "boxcox" dominate the synthetic label distribution)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_counts[class_counts == 0] = 1.0
    weights = (1.0 / class_counts)
    weights = weights / weights.sum() * n_classes

    torch.manual_seed(0)
    model = MLPRecommender(n_features, n_classes).net
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train_t).argmax(1) == y_train_t).float().mean().item()
        test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    torch.save({"state_dict": model.state_dict(),
                "n_features": n_features, "n_classes": n_classes},
               os.path.join(MODELS_DIR, "mlp_recommender.pt"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "mlp_scaler.joblib"))

    return {"model": "mlp", "train_accuracy": train_acc, "test_accuracy": test_acc}


def main(n_samples: int = 4000, seed: int = 0):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Generating {n_samples} synthetic training samples...")
    X, y = generate_dataset(n_samples=n_samples, seed=seed)
    print(f"Got {len(y)} usable samples across {len(set(y))} transform classes.")

    label_encoder = LabelEncoder().fit(y)
    y_enc = label_encoder.transform(y)
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc
        if min(np.bincount(y_enc)) >= 2 else None,
    )

    rf_report = train_rf(X_train, y_train, X_test, y_test, label_encoder)
    print(f"RandomForest  -- train acc: {rf_report['train_accuracy']:.3f}, "
          f"test acc: {rf_report['test_accuracy']:.3f}")

    mlp_report = train_mlp(X_train, y_train, X_test, y_test, label_encoder)
    if mlp_report.get("skipped"):
        print(f"MLP -- skipped: {mlp_report['reason']}")
    else:
        print(f"MLP           -- train acc: {mlp_report['train_accuracy']:.3f}, "
              f"test acc: {mlp_report['test_accuracy']:.3f}")

    return rf_report, mlp_report


if __name__ == "__main__":
    main()
