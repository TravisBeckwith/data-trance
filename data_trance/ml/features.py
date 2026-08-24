"""
data_trance.ml.features
========================
Extracts a fixed-length numeric feature vector describing a column's
distribution shape. This is what the ML/DL recommender models see instead
of the raw data -- cheap to compute, and (unlike the raw values) the same
size regardless of how many rows the column has, so it's what makes
batch-predicting across thousands of columns (or voxels) fast.
"""

import numpy as np
from scipy import stats

FEATURE_NAMES = [
    "n", "mean", "std", "cv", "skew", "kurtosis_excess",
    "min", "max", "range", "iqr_over_range",
    "frac_zero", "frac_negative", "frac_positive",
    "is_integer_like", "normality_p", "log_n",
]


def extract_features(values: np.ndarray) -> dict:
    """Compute the feature vector for one numeric column's values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        raise ValueError("Need at least 3 finite values to extract features")

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    cv = std / abs(mean) if mean != 0 else 0.0

    with np.errstate(all="ignore"):
        skew = float(stats.skew(v))
        kurt = float(stats.kurtosis(v, fisher=True))
    skew = skew if np.isfinite(skew) else 0.0
    kurt = kurt if np.isfinite(kurt) else 0.0

    vmin, vmax = float(np.min(v)), float(np.max(v))
    vrange = vmax - vmin if vmax > vmin else 1.0
    q1, q3 = np.percentile(v, [25, 75])
    iqr_over_range = (q3 - q1) / vrange

    frac_zero = float(np.mean(v == 0))
    frac_negative = float(np.mean(v < 0))
    frac_positive = float(np.mean(v > 0))
    is_integer_like = float(np.allclose(v, np.round(v)))

    if n <= 5000:
        try:
            _s, p = stats.shapiro(v)
        except ValueError:
            p = 1.0
    else:
        _s, p = stats.normaltest(v)
    normality_p = float(p) if np.isfinite(p) else 0.0

    return {
        "n": float(n), "mean": mean, "std": std, "cv": cv,
        "skew": skew, "kurtosis_excess": kurt,
        "min": vmin, "max": vmax, "range": vrange,
        "iqr_over_range": float(iqr_over_range),
        "frac_zero": frac_zero, "frac_negative": frac_negative,
        "frac_positive": frac_positive, "is_integer_like": is_integer_like,
        "normality_p": normality_p, "log_n": float(np.log(n)),
    }


def features_to_vector(feats: dict) -> np.ndarray:
    """Order a feature dict into the fixed vector the models expect."""
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)
