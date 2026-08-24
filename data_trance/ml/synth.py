"""
data_trance.ml.synth
======================
Generates synthetic labeled training data for the ML/DL recommenders.

There's no external "ground truth" dataset of correct transforms, so we
manufacture one: sample many columns from known distributions (normal,
lognormal, gamma, beta, Poisson, mixtures, ...), extract features for each,
and label each with whatever the existing rule-based search in
`steps/recommend.py` would choose. The ML/DL models then learn to
approximate that rule engine's decisions directly from features -- which is
what makes them fast enough to batch-predict across many columns at once,
without re-running the full candidate search every time.
"""

import numpy as np

from ..steps.recommend import evaluate_candidates, pick_recommendation
from .features import extract_features, features_to_vector


def _infer_vtype(v: np.ndarray) -> str:
    """Same lightweight type inference used by detect_type, inlined here so
    synthetic data doesn't need a full DataFrame/Context round-trip."""
    is_integer = np.allclose(v, np.round(v))
    nonneg = (v >= 0).all()
    bounded_01 = (v >= 0).all() and (v <= 1).all()
    bounded_pm1 = (v >= -1).all() and (v <= 1).all() and not bounded_01
    if bounded_01 and not is_integer:
        return "proportion"
    if bounded_pm1:
        return "correlation"
    if is_integer and nonneg:
        return "count"
    return "continuous"


def _sample_distribution(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw one synthetic column from a randomly chosen distribution family
    with randomized parameters, so the training set covers a wide range of
    shapes rather than just a handful of fixed cases."""
    family = rng.choice([
        "normal", "lognormal", "gamma", "exponential", "beta", "uniform",
        "poisson", "chi2", "weibull", "pareto", "mixture_normal",
        "student_t", "triangular",
    ])

    if family == "normal":
        return rng.normal(rng.uniform(-50, 50), rng.uniform(0.5, 20), n)
    if family == "lognormal":
        return rng.lognormal(rng.uniform(-1, 2), rng.uniform(0.3, 1.5), n)
    if family == "gamma":
        return rng.gamma(rng.uniform(0.5, 8), rng.uniform(0.5, 5), n)
    if family == "exponential":
        return rng.exponential(rng.uniform(0.5, 10), n)
    if family == "beta":
        return rng.beta(rng.uniform(0.3, 8), rng.uniform(0.3, 8), n)
    if family == "uniform":
        lo = rng.uniform(-20, 20)
        return rng.uniform(lo, lo + rng.uniform(1, 40), n)
    if family == "poisson":
        return rng.poisson(rng.uniform(0.5, 30), n).astype(float)
    if family == "chi2":
        return rng.chisquare(rng.uniform(1, 10), n)
    if family == "weibull":
        return rng.weibull(rng.uniform(0.5, 5), n) * rng.uniform(1, 10)
    if family == "pareto":
        return (rng.pareto(rng.uniform(1, 5), n) + 1) * rng.uniform(1, 10)
    if family == "mixture_normal":
        n1 = n // 2
        a = rng.normal(rng.uniform(-30, 0), rng.uniform(1, 10), n1)
        b = rng.normal(rng.uniform(0, 30), rng.uniform(1, 10), n - n1)
        return np.concatenate([a, b])
    if family == "student_t":
        return rng.standard_t(rng.uniform(2, 15), n) * rng.uniform(1, 10)
    if family == "triangular":
        lo = rng.uniform(-20, 0)
        hi = lo + rng.uniform(5, 40)
        mode = rng.uniform(lo, hi)
        return rng.triangular(lo, mode, hi, n)
    raise AssertionError(family)


def generate_dataset(n_samples: int = 3000, min_n: int = 30, max_n: int = 2000,
                      alpha: float = 0.05, seed: int = 0):
    """Returns (X, y) for training the recommender models. X is
    (n_samples, n_features); y is a list of chosen transform names."""
    rng = np.random.default_rng(seed)
    X, y = [], []

    attempts = 0
    while len(y) < n_samples and attempts < n_samples * 3:
        attempts += 1
        n = int(rng.integers(min_n, max_n))
        v = _sample_distribution(rng, n)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            continue

        vtype = _infer_vtype(v)
        try:
            feats = extract_features(v)
        except ValueError:
            continue

        candidates = evaluate_candidates(vtype, v)
        if not candidates:
            continue
        chosen, _reason = pick_recommendation(candidates, alpha)

        X.append(features_to_vector(feats))
        y.append(chosen["transform"])

    return np.array(X), y
