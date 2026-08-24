"""Step 2: assess -- descriptive diagnostics and before/after-style plots."""

import os

import numpy as np
import pandas as pd
from scipy import stats

from ..plotting import distribution_panel

STEP_NAME = "assess"


def normality_test(vals: np.ndarray):
    n = len(vals)
    if n < 3:
        return None, None, "n too small"
    if n <= 5000:
        s, p = stats.shapiro(vals)
        return s, p, "Shapiro-Wilk"
    s, p = stats.normaltest(vals)
    return s, p, "D'Agostino-Pearson"


def describe_numeric(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    with np.errstate(all="ignore"):
        skew = float(stats.skew(v))
        kurt = float(stats.kurtosis(v, fisher=True))
    s, p, test_name = normality_test(v)
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)), "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
        "min": float(np.min(v)), "max": float(np.max(v)),
        "skew": skew, "kurtosis_excess": kurt,
        "normality_test": test_name,
        "normality_stat": float(s) if s is not None else None,
        "normality_p": float(p) if p is not None else None,
    }


def assess_categorical(raw: pd.Series) -> dict:
    vc = raw.value_counts(dropna=True)
    return {
        "n": int(len(raw)),
        "missing": int(raw.isna().sum()),
        "n_unique": int(len(vc)),
        "top_categories": {str(k): int(v) for k, v in vc.head(10).items()},
    }


def run(ctx) -> dict:
    df = ctx.df
    types = ctx.load("detect_type")
    result = {}

    for col, info in types.items():
        vtype = info["type"]
        raw = df[col]

        if vtype in ("categorical", "ordinal"):
            result[col] = {"type": vtype, "stats": assess_categorical(raw)}
            continue

        numeric = pd.to_numeric(raw, errors="coerce")
        vals = numeric.dropna().to_numpy(dtype=float)
        if len(vals) < 3:
            result[col] = {"type": vtype, "error": "fewer than 3 numeric values"}
            continue

        stats_dict = describe_numeric(vals)
        plot_path = os.path.join(ctx.config.output_dir, "plots", f"{col}__before.jpg")
        distribution_panel(vals, f"{col} (before transform)", plot_path)

        result[col] = {"type": vtype, "stats": stats_dict, "plot": plot_path}

    ctx.save(STEP_NAME, result)
    return result
