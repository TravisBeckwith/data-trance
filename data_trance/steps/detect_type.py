"""Step 1: detect_type -- guess or apply each column's variable type."""

import numpy as np
import pandas as pd

STEP_NAME = "detect_type"


def _guess(raw: pd.Series, numeric: pd.Series) -> tuple:
    n = len(raw)
    n_unique = raw.nunique(dropna=True)

    if raw.dtype == object or str(raw.dtype) == "category" or raw.dtype == bool:
        if numeric.notna().sum() == 0:
            return "categorical", f"non-numeric column ({n_unique} unique values)"

    n_bad = numeric.isna().sum() - raw.isna().sum()
    if n_bad > 0.3 * n:
        return "categorical", f"most values ({n_bad}/{n}) aren't numeric"

    vals = numeric.dropna()
    if len(vals) == 0:
        return "categorical", "no usable numeric values"

    if n_unique <= 10 and n_unique < 0.05 * n:
        return "categorical", (f"only {n_unique} distinct values across {n} rows "
                                f"-- looks discrete/categorical")

    is_integer = np.allclose(vals, np.round(vals))
    nonneg = (vals >= 0).all()
    bounded_01 = (vals >= 0).all() and (vals <= 1).all()
    bounded_pm1 = (vals >= -1).all() and (vals <= 1).all() and not bounded_01

    if bounded_01 and not is_integer:
        return "proportion", "all values in [0, 1], not all whole numbers"
    if bounded_pm1:
        return "correlation", "all values in (-1, 1)"
    if is_integer and nonneg:
        return "count", "all values are non-negative integers"
    return "continuous", "numeric, real-valued, not obviously bounded"


def run(ctx) -> dict:
    df = ctx.df
    columns = list(ctx.config.columns.keys()) or list(df.columns)
    result = {}

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {ctx.config.input}. "
                              f"Available: {list(df.columns)}")
        override = ctx.config.column_type(col)
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")

        if override != "auto":
            result[col] = {"type": override, "reasoning": "manually specified"}
        else:
            vtype, reasoning = _guess(raw, numeric)
            result[col] = {"type": vtype, "reasoning": reasoning}

    ctx.save(STEP_NAME, result)
    return result
