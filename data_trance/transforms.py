"""
data_trance.transforms
=======================
The actual statistical transform functions, shared by the recommend and
apply_transform steps. Each function takes a 1D numpy array and returns a
1D numpy array of the same length (NaN for values outside the transform's
valid domain, e.g. log of a negative number).
"""

import warnings

import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


def _safe(f, v):
    with np.errstate(all="ignore"):
        return f(v)


# ---- power family ---------------------------------------------------

def t_none(v, **_):
    return v.copy()


def t_square(v, **_):
    return v ** 2


def t_cube(v, **_):
    return v ** 3


def t_sqrt(v, **_):
    return _safe(lambda x: np.where(x >= 0, np.sqrt(x), np.nan), v)


def t_cbrt(v, **_):
    return np.cbrt(v)


def t_reciprocal(v, **_):
    return _safe(lambda x: np.where(x != 0, 1.0 / x, np.nan), v)


def t_power(v, power=0.5, **_):
    p = float(power)
    return _safe(lambda x: np.sign(x) * np.abs(x) ** p, v)


# ---- log family -------------------------------------------------------

def t_log(v, **_):
    return _safe(lambda x: np.where(x > 0, np.log(x), np.nan), v)


def t_log1p(v, **_):
    return _safe(lambda x: np.where(x > -1, np.log1p(x), np.nan), v)


def t_log_shift(v, shift=None, **_):
    if shift is None:
        m = np.nanmin(v)
        shift = (1 - m) if m <= 0 else 0
    return _safe(lambda x: np.where((x + shift) > 0, np.log(x + shift), np.nan), v)


# ---- Box-Cox / Yeo-Johnson --------------------------------------------

def t_boxcox(v, lmbda=None, **_):
    v = np.asarray(v, dtype=float)
    mask = v > 0
    result = np.full_like(v, np.nan)
    if mask.sum() > 1:
        if lmbda is None:
            transformed, _fitted = stats.boxcox(v[mask])
        else:
            transformed = stats.boxcox(v[mask], lmbda=float(lmbda))
        result[mask] = transformed
    return result


def t_yeojohnson(v, lmbda=None, **_):
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    if lmbda is None:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        return pt.fit_transform(v).ravel()
    return np.array([stats.yeojohnson(x, lmbda=float(lmbda)) for x in v.ravel()])


# ---- variance-stabilizing (counts / proportions / correlations) ------

def t_anscombe(v, **_):
    return _safe(lambda x: np.where(x >= -0.375, np.sqrt(x + 0.375), np.nan), v)


def t_freeman_tukey(v, **_):
    return _safe(lambda x: np.where(x >= 0, np.sqrt(x) + np.sqrt(x + 1), np.nan), v)


def t_arcsine_sqrt(v, **_):
    return _safe(lambda x: np.where((x >= 0) & (x <= 1), np.arcsin(np.sqrt(x)), np.nan), v)


def t_logit(v, **_):
    return _safe(lambda x: np.where((x > 0) & (x < 1), np.log(x / (1 - x)), np.nan), v)


def t_probit(v, **_):
    return _safe(lambda x: np.where((x > 0) & (x < 1), stats.norm.ppf(x), np.nan), v)


def t_fisher_z(v, **_):
    return _safe(lambda x: np.where((x > -1) & (x < 1), np.arctanh(x), np.nan), v)


# ---- scaling / normalization -------------------------------------------

def t_zscore(v, **_):
    mu, sd = np.nanmean(v), np.nanstd(v)
    if sd == 0 or np.isnan(sd):
        warnings.warn("zscore: standard deviation is 0")
        return np.full_like(v, np.nan, dtype=float)
    return (v - mu) / sd


def t_minmax(v, **_):
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi == lo:
        warnings.warn("minmax: constant column")
        return np.full_like(v, np.nan, dtype=float)
    return (v - lo) / (hi - lo)


def t_robust_scale(v, **_):
    med = np.nanmedian(v)
    q1, q3 = np.nanpercentile(v, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        warnings.warn("robust_scale: IQR is 0")
        return np.full_like(v, np.nan, dtype=float)
    return (v - med) / iqr


def t_quantile_normal(v, **_):
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    n_q = max(min(1000, len(v)), 10)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_q, random_state=0)
    return qt.fit_transform(v).ravel()


def t_quantile_uniform(v, **_):
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    n_q = max(min(1000, len(v)), 10)
    qt = QuantileTransformer(output_distribution="uniform", n_quantiles=n_q, random_state=0)
    return qt.fit_transform(v).ravel()


# ---- rank-based ---------------------------------------------------------

def _rank(v):
    order = np.argsort(np.argsort(v))
    return order.astype(float) + 1


def t_rank(v, **_):
    return _rank(v)


def t_rank_pct(v, **_):
    return _rank(v) / len(v)


def t_van_der_waerden(v, **_):
    n = len(v)
    r = _rank(v)
    return stats.norm.ppf(r / (n + 1))


# ---- robustness -----------------------------------------------------------

def t_winsorize(v, limits=(0.05, 0.05), **_):
    lo_p, hi_p = float(limits[0]), float(limits[1])
    lo_val = np.nanquantile(v, lo_p)
    hi_val = np.nanquantile(v, 1 - hi_p)
    return np.clip(v, lo_val, hi_val)


# ---- time series ------------------------------------------------------

def t_diff(v, periods=1, **_):
    out = np.full_like(v, np.nan, dtype=float)
    p = int(periods)
    if p < len(v):
        out[p:] = v[p:] - v[:-p]
    return out


def t_pct_change(v, periods=1, **_):
    out = np.full_like(v, np.nan, dtype=float)
    p = int(periods)
    if p < len(v):
        with np.errstate(divide="ignore", invalid="ignore"):
            out[p:] = (v[p:] - v[:-p]) / v[:-p]
    return out


REGISTRY = {
    "none": (t_none, "no transform (identity)"),
    "square": (t_square, "x^2"),
    "cube": (t_cube, "x^3"),
    "sqrt": (t_sqrt, "sqrt(x); requires x >= 0"),
    "cbrt": (t_cbrt, "cube root; handles negatives"),
    "reciprocal": (t_reciprocal, "1/x; requires x != 0"),
    "power": (t_power, "sign(x)*|x|^p; Tukey ladder of powers"),
    "log": (t_log, "natural log; requires x > 0"),
    "log1p": (t_log1p, "log(1+x); requires x > -1"),
    "log_shift": (t_log_shift, "log(x + shift)"),
    "boxcox": (t_boxcox, "Box-Cox family; requires x > 0"),
    "yeojohnson": (t_yeojohnson, "Yeo-Johnson family; handles zero/negative x"),
    "anscombe": (t_anscombe, "sqrt(x + 3/8); for Poisson counts"),
    "freeman_tukey": (t_freeman_tukey, "sqrt(x)+sqrt(x+1); for Poisson counts"),
    "arcsine_sqrt": (t_arcsine_sqrt, "arcsin(sqrt(p)); for proportions in [0,1]"),
    "logit": (t_logit, "log(p/(1-p)); for proportions in (0,1)"),
    "probit": (t_probit, "inverse normal CDF; for proportions in (0,1)"),
    "fisher_z": (t_fisher_z, "arctanh(r); for correlation-like values in (-1,1)"),
    "zscore": (t_zscore, "standardize: (x - mean) / std"),
    "minmax": (t_minmax, "scale to [0, 1]"),
    "robust_scale": (t_robust_scale, "(x - median) / IQR"),
    "quantile_normal": (t_quantile_normal, "rank-based mapping to a normal distribution"),
    "quantile_uniform": (t_quantile_uniform, "rank-based mapping to a uniform distribution"),
    "rank": (t_rank, "replace with rank (1 = smallest)"),
    "rank_pct": (t_rank_pct, "rank scaled to [0, 1]"),
    "van_der_waerden": (t_van_der_waerden, "normal-scores transform via ranks"),
    "winsorize": (t_winsorize, "cap extremes at given percentiles"),
    "diff": (t_diff, "first-order differencing"),
    "pct_change": (t_pct_change, "percent change from prior row"),
}


def apply_transform(name: str, values: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a named transform to a numpy array, returning a new array."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown transform '{name}'. Known: {sorted(REGISTRY)}")
    func, _ = REGISTRY[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = np.asarray(func(np.asarray(values, dtype=float), **kwargs), dtype=float)
    return out, [str(w.message) for w in caught]
