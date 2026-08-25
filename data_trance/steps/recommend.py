"""
Step 3: recommend -- search a shortlist of candidate transforms and pick one.

Deliberately NOT just 'whichever has the lowest skew/kurtosis score': rank
and quantile-based transforms can force near-perfect normality by
construction (they only preserve order, not real distances), so they'd
always "win" a naive comparison even when the raw data was already fine.
Preference order: none (if adequate) > interpretable parametric transform
that normalizes > best partial parametric improvement > rank/quantile
fallback, with an explicit note about the interpretability tradeoff.
"""

import numpy as np
import pandas as pd
from scipy import stats

from ..transforms import apply_transform
from .assess import normality_test

STEP_NAME = "recommend"


def _ml_predict_lazy(*args, **kwargs):
    """Imported lazily so the rule-based-only path never requires sklearn's
    trained model files or torch to be present."""
    from ..ml.predict import predict
    return predict(*args, **kwargs)


ml_predict = _ml_predict_lazy

NONPARAMETRIC = {"quantile_normal", "van_der_waerden"}


def candidates_for(vtype: str, vals: np.ndarray) -> list:
    all_positive = (vals > 0).all()
    nonneg = (vals >= 0).all()

    if vtype == "proportion":
        return ["none", "arcsine_sqrt", "logit", "probit", "yeojohnson",
                "quantile_normal", "van_der_waerden"]
    if vtype == "correlation":
        return ["none", "fisher_z", "yeojohnson", "quantile_normal",
                "van_der_waerden"]
    if vtype == "count":
        base = ["none", "sqrt", "anscombe", "freeman_tukey", "yeojohnson",
                "quantile_normal", "van_der_waerden"]
        if all_positive:
            base[1:1] = ["log1p", "log", "boxcox"]
        return base

    cands = ["none", "cbrt", "yeojohnson", "quantile_normal", "van_der_waerden"]
    if all_positive:
        cands[1:1] = ["log", "sqrt", "boxcox"]
    elif nonneg:
        cands.insert(1, "sqrt")
    else:
        cands.append("square")
    return cands


def evaluate_candidates(vtype: str, vals: np.ndarray) -> list:
    names = candidates_for(vtype, vals)
    rows = []
    for name in names:
        try:
            out, _warnings = apply_transform(name, vals)
            out = out[np.isfinite(out)]
            if len(out) < max(10, 0.5 * len(vals)):
                continue
            with np.errstate(all="ignore"):
                skew = float(stats.skew(out))
                kurt = float(stats.kurtosis(out, fisher=True))
            if not (np.isfinite(skew) and np.isfinite(kurt)):
                continue
            _s, p, test_name = normality_test(out)
            rows.append({
                "transform": name, "n_used": int(len(out)),
                "skew": skew, "kurtosis_excess": kurt,
                "normality_p": float(p) if p is not None else None,
                "normality_test": test_name,
                "score": abs(skew) + abs(kurt),
            })
        except Exception:
            continue
    rows.sort(key=lambda r: r["score"])
    return rows


def pick_recommendation(results: list, alpha: float):
    by_name = {r["transform"]: r for r in results}
    none_r = by_name.get("none")

    def adequate(r):
        if r is None:
            return False
        if r["normality_p"] is not None and r["normality_p"] > alpha:
            return True
        return abs(r["skew"]) < 0.5 and abs(r["kurtosis_excess"]) < 0.5

    if adequate(none_r):
        return none_r, "already_adequate"

    parametric = [r for r in results if r["transform"] not in NONPARAMETRIC
                  and r["transform"] != "none"]
    nonparametric = [r for r in results if r["transform"] in NONPARAMETRIC]
    parametric.sort(key=lambda r: r["score"])
    nonparametric.sort(key=lambda r: r["score"])

    passing = [r for r in parametric if adequate(r)]
    if passing:
        return passing[0], "parametric_normalizes"

    none_score = none_r["score"] if none_r else float("inf")
    best_param = parametric[0] if parametric else None

    if best_param is not None and best_param["score"] < none_score:
        if best_param["score"] < 1.5:
            return best_param, "parametric_improves"
        if nonparametric and nonparametric[0]["score"] < 0.7 * best_param["score"]:
            return nonparametric[0], "nonparametric_fallback"
        return best_param, "parametric_partial"

    if nonparametric and none_r and nonparametric[0]["score"] < none_score:
        return nonparametric[0], "nonparametric_fallback"

    return none_r, "nothing_helps"


def recommend_encoding(n_unique: int, n_rows: int, ordinal: bool) -> tuple:
    if ordinal:
        return "ordinal", "preserves natural ordering, unlike one-hot"
    if n_unique <= 2:
        return "binary", "map to 0/1"
    if n_unique <= 10:
        return "one_hot", "low cardinality -- one-hot won't blow up dimensionality"
    if n_unique <= max(50, 0.05 * n_rows):
        return "target_or_frequency", "moderate cardinality -- one-hot too expensive"
    return "hashing_or_embedding", "high cardinality -- one-hot/target overfit-prone"


def run(ctx) -> dict:
    df = ctx.df
    types = ctx.load("detect_type")
    assessment = ctx.load("assess")
    result = {}

    for col, info in types.items():
        vtype = info["type"]

        if vtype in ("categorical", "ordinal"):
            n_unique = assessment[col]["stats"]["n_unique"]
            n_rows = assessment[col]["stats"]["n"]
            encoding, reason = recommend_encoding(n_unique, n_rows, vtype == "ordinal")
            result[col] = {"type": vtype, "recommendation": encoding, "reason": reason}
            continue

        override = ctx.config.column_transform_override(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        vals = numeric.dropna().to_numpy(dtype=float)
        if len(vals) < 3:
            result[col] = {"type": vtype, "error": "not enough numeric data"}
            continue

        candidates = evaluate_candidates(vtype, vals)
        if not candidates:
            result[col] = {"type": vtype, "error": "no candidate transform could be evaluated"}
            continue

        method = ctx.config.recommend_method
        if override:
            chosen = next((r for r in candidates if r["transform"] == override), None)
            reason = "manual_override"
            if chosen is None:
                out, _ = apply_transform(override, vals)
                out = out[np.isfinite(out)]
                with np.errstate(all="ignore"):
                    skew, kurt = float(stats.skew(out)), float(stats.kurtosis(out, fisher=True))
                chosen = {"transform": override, "n_used": len(out), "skew": skew,
                          "kurtosis_excess": kurt, "normality_p": None,
                          "normality_test": None, "score": abs(skew) + abs(kurt)}
            result[col] = {"type": vtype, "chosen_transform": chosen["transform"],
                            "reason": reason, "chosen_stats": chosen, "candidates": candidates}
            continue

        if method in ("ml", "dl"):
            ml_result = None
            ml_error = None
            try:
                ml_result = ml_predict(vals, method=method)
            except (FileNotFoundError, ImportError) as e:
                ml_error = str(e)

            min_confidence = ctx.config.ml_min_confidence
            if ml_result is not None and ml_result["confidence"] >= min_confidence:
                chosen_name = ml_result["transform"]
                chosen = next((r for r in candidates if r["transform"] == chosen_name), None)
                if chosen is None:
                    # model predicted a transform outside this column's valid
                    # candidate set (e.g. proportion-only transform on
                    # continuous data) -- don't trust it, fall back
                    chosen, fallback_reason = pick_recommendation(candidates, ctx.config.alpha)
                    reason = f"ml_predicted_invalid_candidate_fallback({fallback_reason})"
                else:
                    reason = f"{method}_predicted(confidence={ml_result['confidence']:.2f})"
            else:
                chosen, rule_reason = pick_recommendation(candidates, ctx.config.alpha)
                if ml_error:
                    reason = f"{rule_reason}(ml_unavailable)"
                else:
                    reason = (f"{rule_reason}(ml_low_confidence="
                              f"{ml_result['confidence']:.2f})")
        else:
            chosen, reason = pick_recommendation(candidates, ctx.config.alpha)

        result[col] = {
            "type": vtype,
            "chosen_transform": chosen["transform"],
            "reason": reason,
            "chosen_stats": chosen,
            "candidates": candidates,
        }

    ctx.save(STEP_NAME, result)
    return result
