"""Step 5: validate -- confirm the transform actually helped, with after-plots."""

import os

import pandas as pd

from ..plotting import distribution_panel
from .assess import describe_numeric

STEP_NAME = "validate"


def run(ctx) -> dict:
    applied = ctx.load("apply_transform")
    before = ctx.load("assess")

    out_csv = applied.get("_output_csv")
    df = pd.read_csv(out_csv) if out_csv else ctx.df

    result = {}
    for col, info in applied.items():
        if col.startswith("_"):
            continue
        new_col = info.get("new_column")
        if not new_col:
            result[col] = {"validated": False, "note": info.get("note", "not applicable")}
            continue

        vals = pd.to_numeric(df[new_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 3:
            result[col] = {"validated": False, "note": "not enough data after transform"}
            continue

        after_stats = describe_numeric(vals)
        plot_path = os.path.join(ctx.config.output_dir, "plots", f"{col}__after.jpg")
        distribution_panel(vals, f"{new_col} (after transform)", plot_path)

        before_stats = before.get(col, {}).get("stats", {})
        improved = None
        if before_stats:
            before_badness = abs(before_stats.get("skew", 0)) + abs(before_stats.get("kurtosis_excess", 0))
            after_badness = abs(after_stats["skew"]) + abs(after_stats["kurtosis_excess"])
            improved = after_badness < before_badness

        result[col] = {
            "validated": True,
            "before": before_stats,
            "after": after_stats,
            "improved": improved,
            "plot": plot_path,
        }

    ctx.save(STEP_NAME, result)
    return result
