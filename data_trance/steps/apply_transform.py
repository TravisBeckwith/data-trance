"""Step 4: apply_transform -- apply recommended transforms, write output CSV."""

import os

import pandas as pd

from ..transforms import apply_transform as _apply

STEP_NAME = "apply_transform"


def run(ctx) -> dict:
    df = ctx.df.copy()
    types = ctx.load("detect_type")
    recs = ctx.load("recommend")
    result = {}

    for col, rec in recs.items():
        vtype = rec["type"]

        if vtype in ("categorical", "ordinal"):
            result[col] = {"applied": None, "note": f"encoding recommendation only "
                                                      f"({rec.get('recommendation')}); "
                                                      f"not applied automatically"}
            continue

        chosen = rec.get("chosen_transform")
        if chosen is None or chosen == "none":
            result[col] = {"applied": None, "note": "no transform needed"}
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        out, warnings_ = _apply(chosen, numeric.to_numpy(dtype=float))
        new_col = f"{col}__{chosen}"
        df[new_col] = out
        result[col] = {"applied": chosen, "new_column": new_col, "warnings": warnings_}

    out_path = os.path.join(ctx.config.output_dir, "transformed.csv")
    df.to_csv(out_path, index=False)
    result["_output_csv"] = out_path
    ctx._df = df  # so downstream steps in the same run see the new columns

    ctx.save(STEP_NAME, result)
    return result
