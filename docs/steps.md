# Pipeline steps

Each step is a Python module under `data_trance/steps/` with a single
`run(ctx)` function. `ctx` is a `Context` (see `data_trance/context.py`)
that gives access to the loaded input DataFrame, the parsed config, and
`ctx.load(step_name)` / `ctx.save(step_name, data)` for reading and writing
each step's JSON output.

## 1. `detect_type`

**Reads:** the input CSV.
**Writes:** `detect_type.json` — `{column: {type, reasoning}}`

For each configured column, uses the manual `type:` override from the
config if given, otherwise guesses from the data:

| Signal | Guessed type |
|---|---|
| non-numeric values, or dtype is object/category/bool | `categorical` |
| numeric but few distinct values relative to row count | `categorical` |
| all values in `[0, 1]`, not all whole numbers | `proportion` |
| all values in `(-1, 1)` | `correlation` |
| all values non-negative integers | `count` |
| everything else | `continuous` |

## 2. `assess`

**Reads:** `detect_type.json`.
**Writes:** `assess.json`, plus `plots/<col>__before.jpg` per numeric column.

For numeric columns: n, mean, median, std, min, max, skewness, excess
kurtosis, and a normality test (Shapiro-Wilk for n ≤ 5000, D'Agostino-Pearson
above that). Also renders a 3-panel plot: histogram with a fitted normal
curve overlay, a Q-Q plot, and a boxplot.

For categorical/ordinal columns: n, missing count, number of distinct
categories, and the top 10 categories by frequency.

## 3. `recommend`

**Reads:** `detect_type.json`, `assess.json`.
**Writes:** `recommend.json` — per column, the chosen transform, the reason,
its resulting stats, and the full candidate table.

For numeric columns, evaluates a shortlist of transforms appropriate to the
column's type and actual range (see `data_trance/steps/recommend.py:candidates_for`),
then picks one using this preference order:

1. **No transform**, if the raw data is already adequate (passes the
   normality test, or has low skew/kurtosis).
2. **An interpretable parametric transform** (log, sqrt, Box-Cox,
   Yeo-Johnson, arcsine, logit, ...) if one actually normalizes the data.
3. **The best partial parametric improvement**, if nothing fully
   normalizes but something helps meaningfully.
4. **A rank/quantile transform**, only as a last resort, with an explicit
   note about the interpretability/invertibility tradeoff.

This order exists specifically so the pipeline doesn't get fooled by
rank-based transforms trivially "winning" a naive skew/kurtosis comparison.

A `transform:` override in the config skips this search for that column.

For categorical/ordinal columns, recommends an encoding (binary / one-hot /
target-or-frequency / hashing) based on cardinality relative to row count.

## 4. `apply_transform`

**Reads:** `detect_type.json`, `recommend.json`.
**Writes:** `apply_transform.json`, `transformed.csv`.

Applies the chosen (or overridden) transform to each numeric column,
adding it as a new `<col>__<transform>` column — the original column is
left untouched. Categorical/ordinal columns are not auto-encoded (encoding
touches downstream modeling choices you should make deliberately); the
recommendation is noted but not applied.

## 5. `validate`

**Reads:** `apply_transform.json`, `assess.json`.
**Writes:** `validate.json`, plus `plots/<col>__after.jpg`.

Re-runs the same diagnostics from `assess` on each transformed column,
using the identical plotting function so the before/after images are
directly comparable. Flags whether the transform actually reduced combined
skew + excess kurtosis versus the original.

## 6. `report`

**Reads:** all prior steps' JSON.
**Writes:** `report.md`, `summary.json`.

Assembles everything into one human-readable Markdown report with embedded
before/after plots per column, plus a compact machine-readable summary.
