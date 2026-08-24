# Changelog

## v0.2.0 — ML/DL transform recommender

### Added

**ML/DL recommendation methods** — `recommend_method: ml` or `dl` in the
config swaps the rule-based candidate search for a trained model that
predicts a transform directly from a 16-feature summary of the column
(skew, kurtosis, n, boundedness signals, normality p-value, ...), instead
of applying every candidate transform to the data. This exists primarily
as scaffolding for scaling to many-column/voxel-wise data (e.g. MRI/VBM),
where re-running the full rule-based search per column isn't tractable but
predicting from cheap features is.

- **`ml`** — a scikit-learn RandomForestClassifier. ~82% test accuracy
  across 13 transform classes on held-out synthetic data.
- **`dl`** — a small PyTorch MLP (16 → 96 → 48 → n_classes) trained on the
  same data. ~64% test accuracy — weaker than the RandomForest, included
  as a genuine second model and as infrastructure for future work, gated
  behind the optional `[dl]` extra.
- **Synthetic training data** (`data_trance/ml/synth.py`) — since there's
  no external ground-truth dataset of "correct" transforms, both models
  are trained on thousands of synthetic columns sampled from a mix of
  distribution families (normal, lognormal, gamma, exponential, beta,
  uniform, Poisson, chi-squared, Weibull, Pareto, Student's t, a normal
  mixture, triangular) and labeled with whatever the existing rule-based
  search would have chosen — so the models learn to approximate the rule
  engine's decisions, not invent a separate notion of correctness.
- **Confidence-gated fallback** — every ML/DL prediction includes a
  confidence score; predictions below `ml_min_confidence` (default 0.5),
  predictions outside the column's valid candidate set, or `dl` requests
  when torch isn't installed all fall back to the rule-based search
  automatically, with the reason recorded (e.g.
  `already_adequate(ml_low_confidence=0.31)`,
  `parametric_normalizes(ml_unavailable)`). Nothing silently trusts a weak
  guess or crashes on a missing optional dependency.
- **`data-trance train-recommender [--n-samples N] [--seed S]`** — CLI
  command to regenerate the synthetic dataset and retrain both models.
- Trained model artifacts ship with the package (`data_trance/ml/models/`)
  so `ml`/`dl` work out of the box without a training step first.
- 13 new tests covering feature extraction, synthetic data generation
  (including determinism given a fixed seed), both models' predictions on
  clearly-normal and clearly-skewed data, and a full pipeline integration
  test with `recommend_method: ml`.
- New docs: `docs/ml_recommender.md`.

### Changed

- Core dependencies now include `joblib` (needed for the RandomForest
  regardless of whether the `dl` extra is installed).
- `pyproject.toml` gained a `[dl]` optional-dependencies group (`torch`).

## v0.1.0 — initial release

First release of `data-trance`: a config-driven pipeline that assesses a
column's distribution before transforming it, rather than assuming
`log()` and moving on.

### Added

**Pipeline (6 steps, each independently runnable and chainable via JSON on disk)**
- `detect_type` — auto-detects `continuous` / `count` / `proportion` /
  `correlation` / `categorical` / `ordinal` per column, or takes a manual
  override from the config.
- `assess` — skewness, excess kurtosis, and a normality test (Shapiro-Wilk
  or D'Agostino-Pearson depending on sample size), plus a 3-panel
  distribution plot (histogram with normal-curve overlay, Q-Q plot,
  boxplot) per numeric column. Cardinality stats for categorical columns.
- `recommend` — searches a type-appropriate shortlist of candidate
  transforms and picks one using an explicit preference order: no
  transform (if already adequate) → an interpretable parametric transform
  that normalizes the data → the best partial parametric improvement → a
  rank/quantile fallback as a last resort. This order is deliberate: it
  keeps the pipeline from being fooled by rank-based transforms, which can
  force near-perfect normality by construction and would otherwise win a
  naive skew/kurtosis comparison every time, even against already-fine raw
  data. Categorical columns get an encoding recommendation instead
  (binary / one-hot / target-or-frequency / hashing) based on cardinality.
- `apply_transform` — applies the recommended or manually-overridden
  transform, writing each result as a new `<col>__<transform>` column and
  leaving the original data untouched.
- `validate` — re-runs the same diagnostics on the transformed column
  using the identical plotting function as `assess`, so before/after
  images are directly comparable, and flags whether the transform actually
  reduced combined skew + kurtosis.
- `report` — assembles everything into one Markdown report with embedded
  before/after plots, plus a machine-readable `summary.json`.

**Transform library** (`data_trance/transforms.py`) — 26 named transforms
covering the power family (square, cube, sqrt, cbrt, reciprocal, general
power), the log family (log, log1p, log-shift), Box-Cox and Yeo-Johnson,
variance-stabilizing transforms for counts and proportions (Anscombe,
Freeman-Tukey, arcsine-sqrt, logit, probit, Fisher z), scaling/normalization
(z-score, min-max, robust scale, quantile-to-normal, quantile-to-uniform),
rank-based transforms (rank, rank-percentile, van der Waerden), winsorizing,
and time-series differencing/percent-change. All transforms return NaN for
out-of-domain values instead of raising, so one bad row doesn't kill a run.

**CLI**
- `data-trance run <config.yaml>` — runs all configured steps in order.
- `data-trance step <name> <config.yaml>` — runs a single step, reading
  prior steps' output from disk if it isn't already in memory. A missing
  dependency produces a clear error message naming which step to run
  first, instead of a traceback.

**Config** — one YAML file drives everything: `input`, `output_dir`,
`alpha`, per-column `type` and optional `transform` overrides, and which
`steps` to run.

**Tests** — 30 tests covering the transform library (domain edge cases:
negative inputs to log/sqrt, out-of-range proportions, unknown transform
names), type detection across all six type categories, the recommendation
logic (including a dedicated test that skewed data gets a parametric
transform rather than falling back to rank-based ones), categorical
encoding thresholds, and a full pipeline integration test that runs all
six steps against fixture data and checks every expected output file gets
created, including cross-process step chaining via the on-disk JSON.

**Docs** — `docs/steps.md` (what each step reads/writes/decides) and
`docs/config_reference.md` (every config key).

**Examples** — `examples/sample_dataset.csv` (a synthetic dataset covering
normal, right-skewed, count, proportion, and categorical columns) with a
matching `config.example.yaml`.

### Known limitations

- Categorical/ordinal columns get an encoding *recommendation* only —
  `apply_transform` does not auto-encode them, since encoding choice
  affects downstream modeling in ways that deserve a deliberate decision
  rather than automation.
- The candidate transform shortlist per type is a fixed, hand-picked list
  (see `recommend.py:candidates_for`) rather than an exhaustive search —
  by design, to keep runs fast and results interpretable.
- No support yet for assessing/transforming many columns in a single
  parallelized pass at scale (e.g. voxel-wise data); see project notes on
  a possible future MRI/VBM-oriented mode.
