# data-trance

*Put your data in a different state before you transform it.*

`data-trance` is a config-driven pipeline that looks at a column before
touching it: it detects the variable's type, runs the diagnostics (skew,
kurtosis, a normality test, and a 3-panel distribution plot), recommends a
transform, applies it, and then validates that the transform actually
helped — all from one YAML file.

It exists because "just log it" is bad advice more often than people admit.
The recommendation step deliberately does **not** just chase the lowest
skew/kurtosis score: rank-based transforms can force near-perfect normality
by construction (they only preserve order, not real distances), so a naive
score comparison would pick them every time — even when the raw data was
already fine, or a simple `log`/`sqrt`/Box-Cox would have done the job with
far less cost to interpretability. `data-trance` prefers, in order: no
transform (if the data's already adequate) → an interpretable parametric
transform that actually normalizes the data → the best partial parametric
improvement available → a rank/quantile fallback, only as a last resort,
with an explicit note about what you're trading away.

## Install

```bash
pip install -e .              # base install
pip install -e ".[dl]"        # + PyTorch, for the dl recommender method
# or, without installing the package:
pip install -r requirements.txt
```

## Quickstart

```bash
data-trance run config.example.yaml
```

This runs all six steps against `examples/sample_dataset.csv` and writes
everything to `results/`:

```
results/
├── transformed.csv       # your original columns + new <col>__<transform> columns
├── report.md             # human-readable report with embedded before/after plots
├── summary.json          # machine-readable summary
├── detect_type.json      # per-step intermediate output (one per step)
├── assess.json
├── recommend.json
├── apply_transform.json
├── validate.json
└── plots/
    ├── <col>__before.jpg
    └── <col>__after.jpg
```

## Config

```yaml
input: examples/sample_dataset.csv
output_dir: results
alpha: 0.05                 # significance level for the normality test

columns:
  price:
    type: auto               # or: continuous, count, proportion, correlation, categorical, ordinal
  click_rate:
    type: proportion
  rating:
    type: ordinal
  region:
    type: categorical
    # transform: log        # optional: force a specific transform instead of the recommendation

steps: [detect_type, assess, recommend, apply_transform, validate, report]
```

See [`docs/config_reference.md`](docs/config_reference.md) for every key.

## recommend_method: rule_based | ml | dl

By default the `recommend` step uses the rule-based candidate search
described above. Set `recommend_method` to `ml` or `dl` to use a trained
model instead:

```yaml
recommend_method: ml     # or: dl
ml_min_confidence: 0.5   # below this confidence, fall back to the rule-based search
```

- **`ml`** — a scikit-learn RandomForest, trained on ~4000 synthetic
  columns (drawn from normal, lognormal, gamma, beta, Poisson, mixture,
  and other distribution families) labeled with whatever the rule-based
  search would have chosen. ~82% test accuracy across 13 transform classes.
- **`dl`** — a small PyTorch MLP trained the same way. Requires the `[dl]`
  extra (`pip install -e ".[dl]"`); if torch isn't installed, or if the
  model's top prediction is below `ml_min_confidence`, the pipeline falls
  back to the rule-based search automatically and records why in the
  `reason` field (e.g. `already_adequate(ml_low_confidence=0.31)` or
  `parametric_normalizes(ml_unavailable)`) — it never silently trusts a
  low-confidence guess or crashes because torch is missing.

Both models predict from a fixed 16-feature vector (skew, kurtosis, n,
% zero, % negative, coefficient of variation, normality p-value, ...) — see
`data_trance/ml/features.py`. This is what makes them fast enough to
batch-predict across many columns at once, rather than re-running the full
candidate search every time. That property is also the reason this exists:
the rule-based search doesn't scale to something like voxel-wise MRI data
(hundreds of thousands of columns), but a trained model predicting from
per-voxel features does.

To retrain on more/different synthetic data:

```bash
data-trance train-recommender --n-samples 8000
```



Every step reads what it needs from the previous steps' JSON output on
disk if it isn't already in memory, so each step is runnable on its own —
useful for inspecting or re-running just one part of the pipeline:

```bash
data-trance step detect_type config.example.yaml
data-trance step assess config.example.yaml
data-trance step recommend config.example.yaml
```

If a step's dependency hasn't been run yet, you get a clear error telling
you which step to run first, rather than a crash.

See [`docs/steps.md`](docs/steps.md) for what each step does and produces.

## What's actually going on under the hood

- **detect_type** — guesses `continuous` / `count` / `proportion` /
  `correlation` / `categorical` / `ordinal` from the data's range,
  boundedness, and integer-ness, or uses your override from the config.
- **assess** — descriptive stats, skewness, excess kurtosis, a normality
  test (Shapiro-Wilk or D'Agostino-Pearson depending on sample size), and a
  histogram+Q-Q+boxplot panel per numeric column. Categorical columns get
  cardinality stats instead.
- **recommend** — searches a shortlist of transforms appropriate to the
  column's type and actual range (won't try `log` on negative values, won't
  try `arcsine_sqrt` outside [0,1]), then applies the preference order
  described above. Categorical columns get an encoding recommendation
  (binary / one-hot / target-or-frequency / hashing) based on cardinality.
- **apply_transform** — applies the recommended (or overridden) transform,
  writes `<col>__<transform>` as a new column, leaves the original intact.
- **validate** — re-runs the diagnostics on the transformed column and
  produces a matching after-plot, so you can see the before/after side by
  side and confirm the transform actually helped.
- **report** — assembles everything into one Markdown report with embedded
  plots, plus a machine-readable `summary.json`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
