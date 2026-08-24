# Config reference

```yaml
input: path/to/data.csv       # required. Path to the input CSV.
output_dir: results           # optional, default "results". Where all output goes.
alpha: 0.05                   # optional, default 0.05. Significance level for the normality test.
recommend_method: rule_based  # optional, default "rule_based". One of: rule_based | ml | dl
                               # See docs/ml_recommender.md for what ml/dl do.
ml_min_confidence: 0.5        # optional, default 0.5. Below this confidence, ml/dl
                               # predictions fall back to the rule-based search.

columns:                      # optional. If omitted, every column in the CSV is processed
                               # with auto-detected type and no transform override.
  <column_name>:
    type: auto                # optional, default "auto". One of:
                               #   auto | continuous | count | proportion | correlation
                               #   | categorical | ordinal
    transform: null            # optional. Force a specific transform instead of letting
                               # `recommend` search for one. Must be a name from
                               # data_trance/transforms.py's REGISTRY (e.g. "log", "sqrt",
                               # "boxcox", "yeojohnson", "zscore", "none", ...).
                               # Ignored for categorical/ordinal columns.

steps:                        # optional, default: all six, in this order.
  - detect_type
  - assess
  - recommend
  - apply_transform
  - validate
  - report
```

## Notes

- **`columns` is optional.** If you omit it entirely, every column in the
  input CSV is processed with `type: auto` and no transform override. Use
  `columns` when you want to scope the run to specific columns, force a
  type, or force a transform.
- **`type: auto` vs. an explicit type.** Auto-detection is a heuristic
  (see `docs/steps.md`) and can be wrong for edge cases like a numeric
  Likert scale (1-5) that's really ordinal, or a percentage stored as
  0-100 instead of 0-1. If `detect_type.json`'s reasoning for a column
  looks off, override it explicitly.
- **`transform` overrides skip the search**, but the column still goes
  through `assess` and `validate` normally, so you still get before/after
  diagnostics and plots for it.
- **`steps` lets you run a subset** via `data-trance run config.yaml`, but
  each step still individually depends on its prerequisites having been
  run at least once (their JSON needs to exist in `output_dir`). Use
  `data-trance step <name> config.yaml` to run one step at a time instead.
