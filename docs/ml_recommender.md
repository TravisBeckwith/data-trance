# The ML/DL transform recommender

## Why this exists

The rule-based search in `recommend.py` works by actually applying every
candidate transform to the real data and measuring the result. That's
precise, but it doesn't scale: it can't run per-voxel across an MRI volume
with hundreds of thousands of columns, or anywhere else you need to
recommend a transform for many columns very fast.

The ML/DL recommenders solve that by predicting from a small, fixed-size
feature vector instead of touching the raw data at inference time. That
makes prediction fast and batchable, at the cost of being an
*approximation* of the rule-based search rather than an exact re-derivation
of it.

## Where the training labels come from

There's no external "ground truth" dataset of correct transforms. Instead,
`data_trance/ml/synth.py` generates thousands of synthetic columns from a
mix of distribution families (normal, lognormal, gamma, exponential, beta,
uniform, Poisson, chi-squared, Weibull, Pareto, Student's t, a normal
mixture, triangular) with randomized parameters and sample sizes, and
labels each one with whatever the *existing rule-based search* would have
chosen for it. In other words: the ML/DL models are trained to imitate the
rule-based recommender, not to invent a different notion of "correct."

This means retraining is cheap and self-contained — no manually-labeled
data to maintain — but also means the ML/DL models can never be more
"correct" than the rule-based search they're approximating. If you improve
`pick_recommendation()`'s logic, retrain to propagate that improvement into
the models.

## Features

Sixteen numeric features per column, computed in
`data_trance/ml/features.py`: `n`, `mean`, `std`, `cv` (coefficient of
variation), `skew`, `kurtosis_excess`, `min`, `max`, `range`,
`iqr_over_range`, `frac_zero`, `frac_negative`, `frac_positive`,
`is_integer_like`, `normality_p`, `log_n`.

These are chosen to be cheap to compute (all closed-form, no fitting) and
to capture the same signals the rule-based search's domain-applicability
checks use (positivity, boundedness, integer-ness) plus its scoring
signals (skew, kurtosis, normality).

## Models

- **`ml` — RandomForestClassifier** (scikit-learn). ~82% test accuracy
  across 13 transform classes on held-out synthetic data (vs. a ~7.7%
  random baseline). This is the default/recommended method when you want
  ML over the rule-based search — no extra dependency beyond what's
  already required.
- **`dl` — a small feed-forward network** (PyTorch): 16 → 96 → 48 →
  n_classes, with dropout and class-balanced loss weighting to counter the
  natural imbalance in the synthetic labels (`none` and `boxcox` dominate).
  ~64% test accuracy — meaningfully weaker than the RandomForest. It's
  included because it's a genuine second, independently-trained model
  sharing the same features/data, useful as a comparison point and as
  infrastructure to build on (e.g. a model architecture more suited to
  per-voxel prediction later).

Both are intentionally small. This is a ~16-feature tabular classification
problem, not an image or language task — more capacity wouldn't obviously
help and would cost more to train/ship.

## Confidence-gated fallback

Every prediction returns a confidence score (the predicted class's
softmax/predict_proba probability). The `recommend` step compares this
against `ml_min_confidence` (default 0.5): below that threshold, it
discards the ML/DL prediction and falls back to the rule-based search,
recording why in the `reason` field. The same fallback fires if the
predicted transform isn't actually in that column's valid candidate set
(e.g. a proportion-only transform predicted for continuous data), or if
`dl` is requested but torch isn't installed. The pipeline never silently
trusts a low-confidence guess or crashes for a missing optional dependency.

## Retraining

```bash
data-trance train-recommender --n-samples 8000 --seed 0
```

Regenerates the synthetic dataset, retrains both models, and overwrites
the artifacts in `data_trance/ml/models/`. Do this after changing
`candidates_for()` or `pick_recommendation()` in `recommend.py`, since
those define the labels the models are trained to reproduce.
