import numpy as np
import pandas as pd

from data_trance.steps.detect_type import _guess


def test_detects_categorical_strings():
    raw = pd.Series(["red", "blue", "green", "red"])
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, _ = _guess(raw, numeric)
    assert vtype == "categorical"


def test_detects_proportion():
    raw = pd.Series(np.random.uniform(0.01, 0.99, 100))
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, _ = _guess(raw, numeric)
    assert vtype == "proportion"


def test_detects_count():
    raw = pd.Series(np.random.poisson(5, 100))
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, _ = _guess(raw, numeric)
    assert vtype == "count"


def test_detects_correlation():
    raw = pd.Series(np.random.uniform(-0.9, 0.9, 100))
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, _ = _guess(raw, numeric)
    assert vtype == "correlation"


def test_detects_continuous():
    raw = pd.Series(np.random.normal(50, 10, 100))
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, _ = _guess(raw, numeric)
    assert vtype == "continuous"


def test_low_cardinality_numeric_flagged_categorical():
    raw = pd.Series([1, 2, 3, 1, 2, 3, 1, 2] * 20)  # only 3 distinct values, 160 rows
    numeric = pd.to_numeric(raw, errors="coerce")
    vtype, reasoning = _guess(raw, numeric)
    assert vtype == "categorical"
    assert "distinct" in reasoning
