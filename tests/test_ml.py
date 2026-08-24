import numpy as np
import pytest

from data_trance.ml.features import FEATURE_NAMES, extract_features, features_to_vector
from data_trance.ml.predict import predict
from data_trance.ml.synth import _infer_vtype, generate_dataset


def test_extract_features_returns_all_expected_keys():
    v = np.random.normal(0, 1, 100)
    feats = extract_features(v)
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_extract_features_too_few_values_raises():
    with pytest.raises(ValueError):
        extract_features(np.array([1.0, 2.0]))


def test_features_to_vector_is_ordered_and_fixed_length():
    v = np.random.normal(0, 1, 100)
    feats = extract_features(v)
    vec = features_to_vector(feats)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec[FEATURE_NAMES.index("n")] == 100.0


def test_skewed_data_has_positive_skew_feature():
    v = np.random.lognormal(0, 1, 500)
    feats = extract_features(v)
    assert feats["skew"] > 0.5


def test_infer_vtype_proportion():
    v = np.random.uniform(0.01, 0.99, 100)
    assert _infer_vtype(v) == "proportion"


def test_infer_vtype_count():
    v = np.random.poisson(5, 100).astype(float)
    assert _infer_vtype(v) == "count"


def test_generate_dataset_produces_valid_shapes():
    X, y = generate_dataset(n_samples=30, seed=1)
    assert len(y) > 0
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert all(isinstance(name, str) for name in y)


def test_generate_dataset_is_deterministic_given_seed():
    X1, y1 = generate_dataset(n_samples=20, seed=42)
    X2, y2 = generate_dataset(n_samples=20, seed=42)
    np.testing.assert_array_equal(X1, X2)
    assert y1 == y2


def test_predict_ml_returns_valid_transform_and_confidence():
    v = np.random.lognormal(0, 1, 500)
    result = predict(v, method="ml")
    assert "transform" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["method"] == "ml"


def test_predict_dl_returns_valid_transform_and_confidence():
    v = np.random.lognormal(0, 1, 500)
    result = predict(v, method="dl")
    assert "transform" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["method"] == "dl"


def test_predict_agrees_on_obviously_normal_data():
    v = np.random.normal(0, 1, 2000)
    ml_result = predict(v, method="ml")
    assert ml_result["transform"] == "none"
    assert ml_result["confidence"] > 0.5


def test_predict_unknown_method_raises():
    v = np.random.normal(0, 1, 100)
    with pytest.raises(ValueError):
        predict(v, method="not_a_real_method")
