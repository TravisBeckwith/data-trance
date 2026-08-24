import numpy as np
import pytest

from data_trance.transforms import apply_transform


def test_log_positive():
    v = np.array([1.0, np.e, np.e ** 2])
    out, warnings_ = apply_transform("log", v)
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0], atol=1e-10)
    assert warnings_ == []


def test_log_negative_becomes_nan():
    v = np.array([1.0, -1.0, 2.0])
    out, _ = apply_transform("log", v)
    assert np.isnan(out[1])
    assert not np.isnan(out[0])
    assert not np.isnan(out[2])


def test_sqrt_negative_becomes_nan():
    v = np.array([4.0, -4.0, 9.0])
    out, _ = apply_transform("sqrt", v)
    assert out[0] == 2.0
    assert np.isnan(out[1])
    assert out[2] == 3.0


def test_none_is_identity():
    v = np.array([1.0, 2.0, 3.0])
    out, _ = apply_transform("none", v)
    np.testing.assert_array_equal(out, v)


def test_zscore_mean_zero_std_one():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out, _ = apply_transform("zscore", v)
    assert abs(np.mean(out)) < 1e-10
    assert abs(np.std(out) - 1.0) < 1e-6


def test_minmax_bounds():
    v = np.array([5.0, 10.0, 15.0, 20.0])
    out, _ = apply_transform("minmax", v)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_arcsine_sqrt_domain():
    v = np.array([0.0, 0.5, 1.0, 1.5])
    out, _ = apply_transform("arcsine_sqrt", v)
    assert not np.isnan(out[0])
    assert not np.isnan(out[1])
    assert not np.isnan(out[2])
    assert np.isnan(out[3])  # 1.5 is out of [0,1] domain


def test_logit_open_interval():
    v = np.array([0.0, 0.5, 1.0])
    out, _ = apply_transform("logit", v)
    assert np.isnan(out[0])  # logit undefined at 0
    assert out[1] == 0.0     # logit(0.5) = 0
    assert np.isnan(out[2])  # logit undefined at 1


def test_boxcox_requires_positive():
    v = np.array([1.0, 2.0, -1.0, 3.0])
    out, _ = apply_transform("boxcox", v)
    assert np.isnan(out[2])
    assert not np.isnan(out[0])


def test_rank_orders_correctly():
    v = np.array([30.0, 10.0, 20.0])
    out, _ = apply_transform("rank", v)
    np.testing.assert_array_equal(out, [3.0, 1.0, 2.0])


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        apply_transform("not_a_real_transform", np.array([1.0, 2.0]))


def test_winsorize_caps_extremes():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    out, _ = apply_transform("winsorize", v, limits=(0.2, 0.2))
    assert out.max() < 100.0
