import numpy as np

from data_trance.steps.recommend import (
    evaluate_candidates,
    pick_recommendation,
    recommend_encoding,
)


def test_already_normal_data_recommends_none():
    np.random.seed(0)
    vals = np.random.normal(50, 10, 1000)
    candidates = evaluate_candidates("continuous", vals)
    chosen, reason = pick_recommendation(candidates, alpha=0.05)
    assert reason == "already_adequate"
    assert chosen["transform"] == "none"


def test_right_skewed_data_gets_a_parametric_transform_not_rank():
    np.random.seed(0)
    vals = np.random.lognormal(0, 1, 1000)
    candidates = evaluate_candidates("continuous", vals)
    chosen, reason = pick_recommendation(candidates, alpha=0.05)
    # should NOT fall back to a rank/quantile transform when a simple
    # parametric one (log, boxcox, etc.) works fine
    assert chosen["transform"] not in ("quantile_normal", "van_der_waerden")
    assert reason in ("parametric_normalizes", "parametric_improves")


def test_proportion_data_uses_proportion_candidates():
    np.random.seed(0)
    vals = np.random.beta(2, 5, 1000)
    candidates = evaluate_candidates("proportion", vals)
    names = {c["transform"] for c in candidates}
    assert "arcsine_sqrt" in names or "logit" in names or "probit" in names
    # log/boxcox shouldn't be candidates for proportion data
    assert "log" not in names


def test_recommend_encoding_binary():
    encoding, _ = recommend_encoding(n_unique=2, n_rows=1000, ordinal=False)
    assert encoding == "binary"


def test_recommend_encoding_one_hot_for_low_cardinality():
    encoding, _ = recommend_encoding(n_unique=3, n_rows=1000, ordinal=False)
    assert encoding == "one_hot"


def test_recommend_encoding_respects_ordinal_flag():
    encoding, _ = recommend_encoding(n_unique=5, n_rows=1000, ordinal=True)
    assert encoding == "ordinal"


def test_recommend_encoding_high_cardinality():
    encoding, _ = recommend_encoding(n_unique=800, n_rows=1000, ordinal=False)
    assert encoding == "hashing_or_embedding"
