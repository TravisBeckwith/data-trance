import json
import os
import shutil

import pytest

from data_trance.config import load_config
from data_trance.context import Context
from data_trance.steps import apply_transform, assess, detect_type, recommend, report, validate

FIXTURE_CONFIG = os.path.join(os.path.dirname(__file__), "data", "fixture_config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "_test_results")


@pytest.fixture
def ctx():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    config = load_config(FIXTURE_CONFIG)
    yield Context(config)
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)


def test_full_pipeline_runs_and_produces_expected_files(ctx):
    detect_type.run(ctx)
    assess.run(ctx)
    recommend.run(ctx)
    apply_transform.run(ctx)
    validate.run(ctx)
    report.run(ctx)

    assert os.path.exists(os.path.join(RESULTS_DIR, "transformed.csv"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "report.md"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "summary.json"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "plots", "skewed_col__before.jpg"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "plots", "skewed_col__after.jpg"))


def test_normal_column_gets_no_transform(ctx):
    detect_type.run(ctx)
    assess.run(ctx)
    rec = recommend.run(ctx)
    assert rec["normal_col"]["chosen_transform"] == "none"


def test_skewed_column_gets_a_transform(ctx):
    detect_type.run(ctx)
    assess.run(ctx)
    rec = recommend.run(ctx)
    assert rec["skewed_col"]["chosen_transform"] != "none"


def test_summary_json_is_valid_and_complete(ctx):
    for step in (detect_type, assess, recommend, apply_transform, validate, report):
        step.run(ctx)
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        summary = json.load(f)
    assert "normal_col" in summary
    assert "skewed_col" in summary


def test_step_can_be_rerun_standalone_using_saved_json(ctx):
    detect_type.run(ctx)
    assess.run(ctx)
    recommend.run(ctx)
    # simulate a fresh context (new process) that only has the JSON on disk
    config = load_config(FIXTURE_CONFIG)
    fresh_ctx = Context(config)
    result = apply_transform.run(fresh_ctx)
    assert "normal_col" in result
    assert "skewed_col" in result


ML_FIXTURE_CONFIG = os.path.join(os.path.dirname(__file__), "data", "fixture_config_ml.yaml")
ML_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "_test_results_ml")


@pytest.fixture
def ml_ctx():
    if os.path.exists(ML_RESULTS_DIR):
        shutil.rmtree(ML_RESULTS_DIR)
    config = load_config(ML_FIXTURE_CONFIG)
    yield Context(config)
    shutil.rmtree(ML_RESULTS_DIR, ignore_errors=True)


def test_recommend_method_ml_runs_end_to_end(ml_ctx):
    detect_type.run(ml_ctx)
    assess.run(ml_ctx)
    rec = recommend.run(ml_ctx)
    for col in ("normal_col", "skewed_col"):
        assert rec[col]["chosen_transform"] is not None
        # reason should mention either an ML/DL prediction or a fallback
        # to the rule-based search (both are valid outcomes)
        assert rec[col]["reason"] != ""

