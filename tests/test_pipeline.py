"""Smoke tests for the training pipeline modules. Tasks fill in deeper tests."""

from __future__ import annotations


def test_imports():
    from src import config, ingest, feature_engineering, preprocess, train, register  # noqa: F401


def test_config_paths_resolve():
    from src import config

    assert config.PROJECT_ROOT.exists()
    assert config.RAW_TRAIN_CSV.name == "train.csv"
    assert config.MLFLOW_TRACKING_URI.startswith("sqlite:///")
    assert config.AUTOML_METRIC == "macro_f1"
    assert len(config.AUTOML_ESTIMATORS) >= 4
