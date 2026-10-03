"""Smoke tests for preprocessing and path layout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cansat_ai.config import RAW_DATA_PATH, SCALE_FEATURES
from cansat_ai.preprocess import add_time_features
import pandas as pd


def test_raw_dataset_exists():
    assert RAW_DATA_PATH.exists(), f"Missing raw dataset: {RAW_DATA_PATH}"


def test_time_features_are_added():
    df = pd.read_csv(RAW_DATA_PATH, nrows=5)
    out = add_time_features(df)
    for column in ("Hour", "Day", "Month", "Weekday"):
        assert column in out.columns
    for column in SCALE_FEATURES:
        assert column in out.columns
