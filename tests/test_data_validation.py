"""
Unit tests for data loading, schema detection, and quality auditing.
"""

import pytest
import pandas as pd
import io
from pathlib import Path

from src.data_loader import (
    load_dataset,
    detect_schema_mapping,
    inspect_data_quality,
    validate_required_columns,
    DatasetLoadError,
)
from config import SAMPLE_DATASET_PATH, SYNTHETIC_DATASET_PATH


def test_load_sample_dataset_success():
    """Verify loading default sample dataset succeeds and returns valid metadata."""
    df, meta = load_dataset(SAMPLE_DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Headline" in df.columns
    assert "Topic" in df.columns
    assert meta["is_synthetic"] is False
    assert meta["total_rows"] == len(df)


def test_load_synthetic_demo_flag():
    """Verify synthetic demonstration dataset is correctly flagged."""
    df, meta = load_dataset(SYNTHETIC_DATASET_PATH)
    assert meta["is_synthetic"] is True
    assert "IsSyntheticDemo" in df.columns


def test_load_empty_dataset_raises():
    """Verify attempting to load an empty CSV raises DatasetLoadError."""
    empty_csv = io.StringIO("")
    with pytest.raises(DatasetLoadError):
        load_dataset(empty_csv)


def test_load_nonexistent_file_raises():
    """Verify attempting to load a non-existent path raises DatasetLoadError."""
    with pytest.raises(DatasetLoadError):
        load_dataset(Path("data/non_existent_file_xyz123.csv"))


def test_detect_schema_mapping():
    """Verify automatic detection and mapping of canonical column roles."""
    sample_df = pd.DataFrame({
        "news_headline": ["Global markets rally"],
        "category": ["economy"],
        "timestamp": ["2016-03-01 12:00:00"],
        "outlet": ["Reuters"],
        "facebook_shares": [120]
    })
    mapping = detect_schema_mapping(sample_df)
    assert mapping["text_column"] == "news_headline"
    assert mapping["category_column"] == "category"
    assert mapping["date_column"] == "timestamp"
    assert mapping["source_column"] == "outlet"
    assert "facebook_shares" in mapping["engagement_columns"]


def test_inspect_data_quality():
    """Verify calculation of missing values and duplicate rows."""
    df = pd.DataFrame({
        "Headline": ["Headline A", "Headline A", None],
        "Topic": ["economy", "economy", "obama"]
    })
    audit = inspect_data_quality(df)
    assert audit["missing_counts"]["Headline"] == 1
    assert audit["total_duplicates"] == 1
    assert audit["has_missing"] is True
    assert audit["has_duplicates"] is True


def test_validate_required_columns():
    """Verify column existence validation helper."""
    df = pd.DataFrame({"colA": [1], "colB": [2]})
    valid, missing = validate_required_columns(df, ["colA", "colB"])
    assert valid is True
    assert missing == []

    invalid, missing = validate_required_columns(df, ["colA", "colC"])
    assert invalid is False
    assert missing == ["colC"]

