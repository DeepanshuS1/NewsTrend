"""
Unit tests for text preprocessing, normalization, date parsing, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    clean_text_string,
    preprocess_dataset,
    extract_engagement_features,
)


def test_clean_text_string_normalization():
    """Verify lowercasing, whitespace collapsing, and HTML entity unescaping."""
    raw = "  Obama&#39;s   Speech  &quot;Historic&quot;   \n\t  "
    cleaned = clean_text_string(raw, lowercase=True, remove_punctuation=False)
    assert cleaned == "obama's speech \"historic\""


def test_clean_text_string_empty_and_none():
    """Verify clean_text_string handles None and empty string safely."""
    assert clean_text_string(None) == ""
    assert clean_text_string("") == ""
    assert clean_text_string("   ") == ""


def test_preprocess_dataset_features():
    """Verify feature generation: CleanText, CombinedText, length, date parsing, and engagement."""
    raw_df = pd.DataFrame({
        "Title": ["Market Update"],
        "Headline": ["Stocks rise in early morning trade."],
        "Topic": ["economy"],
        "PublishDate": ["2016-03-01 10:00:00"],
        "Facebook": [10],
        "GooglePlus": [2],
        "LinkedIn": [5],
    })

    cleaned_df = preprocess_dataset(raw_df)
    assert "OriginalText" in cleaned_df.columns
    assert "CleanText" in cleaned_df.columns
    assert "CombinedText" in cleaned_df.columns
    assert "TextLength" in cleaned_df.columns
    assert "WordCount" in cleaned_df.columns
    assert "ParsedDate" in cleaned_df.columns
    assert bool(cleaned_df["HasValidDate"].iloc[0]) is True
    assert cleaned_df["Year"].iloc[0] == 2016
    assert cleaned_df["TotalEngagement"].iloc[0] == 17
    assert cleaned_df["LogTotalEngagement"].iloc[0] == pytest.approx(np.log1p(17), 0.01)

    # Verify automatic schema mapping for custom column names
    custom_df = pd.DataFrame({
        "news_headline": ["Central bank announces monetary policy rate decision."],
        "category": ["Economy"],
        "timestamp": ["2016-05-15 09:30:00"],
    })
    custom_clean = preprocess_dataset(custom_df)
    assert len(custom_clean) == 1
    assert custom_clean["Topic"].iloc[0] == "economy"
    assert bool(custom_clean["HasValidDate"].iloc[0]) is True


def test_preprocess_dataset_negative_engagement_clipping():
    """Verify untracked engagement indicators (-1) are clipped to 0 for sum but flagged."""
    raw_df = pd.DataFrame({
        "Title": ["Headline Title"],
        "Headline": ["Valid news excerpt text here."],
        "Topic": ["microsoft"],
        "PublishDate": ["2016-04-10 14:00:00"],
        "Facebook": [-1],
        "GooglePlus": [-1],
        "LinkedIn": [15],
    })

    cleaned_df = preprocess_dataset(raw_df)
    # Total should equal 0 + 0 + 15 = 15
    assert cleaned_df["TotalEngagement"].iloc[0] == 15
    # Has engagement is True because LinkedIn >= 0
    assert bool(cleaned_df["HasEngagementData"].iloc[0]) is True


def test_extract_engagement_features():
    """Verify pre-publication engagement feature extraction without leakage."""
    df = pd.DataFrame({
        "OriginalText": ["Breaking: Major economic breakthrough announced!"],
        "TextLength": [48],
        "WordCount": [6],
        "SentimentHeadline": [0.35],
        "SentimentTitle": [0.20],
        "Topic": ["economy"],
        "LogTotalEngagement": [np.log1p(100)],
    })

    features, target = extract_engagement_features(df)
    assert "char_length" in features.columns
    assert "word_count" in features.columns
    assert "exclamation_count" in features.columns
    assert features["exclamation_count"].iloc[0] == 1
    assert target.iloc[0] == pytest.approx(np.log1p(100), 0.01)
