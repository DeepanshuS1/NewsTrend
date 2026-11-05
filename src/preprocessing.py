"""
Preprocessing module for text normalization, missing-value handling,
date parsing, and engagement feature engineering.
Strictly separates transformations to prevent data leakage.
"""

from typing import Tuple, Optional, List
import html
import re
import numpy as np
import pandas as pd

from config import (
    COL_HEADLINE,
    COL_TITLE,
    COL_TOPIC,
    COL_DATE,
    COL_SOURCE,
    COL_ENGAGEMENT_FB,
    COL_ENGAGEMENT_GP,
    COL_ENGAGEMENT_LI,
    COL_TOTAL_ENGAGEMENT,
)


def clean_text_string(
    text: Optional[str],
    lowercase: bool = True,
    remove_punctuation: bool = False,
    strip_html_entities: bool = True,
) -> str:
    """
    Normalize and clean an individual text string.
    
    Args:
        text: Raw input text.
        lowercase: Whether to convert text to lowercase.
        remove_punctuation: Whether to strip punctuation characters.
        strip_html_entities: Whether to unescape HTML entities (e.g. &#39; -> ').
        
    Returns:
        str: Normalized text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = text

    # Unescape HTML entities
    if strip_html_entities:
        cleaned = html.unescape(cleaned)

    # Normalize whitespace (replace tabs, newlines, multiple spaces)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Optional punctuation removal (for specific word-count/bag-of-words tasks)
    if remove_punctuation:
        cleaned = re.sub(r"[^\w\s]", "", cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def preprocess_dataset(
    df: pd.DataFrame,
    text_col: str = COL_HEADLINE,
    target_col: Optional[str] = COL_TOPIC,
    date_col: Optional[str] = COL_DATE,
    drop_duplicates: bool = False,
    min_text_len: int = 5,
) -> pd.DataFrame:
    """
    Perform non-leaking end-to-end cleaning and feature standardization on DataFrame.
    Preserves original columns for display while creating cleaned features.
    
    Args:
        df: Input raw DataFrame.
        text_col: Column to use as primary text.
        target_col: Target category column (if available).
        date_col: Datetime column (if available).
        drop_duplicates: Whether to drop identical rows.
        min_text_len: Minimum character length for valid text.
        
    Returns:
        pd.DataFrame: Cleaned and enriched DataFrame.
    """
    data = df.copy()

    # 1. Deduplication if requested
    if drop_duplicates:
        data = data.drop_duplicates()

    # 2. Text Column Handling & Preservation
    if text_col not in data.columns:
        # Fallback to Title if Headline is missing
        if COL_TITLE in data.columns:
            text_col = COL_TITLE
        else:
            raise ValueError(f"Specified text column '{text_col}' not found in dataset.")

    # Preserve original text for user interface display
    data["OriginalText"] = data[text_col].fillna("").astype(str)

    # Generate normalized text
    data["CleanText"] = data["OriginalText"].apply(clean_text_string)

    # If both Title and Headline exist, create an enriched CombinedText feature
    if COL_TITLE in data.columns and COL_HEADLINE in data.columns:
        data["CombinedText"] = (
            data[COL_TITLE].fillna("").astype(str) + " " + data[COL_HEADLINE].fillna("").astype(str)
        ).apply(clean_text_string)
    else:
        data["CombinedText"] = data["CleanText"]

    # Filter out empty or uninformative text
    data["TextLength"] = data["CleanText"].str.len()
    data["WordCount"] = data["CleanText"].apply(lambda s: len(s.split()))

    # Mask valid text records
    valid_text_mask = data["TextLength"] >= min_text_len
    data = data[valid_text_mask].copy()

    # 3. Target Label Handling
    if target_col and target_col in data.columns:
        data[target_col] = data[target_col].astype(str).str.strip().str.lower()
        # Filter out NaN or null target labels
        data = data[data[target_col].notnull() & (data[target_col] != "nan") & (data[target_col] != "")].copy()

    # 4. Date Parsing
    if date_col and date_col in data.columns:
        parsed_dates = pd.to_datetime(data[date_col], errors="coerce")
        data["ParsedDate"] = parsed_dates
        data["HasValidDate"] = parsed_dates.notnull()
        
        # Extract temporal components where valid
        data["Year"] = parsed_dates.dt.year
        data["Month"] = parsed_dates.dt.month
        data["YearMonth"] = parsed_dates.dt.to_period("M").astype(str)
        data["DateOnly"] = parsed_dates.dt.date
        data["DayOfWeek"] = parsed_dates.dt.day_name()
        data["Hour"] = parsed_dates.dt.hour
    else:
        data["HasValidDate"] = False
        data["ParsedDate"] = pd.NaT

    # 5. Engagement Fields Normalization
    for eng_col in [COL_ENGAGEMENT_FB, COL_ENGAGEMENT_GP, COL_ENGAGEMENT_LI]:
        if eng_col in data.columns:
            data[eng_col] = pd.to_numeric(data[eng_col], errors="coerce").fillna(-1)
            # Clip negative untracked indicator (-1) to 0 for non-negative engagement analytics
            data[f"{eng_col}_Valid"] = data[eng_col] >= 0
            data[f"{eng_col}_Count"] = data[eng_col].apply(lambda v: max(0, v))

    # Calculate Total Engagement if any platform exists
    eng_count_cols = [
        f"{c}_Count" for c in [COL_ENGAGEMENT_FB, COL_ENGAGEMENT_GP, COL_ENGAGEMENT_LI] 
        if f"{c}_Count" in data.columns
    ]
    if eng_count_cols:
        data[COL_TOTAL_ENGAGEMENT] = data[eng_count_cols].sum(axis=1)
        data["LogTotalEngagement"] = np.log1p(data[COL_TOTAL_ENGAGEMENT])
        # Track whether the post had tracked engagement
        val_cols = [f"{c}_Valid" for c in [COL_ENGAGEMENT_FB, COL_ENGAGEMENT_GP, COL_ENGAGEMENT_LI] if f"{c}_Valid" in data.columns]
        data["HasEngagementData"] = data[val_cols].any(axis=1)
    else:
        data["HasEngagementData"] = False
        data[COL_TOTAL_ENGAGEMENT] = 0
        data["LogTotalEngagement"] = 0.0

    return data


def extract_engagement_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract strictly pre-publication features to predict engagement without target leakage.
    Features include text length, word count, title/headline sentiment, and one-hot topic.
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Text-based features
    features["char_length"] = df["TextLength"].fillna(0)
    features["word_count"] = df["WordCount"].fillna(0)
    features["exclamation_count"] = df["OriginalText"].apply(lambda s: s.count("!"))
    features["question_count"] = df["OriginalText"].apply(lambda s: s.count("?"))

    # 2. Pre-computed Sentiment features (if present)
    if "SentimentHeadline" in df.columns:
        features["sentiment_headline"] = pd.to_numeric(df["SentimentHeadline"], errors="coerce").fillna(0.0)
    if "SentimentTitle" in df.columns:
        features["sentiment_title"] = pd.to_numeric(df["SentimentTitle"], errors="coerce").fillna(0.0)

    # 3. Topic dummy variables (pre-publication category)
    if COL_TOPIC in df.columns:
        topic_dummies = pd.get_dummies(df[COL_TOPIC], prefix="topic", drop_first=True, dtype=float)
        features = pd.concat([features, topic_dummies], axis=1)

    target = df["LogTotalEngagement"]
    return features, target

