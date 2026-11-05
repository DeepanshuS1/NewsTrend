"""
Data loader module for the Social Media News Trend Analysis and Engagement Prediction project.
Handles dataset loading, validation, schema verification, data quality auditing,
and auto-detection of column roles.
"""

from typing import Dict, Any, Tuple, Optional, Union
import io
import pandas as pd
from pathlib import Path

from config import (
    FULL_DATASET_PATH,
    SAMPLE_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    COL_HEADLINE,
    COL_TITLE,
    COL_TOPIC,
    COL_DATE,
    COL_SOURCE,
    ENGAGEMENT_COLS,
    EXPECTED_CATEGORIES,
)


class DatasetLoadError(Exception):
    """Custom exception raised when dataset loading or validation fails."""
    pass


def load_dataset(
    source: Union[str, Path, io.BytesIO, io.StringIO, pd.DataFrame] = SAMPLE_DATASET_PATH,
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset from a file path, file buffer, or existing DataFrame.
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Loaded DataFrame and dataset metadata dictionary.
    """
    is_synthetic = False
    source_name = "Custom / In-Memory"
    
    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            source_name = "In-Memory DataFrame"
        elif isinstance(source, (io.BytesIO, io.StringIO)):
            df = pd.read_csv(source, nrows=max_rows)
            source_name = getattr(source, "name", "Uploaded CSV Buffer")
        else:
            path = Path(source)
            source_name = path.name
            if not path.exists():
                raise DatasetLoadError(f"Data file not found at: {path}")
            df = pd.read_csv(path, nrows=max_rows)
            if "synthetic" in path.name.lower():
                is_synthetic = True
    except Exception as e:
        if isinstance(e, DatasetLoadError):
            raise
        raise DatasetLoadError(f"Failed to read CSV: {str(e)}") from e

    if df.empty:
        raise DatasetLoadError("The loaded dataset is empty (0 records).")

    # Check for synthetic flags in columns or content
    if "IsSyntheticDemo" in df.columns:
        is_synthetic = bool(df["IsSyntheticDemo"].any())
    elif "[SYNTHETIC" in source_name or "synthetic" in source_name.lower():
        is_synthetic = True

    # Detect schema and mappings
    schema_info = detect_schema_mapping(df)
    
    # Compile quality audit
    quality_audit = inspect_data_quality(df)

    metadata = {
        "source_name": source_name,
        "is_synthetic": is_synthetic,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "schema_mapping": schema_info,
        "quality_audit": quality_audit,
    }

    return df, metadata


def detect_schema_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently inspect and map DataFrame columns to canonical roles:
    text, title, category, date, source, and engagement.
    """
    cols_lower = {col.lower(): col for col in df.columns}
    
    # 1. Headline / Primary Text
    text_col = None
    for candidate in ["headline", "text", "content", "post", "news_headline", "title"]:
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break
            
    # 2. Title
    title_col = cols_lower.get("title", None)
    
    # 3. Category / Topic
    category_col = None
    for candidate in ["topic", "category", "label", "class", "news_category", "section"]:
        if candidate in cols_lower:
            category_col = cols_lower[candidate]
            break

    # 4. Date
    date_col = None
    for candidate in ["publishdate", "publish_date", "date", "timestamp", "datetime", "created_at"]:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break

    # 5. Source / Publisher
    source_col = None
    for candidate in ["source", "publisher", "outlet", "news_source", "author"]:
        if candidate in cols_lower:
            source_col = cols_lower[candidate]
            break

    # 6. Engagement columns
    found_eng_cols = [c for c in ENGAGEMENT_COLS if c in df.columns]
    if not found_eng_cols:
        found_eng_cols = [
            c for c in df.columns 
            if any(term in c.lower() for term in ["facebook", "linkedin", "googleplus", "shares", "likes", "engagement"])
        ]

    return {
        "text_column": text_col,
        "title_column": title_col,
        "category_column": category_col,
        "date_column": date_col,
        "source_column": source_col,
        "engagement_columns": found_eng_cols,
    }


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive statistical data quality report:
    missing values, duplicate rows, memory usage, and empty strings.
    """
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = ((df.isnull().sum() / len(df)) * 100).round(2).to_dict()
    
    # Check duplicates across all columns and across text columns
    total_duplicates = int(df.duplicated().sum())
    
    text_col = COL_HEADLINE if COL_HEADLINE in df.columns else (COL_TITLE if COL_TITLE in df.columns else None)
    text_duplicates = int(df[text_col].duplicated().sum()) if text_col else 0

    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "total_duplicates": total_duplicates,
        "text_duplicates": text_duplicates,
        "memory_mb": memory_mb,
        "has_missing": any(v > 0 for v in missing_counts.values()),
        "has_duplicates": total_duplicates > 0,
    }


def validate_required_columns(
    df: pd.DataFrame, 
    required_cols: list[str]
) -> Tuple[bool, list[str]]:
    """
    Check if required columns exist in the DataFrame.
    Returns (is_valid, missing_columns_list).
    """
    missing = [col for col in required_cols if col not in df.columns]
    return (len(missing) == 0, missing)

