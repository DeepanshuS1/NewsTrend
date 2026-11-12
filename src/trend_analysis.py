"""
Trend Analysis module for temporal and topical news patterns.
Provides timeline aggregations, topic momentum tracking, period-over-period comparisons,
and keyword trend extraction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

from config import COL_TOPIC, COL_DATE


def analyze_temporal_trends(
    df: pd.DataFrame,
    freq: str = "W",
    topic_filter: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform temporal trend aggregation over specified frequency and filters.
    
    Returns:
        Dict[str, Any]: Aggregated time series DataFrame and trend statistics.
    """
    if "HasValidDate" not in df.columns or not df["HasValidDate"].any():
        return {
            "has_temporal_data": False,
            "message": "The active dataset does not contain valid datetime stamps. Temporal trend analysis is unavailable."
        }

    data = df[df["HasValidDate"]].copy()

    # Apply date filters if provided (include entire end day when a date string is supplied)
    if start_date:
        data = data[data["ParsedDate"] >= pd.to_datetime(start_date)]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        data = data[data["ParsedDate"] <= end_dt]

    # Apply topic filter if provided
    if topic_filter and COL_TOPIC in data.columns:
        data = data[data[COL_TOPIC].isin(topic_filter)]

    if data.empty:
        return {
            "has_temporal_data": True,
            "is_empty": True,
            "message": "No news articles match the selected date range and topic filters."
        }

    # Aggregate by time period
    data["Period"] = data["ParsedDate"].dt.to_period(freq).dt.to_timestamp()
    
    overall_ts = data.groupby("Period").size().reset_index(name="Volume")
    
    topic_ts = None
    if COL_TOPIC in data.columns:
        topic_ts = (
            data.groupby(["Period", COL_TOPIC])
            .size()
            .reset_index(name="Volume")
        )

    # Calculate momentum (percentage change over previous period)
    overall_ts["Volume_MoM"] = overall_ts["Volume"].pct_change() * 100

    peak_period = overall_ts.loc[overall_ts["Volume"].idxmax()]

    return {
        "has_temporal_data": True,
        "is_empty": False,
        "total_filtered_articles": len(data),
        "overall_timeseries": overall_ts,
        "topic_timeseries": topic_ts,
        "peak_period": str(peak_period["Period"]),
        "peak_volume": int(peak_period["Volume"]),
        "date_min": str(data["ParsedDate"].min()),
        "date_max": str(data["ParsedDate"].max()),
    }


def plot_trend_momentum_plotly(trend_data: Dict[str, Any]) -> go.Figure:
    """
    Generate an interactive Plotly chart displaying topic volume trends over time.
    """
    if not trend_data.get("has_temporal_data"):
        fig = go.Figure()
        fig.add_annotation(text="Temporal data unavailable for trend plotting.", showarrow=False)
        return fig

    if trend_data.get("is_empty"):
        fig = go.Figure()
        fig.add_annotation(text="No records match current trend filters.", showarrow=False)
        return fig

    topic_ts = trend_data.get("topic_timeseries")
    if topic_ts is not None and not topic_ts.empty:
        fig = px.area(
            topic_ts,
            x="Period",
            y="Volume",
            color=COL_TOPIC,
            title="News Trend Evolution and Topic Share Over Time",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
    else:
        overall_ts = trend_data.get("overall_timeseries")
        fig = px.line(
            overall_ts,
            x="Period",
            y="Volume",
            markers=True,
            title="Overall News Publication Volume Trend",
            template="plotly_white",
        )

    fig.update_layout(
        xaxis_title="Timeline Period",
        yaxis_title="Article Volume",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def extract_top_keywords_per_topic(
    df: pd.DataFrame,
    text_col: str = "CleanText",
    topic_col: str = COL_TOPIC,
    top_n: int = 8
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Discover salient keywords for each topic using class-stratified TF-IDF representation.
    Returns dictionary mapping each category to its highest TF-IDF weighted terms.
    """
    if topic_col not in df.columns or text_col not in df.columns:
        return {}

    topics = df[topic_col].unique()
    results = {}

    for topic in topics:
        sub_df = df[df[topic_col] == topic]
        if len(sub_df) < 5:
            continue

        vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        try:
            tfidf_mat = vec.fit_transform(sub_df[text_col])
            mean_weights = np.asarray(tfidf_mat.mean(axis=0)).flatten()
            feature_names = vec.get_feature_names_out()

            top_indices = mean_weights.argsort()[::-1][:top_n]
            top_pairs = [(feature_names[i], round(float(mean_weights[i]), 4)) for i in top_indices]
            results[str(topic)] = top_pairs
        except Exception:
            continue

    return results

