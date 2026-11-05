"""
Exploratory Data Analysis (EDA) module for news text and social metrics.
Generates statistical summaries, interactive Plotly charts, and Matplotlib figures
with informative titles, axis labels, and analytical context.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    COL_TOPIC,
    COL_DATE,
    COL_SOURCE,
    COL_TOTAL_ENGAGEMENT,
    ENGAGEMENT_COLS,
)

# Standard English stopwords for term frequency calculation without heavy NLTK downloads
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", 
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", 
    "your", "yours", "yourself", "yourselves", "said", "says", "mr", "also", "new", "one", "two", "will"
}


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core statistical overview of the dataset.
    """
    stats = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "total_missing_cells": int(df.isnull().sum().sum()),
        "total_duplicate_rows": int(df.duplicated().sum()),
    }

    if "CleanText" in df.columns:
        stats["avg_headline_char_len"] = round(float(df["TextLength"].mean()), 1)
        stats["avg_word_count"] = round(float(df["WordCount"].mean()), 1)
    
    if COL_TOPIC in df.columns:
        stats["num_unique_topics"] = int(df[COL_TOPIC].nunique())
        stats["topic_counts"] = df[COL_TOPIC].value_counts().to_dict()

    if COL_TOTAL_ENGAGEMENT in df.columns and df["HasEngagementData"].any():
        valid_eng = df[df["HasEngagementData"]][COL_TOTAL_ENGAGEMENT]
        stats["has_engagement"] = True
        stats["engagement_median"] = float(valid_eng.median())
        stats["engagement_mean"] = round(float(valid_eng.mean()), 2)
        stats["engagement_max"] = int(valid_eng.max())
    else:
        stats["has_engagement"] = False

    return stats


def plot_category_distribution_plotly(df: pd.DataFrame, category_col: str = COL_TOPIC) -> go.Figure:
    """
    Generate an interactive bar chart of news category frequencies using Plotly.
    """
    if category_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Category column '{category_col}' not found in dataset.", showarrow=False)
        return fig

    counts = df[category_col].value_counts().reset_index()
    counts.columns = ["Category", "Article Count"]
    counts["Percentage"] = ((counts["Article Count"] / len(df)) * 100).round(1)

    fig = px.bar(
        counts,
        x="Category",
        y="Article Count",
        text=counts["Article Count"].astype(str) + " (" + counts["Percentage"].astype(str) + "%)",
        color="Category",
        title="Distribution of News Articles Across Categories",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Topic / Category",
        yaxis_title="Number of Articles",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
    )
    return fig


def get_top_frequent_words(
    texts: pd.Series,
    top_n: int = 15,
    custom_stopwords: Optional[set] = None
) -> pd.DataFrame:
    """
    Extract the most frequent words across a series of text strings, removing stopwords.
    """
    stopwords = STANDARD_STOPWORDS if custom_stopwords is None else STANDARD_STOPWORDS.union(custom_stopwords)
    counter = Counter()

    for text in texts.dropna():
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in stopwords]
        counter.update(filtered)

    top_items = counter.most_common(top_n)
    return pd.DataFrame(top_items, columns=["Term", "Frequency"])


def plot_frequent_words_plotly(
    texts: pd.Series,
    top_n: int = 15,
    title_suffix: str = ""
) -> go.Figure:
    """
    Create an interactive horizontal bar chart of the most frequent terms.
    """
    freq_df = get_top_frequent_words(texts, top_n=top_n)
    if freq_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No frequent words discovered in provided text.", showarrow=False)
        return fig

    # Invert order for nice top-to-bottom bar chart
    freq_df = freq_df.iloc[::-1]

    fig = px.bar(
        freq_df,
        x="Frequency",
        y="Term",
        orientation="h",
        text="Frequency",
        title=f"Top {top_n} Frequent Terms in News Text {title_suffix}".strip(),
        color="Frequency",
        color_continuous_scale="Teal",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Term Occurrences",
        yaxis_title="Word / Term",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_news_volume_over_time_plotly(
    df: pd.DataFrame,
    date_col: str = "ParsedDate",
    freq: str = "W",
    topic_col: Optional[str] = COL_TOPIC
) -> Optional[go.Figure]:
    """
    Plot news publication volume over time aggregated by Day, Week, or Month.
    """
    if date_col not in df.columns or not df["HasValidDate"].any():
        return None

    valid_df = df[df["HasValidDate"]].copy()
    if valid_df.empty:
        return None

    # Group by frequency
    valid_df["TimeBucket"] = valid_df[date_col].dt.to_period(freq).dt.to_timestamp()
    
    if topic_col and topic_col in valid_df.columns:
        grouped = (
            valid_df.groupby(["TimeBucket", topic_col])
            .size()
            .reset_index(name="Article Count")
        )
        fig = px.line(
            grouped,
            x="TimeBucket",
            y="Article Count",
            color=topic_col,
            markers=True,
            title=f"News Article Publication Volume Over Time ({'Weekly' if freq == 'W' else 'Monthly' if freq == 'M' else 'Daily'} Aggregation)",
            template="plotly_white",
        )
    else:
        grouped = valid_df.groupby("TimeBucket").size().reset_index(name="Article Count")
        fig = px.line(
            grouped,
            x="TimeBucket",
            y="Article Count",
            markers=True,
            title="Total News Publication Volume Over Time",
            template="plotly_white",
        )

    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Number of Articles Published",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_engagement_distribution_plotly(
    df: pd.DataFrame,
    topic_col: str = COL_TOPIC,
    log_scale: bool = True
) -> Optional[go.Figure]:
    """
    Plot box plot of social engagement metrics comparing topics.
    """
    if COL_TOTAL_ENGAGEMENT not in df.columns or not df["HasEngagementData"].any():
        return None

    valid_df = df[df["HasEngagementData"]].copy()
    
    y_col = "LogTotalEngagement" if log_scale else COL_TOTAL_ENGAGEMENT
    y_label = "Log(1 + Total Engagement)" if log_scale else "Total Social Engagement"

    fig = px.box(
        valid_df,
        x=topic_col,
        y=y_col,
        color=topic_col,
        points=False, # hide individual outliers for cleaner rendering
        title=f"Social Engagement Distribution Across Categories ({'Log Transformed' if log_scale else 'Raw Count'})",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="News Category",
        yaxis_title=y_label,
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_platform_breakdown_plotly(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plot comparative aggregate engagement volume across Facebook, LinkedIn, and Google+.
    """
    platforms = [
        ("Facebook", "Facebook_Count"),
        ("LinkedIn", "LinkedIn_Count"),
        ("GooglePlus", "GooglePlus_Count"),
    ]
    
    available = [(name, col) for name, col in platforms if col in df.columns]
    if not available:
        return None

    data = []
    for name, col in available:
        data.append({
            "Platform": name,
            "Total Social Actions": int(df[col].sum()),
            "Average per Article": round(float(df[col].mean()), 2),
        })
    plat_df = pd.DataFrame(data)

    fig = px.bar(
        plat_df,
        x="Platform",
        y="Total Social Actions",
        color="Platform",
        text="Total Social Actions",
        title="Aggregate Social Engagement Actions Across Platforms",
        template="plotly_white",
        color_discrete_sequence=["#1877F2", "#0A66C2", "#DB4437"]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Social Media Platform",
        yaxis_title="Total Actions (Shares/Likes)",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig

