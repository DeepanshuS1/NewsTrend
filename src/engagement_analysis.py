"""
Engagement Analysis module for social news feedback and popularity prediction.
Analyzes multi-platform interactions, evaluates feature correlations,
and provides pre-publication engagement prediction capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    COL_TOPIC,
    COL_TOTAL_ENGAGEMENT,
    COL_ENGAGEMENT_FB,
    COL_ENGAGEMENT_GP,
    COL_ENGAGEMENT_LI,
)


def compute_engagement_benchmarks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute rigorous empirical engagement metrics across platforms and topics.
    """
    if "HasEngagementData" not in df.columns or not df["HasEngagementData"].any():
        return {
            "has_engagement": False,
            "message": "The active dataset does not contain genuine social engagement records."
        }

    valid_df = df[df["HasEngagementData"]].copy()

    # Overall Total Engagement Summary
    overall_stats = {
        "count": len(valid_df),
        "mean": round(float(valid_df[COL_TOTAL_ENGAGEMENT].mean()), 2),
        "median": float(valid_df[COL_TOTAL_ENGAGEMENT].median()),
        "std": round(float(valid_df[COL_TOTAL_ENGAGEMENT].std()), 2),
        "max": int(valid_df[COL_TOTAL_ENGAGEMENT].max()),
        "p75": float(valid_df[COL_TOTAL_ENGAGEMENT].quantile(0.75)),
        "p90": float(valid_df[COL_TOTAL_ENGAGEMENT].quantile(0.90)),
    }

    # Per-platform breakdown
    platform_stats = {}
    for plat_name, col in [("Facebook", "Facebook_Count"), ("LinkedIn", "LinkedIn_Count"), ("GooglePlus", "GooglePlus_Count")]:
        if col in valid_df.columns:
            platform_stats[plat_name] = {
                "mean": round(float(valid_df[col].mean()), 2),
                "median": float(valid_df[col].median()),
                "max": int(valid_df[col].max()),
                "total": int(valid_df[col].sum()),
            }

    # Per-topic engagement breakdown
    topic_breakdown = {}
    if COL_TOPIC in valid_df.columns:
        grouped = valid_df.groupby(COL_TOPIC)[COL_TOTAL_ENGAGEMENT].agg(["count", "mean", "median", "max"]).reset_index()
        for _, row in grouped.iterrows():
            topic_breakdown[row[COL_TOPIC]] = {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 2),
                "median": float(row["median"]),
                "max": int(row["max"]),
            }

    return {
        "has_engagement": True,
        "overall": overall_stats,
        "platforms": platform_stats,
        "topics": topic_breakdown,
    }


def plot_engagement_by_topic_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Plot bar chart comparing Mean and Median engagement across categories.
    """
    if "HasEngagementData" not in df.columns or not df["HasEngagementData"].any() or COL_TOPIC not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Engagement data by topic unavailable.", showarrow=False)
        return fig

    valid_df = df[df["HasEngagementData"]].copy()
    grouped = valid_df.groupby(COL_TOPIC)[COL_TOTAL_ENGAGEMENT].agg(
        Mean="mean", Median="median"
    ).reset_index()

    fig = go.Figure(data=[
        go.Bar(name="Mean Engagement", x=grouped[COL_TOPIC], y=grouped["Mean"], marker_color="#1f77b4"),
        go.Bar(name="Median Engagement", x=grouped[COL_TOPIC], y=grouped["Median"], marker_color="#ff7f0e"),
    ])
    fig.update_layout(
        barmode="group",
        title="Mean vs Median Social Engagement by News Topic",
        xaxis_title="Topic Category",
        yaxis_title="Total Actions (Shares/Likes)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_correlation_heatmap_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Generate interactive correlation matrix between numerical text attributes and engagement.
    """
    candidate_cols = [
        "TextLength",
        "WordCount",
        "SentimentTitle",
        "SentimentHeadline",
        COL_TOTAL_ENGAGEMENT,
        "LogTotalEngagement",
    ]
    present_cols = [c for c in candidate_cols if c in df.columns]

    if len(present_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient numerical features for correlation matrix.", showarrow=False)
        return fig

    corr_df = df[present_cols].corr().round(3)
    
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap: Content Signals vs. Social Engagement",
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
    )
    return fig


def predict_engagement_level(
    model: Any,
    topic: str,
    headline_text: str,
    sentiment_headline: float = 0.0,
    sentiment_title: float = 0.0,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Predict estimated total social engagement for a news post using pre-publication features.
    """
    if model is None:
        return {
            "is_valid": False,
            "error": "Engagement model is not loaded. Please train or load the engagement regressor."
        }

    char_len = len(headline_text)
    word_count = len(headline_text.split())
    exclamation_count = headline_text.count("!")
    question_count = headline_text.count("?")

    # Construct input feature dictionary
    input_dict = {
        "char_length": char_len,
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sentiment_headline": float(sentiment_headline),
        "sentiment_title": float(sentiment_title),
    }

    # Add topic dummy columns matching training columns
    topic_str = str(topic).lower().strip()
    if feature_columns:
        for col in feature_columns:
            if col.startswith("topic_"):
                col_topic = col.replace("topic_", "")
                input_dict[col] = 1.0 if col_topic == topic_str else 0.0
        
        # Ensure all required columns are present in exact order
        row_values = [input_dict.get(c, 0.0) for c in feature_columns]
        X_input = pd.DataFrame([row_values], columns=feature_columns)
    else:
        X_input = pd.DataFrame([input_dict])

    try:
        pred_log = float(model.predict(X_input)[0])
        pred_raw = max(0.0, float(np.expm1(pred_log)))

        # Categorize engagement level
        if pred_raw < 15:
            tier = "Low"
            desc = "Expected standard niche circulation."
        elif pred_raw < 75:
            tier = "Moderate"
            desc = "Average mainstream engagement expected."
        elif pred_raw < 250:
            tier = "High"
            desc = "Strong viral potential across social feeds."
        else:
            tier = "Viral / Very High"
            desc = "Exceptional viral velocity across major networks."

        return {
            "is_valid": True,
            "predicted_log_engagement": round(pred_log, 3),
            "predicted_engagement_actions": int(round(pred_raw)),
            "engagement_tier": tier,
            "tier_description": desc,
            "disclaimer": "Social engagement predictions represent baseline statistical expectations based on pre-publication features. Actual engagement is heavily influenced by publication timing, publisher reach, and platform algorithms."
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Engagement inference failed: {str(e)}"
        }

