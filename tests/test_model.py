"""
Unit tests for model training, evaluation, inference, and metrics computation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from src.evaluate_model import (
    evaluate_classifier,
    plot_confusion_matrix_plotly,
    predict_text,
)
from src.engagement_analysis import predict_engagement_level


@pytest.fixture
def mock_trained_classifier():
    """Create and train a lightweight pipeline fixture on small mock corpus."""
    texts = [
        "Stocks and markets rise on economic stimulus",
        "Inflation eases as central banks maintain rates",
        "Gross domestic product growth beats forecast",
        "Microsoft releases Windows security updates and cloud tools",
        "Azure enterprise cloud revenue increases this quarter",
        "Microsoft Surface hardware announced for enterprise",
        "President Obama delivers speech on foreign diplomacy",
        "Barack Obama signs healthcare and energy initiative",
        "Obama honors veterans at ceremony",
        "Palestine peace negotiations continue under UN supervision",
        "Gaza humanitarian aid delivered by international relief",
        "Palestine cultural folklore celebrated in Ramallah",
    ]
    labels = [
        "economy", "economy", "economy",
        "microsoft", "microsoft", "microsoft",
        "obama", "obama", "obama",
        "palestine", "palestine", "palestine"
    ]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(random_state=42, max_iter=200))
    ])
    pipeline.fit(texts, labels)
    return pipeline


def test_classifier_prediction_valid_input(mock_trained_classifier):
    """Verify inference returns valid structured output with confidence and probabilities."""
    res = predict_text(mock_trained_classifier, "Microsoft unveils new cloud computing tools")
    assert res["is_valid"] is True
    assert res["predicted_category"] == "microsoft"
    assert "confidence" in res
    assert res["confidence"] > 0.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, 0.01)
    assert "top_terms" in res
    assert isinstance(res["top_terms"], list)
    assert len(res["top_terms"]) > 0


def test_classifier_prediction_empty_and_short_input(mock_trained_classifier):
    """Verify input validation catches empty or too-short inputs gracefully."""
    res_empty = predict_text(mock_trained_classifier, "")
    assert res_empty["is_valid"] is False
    assert "empty" in res_empty["error"].lower()

    res_short = predict_text(mock_trained_classifier, "Hi")
    assert res_short["is_valid"] is False
    assert "too short" in res_short["error"].lower()


def test_evaluate_classifier_metrics(mock_trained_classifier):
    """Verify evaluate_classifier produces expected metric keys and valid ranges."""
    test_texts = pd.Series([
        "Markets rally after interest rate cut",
        "Azure cloud services expanded by Microsoft",
        "Barack Obama addresses education leaders",
        "Humanitarian aid sent to Palestine",
    ])
    test_labels = pd.Series(["economy", "microsoft", "obama", "palestine"])

    metrics = evaluate_classifier(mock_trained_classifier, test_texts, test_labels)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 4
    assert "classification_report" in metrics

    fig = plot_confusion_matrix_plotly(metrics["confusion_matrix"], metrics["class_labels"])
    assert fig is not None
    assert len(fig.data) > 0


def test_predict_engagement_level():
    """Verify engagement regressor prediction output formatting and tier logic."""
    # Create simple mock Ridge model
    X_mock = pd.DataFrame({
        "char_length": [50, 100],
        "word_count": [8, 16],
        "exclamation_count": [0, 1],
        "question_count": [0, 0],
        "sentiment_headline": [0.1, 0.2],
        "sentiment_title": [0.0, 0.1],
        "topic_microsoft": [1.0, 0.0],
        "topic_obama": [0.0, 1.0],
        "topic_palestine": [0.0, 0.0]
    })
    y_mock = pd.Series([np.log1p(25), np.log1p(150)])
    reg = Ridge().fit(X_mock, y_mock)

    feature_cols = list(X_mock.columns)
    pred_res = predict_engagement_level(
        model=reg,
        topic="obama",
        headline_text="President Obama speaks on national infrastructure investment.",
        sentiment_headline=0.2,
        sentiment_title=0.1,
        feature_columns=feature_cols,
    )

    assert pred_res["is_valid"] is True
    assert "predicted_engagement_actions" in pred_res
    assert pred_res["predicted_engagement_actions"] >= 0
    assert "engagement_tier" in pred_res
    assert pred_res["engagement_tier"] in ["Low", "Moderate", "High", "Viral / Very High"]

