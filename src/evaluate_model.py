"""
Model evaluation module for the news classification pipeline.
Calculates empirical accuracy, precision, recall, F1 scores, confusion matrix,
and provides prediction helpers with class probability distributions.
"""

from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from config import CLASSIFIER_METRICS_PATH
from src.preprocessing import clean_text_string


def evaluate_classifier(
    pipeline: Any,
    X_test: pd.Series,
    y_test: pd.Series,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform rigorous evaluation of the trained text classifier on unseen test data.
    
    Returns:
        Dict[str, Any]: Full metrics suite including scalar metrics, per-class breakdown,
                        and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    
    if class_labels is None:
        if hasattr(pipeline, "classes_"):
            class_labels = list(pipeline.classes_)
        elif hasattr(pipeline.named_steps.get("clf"), "classes_"):
            class_labels = list(pipeline.named_steps["clf"].classes_)
        else:
            class_labels = sorted(list(set(y_test.unique())))

    # Global accuracy
    acc = float(accuracy_score(y_test, y_pred))

    # Macro and weighted metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Detailed per-class report
    clf_report_dict = classification_report(
        y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    # Class balance in test set
    test_distribution = y_test.value_counts().to_dict()

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "total_test_samples": len(y_test),
        "class_labels": class_labels,
        "test_distribution": test_distribution,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report_dict,
    }

    return metrics


def save_metrics_to_json(metrics: Dict[str, Any], path=CLASSIFIER_METRICS_PATH) -> None:
    """Save evaluation metrics dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics_from_json(path=CLASSIFIER_METRICS_PATH) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics dictionary from JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def plot_confusion_matrix_plotly(
    cm_data: List[List[int]],
    class_labels: List[str]
) -> go.Figure:
    """
    Generate an annotated heatmap of the confusion matrix using Plotly Express.
    Compatible with Plotly 5.x and 6.x.
    """
    cm = np.array(cm_data)

    fig = px.imshow(
        cm,
        x=class_labels,
        y=class_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(
            x="Predicted Topic Category",
            y="Actual Topic Category",
            color="Articles",
        ),
    )

    fig.update_layout(
        title="Confusion Matrix on Unseen Test Split",
        xaxis=dict(title="Predicted Topic Category", tickmode="linear", side="bottom"),
        yaxis=dict(title="Actual Topic Category", tickmode="linear"),
        margin=dict(l=60, r=40, t=50, b=50),
        template="plotly_white",
    )
    return fig


def predict_text(
    pipeline: Any,
    text: str
) -> Dict[str, Any]:
    """
    Predict news category for user-provided headline or post.
    Validates input, normalizes text consistently with training preprocessing,
    computes class probabilities and top contributing terms, and returns structured results.
    """
    if not text or not text.strip():
        return {
            "is_valid": False,
            "error": "Input text is empty. Please enter a valid news headline or article excerpt."
        }

    raw_trimmed = text.strip()
    cleaned = clean_text_string(raw_trimmed)
    if len(cleaned) < 3:
        return {
            "is_valid": False,
            "error": "Input text is too short to classify meaningfully (minimum 3 characters required)."
        }

    try:
        predicted_class = pipeline.predict([cleaned])[0]
        
        # Check if pipeline supports predict_proba
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            probabilities = {
                cls: round(float(prob), 4) for cls, prob in zip(classes, probs)
            }
            # Sort descending
            probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        # Extract top contributing TF-IDF terms for explainability
        top_terms: List[str] = []
        if hasattr(pipeline, "named_steps") and "tfidf" in pipeline.named_steps and "clf" in pipeline.named_steps:
            tfidf_step = pipeline.named_steps["tfidf"]
            clf_step = pipeline.named_steps["clf"]
            if hasattr(tfidf_step, "transform") and hasattr(clf_step, "coef_"):
                vec = tfidf_step.transform([cleaned]).toarray()[0]
                nonzero_idx = np.where(vec > 0)[0]
                if len(nonzero_idx) > 0:
                    feature_names = tfidf_step.get_feature_names_out()
                    class_idx = list(clf_step.classes_).index(predicted_class)
                    coefs = clf_step.coef_[class_idx] if clf_step.coef_.shape[0] > 1 else clf_step.coef_[0]
                    contributions = [(feature_names[i], float(vec[i] * coefs[i])) for i in nonzero_idx]
                    contributions.sort(key=lambda item: item[1], reverse=True)
                    top_terms = [term for term, score in contributions[:5] if score > 0] or [
                        term for term, _ in contributions[:3]
                    ]

        return {
            "is_valid": True,
            "input_text": raw_trimmed,
            "cleaned_text": cleaned,
            "predicted_category": str(predicted_class),
            "confidence": probabilities.get(predicted_class, None),
            "probabilities": probabilities,
            "top_terms": top_terms,
            "disclaimer": "Predictions are probabilistic estimates generated by Logistic Regression and should not be treated as absolute facts."
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Model inference error: {str(e)}"
        }

