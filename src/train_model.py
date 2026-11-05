"""
Model training script for the news classification pipeline and engagement regressor.
Executes train-test splitting with fixed random seeds, scikit-learn Pipeline construction,
empirical evaluation on holdouts, and joblib artifact serialization.
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from config import (
    FULL_DATASET_PATH,
    SAMPLE_DATASET_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_METRICS_PATH,
    ENGAGEMENT_MODEL_PATH,
    ENGAGEMENT_METRICS_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGREG_MAX_ITER,
    LOGREG_C,
    COL_TOPIC,
    COL_HEADLINE,
    COL_TOTAL_ENGAGEMENT,
)
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset, extract_engagement_features
from src.evaluate_model import evaluate_classifier, save_metrics_to_json


def train_text_classifier(
    df: pd.DataFrame,
    text_col: str = "CombinedText",
    target_col: str = COL_TOPIC,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    max_features: int = TFIDF_MAX_FEATURES,
    C: float = LOGREG_C,
    model_output_path: Path = CLASSIFIER_MODEL_PATH,
    metrics_output_path: Path = CLASSIFIER_METRICS_PATH,
) -> tuple:
    """
    Train and evaluate the TF-IDF + Logistic Regression classification pipeline.
    """
    print(f"\n--- [1/2] Training News Text Classifier ({target_col}) ---")
    
    # Filter valid rows
    valid_data = df[df[text_col].notnull() & (df[text_col].str.len() > 3) & df[target_col].notnull()].copy()
    print(f"Total training candidate records: {len(valid_data)}")
    print(f"Class distribution:\n{valid_data[target_col].value_counts()}")

    X = valid_data[text_col]
    y = valid_data[target_col]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train partition: {len(X_train)} samples | Test partition: {len(X_test)} samples")

    # Scikit-learn Pipeline (encapsulates vectorization to avoid data leakage)
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=TFIDF_NGRAM_RANGE,
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                max_iter=LOGREG_MAX_ITER,
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])

    print("Fitting Pipeline on X_train...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on unseen X_test...")
    metrics = evaluate_classifier(pipeline, X_test, y_test)
    metrics["text_column_used"] = text_col
    metrics["target_column"] = target_col
    metrics["total_records"] = len(valid_data)
    metrics["train_records"] = len(X_train)
    metrics["random_state"] = random_state

    print(f"\n>>> Classification Results:")
    print(f"    Accuracy:           {metrics['accuracy']:.4f}")
    print(f"    Macro F1-Score:     {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1-Score:  {metrics['weighted_f1']:.4f}")

    # Persist artifacts
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Saved model pipeline to: {model_output_path}")

    save_metrics_to_json(metrics, metrics_output_path)
    print(f"Saved evaluation metrics to: {metrics_output_path}")

    return pipeline, metrics


def train_engagement_regressor(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    model_output_path: Path = ENGAGEMENT_MODEL_PATH,
    metrics_output_path: Path = ENGAGEMENT_METRICS_PATH,
) -> tuple:
    """
    Train and evaluate a baseline Ridge Regression model predicting LogTotalEngagement
    strictly from pre-publication features.
    """
    print(f"\n--- [2/2] Training Engagement Regressor (LogTotalEngagement) ---")

    # Filter records with genuine engagement data
    if "HasEngagementData" not in df.columns or not df["HasEngagementData"].any():
        print("Dataset has no genuine engagement records. Skipping engagement model.")
        return None, None

    valid_eng = df[df["HasEngagementData"]].copy()
    print(f"Records with valid engagement: {len(valid_eng)}")

    X_eng, y_eng = extract_engagement_features(valid_eng)

    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_eng, test_size=test_size, random_state=random_state
    )

    regressor = Ridge(alpha=1.0, random_state=random_state)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # Invert log1p for raw scale evaluation
    y_test_raw = np.expm1(y_test)
    y_pred_raw = np.maximum(0, np.expm1(y_pred))

    log_mae = float(mean_absolute_error(y_test, y_pred))
    log_rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    raw_mae = float(mean_absolute_error(y_test_raw, y_pred_raw))

    metrics = {
        "log_mae": round(log_mae, 4),
        "log_rmse": round(log_rmse, 4),
        "r2_score": round(r2, 4),
        "raw_mae": round(raw_mae, 2),
        "total_samples": len(valid_eng),
        "test_samples": len(y_test),
        "features": list(X_eng.columns),
        "feature_coefficients": {col: round(float(coef), 4) for col, coef in zip(X_eng.columns, regressor.coef_)},
        "intercept": round(float(regressor.intercept_), 4),
    }

    print(f"\n>>> Engagement Regression Results:")
    print(f"    Log MAE:   {metrics['log_mae']:.4f}")
    print(f"    Log RMSE:  {metrics['log_rmse']:.4f}")
    print(f"    R² Score:  {metrics['r2_score']:.4f}")
    print(f"    Raw MAE:   {metrics['raw_mae']} actions")

    # Persist model & metrics
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, model_output_path)
    print(f"Saved engagement model to: {model_output_path}")

    save_metrics_to_json(metrics, metrics_output_path)
    print(f"Saved engagement metrics to: {metrics_output_path}")

    return regressor, metrics


def main():
    parser = argparse.ArgumentParser(description="Train text classifier and engagement regressor.")
    parser.add_argument("--use-sample", action="store_true", help="Train on the 5k sample dataset for rapid execution.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Custom dataset path.")
    args = parser.parse_args()

    if args.dataset_path:
        data_path = Path(args.dataset_path)
    elif args.use_sample:
        data_path = SAMPLE_DATASET_PATH
    else:
        # Default to full dataset if present, else fallback to sample
        data_path = FULL_DATASET_PATH if FULL_DATASET_PATH.exists() else SAMPLE_DATASET_PATH

    print(f"Loading dataset from: {data_path}")
    raw_df, meta = load_dataset(data_path)
    print(f"Loaded {len(raw_df)} records. Is synthetic demo: {meta['is_synthetic']}")

    print("Preprocessing data...")
    clean_df = preprocess_dataset(raw_df)

    # 1. Train Text Classifier
    train_text_classifier(clean_df)

    # 2. Train Engagement Regressor
    train_engagement_regressor(clean_df)

    print("\n[SUCCESS] Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
