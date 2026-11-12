"""
Configuration module for the Social Media News Trend Analysis and Engagement Prediction project.
Provides centralized paths, random seeds, column mappings, and model parameters.
"""

from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"

# Data File Paths
FULL_DATASET_PATH = DATA_DIR / "News_Final.csv"
SAMPLE_DATASET_PATH = DATA_DIR / "news_sample_real.csv"
CUSTOM_DEMO_DATASET_PATH = DATA_DIR / "demo_custom_upload_sample.csv"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_demo_sample.csv"

# Model Artifact Paths
CLASSIFIER_MODEL_PATH = MODELS_DIR / "text_classifier_pipeline.joblib"
CLASSIFIER_METRICS_PATH = ARTIFACTS_DIR / "classifier_metrics.json"
ENGAGEMENT_MODEL_PATH = MODELS_DIR / "engagement_regressor_pipeline.joblib"
ENGAGEMENT_METRICS_PATH = ARTIFACTS_DIR / "engagement_metrics.json"

# General Reproducibility Settings
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Canonical Schema Definitions
COL_ID = "IDLink"
COL_TITLE = "Title"
COL_HEADLINE = "Headline"
COL_SOURCE = "Source"
COL_TOPIC = "Topic"
COL_DATE = "PublishDate"
COL_SENTIMENT_TITLE = "SentimentTitle"
COL_SENTIMENT_HEADLINE = "SentimentHeadline"
COL_ENGAGEMENT_FB = "Facebook"
COL_ENGAGEMENT_GP = "GooglePlus"
COL_ENGAGEMENT_LI = "LinkedIn"
COL_TOTAL_ENGAGEMENT = "TotalEngagement"

# Default Column Roles
DEFAULT_TEXT_COL = COL_HEADLINE
DEFAULT_FALLBACK_TEXT_COL = COL_TITLE
DEFAULT_TARGET_COL = COL_TOPIC
DEFAULT_DATE_COL = COL_DATE
ENGAGEMENT_COLS = [COL_ENGAGEMENT_FB, COL_ENGAGEMENT_GP, COL_ENGAGEMENT_LI]

# Known categories for the UCI News Popularity dataset
EXPECTED_CATEGORIES = ["economy", "microsoft", "obama", "palestine"]

# Model Hyperparameters
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0

# Engagement Regressor Hyperparameters
ENGAGEMENT_TARGET_COL = "TotalEngagement"

