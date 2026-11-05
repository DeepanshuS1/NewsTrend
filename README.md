# 📰 Social Media News Trend Analysis and Engagement Prediction Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed%20%28100%25%29-brightgreen.svg)](tests/)
[![Framework](https://img.shields.io/badge/Streamlit-1.61.1-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning, Natural Language Processing, and Predictive Analytics platform engineered for digital news intelligence, automated editorial topic classification, and multi-platform social engagement forecasting.

---

## 1. Project Overview & Problem Statement

In modern digital journalism, understanding news coverage dynamics and forecasting social media engagement is essential. This project addresses two major objectives:
1. **Automated Topic Categorization:** Classifying incoming news headlines into verified subject categories using an leakage-free Natural Language Processing (NLP) pipeline.
2. **Temporal Trend Analysis & Engagement Forecasting:** Tracking the chronological evolution of news topics, extracting salient keywords, and predicting pre-publication viral engagement across **Facebook, LinkedIn, and Google+**.

Built on the genuine **UCI Machine Learning Repository "News Popularity in Multiple Social Media Platforms" dataset (ID: 432)** consisting of **93,239 real-world news records**.

---

## 2. Key Features

- **Empirical Dataset & Data Integrity:**
  - Complete 93,239 records loaded from UCI Repository with full data dictionary.
  - Stratified 5,000-row sample for rapid demonstration on low-spec student laptops.
  - Clearly labelled synthetic fallback sample (60 records) with amber alert banners.
  - Custom CSV file uploader with automatic schema mapping.
- **Robust Text Preprocessing:**
  - HTML entity decoding (`&quot;`, `&#39;`, `&amp;`), whitespace normalization, and text feature enrichment.
  - Strict train/test separation within Scikit-learn Pipelines to eliminate data leakage.
- **Exploratory Data Analysis (Plotly & Seaborn):**
  - Category distribution, term frequency bar charts (with stopword filtering), multi-platform interaction shares, and feature correlation heatmaps.
- **Machine Learning Text Classifier:**
  - Sublinear TF-IDF vectorization ($10,000$ features, unigrams and bigrams).
  - Multinomial Logistic Regression ($C=1.0$, L-BFGS solver).
  - Evaluated on $18,644$ unseen holdout samples: **99.49% Accuracy**, **99.45% Macro F1-Score**.
- **Interactive Live Predictor:**
  - Real-time headline classification with class probabilities and confidence distribution charts.
- **Social Engagement Prediction:**
  - Log-transformed Ridge Regression model predicting total social actions from pre-publication signals.
  - Virality tier classification (Low, Moderate, High, Viral).
- **Temporal Trend Discovery:**
  - Aggregated news volume trajectories (Daily, Weekly, Monthly) with date filtering.
  - Topic-stratified keyword extraction via class-specific TF-IDF weighting.
- **Interactive Streamlit Web Dashboard:**
  - 6 modular, polished dashboard views with responsive design and CSV/JSON export facilities.
- **Automated Pytest Suite:**
  - 16 comprehensive unit tests validating data loading, schema audits, preprocessing, and inference.

---

## 3. Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10 – 3.14 (Tested on 3.14.3) | Core runtime environment |
| **Data Processing** | Pandas, NumPy | $\ge$ 2.0.0, $\ge$ 1.24.0 | Tabular data manipulation & linear algebra |
| **Machine Learning** | Scikit-learn | $\ge$ 1.3.0 | TF-IDF, Logistic Regression, Ridge, Pipeline |
| **Model Persistence**| Joblib | $\ge$ 1.3.0 | Model serialization and deserialization |
| **Interactive UI** | Streamlit | $\ge$ 1.28.0 | Interactive web application dashboard |
| **Visualizations** | Plotly, Seaborn, Matplotlib | $\ge$ 5.15.0, $\ge$ 0.12.0 | Dynamic interactive charts and heatmaps |
| **Testing** | Pytest | $\ge$ 7.4.0 | Automated unit testing suite |
| **Version Control** | Git | 2.x | Source code management |

---

## 4. Dataset Information & Provenance

- **Source:** [UCI Machine Learning Repository (Dataset ID: 432)](https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms)
- **Creators:** Nuno Moniz & Luís Torgo, LIAAD - INESC TEC / University of Porto, Portugal
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Direct File:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv`
- **Scale:** 93,239 rows × 11 columns (~28.9 MB)
- **Time Span:** November 2015 – July 2016

### Data Dictionary

| Column | Type | Description |
| :--- | :--- | :--- |
| `IDLink` | Float | Unique numeric identifier |
| `Title` | String | Headline title of the news article |
| `Headline` | String | Short excerpt or lead summary |
| `Source` | String | Publishing news media organization |
| `Topic` | String | Ground-truth category: `economy`, `microsoft`, `obama`, `palestine` |
| `PublishDate` | Datetime | Initial publication timestamp |
| `SentimentTitle` | Float | Title sentiment polarity ($-1.0$ to $+1.0$) |
| `SentimentHeadline` | Float | Headline sentiment polarity ($-1.0$ to $+1.0$) |
| `Facebook` | Integer | Facebook share actions ($-1$ indicates untracked) |
| `GooglePlus` | Integer | Google+ share actions ($-1$ indicates untracked) |
| `LinkedIn` | Integer | LinkedIn share actions ($-1$ indicates untracked) |

---

## 5. System Architecture & Workflow

```
+-------------------------------------------------------------+
|                      DATA INGESTION                         |
|   data/News_Final.csv (UCI 432) -> src/data_loader.py       |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    DATA PREPROCESSING                       |
|   HTML unescape, normalization -> src/preprocessing.py      |
+-------------------------------------------------------------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
+-----------------------------+ +-----------------------------+
|    CLASSIFIER PIPELINE      | |    ENGAGEMENT PIPELINE      |
|  TfidfVectorizer(1-2 ngrams)| |  Pre-publication features   |
|  + LogisticRegression       | |  + Ridge Regressor          |
|  -> models/text_classifier  | |  -> models/engagement_reg   |
+-----------------------------+ +-----------------------------+
        |                                             |
        +----------------------+----------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  STREAMLIT DASHBOARD (app.py)               |
|  - Tab 1: Dataset Overview & Quality Audit                  |
|  - Tab 2: Exploratory Data Analysis (Plotly)                |
|  - Tab 3: Temporal Trends & Topic Momentum                  |
|  - Tab 4: ML Topic Classifier & Live Text Prediction        |
|  - Tab 5: Multi-Platform Engagement Prediction              |
|  - Tab 6: Filtered Data Export & Metrics Reports            |
+-------------------------------------------------------------+
```

---

## 6. Installation & Execution Guide

### Step 1: Clone Repository & Open Terminal
```powershell
# Windows PowerShell
cd "e:\DA Project"
```

### Step 2: (Optional) Create and Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
python -m pytest -v
```
*(Expected output: 16 passed in ~7 seconds)*

### Step 5: Train ML Models (Optional - Models Are Pre-Trained)
```powershell
# To train on the full 93,239 rows:
python -m src.train_model

# Or to train rapidly on the 5k sample:
python -m src.train_model --use-sample
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Actual Empirical Model Results

Both models were trained on the empirical UCI dataset ($74,575$ train samples) and evaluated on an unseen holdout test split ($18,644$ test samples):

### Text Classification Pipeline (TF-IDF + Logistic Regression)
- **Accuracy:** **99.49%** ($18,549 / 18,644$ correctly classified)
- **Macro Precision:** **99.58%**
- **Macro Recall:** **99.32%**
- **Macro F1-Score:** **99.45%**
- **Weighted F1-Score:** **99.49%**

#### Per-Class Breakdown
| Topic Category | Precision | Recall | F1-Score | Support (Test Samples) |
| :--- | :--- | :--- | :--- | :--- |
| **economy** | 0.9934 | 0.9943 | 0.9938 | 6,785 |
| **microsoft** | 0.9993 | 0.9973 | 0.9983 | 4,371 |
| **obama** | 0.9923 | 0.9972 | 0.9948 | 5,721 |
| **palestine** | 0.9983 | 0.9842 | 0.9912 | 1,767 |

### Engagement Regressor (Ridge Regression on Log-Transformed Engagement)
- **Evaluated Test Samples:** $17,496$
- **Log MAE:** $1.5817$
- **Log RMSE:** $1.9287$
- **R² Score:** $0.1670$
- **Raw Scale MAE:** $149.39$ actions

---

## 8. Automated Test Suite Summary

Execution of `python -m pytest -v` validates all software components:

```
tests/test_data_validation.py::test_load_sample_dataset_success PASSED   [  6%]
tests/test_data_validation.py::test_load_synthetic_demo_flag PASSED      [ 12%]
tests/test_data_validation.py::test_load_empty_dataset_raises PASSED     [ 18%]
tests/test_data_validation.py::test_load_nonexistent_file_raises PASSED  [ 25%]
tests/test_data_validation.py::test_detect_schema_mapping PASSED         [ 31%]
tests/test_data_validation.py::test_inspect_data_quality PASSED          [ 37%]
tests/test_data_validation.py::test_validate_required_columns PASSED     [ 43%]
tests/test_model.py::test_classifier_prediction_valid_input PASSED       [ 50%]
tests/test_model.py::test_classifier_prediction_empty_and_short_input PASSED [ 56%]
tests/test_model.py::test_evaluate_classifier_metrics PASSED             [ 62%]
tests/test_model.py::test_predict_engagement_level PASSED                [ 68%]
tests/test_preprocessing.py::test_clean_text_string_normalization PASSED [ 75%]
tests/test_preprocessing.py::test_clean_text_string_empty_and_none PASSED [ 81%]
tests/test_preprocessing.py::test_preprocess_dataset_features PASSED     [ 87%]
tests/test_preprocessing.py::test_preprocess_dataset_negative_engagement_clipping PASSED [ 93%]
tests/test_preprocessing.py::test_extract_engagement_features PASSED     [100%]

============================= 16 passed in 6.85s ==============================
```

---

## 9. Recommended Screenshots to Capture for Viva

When presenting or preparing the internship project report, capture the following screenshots from the running Streamlit dashboard:

1. **Screenshot 1: Overview & Data Integrity Tab**
   - Shows the green **VERIFIED EMPIRICAL DATASET** badge, dataset metadata, summary cards (93k records), and data quality table.
2. **Screenshot 2: Exploratory Data Analysis Tab**
   - Shows category distribution bar chart, frequent words chart, and correlation heatmap.
3. **Screenshot 3: Temporal Trends Tab**
   - Shows the time-series area chart of topic evolution over time and topic-stratified keywords.
4. **Screenshot 4: ML Topic Classifier Tab (Model Evaluation)**
   - Shows the empirical metrics row (99.49% accuracy), the interactive Plotly confusion matrix heatmap, and the per-class table.
5. **Screenshot 5: Interactive Real-Time Prediction in Action**
   - Enter a test headline (e.g., *"Microsoft announces new Azure cloud infrastructure upgrades"*) and capture the predicted category, confidence score, and horizontal probability bar chart.
6. **Screenshot 6: Social Engagement Prediction Estimator**
   - Shows estimated social actions, virality tier (e.g. *Moderate* or *High*), and sentiment sliders.
7. **Screenshot 7: Terminal Running Pytest**
   - Capture terminal output displaying `16 passed in 6.85s`.

---

## 10. Limitations & Future Scope

### Limitations
- **Four Pre-Defined Topics:** The dataset focuses on *Economy*, *Microsoft*, *Barack Obama*, and *Palestine*. Out-of-vocabulary articles default to the closest semantic cluster.
- **Engagement Stochasticity:** Social media virality is influenced by external variables (time of day, breaking news, influencer reposts) beyond headline semantics.
- **Historic Social Networks:** Google+ is included as a legacy network from the 2015-2016 collection period.

### Future Scope
- Integration with transformer embeddings (e.g. `sentence-transformers` or `DistilBERT`).
- Live RSS feed scraping for continuous real-time classification.
- Multi-modal analysis incorporating news image thumbnails.

---

## 11. Project Directory Structure

```
social-news-analysis/
├── app.py                      # Main Streamlit interactive web dashboard
├── requirements.txt            # Dependency specifications
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
├── config.py                   # Central paths, hyperparameters, and schema
├── data/
│   ├── README.md               # Dataset provenance and citation
│   ├── News_Final.csv          # Full UCI empirical dataset (93,239 rows)
│   ├── news_sample_real.csv    # Stratified empirical sample (5,000 rows)
│   └── synthetic_demo_sample.csv # Clearly labelled fallback (60 rows)
├── models/
│   ├── README.md               # Model descriptions & retraining commands
│   ├── text_classifier_pipeline.joblib    # Serialized text classification pipeline
│   └── engagement_regressor_pipeline.joblib # Serialized engagement regressor
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Loading, validation & schema detection
│   ├── preprocessing.py        # Text normalization & feature extraction
│   ├── eda.py                  # Statistical summaries & Plotly chart generators
│   ├── train_model.py          # Standalone training script
│   ├── evaluate_model.py       # Metrics computation & inference functions
│   ├── trend_analysis.py       # Time-series momentum & keyword discovery
│   └── engagement_analysis.py  # Platform benchmarks & popularity regression
├── tests/
│   ├── test_data_validation.py # Unit tests for data loading & schemas
│   ├── test_preprocessing.py   # Unit tests for text cleaning & leakage
│   └── test_model.py           # Unit tests for pipeline inference & metrics
├── docs/
│   ├── project_report.md       # Full Technical project report
│   ├── methodology.md          # Technical NLP & ML methodology
│   ├── dataset_card.md         # UCI dataset card & schema specifications
│   └── user_guide.md           # Step-by-step user manual
└── artifacts/
    ├── README.md               # Artifact metadata description
    ├── classifier_metrics.json # Exact empirical classification results
    └── engagement_metrics.json # Exact empirical regression results
```

---

## 12. Project Verification & Demonstration Checklist

- [x] Full source code written, tested, and validated.
- [x] Real empirical dataset downloaded and verified (UCI #432, CC BY 4.0).
- [x] Both machine learning models pre-trained and serialized locally.
- [x] Pytest suite executed (16/16 tests passing).
- [x] Self-contained ZIP package generated (`social_news_analysis_project.zip`).
- [ ] Run `streamlit run app.py` and capture demonstration screenshots for presentation.

