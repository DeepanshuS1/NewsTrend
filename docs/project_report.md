# PROJECT REPORT & TECHNICAL SPECIFICATION

---

## SOCIAL MEDIA NEWS TREND ANALYSIS AND ENGAGEMENT PREDICTION USING MACHINE LEARNING

**Enterprise Machine Learning & Predictive Analytics Suite**  
**Technical Architecture, Empirical Methodology, and Production Documentation**

---

<div style="page-break-after: always;"></div>

## EXECUTIVE SUMMARY & ABSTRACT

In the contemporary digital information ecosystem, social media platforms serve as primary conduits for news dissemination, public discourse, and opinion formation. Understanding the temporal dynamics of news coverage and predicting social engagement across heterogeneous social networks is crucial for journalists, media analysts, and computational social scientists. This project presents a comprehensive, reproducible, end-to-end Machine Learning and Natural Language Processing (NLP) system designed to analyze news trends and predict multi-platform social engagement.

The system is constructed upon the verified **UCI Machine Learning Repository "News Popularity in Multiple Social Media Platforms" dataset (ID: 432)**, comprising **93,239 empirical news observations** tracked over eight months (November 2015 to July 2016) across four major topic categories (*Economy*, *Microsoft*, *Barack Obama*, and *Palestine*), paired with social interaction metrics across **Facebook**, **LinkedIn**, and **Google+**.

The engineering pipeline integrates:
1. **Automated Data Validation & Schema Ingestion:** Ensuring schema integrity, missing-data auditing, duplicate detection, and explicit disclosure of empirical vs. demonstration records.
2. **Robust Text Normalization:** HTML entity unescaping (`&quot;`, `&#39;`, `&amp;`), whitespace collapsing, and text feature enrichment without data leakage.
3. **Multi-Class Text Classification:** Utilizing a Scikit-learn `Pipeline` combining sublinear TF-IDF vectorization ($10,000$ features, unigrams and bigrams) and multinomial Logistic Regression. Evaluated on an unseen stratified $20\%$ test split ($18,644$ samples), the classifier achieves an **empirical accuracy of 99.49%**, a **macro F1-score of 99.45%**, and a **weighted F1-score of 99.49%**.
4. **Multi-Platform Social Engagement Regression:** Mitigating severe right-skewness via logarithmic target transformation ($\text{log1p}$) and fitting a regularized Ridge Regressor on strictly pre-publication content signals (character length, word count, sentiment polarity, topic), achieving a test **Log Mean Absolute Error of 1.5817**.
5. **Interactive Streamlit Web Dashboard:** Featuring six modules including dynamic exploratory visualizations with Plotly, temporal trend aggregations, topic-stratified keyword discovery, real-time headline categorization with confidence probabilities, and filtered dataset export.

The codebase incorporates automated unit testing via `pytest` ($16/16$ tests passed, $100\%$ pass rate) and follows clean software engineering principles.

---

<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

- **Executive Summary & Abstract**
- **Table of Contents**
- **List of Figures & Tables**
- **Chapter 1: Introduction**
  - 1.1 Project Background
  - 1.2 Motivation
  - 1.3 Scope and Objectives
- **Chapter 2: Engineering Environment & System Context**
  - 2.1 Technology Architecture
  - 2.2 Data Science Toolchain & Dependencies
- **Chapter 3: Problem Definition & Requirements**
  - 3.1 Problem Statement
  - 3.2 Functional Requirements
  - 3.3 Non-Functional Requirements
  - 3.4 Hardware and Software Prerequisites
- **Chapter 4: Literature Review & Theoretical Foundation**
  - 4.1 Natural Language Processing in News Analytics
  - 4.2 TF-IDF Mathematical Formulation
  - 4.3 Multi-Class Logistic Regression & Regularization
  - 4.4 Social Media Engagement Dynamics & Skewness
  - 4.5 Prevention of Data Leakage in Machine Learning
- **Chapter 5: System Architecture & Design**
  - 5.1 Architectural Blueprint
  - 5.2 Data Ingestion & Validation Pipeline
  - 5.3 Classification Pipeline Design
  - 5.4 Pre-Publication Engagement Regressor Design
  - 5.5 Interactive Dashboard Architecture
- **Chapter 6: Implementation Details**
  - 6.1 Project Directory Structure
  - 6.2 Preprocessing & Feature Extraction (`src/preprocessing.py`)
  - 6.3 Model Training & Serialization (`src/train_model.py`)
  - 6.4 Model Evaluation & Inference (`src/evaluate_model.py`)
  - 6.5 Temporal Trend Discovery (`src/trend_analysis.py`)
  - 6.6 Streamlit Application (`app.py`)
- **Chapter 7: Results, Evaluation & Discussion**
  - 7.1 Dataset Schema & Audit Findings
  - 7.2 Classification Performance Metrics (Actual Empirical Results)
  - 7.3 Confusion Matrix & Per-Class Precision/Recall
  - 7.4 Engagement Regression Metrics & Discussion
  - 7.5 Temporal Trend & Topic Momentum Observations
- **Chapter 8: Automated Software Testing**
  - 8.1 Testing Methodology & Pytest Suite
  - 8.2 Execution Results ($16/16$ Test Cases Passed)
  - 8.3 Input Validation & Graceful Degradation Testing
- **Chapter 9: Limitations & Challenges**
  - 9.1 Dataset Scope and Class Inherent Boundaries
  - 9.2 Engagement Prediction Stochasticity
- **Chapter 10: Conclusion & Future Scope**
  - 10.1 Conclusion
  - 10.2 Future Scope
- **References**
- **Appendices**
  - Appendix A: Complete Pytest Verification Logs
  - Appendix B: Engineering Implementation Milestones

---

<div style="page-break-after: always;"></div>

## CHAPTER 1: INTRODUCTION

### 1.1 Project Background
In an era dominated by instantaneous digital communication, online news outlets and social media channels are intrinsically interconnected. News agencies broadcast articles across social platforms, while social interactions (likes, retweets, shares, and discussions) dictate information velocity and algorithmic prominence. 

Extracting actionable insights from high-velocity digital news requires automated mechanisms that can categorize content accurately, track topic momentum over chronological periods, and analyze the drivers of user engagement.

### 1.2 Motivation
Building production-grade machine learning applications requires moving beyond isolated experimental notebooks to reliable, maintainable systems:
1. Handling messy empirical data with missing values, character encodings, and platform-specific anomalies.
2. Formulating robust preprocessing pipelines that strictly eliminate data leakage between train and test partitions.
3. Rigorously evaluating models on unseen holdout test sets using comprehensive metrics beyond overall accuracy.
4. Packaging analytical models into intuitive, interactive user interfaces for domain stakeholders.

### 1.3 Scope and Objectives
The primary objective of this project is to architect, develop, verify, and document an end-to-end data science application titled **"Social Media News Trend Analysis and Engagement Prediction Using Machine Learning"**.

Key technical goals include:
- Ingesting and auditing the verified **UCI News Popularity in Multiple Social Media Platforms dataset (ID: 432)**.
- Formulating a Scikit-learn text classification pipeline using **TF-IDF Vectorization** and **Logistic Regression**.
- Implementing an empirical pre-publication regression model to predict aggregate social feedback across **Facebook, LinkedIn, and Google+**.
- Implementing temporal trend analysis and topic-stratified keyword discovery.
- Designing and deploying a reactive web dashboard using **Streamlit** and **Plotly**.
- Enforcing software reliability through automated test suites using **Pytest**.

---

<div style="page-break-after: always;"></div>

## CHAPTER 2: ENGINEERING ENVIRONMENT & SYSTEM CONTEXT

### 2.1 Technology Architecture
The development architecture adheres strictly to reproducible data science engineering standards:
- **Core Language & Runtime:** Python 3.10+ (Tested and verified on Python 3.14.3)
- **Data Engineering:** Pandas 3.0.5, NumPy 2.5.2
- **Machine Learning & NLP:** Scikit-learn 1.9.0, Joblib 1.5.3
- **Data Visualization Engine:** Plotly 7.1.0, Matplotlib 3.11.1, Seaborn 0.13.2
- **Web Application Framework:** Streamlit 1.61.1
- **Quality Assurance & Testing:** Pytest 9.1.1
- **Version Control:** Git

### 2.2 System Operating Standards
1. **Local Reproducibility:** Operates locally on standard computing hardware without requiring external cloud accounts or paid proprietary APIs.
2. **Modular Code Organization:** Decoupled modules separating data ingestion, preprocessing, training, evaluation, trend analysis, and dashboard rendering.
3. **Artifact Persistence:** Trained models, transformation pipelines, and evaluation metrics are saved to disk in portable `.joblib` and `.json` formats.

---

<div style="page-break-after: always;"></div>

## CHAPTER 3: PROBLEM DEFINITION & REQUIREMENTS

### 3.1 Problem Statement
Online news platforms produce massive volumes of textual articles daily. Newsrooms and analysts face two primary challenges:
1. **Automated Topic Categorization:** Manually sorting incoming news feeds into topic categories is resource-intensive and error-prone. An automated classifier must accurately categorize headlines into verified subject topics.
2. **Engagement Forecasting & Trend Detection:** Prior to publishing, editors lack objective signals estimating how an article might circulate on social media. Developing pre-publication models to predict social engagement while tracking chronological topic trends provides actionable editorial insights.

### 3.2 Functional Requirements
1. **Data Ingestion & Integrity Auditing:** Load CSV datasets, audit missing values, detect duplicates, and clearly distinguish between empirical records and demonstration fallbacks.
2. **Preprocessing Pipeline:** Clean HTML entities, collapse redundant whitespace, concatenate headline fields, and convert datetime fields without data leakage.
3. **Exploratory Data Analysis:** Render interactive bar charts, vocabulary distributions, log-scale engagement box plots, and feature correlation heatmaps.
4. **Machine Learning Classification:** Train a TF-IDF + Logistic Regression pipeline on an $80\%$ stratified split, reporting real empirical holdout accuracy, precision, recall, F1-scores, and confusion matrices.
5. **Interactive Topic Inference:** Provide an interactive text area allowing users to enter custom headlines, returning the predicted class and full class probability distributions.
6. **Pre-Publication Engagement Modeling:** Provide an engagement regressor predicting estimated social interactions and engagement tiers from pre-publication signals.
7. **Temporal Trend Analysis:** Aggregate publication volume over time with selectable frequency (Daily, Weekly, Monthly) and extract characteristic keywords.
8. **Export Facilities:** Allow users to export filtered datasets and evaluation metrics in CSV and JSON formats.

### 3.3 Hardware & Software Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Processor:** Intel Core i3/i5/i7 or AMD Ryzen equivalent
- **RAM:** Minimum 4 GB (8 GB recommended for full 93k row training)
- **Storage:** 150 MB free disk space for codebase, dependencies, and model artifacts
- **Python Version:** Python 3.10 to 3.14

---

<div style="page-break-after: always;"></div>

## CHAPTER 4: LITERATURE REVIEW & THEORETICAL FOUNDATION

### 4.1 Natural Language Processing in News Analytics
Natural Language Processing (NLP) in news analytics has evolved from basic keyword matching to statistical vector space models and deep contextual embeddings. While deep transformer architectures offer rich semantic representations, linear models coupled with TF-IDF remain the industry standard benchmark for production systems due to:
- High computational throughput (sub-millisecond inference times).
- Negligible memory footprint.
- Full mathematical interpretability.
- Exceptional accuracy on well-defined topical news classification tasks.

### 4.2 TF-IDF Mathematical Formulation
Term Frequency-Inverse Document Frequency evaluates the relative importance of a word $t$ within a document $d$ belonging to a corpus $D$.

$$\text{tf}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for } \text{count}(t, d) > 0$$

$$\text{idf}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

Using Euclidean $L_2$ normalization ensures documents of varying lengths are projected onto the unit hypersphere:

$$v_{\text{norm}} = \frac{v}{\|v\|_2}$$

### 4.3 Multi-Class Logistic Regression & Regularization
For $K$ classes, multinomial logistic regression computes probabilities using the Softmax function:

$$P(Y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$

Optimization minimizes penalized negative log-likelihood:

$$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log P(Y = k \mid x_i) + \frac{1}{2C} \sum_{k=1}^K \|w_k\|_2^2$$

The L-BFGS solver iteratively updates weights by approximating the inverse Hessian matrix, ensuring quadratic convergence.

---

<div style="page-break-after: always;"></div>

## CHAPTER 5: SYSTEM ARCHITECTURE & DESIGN

### 5.1 Architectural Blueprint

The architecture follows a modular, decoupled design where data processing, model training, evaluation, and dashboard presentation are cleanly separated:

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

<div style="page-break-after: always;"></div>

## CHAPTER 6: IMPLEMENTATION DETAILS

### 6.1 Project Directory Structure
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

<div style="page-break-after: always;"></div>

## CHAPTER 7: RESULTS, EVALUATION & DISCUSSION

### 7.1 Dataset Schema & Audit Findings
The empirical dataset from the UCI Machine Learning Repository (Dataset #432) was audited:
- **Total Records:** 93,239
- **Columns:** 11 (`IDLink`, `Title`, `Headline`, `Source`, `Topic`, `PublishDate`, `SentimentTitle`, `SentimentHeadline`, `Facebook`, `GooglePlus`, `LinkedIn`)
- **Missing Values:** Only 15 missing entries in `Headline` (0.02%) and 279 in `Source` (0.30%). Zero missing values in `Title`, `Topic`, `PublishDate`, or engagement metrics.
- **Tracked Engagement Records:** 87,477 records (93.8%) contain valid positive social interactions.

### 7.2 Classification Performance Metrics (Actual Empirical Results)
The classification model was trained on the $80\%$ partition ($74,575$ samples) and evaluated on the unseen $20\%$ holdout test partition ($18,644$ samples).

| Metric | Empirical Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | **0.9949** | **99.49%** |
| **Macro Precision** | **0.9958** | **99.58%** |
| **Macro Recall** | **0.9932** | **99.32%** |
| **Macro F1-Score** | **0.9945** | **99.45%** |
| **Weighted Precision** | **0.9949** | **99.49%** |
| **Weighted Recall** | **0.9949** | **99.49%** |
| **Weighted F1-Score** | **0.9949** | **99.49%** |

### 7.3 Per-Class Breakdown & Confusion Matrix

#### Per-Class Performance
| Category | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **economy** | 0.9934 | 0.9943 | 0.9938 | 6,785 |
| **microsoft** | 0.9993 | 0.9973 | 0.9983 | 4,371 |
| **obama** | 0.9923 | 0.9972 | 0.9948 | 5,721 |
| **palestine** | 0.9983 | 0.9842 | 0.9912 | 1,767 |
| **Macro Avg** | **0.9958** | **0.9932** | **0.9945** | **18,644** |

#### Confusion Matrix ($N_{\text{test}} = 18,644$)
```
Actual \ Predicted    economy    microsoft    obama    palestine
economy                 6,746            0       37            2
microsoft                   9        4,359        2            1
obama                      15            1    5,705            0
palestine                  21            2        5        1,739
```

The confusion matrix reveals near-flawless discrimination across categories. The highest confusion occurs between `economy` and `obama` (37 economy articles classified as obama, and 15 obama articles classified as economy), attributable to overlapping policy headlines concerning the federal budget and economic stimuli.

### 7.4 Engagement Regression Metrics
The Ridge Regressor predicting log-transformed social engagement ($\text{log1p}$) on $17,496$ unseen test records yielded:
- **Log Mean Absolute Error (MAE):** $1.5817$
- **Log Root Mean Squared Error (RMSE):** $1.9287$
- **Coefficient of Determination ($R^2$):** $0.1670$
- **Raw Scale MAE:** $149.39$ actions

#### Discussion on Engagement Modeling
Social media engagement exhibits high stochasticity; pre-publication textual attributes account for approximately $16.7\%$ of variance ($R^2 = 0.167$). Factors including publication timing, influencer amplification, recommendation algorithms, and breaking news cycles introduce substantial external variance. The model successfully stratifies articles into meaningful virality tiers (Low, Moderate, High, Viral) without overpromising deterministic accuracy.

---

<div style="page-break-after: always;"></div>

## CHAPTER 8: AUTOMATED SOFTWARE TESTING

### 8.1 Testing Methodology & Pytest Suite
To ensure enterprise software quality, automated unit tests were created across three test suites:
1. `tests/test_data_validation.py`: Schema mapping, missing-data audits, empty CSV detection, synthetic flag tracking.
2. `tests/test_preprocessing.py`: Text cleaning, HTML unescaping, date parsing, engagement clipping, and feature extraction.
3. `tests/test_model.py`: Pipeline fitting, prediction formatting, confidence probability validation, metrics calculation, and engagement inference.

### 8.2 Execution Results
The test suite was executed via the terminal command:
```powershell
python -m pytest -v
```

**Results:**
- **Collected Test Cases:** 16
- **Passed:** 16
- **Failed:** 0
- **Pass Rate:** **100%**
- **Execution Time:** 6.85 seconds

---

<div style="page-break-after: always;"></div>

## CHAPTER 9: LIMITATIONS & CHALLENGES

1. **Topical Scope:** The UCI dataset focuses on four specific topics (*Economy*, *Microsoft*, *Barack Obama*, *Palestine*). Classifying articles outside these domains defaults to the closest probabilistic match among the four known categories.
2. **Social Platform Evolution:** The dataset tracks Facebook, LinkedIn, and Google+ (which has since been deprecated). Modern platforms like X (formerly Twitter) or Threads are not captured in the 2015-2016 historical window.
3. **Engagement Stochasticity:** Social media virality cannot be fully determined by headline text alone; platform algorithms, account follower count, and temporal posting time play significant roles.

---

<div style="page-break-after: always;"></div>

## CHAPTER 10: CONCLUSION & FUTURE SCOPE

### 10.1 Conclusion
The project successfully designed, implemented, and verified a complete, end-to-end Machine Learning and NLP software application. By utilizing the verified UCI News Popularity dataset, implementing leakage-free Scikit-learn pipelines, and achieving **99.49% accuracy** on holdout text classification, the system satisfies all rigorous Data Science engineering standards.

### 10.2 Future Scope
1. **Transformer Integration:** Evaluating lightweight transformer models (e.g., DistilBERT) for fine-grained sentiment and transfer learning.
2. **Real-Time RSS Ingestion:** Incorporating live RSS feeds from Google News or Reuters for real-time streaming classification.
3. **Multi-Modal Features:** Extracting image thumbnail embeddings to enhance engagement regression models.

---

<div style="page-break-after: always;"></div>

## REFERENCES

1. Moniz, N., & Torgo, L. (2018). *News Popularity in Multiple Social Media Platforms*. UCI Machine Learning Repository. https://doi.org/10.24432/C5160Q
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.
4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.
5. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.
6. Streamlit Inc. (2024). *Streamlit: An open-source app framework for Machine Learning and Data Science*. https://streamlit.io

---

<div style="page-break-after: always;"></div>

## APPENDICES

### Appendix A: Pytest Verification Logs
```text
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\DA Project
plugins: anyio-4.14.2
collected 16 items

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

### Appendix B: Engineering Implementation Milestones

| Milestone | Phase | Description | Key Deliverables & Artifacts |
| :--- | :--- | :--- | :--- |
| **M1** | Dataset Audit & Ingestion | Ingested and audited UCI Dataset #432; evaluated schemas and licenses. | `data/README.md`, Data Dictionary |
| **M2** | Preprocessing Engine | Built text normalization, HTML decoding, and date parsing routines. | `src/data_loader.py`, `src/preprocessing.py` |
| **M3** | Exploratory Analytics | Implemented statistical summaries, Plotly charts, and engagement breakdowns. | `src/eda.py` |
| **M4** | Text Classifier Pipeline | Formulated TF-IDF + Logistic Regression pipeline; eliminated data leakage. | `src/train_model.py`, `models/` |
| **M5** | Model Evaluation & Regression | Evaluated holdout metrics (99.49% accuracy); trained Ridge engagement regressor. | `src/evaluate_model.py`, `artifacts/` |
| **M6** | Temporal & Topical Trends | Developed timeline aggregation and topic-stratified keyword discovery. | `src/trend_analysis.py` |
| **M7** | Dashboard Deployment | Built multi-page Streamlit dashboard with real-time prediction and Plotly charts. | `app.py` |
| **M8** | Quality Assurance & Documentation | Authored Pytest suites (16/16 passed), finalized technical documentation. | `tests/`, `README.md`, `project_report.md` |
