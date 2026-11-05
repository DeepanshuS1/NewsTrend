# Technical Methodology: News Trend Analysis & Social Engagement Prediction

## 1. Mathematical Formulation & Architecture Overview

This project implements an end-to-end Natural Language Processing (NLP) and Predictive Analytics system. The workflow encompasses automated schema ingestion, text normalization, n-gram TF-IDF feature extraction, multi-class text categorization via regularized Logistic Regression, temporal trend modeling, and pre-publication engagement regression.

```
+-------------------------------------------------------------+
|                Empirical Data Ingestion                      |
| (UCI Machine Learning Repository Dataset #432: 93,239 rows)  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Data Cleaning & Preprocessing                |
| - HTML entity decoding (&quot;, &#39;, &amp;)               |
| - Whitespace collapsing & text standardization               |
| - Title + Headline enrichment                                |
| - Untracked flag clipping (-1 to 0 for sum calculation)      |
| - ISO-8601 Datetime parsing                                  |
+-------------------------------------------------------------+
                               |
               +---------------+---------------+
               |                               |
               v                               v
+-----------------------------+ +-----------------------------+
|    Text Classification      | |    Engagement Regression    |
| - Stratified Split (80/20)  | | - Pre-publication features  |
| - Scikit-learn Pipeline     | | - Character & word metrics  |
| - Sublinear TF-IDF (10k fts)| | - Sentiment polarities      |
| - Logistic Regression       | | - Category one-hot dummies  |
| - Unseen Test Holdout:      | | - Log1p transformation      |
|   * Accuracy: 99.49%        | | - Ridge Regression          |
|   * Macro F1: 99.45%        | |   * Log MAE: 1.5817         |
+-----------------------------+ +-----------------------------+
               |                               |
               +---------------+---------------+
                               |
                               v
+-------------------------------------------------------------+
|               Interactive Streamlit Dashboard               |
| - Exploratory Data Analysis & Plotly Visualizations         |
| - Real-time Headline Categorization & Confidence Probabilities |
| - Temporal Trend Discovery & Class-Stratified Keywords      |
| - Social Engagement Estimator & Filtered Data Export        |
+-------------------------------------------------------------+
```

---

## 2. Text Normalization and Cleaning

Raw text data scraped from internet news feeds contains formatting artifacts, HTML character entities, and punctuation irregularities. The preprocessing pipeline applies:

1. **HTML Entity Decoding:** Replaces escaped representations such as `&quot;`, `&#39;`, and `&amp;` with canonical Unicode characters via `html.unescape`.
2. **Whitespace Collapsing:** Regular expression substitution `re.sub(r"\s+", " ", text)` normalizes multi-space gaps, tabs, and newline characters into single spaces.
3. **Case Normalization:** Standardizes characters to lowercase for invariant vocabulary matching during vectorization.
4. **Feature Enrichment:** Concatenates `Title` and `Headline` strings (`Title + " " + Headline`) into a unified contextual document representation (`CombinedText`).

---

## 3. Feature Extraction: Term Frequency - Inverse Document Frequency (TF-IDF)

To transform unstructured textual strings into numerical feature matrices suitable for linear classification, we utilize the TF-IDF representation with sublinear scaling and n-gram extraction.

### Term Frequency (Sublinear Scaling)
Instead of raw frequency counts, sublinear term frequency dampens the effect of repetitive words:

$$\text{tf}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for } \text{count}(t, d) > 0$$

### Inverse Document Frequency (Smooth IDF)
IDF penalizes ubiquitous terms that appear across many documents:

$$\text{idf}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Final Weight and L2 Normalization
$$\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

Each document vector is normalized using the Euclidean norm ($\|v\|_2 = 1$), ensuring that document length does not artificially bias classification margins.

### Hyperparameters Configured
- **N-gram Range:** $(1, 2)$ — extracts both single words (unigrams) and word pairs (bigrams).
- **Max Features:** $10,000$ most informative n-grams ordered by term frequency across the corpus.
- **Stop Words:** Built-in standard English dictionary to eliminate uninformative stop tokens.
- **Sublinear TF:** Enabled (`sublinear_tf=True`).

---

## 4. Multi-Class Classification: Logistic Regression

The multi-class classification task assigns a news article $x$ to one of $K = 4$ mutually exclusive topic categories: $\mathcal{C} = \{\text{economy}, \text{microsoft}, \text{obama}, \text{palestine}\}$.

### Multinomial Formulation
The conditional probability of article $x$ belonging to class $k$ is given by the Softmax function:

$$P(Y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$

### Objective Function with L2 Regularization
The model parameters $W = [w_1, \dots, w_K]$ are optimized by minimizing the cross-entropy loss with an L2 weight penalty:

$$\min_{W, b} -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \log P(Y = k \mid x_i) + \frac{1}{2C} \sum_{k=1}^K \|w_k\|_2^2$$

- **Regularization Inverse Strength:** $C = 1.0$
- **Optimization Algorithm:** Limited-memory BFGS (`lbfgs`)
- **Max Iterations:** $1000$ to ensure complete gradient convergence.

---

## 5. Prevention of Data Leakage

A common vulnerability in student data science projects is data leakage, where global statistics from the entire dataset leak into the model training pipeline. We strictly guard against this:

1. **Stratified Train/Test Split First:** The full dataset is partitioned into an $80\%$ training split ($N_{\text{train}} = 74,575$) and a $20\%$ test holdout ($N_{\text{test}} = 18,644$) with fixed random seed $42$. Stratification preserves class balance across partitions.
2. **Scikit-Learn Pipeline Encapsulation:** The `TfidfVectorizer` and `LogisticRegression` classifier are bound within a `Pipeline`. Vocabulary discovery (`fit`) occurs **strictly** on $X_{\text{train}}$. The test set $X_{\text{test}}$ is only transformed (`transform`), ensuring zero out-of-fold vocabulary leakage.

---

## 6. Social Engagement Regression Formulation

In addition to text classification, we analyze and predict multi-platform social engagement ($Facebook + LinkedIn + Google+$).

### Challenge: Heavy-Tailed Skewness
Empirical social media engagement exhibits extreme right-skewness: a majority of articles receive modest engagement ($1-50$ actions), while viral outliers accumulate tens of thousands of shares. Fitting ordinary least squares directly on raw engagement yields unstable gradients dominated by outliers.

### Solution: Logarithmic Target Transformation
We apply the monotonic transformation:

$$y_{\text{transformed}} = \log(1 + \text{TotalEngagement}) = \text{log1p}(\text{TotalEngagement})$$

### Pre-Publication Feature Extraction
To prevent **target leakage**, no post-publication features (such as user comments, retweets over time, or subsequent engagement velocity) are utilized. We extract strictly pre-publication attributes:
- Headline character length (`char_length`)
- Headline word count (`word_count`)
- Punctuation indicators (`exclamation_count`, `question_count`)
- Headline and Title sentiment polarity scores (`SentimentHeadline`, `SentimentTitle`)
- One-hot encoded category indicators (`topic_microsoft`, `topic_obama`, `topic_palestine`)

### Ridge Regression Model
$$\min_{\beta} \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \quad (\alpha = 1.0)$$

Predictions are converted back to estimated actions via $\hat{y}_{\text{raw}} = \exp(\hat{y}) - 1 = \text{expm1}(\hat{y})$.

---

## 7. Temporal Trend Analysis Mechanics

When valid datetime stamps are available, the dataset is indexed temporally:
1. **Time-Series Aggregation:** Articles are grouped into temporal buckets: Daily ($D$), Weekly ($W$), or Monthly ($M$).
2. **Category Momentum:** Period-over-period percentage growth in article volume:
   $$\text{MoM} = \frac{V_t - V_{t-1}}{V_{t-1}} \times 100$$
3. **Class-Stratified Keyword Discovery:** Salient terminology is identified per category by calculating average TF-IDF weights within each topic subset, surfacing characteristic vocabulary distinct from global corpus terms.

