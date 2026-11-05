# User Guide: Social Media News Trend Analysis and Engagement Prediction Dashboard

This guide walks through every section of the interactive Streamlit dashboard.

---

## 1. Launching the Application

Ensure dependencies are installed and run the application from the project root:

```powershell
# Windows PowerShell
streamlit run app.py
```

```bash
# Linux / macOS
streamlit run app.py
```

The application opens automatically in your web browser at `http://localhost:8501`.

---

## 2. Navigating the Dashboard

The dashboard is organized into six functional modules accessible via the left sidebar:

### Tab 1: Overview & Data Integrity
- **Real vs Demo Indicator:** A prominent banner indicates whether you are analyzing genuine empirical records or the fallback synthetic sample.
- **Provenance & Licensing:** Review official repository metadata, DOI citation, and CC BY 4.0 licensing.
- **Data Quality Audit:** Check missing value counts, duplicate rows, and memory usage.
- **Data Dictionary & Raw Preview:** Browse column descriptions and examine raw records.

### Tab 2: Exploratory Data Analysis (EDA)
- **Topic Filter:** Select or deselect topics (`economy`, `microsoft`, `obama`, `palestine`) to dynamically filter the charts.
- **Category Distribution Chart:** Interactive Plotly bar chart with exact counts and percentage shares.
- **Top Frequent Vocabulary:** Frequency bar chart with English stopwords removed.
- **Social Engagement Distribution:** Log-scaled box plots comparing engagement across categories.
- **Multi-Platform Actions:** Aggregate engagement comparisons across Facebook, LinkedIn, and Google+.
- **Correlation Matrix:** Interactive heatmap displaying correlations between character length, word count, sentiment scores, and engagement.

### Tab 3: Trend Analysis & Topic Evolution
- **Temporal Aggregation:** Toggle between Weekly (`W`), Daily (`D`), and Monthly (`M`) time buckets.
- **Date Range Selector:** Focus on specific periods within the 2015–2016 timeframe.
- **Volume Trajectory Chart:** Area chart showing overall news activity and topic share over time.
- **Key Trend Metrics:** View total articles in range, peak activity date, and peak volume.
- **Topic-Stratified Keywords:** Inspect top TF-IDF weighted terms characteristic of each topic.

### Tab 4: ML Topic Classifier & Live Predictor
- **Empirical Model Performance:** Review held-out test set accuracy (99.49%), macro precision, recall, and F1-score.
- **Interactive Confusion Matrix:** Plotly heatmap of true vs. predicted classes.
- **Per-Class Breakdown:** Precision, recall, and support for all 4 categories.
- **Live Text Inference:** Enter any news headline or social media post draft and click **Classify Topic** to receive:
  - Predicted category
  - Confidence percentage
  - Full class probability distribution bar chart

### Tab 5: Social Engagement Prediction
- **Empirical Benchmarks:** Mean, median, and peak engagement actions observed across platforms.
- **Mean vs. Median Comparison:** Grouped bar chart comparing engagement across topics.
- **Pre-Publication Engagement Estimator:**
  - Select category and draft headline.
  - Adjust title and headline sentiment polarity sliders.
  - Click **Estimate Social Engagement** to calculate predicted share actions, engagement tier (Low, Moderate, High, Viral), and interpretability notes.

### Tab 6: Data Export & Reports
- **Download Filtered Data (CSV):** Export active, preprocessed dataset with all engineered features.
- **Export Classification Metrics (JSON):** Download raw model evaluation metrics.
- **Architecture Specifications:** Review tech stack and configuration parameters.

---

## 3. Switching Data Sources

In the sidebar **Dataset Configuration** dropdown:
- `UCI News Popularity (Sample - 5k rows)`: Fast, lightweight stratified sample recommended for quick demonstrations.
- `UCI News Popularity (Full - 93k rows)`: Complete dataset with 93,239 empirical records.
- `Synthetic Demo Fallback (60 rows)`: Offline fallback dataset clearly labelled with alert banners.
- `Upload Custom CSV`: Upload any compatible CSV file; the application auto-detects columns or maps text, topic, date, and engagement fields.

