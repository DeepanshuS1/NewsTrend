# Dataset Information & Integrity Record

This directory contains the datasets used for the **Social Media News Trend Analysis and Engagement Prediction** project.

## 1. Primary Dataset: UCI News Popularity in Multiple Social Media Platforms

- **Dataset Identifier:** UCI Machine Learning Repository Dataset #432
- **Direct UCI Source URL:** `https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms`
- **Direct Data File:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv`
- **Donated:** February 19, 2018
- **Authors / Creators:** Nuno Moniz and Luís Torgo, LIAAD - INESC TEC / Faculty of Sciences, University of Porto, Portugal
- **Citation:**
  > Moniz, N., & Torgo, L. (2018). *News Popularity in Multiple Social Media Platforms*. UCI Machine Learning Repository. https://doi.org/10.24432/C5160Q
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Scale:** 93,239 rows × 11 columns (~28.9 MB)
- **Time Span:** November 2015 – July 2016

### Data Dictionary

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `IDLink` | Float/Int | Unique identifier assigned to each news item |
| `Title` | String | Headline text of the news article |
| `Headline` | String | Excerpt / short summary text of the news article |
| `Source` | String | Publishing news outlet or media organization (e.g. Bloomberg, Reuters) |
| `Topic` | String | Ground-truth topic category: `economy`, `microsoft`, `obama`, `palestine` |
| `PublishDate` | Datetime (string) | Publication timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `SentimentTitle` | Float | Sentiment polarity score of the title (-1.0 to +1.0) |
| `SentimentHeadline` | Float | Sentiment polarity score of the headline (-1.0 to +1.0) |
| `Facebook` | Integer | Facebook social engagement score / share count (-1 indicates untracked) |
| `GooglePlus` | Integer | Google+ social engagement score (-1 indicates untracked) |
| `LinkedIn` | Integer | LinkedIn social engagement score (-1 indicates untracked) |

---

## 2. Real Stratified Sample (`news_sample_real.csv`)

- **Rows:** 4,999 observations drawn via stratified random sampling (`random_state=42`) from `News_Final.csv`
- **Purpose:** Enables rapid local training, lightweight dashboard testing, and low-RAM demonstration on standard student laptops.
- **Integrity:** 100% genuine empirical records; zero synthetic data.

---

## 3. Synthetic Demonstration Fallback (`synthetic_demo_sample.csv`)

- **Rows:** 60 records
- **Purpose:** Optional fallback demonstration only, used if network access is restricted or large CSVs cannot be loaded.
- **Disclosure:** Explicitly flagged with `IsSyntheticDemo: True` and clearly demarcated in the dashboard UI with amber warning banners. Never mixed with real empirical records.

