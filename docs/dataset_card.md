# Dataset Card: News Popularity in Multiple Social Media Platforms

## Summary
The **News Popularity in Multiple Social Media Platforms** dataset is a multivariate, text, and time-series resource hosted on the **UCI Machine Learning Repository** (Dataset ID: 432). It tracks online news articles published between November 2015 and July 2016 across four prominent international topics, paired with feedback metrics across three major social media networks (Facebook, Google+, and LinkedIn).

---

## Dataset Details

- **Dataset Identifier:** UCI Machine Learning Repository Dataset #432
- **Direct Link:** https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms
- **Data File Download:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv`
- **Donated Date:** February 19, 2018
- **Primary Authors:** Nuno Moniz and Luís Torgo
- **Institution:** LIAAD - INESC TEC / Department of Computer Science, Faculty of Sciences, University of Porto, Portugal
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Citation
```bibtex
@misc{moniz_torgo_2018_newspop,
  author       = {Moniz, Nuno and Torgo, Luís},
  title        = {News Popularity in Multiple Social Media Platforms},
  year         = {2018},
  publisher    = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5160Q}
}
```

---

## Data Structure & Schema

The dataset comprises **93,239 rows** and **11 columns** (~28.9 MB in CSV format):

| Column Name | Type | Description | Range / Values | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `IDLink` | Float | Unique numeric identifier for the news item | Numeric ($1.0 - 104,802.0$) | 0 |
| `Title` | String | Headline title of the news article | Text (Avg length: 57 chars) | 0 |
| `Headline` | String | Article excerpt or short lead summary | Text (Avg length: 165 chars) | 15 (0.02%) |
| `Source` | String | Publishing news media organization | Categorical (e.g. Bloomberg, Reuters) | 279 (0.30%) |
| `Topic` | String | Ground-truth topic category | `economy`, `obama`, `microsoft`, `palestine` | 0 |
| `PublishDate` | Datetime | Publication date and time | 2002-04-02 to 2016-07-07 | 0 |
| `SentimentTitle` | Float | Title sentiment polarity | Continuous ($-1.0$ to $+1.0$) | 0 |
| `SentimentHeadline` | Float | Headline sentiment polarity | Continuous ($-1.0$ to $+1.0$) | 0 |
| `Facebook` | Integer | Facebook social engagement actions | $-1$ (untracked) or $\ge 0$ (up to 49,211) | 0 |
| `GooglePlus` | Integer | Google+ social engagement actions | $-1$ (untracked) or $\ge 0$ (up to 1,267) | 0 |
| `LinkedIn` | Integer | LinkedIn social engagement actions | $-1$ (untracked) or $\ge 0$ (up to 20,341) | 0 |

---

## Category Class Distribution

The target variable `Topic` contains 4 distinct classes:

- **economy:** 33,928 instances (36.39%)
- **obama:** 28,610 instances (30.68%)
- **microsoft:** 21,858 instances (23.44%)
- **palestine:** 8,843 instances (9.48%)
- **Total:** 93,239 instances

### Imbalance Note
While `economy`, `obama`, and `microsoft` each account for 23% to 36% of the corpus, `palestine` comprises ~9.5%. Stratified splitting is applied to maintain consistent target ratios across train and test partitions.

---

## Subsets Provided in This Project

1. **Full UCI Dataset (`data/News_Final.csv`):** Complete 93,239 rows downloaded directly from the official UCI storage server.
2. **Stratified Fast Sample (`data/news_sample_real.csv`):** 4,999 rows sampled via stratified random sampling (`random_state=42`). 100% genuine empirical records.
3. **Synthetic Demonstration Fallback (`data/synthetic_demo_sample.csv`):** 60 synthetic rows created strictly for offline testing or demonstration fallback. Distinctly flagged with `IsSyntheticDemo: True` and presented with amber alert banners.

