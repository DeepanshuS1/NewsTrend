# Models Directory

This directory stores serialized machine learning models and pipelines trained locally using `joblib`.

## Artifacts Generated Locally

1. `text_classifier_pipeline.joblib`:
   - Scikit-learn Pipeline consisting of `TfidfVectorizer` (sublinear TF, n-grams 1-2, max_features 10000) and `LogisticRegression(max_iter=1000, random_state=42)`.
   - Trained on `Topic` classification across the 4 news classes: `economy`, `microsoft`, `obama`, and `palestine`.
   - Fitted strictly on the training partition (`X_train`) to avoid data leakage.

2. `engagement_regressor_pipeline.joblib`:
   - Scikit-learn Pipeline (or ColumnTransformer + Regressor) predicting `TotalEngagement` or log-transformed engagement using pre-publication features: Topic (one-hot), Headline/Title sentiment polarity, Headline character length, and word count.
   - Strictly pre-publication features to prevent target leakage.

## Local Training Command

To train or retrain these models locally on the dataset:

```powershell
python -m src.train_model
```

