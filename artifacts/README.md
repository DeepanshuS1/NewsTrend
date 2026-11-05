# Evaluation Artifacts Directory

This directory stores quantitative evaluation outputs generated during model training and evaluation runs.

## Files

1. `classifier_metrics.json`:
   - Exact empirical metrics calculated on the unseen test set (`X_test`, `y_test`):
     - Accuracy
     - Macro and Weighted Precision
     - Macro and Weighted Recall
     - Macro and Weighted F1-Score
     - Per-class precision, recall, and F1-score
     - Confusion Matrix array and class labels
     - Test split record counts and class balance

2. `engagement_metrics.json`:
   - Exact regression metrics on unseen test holdout:
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R² Score
     - Train/test sample sizes

