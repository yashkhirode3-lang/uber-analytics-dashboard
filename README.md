# Uber Ride Analytics: Multi-Feature Predictive Dashboard

Mini-Project 2. A Streamlit dashboard on the Kaggle **Uber Ride Analytics Dashboard** dataset
(`ncr_ride_bookings.csv`, ~150k bookings, NCR India, 2024).

## Quick start

```bash
pip install -r requirements.txt
# put the Kaggle file at data/ncr_ride_bookings.csv   (download from Kaggle -> Uber Ride Analytics Dashboard)
streamlit run app.py
```

No Kaggle file yet? `python generate_sample_data.py` creates a synthetic file with the same
columns; the app falls back to it and shows a warning banner. You can also upload the CSV in the sidebar.
Numbers shown while on sample data are NOT real results.

## Pipeline (`pipeline.py`)

| Stage | What it does |
|---|---|
| Extraction & cleaning | Parses Date+Time, strips quoted IDs, de-duplicates bookings, coerces numerics |
| Feature engineering | Hour, weekday, month, peak-hour and weekend flags, fare per km, completed/cancelled flags |
| Correlation mapping | Pearson/Spearman heatmap plus ranked strongest pairs |
| Regression | Predict fare (`Booking Value`) of completed trips: Linear, Random Forest, Gradient Boosting. MAE/RMSE/R², permutation importance |
| Time series | Daily bookings or revenue. Seasonal-naive, 7-day moving average and Holt-Winters compared on a 28-day holdout; best model is refit and projected with an interval |
| Classification | Predict whether a booking is cancelled (customer or driver): Logistic Regression and Random Forest. Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrix, ROC curve |

## Design decisions worth citing in your report

- **No target leakage in classification.** `Ride Distance` and `Booking Value` are empty for cancelled trips,
  so the classifier uses only booking-time features (time, vehicle, locations, pickup ETA `Avg VTAT`).
- **Class imbalance** is handled with balanced class weights, and models are ranked by ROC-AUC rather than accuracy.
- **Fare regression is dominated by distance** (fare is mostly rate x distance). Expect a high R²; this is honest, not a bug.
  Linear regression is the baseline to show how much the tree models add.
- **Time-series validation** uses a chronological holdout (never shuffled) and compares against naive baselines.
- Sidebar filters affect the Overview, Correlation and Forecast tabs; models train on the full dataset.

## Files

```
app.py                    Streamlit dashboard (6 tabs)
pipeline.py               cleaning, features, correlations, models
generate_sample_data.py   synthetic data in the Kaggle schema
requirements.txt
data/                     put ncr_ride_bookings.csv here
```

## Ideas to extend

Location-pair clustering, driver-rating prediction, SHAP explanations, surge-pricing proxy from VTAT,
deploy to Streamlit Community Cloud.
