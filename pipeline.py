"""Data pipeline: extraction -> cleaning -> features -> correlation -> models.

Works with the Kaggle "Uber Ride Analytics Dashboard" file (ncr_ride_bookings.csv).
Every function is pure (DataFrame in, results out) so the Streamlit app can cache them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
PEAK_HOURS = [7, 8, 9, 17, 18, 19, 20]


# --------------------------------------------------------------------------- #
# 1. Extraction + cleaning
# --------------------------------------------------------------------------- #
def load_and_clean(source) -> pd.DataFrame:
    """Read the raw CSV (path or file-like) and return a tidy DataFrame."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    # IDs in the Kaggle file are wrapped in literal quotes: "CNR1234"
    for col in ("Booking ID", "Customer ID"):
        if col in df:
            df[col] = df[col].astype(str).str.replace('"', "", regex=False)

    df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    df = df.dropna(subset=["Datetime"]).drop_duplicates(subset="Booking ID")

    num_cols = ["Avg VTAT", "Avg CTAT", "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- feature engineering (only time/context features) ---
    df["Date"] = df["Datetime"].dt.normalize()
    df["Hour"] = df["Datetime"].dt.hour
    df["Weekday"] = df["Datetime"].dt.dayofweek          # 0 = Monday
    df["Weekday Name"] = df["Datetime"].dt.day_name()
    df["Month"] = df["Datetime"].dt.month
    df["Is Weekend"] = (df["Weekday"] >= 5).astype(int)
    df["Is Peak"] = df["Hour"].isin(PEAK_HOURS).astype(int)
    df["Completed"] = (df["Booking Status"] == "Completed").astype(int)
    df["Cancelled"] = df["Booking Status"].str.contains("Cancelled", na=False).astype(int)

    # a real fare/km is only defined for completed trips
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Fare per km"] = np.where(df["Ride Distance"] > 0, df["Booking Value"] / df["Ride Distance"], np.nan)
    return df.sort_values("Datetime").reset_index(drop=True)


def filter_df(df, date_range=None, vehicles=None, statuses=None) -> pd.DataFrame:
    out = df
    if date_range and len(date_range) == 2:
        out = out[(out["Date"] >= pd.Timestamp(date_range[0])) & (out["Date"] <= pd.Timestamp(date_range[1]))]
    if vehicles:
        out = out[out["Vehicle Type"].isin(vehicles)]
    if statuses:
        out = out[out["Booking Status"].isin(statuses)]
    return out


def kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    done = df[df["Completed"] == 1]
    return {
        "Total bookings": n,
        "Completion rate": df["Completed"].mean() if n else 0,
        "Cancellation rate": df["Cancelled"].mean() if n else 0,
        "Revenue": done["Booking Value"].sum(),
        "Avg fare": done["Booking Value"].mean(),
        "Avg distance (km)": done["Ride Distance"].mean(),
        "Avg driver rating": done["Driver Ratings"].mean(),
    }


# --------------------------------------------------------------------------- #
# 2. Correlation mapping
# --------------------------------------------------------------------------- #
CORR_COLS = ["Booking Value", "Ride Distance", "Avg VTAT", "Avg CTAT", "Driver Ratings",
             "Customer Rating", "Hour", "Weekday", "Is Peak", "Is Weekend", "Fare per km"]


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    cols = [c for c in CORR_COLS if c in df.columns and df[c].notna().sum() > 10]
    return df[cols].corr(method=method)


def top_correlations(corr: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    m = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().reset_index()
    m.columns = ["Feature A", "Feature B", "Correlation"]
    m["abs"] = m["Correlation"].abs()
    return m.sort_values("abs", ascending=False).head(k).drop(columns="abs").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3a. Regression: predict Booking Value of a completed trip
# --------------------------------------------------------------------------- #
REG_NUM = ["Ride Distance", "Hour", "Weekday", "Month", "Is Peak", "Is Weekend"]
REG_CAT = ["Vehicle Type", "Pickup Location", "Drop Location"]


def _preprocessor(num, cat, scale=False):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
    ])


@dataclass
class RegressionResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame      # actual vs predicted for best model
    importances: pd.DataFrame
    best_name: str
    best_model: Pipeline


def run_regression(df: pd.DataFrame, max_rows: int = 10_000) -> RegressionResult:
    d = df[(df["Completed"] == 1)].dropna(subset=["Booking Value", "Ride Distance"])
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=RANDOM_STATE)
    X, y = d[REG_NUM + REG_CAT], d["Booking Value"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": Pipeline([("pre", _preprocessor(REG_NUM, REG_CAT, scale=True)),
                                       ("m", LinearRegression())]),
        "Random Forest": Pipeline([("pre", _preprocessor(REG_NUM, REG_CAT)),
                                   ("m", RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                                               n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("pre", _preprocessor(REG_NUM, REG_CAT)),
                                       ("m", GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                                       learning_rate=0.08,
                                                                       random_state=RANDOM_STATE))]),
    }
    rows, fitted, preds = [], {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        fitted[name], preds[name] = mdl, p
        rows.append({"Model": name, "MAE": mean_absolute_error(yte, p),
                     "RMSE": np.sqrt(mean_squared_error(yte, p)), "R²": r2_score(yte, p)})
    metrics = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    best = metrics.loc[0, "Model"]

    pred_df = pd.DataFrame({"Actual": yte.values, "Predicted": preds[best]})
    imp = _importance(fitted["Random Forest"], Xte, yte)
    return RegressionResult(metrics, pred_df, imp, best, fitted[best])


def _importance(pipe: Pipeline, X, y, n=800) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL columns (so one-hots are grouped)."""
    from sklearn.inspection import permutation_importance
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X), min(n, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    r = permutation_importance(pipe, Xs, ys, n_repeats=4, random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"Feature": X.columns, "Importance": r.importances_mean})
            .sort_values("Importance", ascending=True).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 3b. Time series: daily demand forecast
# --------------------------------------------------------------------------- #
def daily_series(df: pd.DataFrame, metric: str = "Bookings") -> pd.Series:
    g = df.groupby("Date")
    s = g.size() if metric == "Bookings" else g["Booking Value"].sum()
    s = s.asfreq("D").fillna(0).astype(float)
    s.name = metric
    return s


@dataclass
class ForecastResult:
    history: pd.Series
    test_actual: pd.Series
    test_pred: dict
    metrics: pd.DataFrame
    future: pd.DataFrame           # forecast + interval
    best_name: str


def run_forecast(series: pd.Series, horizon: int = 30, holdout: int = 28) -> ForecastResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(series) < holdout + 28:
        raise ValueError("Need at least ~2 months of daily data to forecast.")
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    def seasonal_naive(tr, h):
        last = tr.iloc[-7:].values
        return np.resize(last, h)

    def hw(tr, h):
        m = ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=7, initialization_method="estimated").fit()
        return m.forecast(h).values

    def ma7(tr, h):
        return np.repeat(tr.iloc[-7:].mean(), h)

    methods = {"Seasonal naive (7d)": seasonal_naive, "7-day moving avg": ma7, "Holt-Winters": hw}
    preds, rows = {}, []
    for name, fn in methods.items():
        p = fn(train, holdout)
        preds[name] = pd.Series(p, index=test.index)
        rows.append({"Model": name, "MAE": mean_absolute_error(test, p),
                     "RMSE": np.sqrt(mean_squared_error(test, p)),
                     "MAPE %": float(np.mean(np.abs((test - p) / test.replace(0, np.nan))) * 100)})
    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best = metrics.loc[0, "Model"]

    # refit best on all data and project forward; interval from holdout residual spread
    fut_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fc = methods[best](series, horizon)
    sigma = float(np.std(test.values - preds[best].values))
    width = 1.96 * sigma * np.sqrt(1 + np.arange(horizon) / 30)
    future = pd.DataFrame({"Forecast": fc, "Lower": np.maximum(fc - width, 0), "Upper": fc + width}, index=fut_idx)
    return ForecastResult(series, test, preds, metrics, future, best)


# --------------------------------------------------------------------------- #
# 3c. Classification: will this booking be cancelled?
# --------------------------------------------------------------------------- #
# Only features known when the booking is made -> no target leakage
# (Ride Distance / Booking Value are empty for cancelled trips in the raw data).
CLF_NUM = ["Hour", "Weekday", "Month", "Is Peak", "Is Weekend", "Avg VTAT"]
CLF_CAT = ["Vehicle Type", "Pickup Location", "Drop Location"]


@dataclass
class ClassificationResult:
    metrics: pd.DataFrame
    roc: dict
    cm: np.ndarray
    importances: pd.DataFrame
    best_name: str
    best_model: Pipeline
    base_rate: float


def run_classification(df: pd.DataFrame, max_rows: int = 15_000) -> ClassificationResult:
    d = df.copy()
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=RANDOM_STATE)
    X, y = d[CLF_NUM + CLF_CAT], d["Cancelled"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": Pipeline([("pre", _preprocessor(CLF_NUM, CLF_CAT, scale=True)),
                                         ("m", LogisticRegression(max_iter=500, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", _preprocessor(CLF_NUM, CLF_CAT)),
                                   ("m", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=10,
                                                                class_weight="balanced", n_jobs=-1,
                                                                random_state=RANDOM_STATE))]),
    }
    rows, roc, fitted, probs = [], {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        pr = mdl.predict_proba(Xte)[:, 1]
        pred = (pr >= 0.5).astype(int)
        fitted[name], probs[name] = mdl, pr
        fpr, tpr, _ = roc_curve(yte, pr)
        roc[name] = (fpr, tpr)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
                     "ROC-AUC": roc_auc_score(yte, pr)})
    metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best = metrics.loc[0, "Model"]
    cm = confusion_matrix(yte, (probs[best] >= 0.5).astype(int))
    imp = _importance_clf(fitted["Random Forest"], Xte, yte)
    return ClassificationResult(metrics, roc, cm, imp, best, fitted[best], float(y.mean()))


def _importance_clf(pipe, X, y, n=1500) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X), min(n, len(X)), replace=False)
    r = permutation_importance(pipe, X.iloc[idx], y.iloc[idx], n_repeats=4, scoring="roc_auc",
                               random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"Feature": X.columns, "Importance": r.importances_mean})
            .sort_values("Importance", ascending=True).reset_index(drop=True))
