"""Streamlit dashboard: Uber ride analytics + predictive models.
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Uber Ride Analytics", page_icon="🚕", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
REAL = DATA_DIR / "ncr_ride_bookings.csv"
SAMPLE = DATA_DIR / "sample_ncr_ride_bookings.csv"
ACCENT, MUTED, WARN = "#2563eb", "#94a3b8", "#dc2626"
import plotly.io as pio
pio.templates["plotly_white"].layout.margin = dict(l=10, r=10, t=50, b=10)
PLOT = dict(template="plotly_white")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ------------------------------------------------------------------ data ---
@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data(path_or_bytes, name):
    return pl.load_and_clean(path_or_bytes)


@st.cache_resource(show_spinner="Training regression models…")
def get_regression(df):
    return pl.run_regression(df)


@st.cache_resource(show_spinner="Training classifiers…")
def get_classification(df):
    return pl.run_classification(df)


st.sidebar.title("🚕 Uber Analytics")
DATA_DIR.mkdir(exist_ok=True)
# any CSV in data/ except the synthetic sample counts as the real dataset (largest file wins)
real_files = sorted((f for f in DATA_DIR.glob("*.csv") if f.name != SAMPLE.name),
                    key=lambda f: f.stat().st_size, reverse=True)
using_sample = False
if real_files:
    csv_path = real_files[0]
    data = get_data(str(csv_path), f"{csv_path.name}-{csv_path.stat().st_mtime}")
    st.sidebar.caption(f"Data file: `data/{csv_path.name}`")
elif SAMPLE.exists():
    data = get_data(str(SAMPLE), "sample")
    using_sample = True
else:
    st.error(f"No CSV found. Copy your downloaded dataset into this folder:\n\n`{DATA_DIR}`")
    st.stop()

if using_sample:
    st.warning("Showing **synthetic sample data**. Copy your downloaded Kaggle CSV into the "
               f"`data/` folder (`{DATA_DIR}`) and refresh the page.")

st.sidebar.markdown("### Filters")
dmin, dmax = data["Date"].min().date(), data["Date"].max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
veh = st.sidebar.multiselect("Vehicle type", sorted(data["Vehicle Type"].dropna().unique()))
stat = st.sidebar.multiselect("Booking status", sorted(data["Booking Status"].dropna().unique()))
df = pl.filter_df(data, dr if len(dr) == 2 else None, veh, stat)
st.sidebar.caption(f"{len(df):,} of {len(data):,} bookings selected. "
                   "Filters apply to Overview, Correlations and Forecast. "
                   "Models are trained on the full dataset.")
if df.empty:
    st.info("No rows match the filters.")
    st.stop()

st.title("Uber Ride Analytics & Prediction")
tabs = st.tabs(["Overview", "Correlations", "Fare regression", "Demand forecast",
                "Cancellation classifier", "Data"])

# -------------------------------------------------------------- overview ---
with tabs[0]:
    k = pl.kpis(df)
    c = st.columns(5)
    c[0].metric("Bookings", f"{k['Total bookings']:,}")
    c[1].metric("Completion rate", f"{k['Completion rate']:.1%}")
    c[2].metric("Cancellation rate", f"{k['Cancellation rate']:.1%}")
    c[3].metric("Revenue", f"₹{k['Revenue']:,.0f}")
    c[4].metric("Avg fare", f"₹{k['Avg fare']:,.0f}" if pd.notna(k["Avg fare"]) else "–")

    a, b = st.columns(2)
    daily = df.groupby("Date").size().reset_index(name="Bookings")
    fig = px.line(daily, x="Date", y="Bookings", title="Daily bookings", **PLOT)
    fig.update_traces(line_color=ACCENT)
    a.plotly_chart(fig, width="stretch")

    hw = df.groupby(["Weekday Name", "Hour"]).size().reset_index(name="n")
    hw["Weekday Name"] = pd.Categorical(hw["Weekday Name"], DAYS, ordered=True)
    heat = hw.pivot(index="Weekday Name", columns="Hour", values="n")
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                    title="Demand by weekday and hour", labels=dict(color="Bookings"))
    fig.update_layout(**PLOT)
    b.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    st_ = df["Booking Status"].value_counts().reset_index()
    st_.columns = ["Status", "Bookings"]
    fig = px.bar(st_, x="Bookings", y="Status", orientation="h", title="Booking status", **PLOT)
    fig.update_traces(marker_color=ACCENT)
    fig.update_yaxes(categoryorder="total ascending")
    a.plotly_chart(fig, width="stretch")

    vt = (df.groupby("Vehicle Type")
            .agg(Bookings=("Completed", "size"), Completion=("Completed", "mean"),
                 Avg_fare=("Booking Value", "mean")).reset_index())
    fig = px.bar(vt.sort_values("Avg_fare"), x="Avg_fare", y="Vehicle Type", orientation="h",
                 title="Average fare by vehicle type (completed trips)", labels={"Avg_fare": "₹"}, **PLOT)
    fig.update_traces(marker_color=ACCENT)
    b.plotly_chart(fig, width="stretch")

    reasons = df["Reason for cancelling by Customer"].dropna().value_counts().reset_index()
    if not reasons.empty:
        reasons.columns = ["Reason", "Cancellations"]
        fig = px.bar(reasons, x="Cancellations", y="Reason", orientation="h",
                     title="Why customers cancel", **PLOT)
        fig.update_traces(marker_color=WARN)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------- correlations ---
with tabs[1]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = pl.correlation_matrix(df, method)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    aspect="auto", title=f"{method.title()} correlation (completed-trip fields are empty for cancellations)")
    fig.update_layout(height=600, **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Strongest relationships")
    st.dataframe(pl.top_correlations(corr, 10).style.format({"Correlation": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    st.caption("Correlation is not causation. Highly correlated pairs (e.g. distance and fare) "
               "are also why a single feature dominates the regression below.")

# ------------------------------------------------------------ regression ---
with tabs[2]:
    st.subheader("Predict the fare of a trip")
    reg = get_regression(data)
    st.dataframe(reg.metrics.style.format({"MAE": "₹{:.1f}", "RMSE": "₹{:.1f}", "R²": "{:.3f}"})
                 .highlight_min(subset=["MAE", "RMSE"], color="#dcfce7")
                 .highlight_max(subset=["R²"], color="#dcfce7"),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    p = reg.predictions.sample(min(3000, len(reg.predictions)), random_state=1)
    fig = px.scatter(p, x="Actual", y="Predicted", opacity=.35, title=f"{reg.best_name}: actual vs predicted (test set)", **PLOT)
    lim = [p.min().min(), p.max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=WARN, dash="dash"), name="perfect"))
    fig.update_traces(marker_color=ACCENT, selector=dict(mode="markers"))
    a.plotly_chart(fig, width="stretch")
    fig = px.bar(reg.importances, x="Importance", y="Feature", orientation="h",
                 title="Permutation importance (drop in R² when shuffled)", **PLOT)
    fig.update_traces(marker_color=ACCENT)
    b.plotly_chart(fig, width="stretch")

    st.markdown("**Try it:** estimate a fare")
    c = st.columns(4)
    dist = c[0].number_input("Distance (km)", 1.0, 80.0, 12.0)
    vt_ = c[1].selectbox("Vehicle", sorted(data["Vehicle Type"].dropna().unique()))
    hr = c[2].slider("Hour", 0, 23, 18)
    wd = c[3].selectbox("Weekday", DAYS, index=4)
    c = st.columns(3)
    pk = c[0].selectbox("Pickup", sorted(data["Pickup Location"].dropna().unique()))
    dp = c[1].selectbox("Drop", sorted(data["Drop Location"].dropna().unique()), index=1)
    mo = c[2].slider("Month", 1, 12, int(data["Month"].mode()[0]))
    row = pd.DataFrame([{"Ride Distance": dist, "Hour": hr, "Weekday": DAYS.index(wd), "Month": mo,
                         "Is Peak": int(hr in pl.PEAK_HOURS), "Is Weekend": int(DAYS.index(wd) >= 5),
                         "Vehicle Type": vt_, "Pickup Location": pk, "Drop Location": dp}])
    st.metric("Predicted fare", f"₹{reg.best_model.predict(row)[0]:,.0f}",
              help=f"±₹{reg.metrics.loc[0, 'MAE']:.0f} typical error (test MAE)")

# -------------------------------------------------------------- forecast ---
with tabs[3]:
    c = st.columns(3)
    metric = c[0].selectbox("Forecast", ["Bookings", "Revenue"])
    horizon = c[1].slider("Days ahead", 7, 90, 30)
    s = pl.daily_series(df if metric == "Bookings" else df[df["Completed"] == 1],
                        "Bookings" if metric == "Bookings" else "Revenue")
    try:
        fc = pl.run_forecast(s, horizon)
    except ValueError as e:
        st.info(str(e))
    else:
        st.dataframe(fc.metrics.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE %": "{:.1f}"})
                     .highlight_min(subset=["MAE", "RMSE", "MAPE %"], color="#dcfce7"),
                     hide_index=True, width="stretch")
        st.caption(f"Methods compared on the last 28 days (held out). Best: **{fc.best_name}**, "
                   "refit on all data for the forecast.")
        fig = go.Figure()
        hist = s.iloc[-120:]
        fig.add_scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color=MUTED))
        fig.add_scatter(x=fc.test_pred[fc.best_name].index, y=fc.test_pred[fc.best_name].values,
                        name="Back-test", line=dict(color=ACCENT, dash="dot"))
        fig.add_scatter(x=fc.future.index, y=fc.future["Upper"], line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc.future.index, y=fc.future["Lower"], line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(37,99,235,.15)", name="95% interval")
        fig.add_scatter(x=fc.future.index, y=fc.future["Forecast"], name="Forecast", line=dict(color=ACCENT))
        fig.update_layout(title=f"Daily {metric.lower()}: history and {horizon}-day forecast", **PLOT)
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download forecast CSV", fc.future.round(1).to_csv().encode(), "forecast.csv")

# --------------------------------------------------------- classification ---
with tabs[4]:
    st.subheader("Will a booking be cancelled?")
    clf = get_classification(data)
    st.caption(f"Base cancellation rate: {clf.base_rate:.1%}. Features are limited to what is known at booking "
               "time (no fare or distance, which are empty for cancelled trips).")
    st.dataframe(clf.metrics.style.format({c: "{:.3f}" for c in clf.metrics.columns[1:]})
                 .highlight_max(subset=list(clf.metrics.columns[1:]), color="#dcfce7"),
                 hide_index=True, width="stretch")
    a, b, c3 = st.columns(3)
    fig = go.Figure()
    for n, (f, t) in clf.roc.items():
        fig.add_scatter(x=f, y=t, name=n)
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=MUTED), name="chance")
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", **PLOT)
    a.plotly_chart(fig, width="stretch")
    fig = px.imshow(clf.cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Not cancelled", "Cancelled"], y=["Not cancelled", "Cancelled"],
                    title=f"Confusion matrix: {clf.best_name}", labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(**PLOT)
    b.plotly_chart(fig, width="stretch")
    fig = px.bar(clf.importances, x="Importance", y="Feature", orientation="h",
                 title="Permutation importance (drop in AUC)", **PLOT)
    fig.update_traces(marker_color=ACCENT)
    c3.plotly_chart(fig, width="stretch")

    st.markdown("**Try it:** cancellation risk for a booking")
    c = st.columns(4)
    hr = c[0].slider("Hour of day", 0, 23, 8, key="ch")
    wd = c[1].selectbox("Weekday", DAYS, key="cw")
    vt_ = c[2].selectbox("Vehicle", sorted(data["Vehicle Type"].dropna().unique()), key="cv")
    vtat = c[3].slider("Avg pickup ETA (min)", 1.0, 30.0, float(data["Avg VTAT"].median()), key="cvt")
    c = st.columns(2)
    pk = c[0].selectbox("Pickup", sorted(data["Pickup Location"].dropna().unique()), key="cp")
    dp = c[1].selectbox("Drop", sorted(data["Drop Location"].dropna().unique()), index=1, key="cd")
    row = pd.DataFrame([{"Hour": hr, "Weekday": DAYS.index(wd), "Month": int(data["Month"].mode()[0]),
                         "Is Peak": int(hr in pl.PEAK_HOURS), "Is Weekend": int(DAYS.index(wd) >= 5),
                         "Avg VTAT": vtat, "Vehicle Type": vt_, "Pickup Location": pk, "Drop Location": dp}])
    prob = float(clf.best_model.predict_proba(row)[0, 1])
    st.metric("Cancellation probability", f"{prob:.1%}", f"{prob - clf.base_rate:+.1%} vs average", delta_color="inverse")
    st.caption("Class weights are balanced, so probabilities run higher than the raw cancellation rate; "
               "read them as relative risk.")

# ------------------------------------------------------------------ data ---
with tabs[5]:
    st.dataframe(df.head(1000), width="stretch")
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode(), "filtered_rides.csv")
