"""
Phase 6 - Deployment
Streamlit app: pick a store + forecast horizon (7/14/30 days),
see historical sales + XGBoost forecast on one chart.

Run with:  streamlit run app.py

Requires train.csv and store.csv (raw Kaggle files) in the same folder.
The app rebuilds cleaning + features + model itself so it's self-contained
and doesn't depend on having run Phases 1-5 as separate scripts first.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Sales Forecast", layout="wide", page_icon="📈")

# Palette: teal for actuals, amber for the forecast, coral for closed days,
# violet for accent metrics -- chosen for clear separation against the
# dark background and to stay distinguishable for common color-vision
# deficiencies (teal/amber/violet are spread across hue, not just red/green).
COLOR_ACTUAL = "#00C2A8"    # teal
COLOR_FORECAST = "#FFB020"  # amber
COLOR_CLOSED = "#FF5C5C"    # coral/red
COLOR_ACCENT = "#8B7CF6"    # violet
BG_DARK = "#0E1117"
PANEL_DARK = "#1B1F2A"
GRID_COLOR = "#333844"
TEXT_LIGHT = "#E6E6E6"

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": PANEL_DARK,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_LIGHT,
    "text.color": TEXT_LIGHT,
    "xtick.color": TEXT_LIGHT,
    "ytick.color": TEXT_LIGHT,
    "grid.color": GRID_COLOR,
    "legend.facecolor": PANEL_DARK,
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_LIGHT,
})

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday", "Promo", "SchoolHoliday",
]


# ---------------------------------------------------------------
# Data loading + cleaning (Phase 2 logic, condensed)
# ---------------------------------------------------------------
@st.cache_data
def load_clean_data():
    df = pd.read_csv("train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv("store.csv")
    df = df.sort_values(["Store", "Date"])

    def reindex_store(g):
        g = g.set_index("Date")
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_range)
        g.index.name = "Date"
        return g

    df = (
        df.groupby("Store", group_keys=True)
        .apply(reindex_store, include_groups=False)
        .reset_index()
        .rename(columns={"level_1": "Date"})
    )
    if "Store" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Store"})

    is_gap = df["Sales"].isna() & df["Open"].isna()
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df.loc[is_gap & (df["DayOfWeek"] == 7), "Open"] = 0
    df.loc[is_gap & (df["DayOfWeek"] != 7), "Open"] = 1

    closed = df["Open"] == 0
    df.loc[closed, "Sales"] = df.loc[closed, "Sales"].fillna(0)
    df.loc[closed, "Customers"] = df.loc[closed, "Customers"].fillna(0)
    df["Sales"] = df.groupby("Store")["Sales"].transform(
        lambda s: s.interpolate(method="linear", limit=3)
    )
    for col, default in [("Promo", 0), ("StateHoliday", "0"), ("SchoolHoliday", 0)]:
        df[col] = df[col].fillna(default)

    df = df.merge(store, on="Store", how="left")
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].max() * 2
        )
    df = df.dropna(subset=["Sales"])
    return df


# ---------------------------------------------------------------
# Feature engineering (Phase 4 logic, condensed)
# ---------------------------------------------------------------
def add_calendar_features(df):
    df = df.copy()
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
    return df


def add_lag_roll_features(df):
    df = df.sort_values(["Store", "Date"]).copy()
    for lag in [1, 7, 14, 30]:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    for window in [7, 30]:
        shifted = df.groupby("Store")["Sales"].shift(1)
        df[f"sales_roll_mean_{window}"] = shifted.groupby(df["Store"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"sales_roll_std_{window}"] = shifted.groupby(df["Store"]).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


@st.cache_resource
def train_model(df):
    feat = add_calendar_features(df)
    feat = add_lag_roll_features(feat)
    feat = feat[feat["Open"] == 1].dropna(subset=FEATURE_COLS + ["Sales"])

    X, y = feat[FEATURE_COLS], feat["Sales"]
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------
# Recursive multi-step forecast
# ---------------------------------------------------------------
# WHY RECURSIVE: lag/rolling features (e.g. sales_lag_1, sales_roll_mean_7)
# require actual sales values from the recent past. For day 1 of the
# forecast we have real history to compute them from, but for day 2 we
# don't have a real "yesterday" yet -- so we feed the model's OWN day-1
# prediction back in as if it were observed, then predict day 2, and so
# on. This is standard for multi-step forecasting with a model trained
# for one-step-ahead prediction, but note that errors can compound the
# further out the horizon goes (day 30's forecast rests partly on 29
# earlier predictions, not ground truth).
def recursive_forecast(model, store_history, horizon_days):
    history = store_history.sort_values("Date").copy()
    last_date = history["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # Infer which weekdays this SPECIFIC store is normally open on, from
    # its own history (Phase 2 found Rossmann stores commonly close on a
    # fixed weekday, e.g. Store 1 is closed every single Sunday). Without
    # this, a future Sunday would be fed to the model as a normal open
    # day, producing a nonsense mid-range prediction instead of the
    # correct $0 -- this is exactly the bug caught in testing below.
    open_rate_by_dow = history.assign(dow=history["Date"].dt.dayofweek).groupby("dow")["Open"].mean()

    working = history[["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
    predictions = []

    for future_date in future_dates:
        dow = future_date.dayofweek
        # Treat the store as closed on this weekday if it was historically
        # open less than half the time on that weekday.
        is_open = int(open_rate_by_dow.get(dow, 1.0) >= 0.5)

        # Future Promo/StateHoliday are not actually known in advance;
        # assume no promo / no special holiday as a neutral default.
        # Weekday IS knowable exactly (calendar math), which combined with
        # the inferred Open pattern is the dominant seasonal driver from
        # our EDA.
        new_row = pd.DataFrame([{
            "Store": working["Store"].iloc[0],
            "Date": future_date,
            "Sales": 0.0 if not is_open else np.nan,
            "Open": is_open,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }])
        working = pd.concat([working, new_row], ignore_index=True)

        if is_open:
            feat = add_calendar_features(working)
            feat = add_lag_roll_features(feat)
            row = feat.iloc[[-1]]
            pred = max(0.0, model.predict(row[FEATURE_COLS])[0])
            working.loc[working.index[-1], "Sales"] = pred
        else:
            pred = 0.0

        predictions.append({"Date": future_date, "Predicted_Sales": pred, "Open": is_open})

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.title("📈 Store Sales Forecast")
st.caption("XGBoost model trained on lag / rolling / calendar features (Rossmann Store Sales)")
st.divider()

with st.spinner("Loading and cleaning data..."):
    data = load_clean_data()

store_ids = sorted(data["Store"].unique())

col1, col2 = st.columns(2)
with col1:
    selected_store = st.selectbox("Select a store", store_ids, index=0)
with col2:
    horizon = st.selectbox("Forecast horizon (days)", [7, 14, 30], index=0)

with st.spinner("Training model (cached after first run)..."):
    model = train_model(data)

store_history = data[data["Store"] == selected_store].sort_values("Date")

if st.button("Generate Forecast", type="primary"):
    with st.spinner(f"Forecasting next {horizon} days for Store {selected_store}..."):
        forecast_df = recursive_forecast(model, store_history, horizon)

    open_days = forecast_df[forecast_df["Open"] == 1]
    closed_days = forecast_df[forecast_df["Open"] == 0]

    # KPI cards for a quick read before looking at the chart
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg forecast (open days)", f"{open_days['Predicted_Sales'].mean():,.0f}"
               if len(open_days) else "—")
    m2.metric("Peak day", f"{forecast_df['Predicted_Sales'].max():,.0f}")
    m3.metric("Closed days in horizon", int((forecast_df["Open"] == 0).sum()))
    last_hist_avg = store_history["Sales"].tail(30).mean()
    delta_pct = (
        (open_days["Predicted_Sales"].mean() - last_hist_avg) / last_hist_avg * 100
        if len(open_days) and last_hist_avg else 0
    )
    m4.metric("vs last 30-day avg", f"{delta_pct:+.1f}%")

    # Plot: last 90 days of actual history + forecast, dark theme palette
    history_window = store_history.tail(90)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history_window["Date"], history_window["Sales"],
            label="Historical Sales", color=COLOR_ACTUAL, linewidth=2)
    ax.plot(open_days["Date"], open_days["Predicted_Sales"],
            label=f"Forecast (next {horizon} days)", color=COLOR_FORECAST,
            linestyle="--", marker="o", markersize=5, linewidth=2)
    if len(closed_days):
        ax.scatter(closed_days["Date"], closed_days["Predicted_Sales"],
                    label="Predicted closed", color=COLOR_CLOSED, marker="x", s=60, zorder=5)
    ax.axvline(history_window["Date"].max(), color=COLOR_ACCENT, linestyle=":", linewidth=1.5)
    ax.set_title(f"Store {selected_store} — Historical Sales & {horizon}-Day Forecast", color=TEXT_LIGHT)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Forecast values")
    display_df = forecast_df.rename(
        columns={"Predicted_Sales": "Predicted Sales", "Open": "Store Open"}
    )
    display_df["Store Open"] = display_df["Store Open"].map({1: "Yes", 0: "No (closed)"})
    st.dataframe(
        display_df.style.format({"Predicted Sales": "{:.0f}"}),
        use_container_width=True,
    )
else:
    st.info("Select a store and horizon, then click **Generate Forecast**.")
    fig, ax = plt.subplots(figsize=(14, 4))
    recent = store_history.tail(180)
    ax.plot(recent["Date"], recent["Sales"], color=COLOR_ACTUAL, linewidth=1.5)
    ax.set_title(f"Store {selected_store} — Last 180 Days", color=TEXT_LIGHT)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
