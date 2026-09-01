"""
Phase 6 - Deployment
Streamlit app: pick a store + forecast horizon (7/14/30 days),
see historical sales + XGBoost forecast on one chart.

Run with:  streamlit run app.py

Requires train.csv and store.csv (raw Kaggle files) in the same folder,
plus store_health_scores.csv (output of combine_store_health.py). The app
rebuilds cleaning + features + the point-forecast model itself so it's
self-contained and doesn't depend on having run Phases 1-5 as separate
scripts first. store.csv is only read by the What-If Simulator tab (for
StoreType/CompetitionDistance) -- FEATURE_COLS still doesn't use any of
its columns for the forecast model itself.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
import shap

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
# Data loading + cleaning (Phase 2 logic, condensed for low memory)
# ---------------------------------------------------------------
# MEMORY NOTES (this app runs on Streamlit Community Cloud's free tier,
# capped at ~1GB RAM -- the original version crashed with "gone over
# resource limits" because of this function):
#   1. store.csv was merged in but none of its columns (StoreType,
#      CompetitionDistance, etc.) are in FEATURE_COLS below -- pure waste.
#      Dropped entirely; is_holiday only needs StateHoliday from train.csv.
#   2. `Customers` isn't used anywhere downstream -- dropped at load time
#      via usecols instead of carrying a whole extra float column through
#      every later copy.
#   3. groupby("Store").apply(reindex_store) builds ~1,115 separate
#      DataFrames (one per store) and holds them ALL in memory before
#      concatenating them -- a well-known pandas memory spike. Replaced
#      with a single vectorized MultiIndex reindex (Store x full date
#      range), which does the same thing in one allocation.
#   4. float64/int64 are pandas' default dtypes but are 2x the size we
#      need here -- downcast to float32/int8/int16 right after loading.
@st.cache_data
def load_clean_data():
    df = pd.read_csv(
        "train.csv",
        usecols=["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"],
        parse_dates=["Date"],
        dtype={"Store": "int16", "Promo": "int8", "SchoolHoliday": "int8", "StateHoliday": "category"},
        low_memory=False,
    )
    df = df.sort_values(["Store", "Date"])

    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product([df["Store"].unique(), full_dates], names=["Store", "Date"])
    df = df.set_index(["Store", "Date"]).reindex(full_index).reset_index()

    is_gap = df["Sales"].isna() & df["Open"].isna()
    dow = df["Date"].dt.dayofweek + 1  # Mon=1..Sun=7
    df.loc[is_gap & (dow == 7), "Open"] = 0
    df.loc[is_gap & (dow != 7), "Open"] = 1

    closed = df["Open"] == 0
    df.loc[closed, "Sales"] = df.loc[closed, "Sales"].fillna(0)
    df["Sales"] = df.groupby("Store")["Sales"].transform(
        lambda s: s.interpolate(method="linear", limit=3)
    )
    df["Promo"] = df["Promo"].fillna(0)
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0)
    df["StateHoliday"] = df["StateHoliday"].astype(object).fillna("0").astype("category")

    df = df.dropna(subset=["Sales"])

    # Downcast after all fills/interpolation so no precision is lost mid-computation.
    df["Sales"] = df["Sales"].astype("float32")
    df["Open"] = df["Open"].astype("int8")
    df["Promo"] = df["Promo"].astype("int8")
    df["SchoolHoliday"] = df["SchoolHoliday"].astype("int8")

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


@st.cache_data
def build_training_matrix(df):
    """
    Feature engineering, done ONCE and cached, then reused by both
    train_model and train_quantile_models below. Three separate XGBoost
    models (point + 10th/90th percentile) each independently recomputing
    add_calendar_features/add_lag_roll_features on the full ~1M-row
    dataframe would triple both compute time and peak memory for no
    benefit, since all three train on identical X/y -- exactly the kind
    of redundant allocation the earlier memory-crash fix was about
    eliminating.
    """
    feat = add_calendar_features(df)
    feat = add_lag_roll_features(feat)
    feat = feat[feat["Open"] == 1].dropna(subset=FEATURE_COLS + ["Sales"])
    X = feat[FEATURE_COLS].astype("float32")
    y = feat["Sales"].astype("float32")
    return X, y


@st.cache_resource
def train_model(df):
    X, y = build_training_matrix(df)
    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist",  # histogram-binned splits: much lower memory than the exact method
    )
    model.fit(X, y)
    return model


@st.cache_resource
def train_quantile_models(df):
    """
    Trains two extra XGBoost models at the 10th and 90th percentiles
    (native XGBoost quantile regression, objective="reg:quantileerror")
    to produce a prediction INTERVAL alongside the point forecast, rather
    than a single number that implies false precision. Trained on the
    exact same X/y as the point model (via the shared, cached
    build_training_matrix) for consistency and to avoid recomputing
    features a second time.
    """
    X, y = build_training_matrix(df)
    common = dict(n_estimators=150, max_depth=5, learning_rate=0.08,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist")
    model_lo = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **common)
    model_hi = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **common)
    model_lo.fit(X, y)
    model_hi.fit(X, y)
    return model_lo, model_hi


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
def recursive_forecast(model, store_history, horizon_days, quantile_models=None):
    """
    Returns (forecast_df, X_forecast):
      - forecast_df: Date, Predicted_Sales, Open, and (if quantile_models
        given) Predicted_Sales_Lower/Upper for the 10th/90th percentile band.
      - X_forecast: the FEATURE_COLS row used for each OPEN forecasted day,
        in order -- exposed so a SHAP explainer can be run on the exact
        inputs that produced these predictions (used by the "Why this
        forecast?" panel), without recomputing feature engineering twice.
    """
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
    feature_rows = []

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

        pred_lo = pred_hi = None
        if is_open:
            feat = add_calendar_features(working)
            feat = add_lag_roll_features(feat)
            row = feat.iloc[[-1]]
            pred = max(0.0, model.predict(row[FEATURE_COLS])[0])
            working.loc[working.index[-1], "Sales"] = pred
            feature_rows.append(row[FEATURE_COLS])

            if quantile_models is not None:
                model_lo, model_hi = quantile_models
                # Quantile models score the SAME feature row as the point
                # model (built from the median trajectory), not independent
                # quantile paths -- a documented simplification. It keeps
                # the interval anchored to a single self-consistent
                # forecast path instead of requiring 3x the recursive
                # rollouts, at the cost of the band not fully reflecting
                # uncertainty that would compound differently at each
                # quantile over a multi-day horizon.
                pred_lo = max(0.0, model_lo.predict(row[FEATURE_COLS])[0])
                pred_hi = max(pred_lo, model_hi.predict(row[FEATURE_COLS])[0])
        else:
            pred = 0.0
            if quantile_models is not None:
                pred_lo = pred_hi = 0.0

        pred_row = {"Date": future_date, "Predicted_Sales": pred, "Open": is_open}
        if quantile_models is not None:
            pred_row["Predicted_Sales_Lower"] = pred_lo
            pred_row["Predicted_Sales_Upper"] = pred_hi
        predictions.append(pred_row)

    forecast_df = pd.DataFrame(predictions)
    X_forecast = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame(columns=FEATURE_COLS)
    return forecast_df, X_forecast


# ---------------------------------------------------------------
# Extra palette entries for the multi-series trend/seasonality charts
# in the "Upload Your Own Data" tab (kept distinct in hue from the
# forecast-tab colors above so nothing clashes if both tabs are open).
# ---------------------------------------------------------------
COLOR_SKY = "#4CC3FF"
COLOR_ROLL7 = "#FFB020"   # amber, reused: short rolling window
COLOR_ROLL30 = "#8B7CF6"  # violet, reused: long rolling window


def _guess_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return df.columns[0]


def _guess_value_column(df, date_col):
    numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    for kw in ["sales", "revenue", "amount", "value", "qty", "quantity", "total"]:
        for c in numeric_cols:
            if kw in c.lower():
                return c
    return numeric_cols[0] if numeric_cols else None


# ---------------------------------------------------------------
# What-If Simulator helpers
# ---------------------------------------------------------------
# combine_store_score / assign_tier are duplicated here (not imported)
# from combine_store_health.py, kept identical on purpose: importing that
# script would re-run its full offline training pipeline as a side effect
# of the import, which is exactly what we avoided for the deployed app's
# memory budget. Keep these two in sync by hand if the scoring formula
# changes in combine_store_health.py.
def combine_store_score(forecast_mape, anomaly_rate, sentiment_score):
    if forecast_mape is None or (isinstance(forecast_mape, float) and np.isnan(forecast_mape)):
        forecast_mape = 50.0
    reliability = np.clip(100 - forecast_mape * 2, 0, 100)
    anomaly_component = np.clip(100 - anomaly_rate * 400, 0, 100)
    sentiment_component = np.clip(sentiment_score * 100, 0, 100)
    return round(0.5 * reliability + 0.3 * anomaly_component + 0.2 * sentiment_component, 1)


def assign_tier(score):
    if score <= 40:
        return "Needs Attention"
    elif score <= 70:
        return "Monitor"
    return "Performing Well"


# Lightweight, dependency-free keyword sentiment scorer for the live text
# box below -- deliberately NOT the HuggingFace transformer pipeline used
# in generate_and_analyze_reviews.py. Loading a transformer model inside
# the deployed app risks the exact memory crash already fixed once; this
# runs in microseconds with zero extra RAM.
POSITIVE_WORDS = {
    "good", "great", "excellent", "friendly", "clean", "helpful", "fast", "quick",
    "convenient", "love", "loved", "nice", "fair", "fresh", "organized", "polite",
    "amazing", "wonderful", "well-stocked", "stocked", "affordable", "easy", "pleasant",
    "best", "recommend", "efficient", "welcoming", "spacious", "reliable",
}
NEGATIVE_WORDS = {
    "bad", "dirty", "rude", "slow", "expensive", "overpriced", "empty", "messy",
    "unhelpful", "poor", "terrible", "awful", "disorganized", "cramped", "dark",
    "worst", "avoid", "disappointing", "unfriendly", "understaffed", "chaotic",
    "outdated", "cluttered", "sold-out", "closed", "long", "wait", "hassle",
}


def simple_sentiment_score(text):
    words = set(w.strip(".,!?;:'\"").lower() for w in text.split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.5  # neutral default: no recognized sentiment-bearing words
    return pos / (pos + neg)


@st.cache_data
def load_health_with_metadata():
    health = pd.read_csv("store_health_scores.csv")
    store_meta = pd.read_csv("store.csv", usecols=["Store", "StoreType", "Assortment", "CompetitionDistance"])
    return health.merge(store_meta, on="Store", how="left")


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.title("📈 Sales Forecast & Trend Explorer")
st.caption("XGBoost model trained on lag / rolling / calendar features (Rossmann Store Sales)")

with st.expander("ℹ️ How this app works — read this first", expanded=False):
    st.markdown("""
This app has two independent tools, both built on the same idea: **past sales
patterns predict near-future sales**, because real-world demand isn't random
— it repeats on a weekly rhythm, drifts with seasons, and reacts to
promotions and holidays.

#### 🏬 Tab 1 — Rossmann Store Forecast
1. **Data cleaning**: 1,115 German drugstores' daily sales (2013–2015) are
   loaded and every store's calendar is filled to a continuous daily range —
   including days missing from the raw export. Days a store was genuinely
   closed (many close every Sunday) keep `Sales = 0`, since that's a real,
   recurring fact the model needs to learn — not something to hide or
   average away.
2. **Feature engineering**: for every day, the model is given the store's
   own sales from **1, 7, 14, and 30 days ago**, plus **7-day and 30-day
   rolling averages/volatility**, plus calendar signals (day of week, month,
   is-weekend, is-holiday, is-promo). These are the columns an
   [XGBoost](https://en.wikipedia.org/wiki/XGBoost) model is trained on —
   it has no built-in sense of time, so lag/rolling values are literally how
   it "remembers" the recent past.
3. **Forecasting**: predicting more than 1 day ahead is done **recursively**
   — day 1's prediction is fed back in as if it were real, so day 2 can be
   predicted from it, and so on. This is why longer horizons (30 days) are
   less reliable than shorter ones (7 days): later predictions are built
   partly on earlier *predictions*, not ground truth, so small errors can
   compound.
4. Days the model expects a store to be **closed** (inferred from that
   store's own weekday history, e.g. always-closed Sundays) are forced to a
   $0 forecast rather than asking the model to guess — a closure is a
   scheduling fact, not something to predict statistically.

#### 📁 Tab 2 — Upload Your Own Data
This tool doesn't use the trained model at all — it's a **generic
exploratory analysis**, not a forecast. Upload any CSV with a date column
and a numeric column (sales, revenue, orders, web traffic — anything), and
the app:
- auto-detects which columns are the date and the value (you can override it),
- plots the raw values with **7-period and 30-period rolling averages** so
  you can see the underlying trend through the day-to-day noise,
- averages by day-of-week and by month to reveal **seasonality patterns**,
  highlighting the strongest day/month,
- shows a distribution histogram and summary stats.

It answers *"what patterns exist in this data?"* — a first step before
anyone would build a forecasting model on it, which is exactly the kind of
exploration Tab 1's model was originally built on top of.

#### 📊 Tab 3 — Store Health
Combines three independent signals into one 0-100 score per store:
forecast reliability (how low the XGBoost model's error is on that
store's held-out days), anomaly rate (how often an LSTM Autoencoder
flags irregular sales days), and review sentiment (**synthetic** — the
Rossmann dataset has no real customer reviews). Forecast reliability is
weighted highest, since an unpredictable store is an operational risk
regardless of the underlying cause.

#### 🔮 Tab 4 — What-If: New Store Simulator
Projects a Store Health Score for a **hypothetical** store that has no
sales history yet. Forecast reliability and anomaly rate are estimated
from existing stores sharing the entered StoreType and a similar
CompetitionDistance — an approximation, not a guarantee. Review
sentiment for the entered sample text is scored by a lightweight keyword
matcher (not the transformer model Tab 3's underlying data uses), so the
live simulator stays fast and doesn't add memory overhead to the
deployed app.

#### 🗺️ Tab 5 — Portfolio Overview
Zooms out to all 1,115 stores at once — a chain-wide view for "where should
I focus this week?" instead of inspecting one store at a time. Precomputed
offline (forecasting every store live was measured at ~11 minutes, too slow
for a tab), so this reads `portfolio_forecast.csv` directly. **Read the
warning banner on that tab** — the % change figures compare against a
period with a very different school-holiday rate than the forecast assumes,
so treat them as a rough signal, not a precise number.
""")

st.divider()

tab_forecast, tab_upload, tab_health, tab_whatif, tab_portfolio = st.tabs(
    ["🏬 Rossmann Store Forecast", "📁 Upload Your Own Data", "📊 Store Health",
     "🔮 What-If: New Store Simulator", "🗺️ Portfolio Overview"]
)

# =================================================================
# TAB 1 — existing Rossmann forecast (unchanged behavior)
# =================================================================
with tab_forecast:
    st.caption(
        "An XGBoost model trained on this store's own sales history (lag + rolling-average "
        "features) predicts sales day-by-day, feeding each prediction back in to forecast "
        "the next — accuracy is highest in the first few days of the horizon."
    )
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
        model_lo, model_hi = train_quantile_models(data)

    store_history = data[data["Store"] == selected_store].sort_values("Date")

    # Cache the forecast in session_state, keyed by (store, horizon), so
    # widgets rendered AFTER this point (like the staffing-ratio input
    # below) can be changed without needing to re-click "Generate
    # Forecast" -- any other widget change triggers a Streamlit rerun,
    # and st.button() only evaluates True on the actual click event, so
    # without this the whole results section would vanish the moment the
    # ratio slider moved.
    forecast_key = (selected_store, horizon)
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Forecasting next {horizon} days for Store {selected_store}..."):
            forecast_df, X_forecast = recursive_forecast(
                model, store_history, horizon, quantile_models=(model_lo, model_hi)
            )
        st.session_state["tab1_forecast"] = (forecast_df, X_forecast, forecast_key)

    cached = st.session_state.get("tab1_forecast")
    if cached is not None and cached[2] == forecast_key:
        forecast_df, X_forecast, _ = cached
        open_days = forecast_df[forecast_df["Open"] == 1]
        closed_days = forecast_df[forecast_df["Open"] == 0]

        # =========================================================
        # Business Impact — translate the raw forecast into decision-
        # relevant language, shown ABOVE the chart since that's what a
        # store manager actually needs first.
        # =========================================================
        st.markdown("### 💼 Business Impact")

        forecast_total = open_days["Predicted_Sales"].sum()
        prev_period = store_history.tail(horizon)
        prev_total = prev_period["Sales"].sum()
        period_pct_change = (forecast_total - prev_total) / prev_total * 100 if prev_total else 0.0

        bi1, bi2, bi3 = st.columns(3)
        bi1.metric(f"Forecasted total ({horizon}d)", f"{forecast_total:,.0f}",
                   f"{period_pct_change:+.1f}% vs prior {horizon}d")

        staff_ratio = st.number_input(
            "Units per staff member (editable)", min_value=50, max_value=5000,
            value=500, step=50, key="staff_ratio",
            help="Adjust to your own staffing rule of thumb; suggested staffing recalculates live.",
        )
        avg_daily_forecast = open_days["Predicted_Sales"].mean() if len(open_days) else 0.0
        suggested_staff = int(np.ceil(avg_daily_forecast / staff_ratio)) if staff_ratio else 0
        bi2.metric("Suggested staffing (avg day)", f"{suggested_staff} staff",
                   help=f"= avg daily forecast ({avg_daily_forecast:,.0f}) / {staff_ratio} units per staff")
        bi3.metric("Suggested inventory order", f"{forecast_total:,.0f} units",
                   help=f"Sum of forecasted demand across all {horizon} days")

        # Flag days whose forecast is unusually high/low vs. this store's
        # own typical pattern for that weekday (not vs. a generic average
        # -- Monday always looks different from Wednesday, that's not an
        # anomaly, so the comparison has to be weekday-specific).
        weekday_avg = (
            store_history[store_history["Open"] == 1]
            .assign(dow=store_history["Date"].dt.dayofweek)
            .groupby("dow")["Sales"].mean()
        )
        flagged = []
        for i, (_, row) in enumerate(open_days.iterrows()):
            typical = weekday_avg.get(row["Date"].dayofweek, np.nan)
            if pd.isna(typical) or typical <= 0:
                continue
            diff_pct = (row["Predicted_Sales"] - typical) / typical * 100
            if abs(diff_pct) < 25:
                continue
            reason = "unusually high demand expected" if diff_pct > 0 else "unusually low demand expected"
            if i < len(X_forecast):
                feat_row = X_forecast.iloc[i]
                if feat_row.get("is_holiday", 0) == 1:
                    reason = "likely holiday effect"
                elif feat_row.get("Promo", 0) == 1 and diff_pct > 0:
                    reason = "active promo likely driving higher sales"
            flagged.append(f"**{row['Date'].strftime('%b %d')}** forecast is {diff_pct:+.0f}% vs. typical "
                            f"{row['Date'].strftime('%A')} — {reason}")
        if flagged:
            st.markdown("\n".join(f"- {line}" for line in flagged[:5]))

        # =========================================================
        # KPI cards
        # =========================================================
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

        # =========================================================
        # Signature interaction: Plotly chart with a shaded 80%
        # confidence band, kept at a stable `key` so Streamlit updates
        # the SAME chart in place (Plotly.react) rather than replacing
        # it -- combined with layout.transition, changing the horizon
        # animates the forecast line smoothly extending/retracting
        # instead of the chart instantly redrawing.
        # =========================================================
        history_window = store_history.tail(90)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_window["Date"], y=history_window["Sales"],
            mode="lines", name="Historical Sales",
            line=dict(color=COLOR_ACTUAL, width=2),
        ))
        if len(open_days) and "Predicted_Sales_Upper" in open_days.columns:
            band_x = list(open_days["Date"]) + list(open_days["Date"][::-1])
            band_y = list(open_days["Predicted_Sales_Upper"]) + list(open_days["Predicted_Sales_Lower"][::-1])
            fig.add_trace(go.Scatter(
                x=band_x, y=band_y, fill="toself", fillcolor="rgba(255,176,32,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="80% interval", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=open_days["Date"], y=open_days["Predicted_Sales"],
            mode="lines+markers", name=f"Forecast (next {horizon} days)",
            line=dict(color=COLOR_FORECAST, width=2, dash="dash"), marker=dict(size=6),
        ))
        if len(closed_days):
            fig.add_trace(go.Scatter(
                x=closed_days["Date"], y=closed_days["Predicted_Sales"],
                mode="markers", name="Predicted closed",
                marker=dict(color=COLOR_CLOSED, symbol="x", size=10),
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG_DARK, plot_bgcolor=PANEL_DARK,
            font=dict(color=TEXT_LIGHT),
            title=f"Store {selected_store} — Historical Sales & {horizon}-Day Forecast",
            xaxis_title="Date", yaxis_title="Sales",
            legend=dict(x=0.01, y=0.99),
            margin=dict(t=60, b=40),
            transition=dict(duration=600, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True, key="tab1_forecast_chart")
        st.caption(
            "Shaded band = 80% prediction interval (10th-90th percentile quantile models) — "
            "actual sales are expected to fall in this range 80% of the time."
        )

        # =========================================================
        # "Why this forecast?" — SHAP explainability
        # =========================================================
        with st.expander("🔍 Why this forecast?"):
            if len(X_forecast) == 0:
                st.info("No open forecasted days to explain (store predicted closed for the full horizon).")
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_forecast)
                mean_signed = shap_values.mean(axis=0)
                top_idx = np.argsort(np.abs(mean_signed))[-5:][::-1]

                shap_df = pd.DataFrame({
                    "feature": [FEATURE_COLS[i] for i in top_idx],
                    "contribution": [mean_signed[i] for i in top_idx],
                }).sort_values("contribution")

                fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                bar_colors = [COLOR_ACTUAL if v >= 0 else COLOR_CLOSED for v in shap_df["contribution"]]
                ax_shap.barh(shap_df["feature"], shap_df["contribution"], color=bar_colors)
                ax_shap.set_xlabel("Avg. contribution to forecast (sales units)")
                ax_shap.set_title(f"Top drivers of Store {selected_store}'s forecast", color=TEXT_LIGHT)
                ax_shap.axvline(0, color=GRID_COLOR, linewidth=1)
                ax_shap.grid(alpha=0.25, axis="x")
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                st.caption(
                    "Positive bars push the forecast higher than the model's baseline; negative bars pull "
                    "it lower. Averaged (SHAP values) across all open days in the selected horizon."
                )

        st.subheader("Forecast values")
        display_cols = {"Predicted_Sales": "Predicted Sales", "Open": "Store Open"}
        if "Predicted_Sales_Lower" in forecast_df.columns:
            display_cols["Predicted_Sales_Lower"] = "80% Interval Low"
            display_cols["Predicted_Sales_Upper"] = "80% Interval High"
        display_df = forecast_df.rename(columns=display_cols)
        display_df["Store Open"] = display_df["Store Open"].map({1: "Yes", 0: "No (closed)"})
        st.dataframe(
            display_df.style.format({c: "{:.0f}" for c in display_cols.values() if c != "Store Open"}),
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
        plt.close(fig)

# =================================================================
# TAB 2 — upload any CSV, auto-explore trends/seasonality
# =================================================================
with tab_upload:
    st.subheader("Upload a CSV to explore its trends")
    st.caption(
        "This is exploratory analysis, not a forecast — it reveals trend and seasonality "
        "patterns in whatever you upload. Works with any date + numeric-value time series "
        "(not just Rossmann data) — e.g. your own sales export, web traffic, orders, etc."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV with at least one date column and one numeric column to get started.")
    else:
        try:
            raw_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as CSV: {e}")
            st.stop()

        if raw_upload.empty or raw_upload.shape[1] < 2:
            st.error("The file needs at least two columns: a date and a numeric value.")
            st.stop()

        st.write(f"Loaded **{raw_upload.shape[0]:,} rows × {raw_upload.shape[1]} columns**")
        st.dataframe(raw_upload.head(), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            date_col = st.selectbox(
                "Date column", raw_upload.columns,
                index=list(raw_upload.columns).index(_guess_date_column(raw_upload)),
            )
        # Guess the value column AFTER date_col is chosen, so it's never the same column
        numeric_candidates = [c for c in raw_upload.columns if c != date_col]
        guessed_value = _guess_value_column(raw_upload, date_col)
        with col_b:
            value_col = st.selectbox(
                "Value column to analyze", numeric_candidates,
                index=numeric_candidates.index(guessed_value) if guessed_value in numeric_candidates else 0,
            )

        df_u = raw_upload[[date_col, value_col]].rename(columns={date_col: "Date", value_col: "Value"})
        df_u["Date"] = pd.to_datetime(df_u["Date"], errors="coerce")
        df_u["Value"] = pd.to_numeric(df_u["Value"], errors="coerce")
        n_before = len(df_u)
        df_u = df_u.dropna(subset=["Date", "Value"]).sort_values("Date")
        n_dropped = n_before - len(df_u)
        if n_dropped:
            st.warning(f"Dropped {n_dropped} row(s) with unparseable date/value.")

        if len(df_u) < 2:
            st.error("Not enough valid rows left to analyze after parsing dates/values.")
            st.stop()

        # Collapse duplicate dates (e.g. multiple stores/rows per day) by summing,
        # since trend/seasonality analysis needs one value per date.
        df_u = df_u.groupby("Date", as_index=False)["Value"].sum()

        # --- Summary stats ---
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Date range", f"{(df_u['Date'].max() - df_u['Date'].min()).days} days")
        s2.metric("Mean", f"{df_u['Value'].mean():,.1f}")
        s3.metric("Min / Max", f"{df_u['Value'].min():,.0f} / {df_u['Value'].max():,.0f}")
        s4.metric("Total", f"{df_u['Value'].sum():,.0f}")

        # --- Raw series + rolling trend ---
        st.markdown("#### Raw values over time, with rolling trend")
        df_u["roll_7"] = df_u["Value"].rolling(7, min_periods=1).mean()
        df_u["roll_30"] = df_u["Value"].rolling(30, min_periods=1).mean()

        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df_u["Date"], df_u["Value"], color=COLOR_SKY, alpha=0.5, linewidth=1, label=value_col)
        ax1.plot(df_u["Date"], df_u["roll_7"], color=COLOR_ROLL7, linewidth=2, label="7-period rolling mean")
        ax1.plot(df_u["Date"], df_u["roll_30"], color=COLOR_ROLL30, linewidth=2, label="30-period rolling mean")
        ax1.set_xlabel("Date")
        ax1.set_ylabel(value_col)
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.25)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # --- Weekly & monthly seasonality (only meaningful if data spans enough time) ---
        st.markdown("#### Seasonality patterns")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        month_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]

        df_u["DayOfWeek"] = df_u["Date"].dt.day_name()
        df_u["Month"] = df_u["Date"].dt.month_name()

        weekly_avg_u = df_u.groupby("DayOfWeek")["Value"].mean().reindex(weekday_order)
        monthly_avg_u = df_u.groupby("Month")["Value"].mean().reindex(month_order).dropna()

        seas_col1, seas_col2 = st.columns(2)
        with seas_col1:
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            colors_wd = [COLOR_ACTUAL if v == weekly_avg_u.max() else COLOR_SKY for v in weekly_avg_u]
            weekly_avg_u.plot(kind="bar", ax=ax2, color=colors_wd)
            ax2.set_title("Average by Day of Week", color=TEXT_LIGHT)
            ax2.set_ylabel(value_col)
            ax2.grid(alpha=0.25, axis="y")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with seas_col2:
            if len(monthly_avg_u) >= 2:
                fig3, ax3 = plt.subplots(figsize=(7, 4.5))
                colors_m = [COLOR_FORECAST if v == monthly_avg_u.max() else COLOR_ACCENT for v in monthly_avg_u]
                monthly_avg_u.plot(kind="bar", ax=ax3, color=colors_m)
                ax3.set_title("Average by Month", color=TEXT_LIGHT)
                ax3.set_ylabel(value_col)
                ax3.grid(alpha=0.25, axis="y")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("Not enough distinct months in this data to show monthly seasonality.")

        # --- Distribution ---
        st.markdown("#### Distribution of values")
        fig4, ax4 = plt.subplots(figsize=(14, 3.5))
        ax4.hist(df_u["Value"], bins=40, color=COLOR_ACCENT, edgecolor=BG_DARK)
        ax4.set_xlabel(value_col)
        ax4.set_ylabel("Frequency")
        ax4.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        with st.expander("View cleaned data used for these charts"):
            st.dataframe(df_u, use_container_width=True)

# =================================================================
# TAB 3 — combined Store Health Score
# (forecast reliability + LSTM anomaly rate + synthetic review sentiment)
# =================================================================
with tab_health:
    st.subheader("Store Health Score")
    st.caption(
        "Combines three independent signals into one 0-100 score: how reliable this store's "
        "forecast is (XGBoost MAPE on held-out days it wasn't trained on), how many "
        "anomalous sales days an LSTM Autoencoder flagged in the last 90 days, and review "
        "sentiment. **Review sentiment is synthetically generated for demonstration** — the "
        "Rossmann dataset has no real customer review data."
    )

    try:
        health_df = pd.read_csv("store_health_scores.csv")
    except FileNotFoundError:
        st.warning(
            "`store_health_scores.csv` not found. Run, in order: "
            "`train_anomaly_model.py` → `generate_and_analyze_reviews.py` → "
            "`combine_store_health.py`, then reload this app."
        )
        st.stop()

    TIER_ORDER = ["Needs Attention", "Monitor", "Performing Well"]
    TIER_COLORS = {
        "Needs Attention": COLOR_CLOSED,
        "Monitor": COLOR_FORECAST,
        "Performing Well": COLOR_ACTUAL,
    }

    # --- Chain-wide overview ---
    st.markdown("#### Chain-wide overview")
    tier_counts = health_df["tier"].value_counts().reindex(TIER_ORDER).fillna(0)

    ov1, ov2 = st.columns([1, 2])
    with ov1:
        for tier in TIER_ORDER:
            st.metric(tier, int(tier_counts[tier]))
    with ov2:
        fig_ov, ax_ov = plt.subplots(figsize=(7, 4))
        ax_ov.bar(TIER_ORDER, tier_counts.values, color=[TIER_COLORS[t] for t in TIER_ORDER])
        ax_ov.set_ylabel("Number of stores")
        ax_ov.set_title("Stores by Health Tier", color=TEXT_LIGHT)
        ax_ov.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        st.pyplot(fig_ov)
        plt.close(fig_ov)

    st.divider()

    # --- Individual store lookup ---
    health_store_ids = sorted(health_df["Store"].unique())
    health_selected_store = st.selectbox(
        "Select a store to inspect", health_store_ids, key="health_store_select"
    )

    row = health_df[health_df["Store"] == health_selected_store].iloc[0]
    score = row["health_score"]
    tier = row["tier"]
    tier_color = TIER_COLORS.get(tier, TEXT_LIGHT)

    st.markdown(
        f"<div style='display:flex; align-items:baseline; gap:16px; margin-top:8px;'>"
        f"<span style='font-size:64px; font-weight:700; color:{tier_color};'>{score:.0f}</span>"
        f"<span style='font-size:20px; color:{TEXT_LIGHT};'>/ 100</span>"
        f"<span style='background:{tier_color}; color:{BG_DARK}; padding:4px 14px; "
        f"border-radius:14px; font-weight:600; font-size:14px;'>{tier}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"*{row['explanation']}*")

    h1, h2, h3 = st.columns(3)
    reliability = max(0.0, 100 - row["forecast_mape"] * 2)
    h1.metric("Forecast Reliability", f"{reliability:.0f}/100",
              help=f"XGBoost MAPE on this store's last 30 held-out days: {row['forecast_mape']:.1f}%")
    h2.metric("Anomaly Rate (90d)", f"{row['anomaly_rate'] * 100:.1f}%",
              help=f"{int(row['anomaly_count_90d'])} day(s) flagged by the LSTM Autoencoder")
    h3.metric("Review Sentiment", f"{row['avg_sentiment'] * 100:.0f}/100",
              help=f"Dominant theme: {row['dominant_theme']} (synthetic reviews)")

    st.divider()

    # --- Same forecast chart as Tab 1, for this store, for context ---
    st.markdown(f"#### Forecast context for Store {health_selected_store}")
    health_store_history = data[data["Store"] == health_selected_store].sort_values("Date")
    health_forecast_df, _ = recursive_forecast(model, health_store_history, 14)
    health_open = health_forecast_df[health_forecast_df["Open"] == 1]
    health_closed = health_forecast_df[health_forecast_df["Open"] == 0]

    fig_h, ax_h = plt.subplots(figsize=(14, 5))
    hist_window = health_store_history.tail(90)
    ax_h.plot(hist_window["Date"], hist_window["Sales"], label="Historical Sales",
              color=COLOR_ACTUAL, linewidth=2)
    ax_h.plot(health_open["Date"], health_open["Predicted_Sales"], label="14-Day Forecast",
              color=COLOR_FORECAST, linestyle="--", marker="o", markersize=4, linewidth=2)
    if len(health_closed):
        ax_h.scatter(health_closed["Date"], health_closed["Predicted_Sales"], label="Predicted closed",
                     color=COLOR_CLOSED, marker="x", s=50, zorder=5)
    ax_h.set_title(f"Store {health_selected_store} — Sales & Forecast (health context)", color=TEXT_LIGHT)
    ax_h.legend(loc="upper left")
    ax_h.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_h)
    plt.close(fig_h)

# =================================================================
# TAB 4 — What-If: New Store Simulator
# =================================================================
with tab_whatif:
    st.subheader("What-If: New Store Simulator")
    st.caption(
        "Project a Store Health Score for a hypothetical store that doesn't exist yet — "
        "useful for \"should we open a store here\" questions. A new store has no sales "
        "history, so forecast reliability and anomaly rate are estimated from existing "
        "stores with similar characteristics, not calculated directly."
    )

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        wi_store_type = st.selectbox("Store Type", ["a", "b", "c", "d"], key="wi_store_type")
        wi_assortment = st.selectbox("Assortment Level", ["a", "b", "c"], key="wi_assortment")
    with wi_col2:
        wi_competition_distance = st.number_input(
            "Competition Distance (meters)", min_value=0, max_value=50000, value=1000, step=100,
            key="wi_competition_distance",
        )
        wi_promo2 = st.checkbox("Will run Promo2 (recurring promotion)?", key="wi_promo2")

    wi_review_text = st.text_area(
        "Sample customer review text (simulates expected sentiment)",
        placeholder="e.g. Friendly staff, always well stocked, but a bit expensive.",
        key="wi_review_text",
    )

    if st.button("Run Simulation", type="primary"):
        health_meta = load_health_with_metadata()

        # Comparable stores: same StoreType, CompetitionDistance within a
        # tolerance band. If too few match, widen to the nearest N by
        # distance within the same StoreType instead of failing outright.
        same_type = health_meta[health_meta["StoreType"] == wi_store_type]
        tolerance = 1000
        similar = same_type[(same_type["CompetitionDistance"] - wi_competition_distance).abs() <= tolerance]
        if len(similar) < 5 and len(same_type) > 0:
            similar = same_type.assign(
                _dist_diff=(same_type["CompetitionDistance"] - wi_competition_distance).abs()
            ).nsmallest(12, "_dist_diff")
        if len(similar) == 0:
            similar = health_meta  # ultimate fallback: chain-wide average

        est_mape = float(similar["forecast_mape"].median())
        est_anomaly_rate = float(similar["anomaly_rate"].median())
        sentiment_score = simple_sentiment_score(wi_review_text) if wi_review_text.strip() else 0.5

        st.caption(
            f"Estimated based on {len(similar)} existing store(s) with StoreType='{wi_store_type}' "
            f"and similar CompetitionDistance."
        )

        projected_score = combine_store_score(est_mape, est_anomaly_rate, sentiment_score)
        projected_tier = assign_tier(projected_score)
        proj_color = TIER_COLORS.get(projected_tier, TEXT_LIGHT)

        st.markdown(
            f"<div style='display:flex; align-items:baseline; gap:16px; margin-top:8px;'>"
            f"<span style='font-size:64px; font-weight:700; color:{proj_color};'>{projected_score:.0f}</span>"
            f"<span style='font-size:20px; color:{TEXT_LIGHT};'>/ 100 (projected)</span>"
            f"<span style='background:{proj_color}; color:{BG_DARK}; padding:4px 14px; "
            f"border-radius:14px; font-weight:600; font-size:14px;'>{projected_tier}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        w1, w2, w3 = st.columns(3)
        w1.metric("Est. Forecast Reliability", f"{max(0.0, 100 - est_mape * 2):.0f}/100",
                   help=f"Estimated MAPE from comparable stores: {est_mape:.1f}%")
        w2.metric("Est. Anomaly Rate", f"{est_anomaly_rate * 100:.1f}%")
        w3.metric("Review Sentiment", f"{sentiment_score * 100:.0f}/100",
                   help="Scored by a lightweight keyword matcher, not a full ML model")

        st.warning(
            "⚠️ This is a projection based on similar existing stores, not a guarantee — "
            "this hypothetical store has no actual sales history yet."
        )

# =================================================================
# TAB 5 — Portfolio Overview (chain-wide, all stores at once)
# =================================================================
with tab_portfolio:
    st.subheader("Portfolio Overview")
    st.caption(
        "All 1,115 stores at once — answers \"where should I focus this week?\" without "
        "clicking into individual stores. Precomputed offline (`generate_portfolio_forecast.py`) "
        "since forecasting all stores live was measured at ~11 minutes — too slow for a Streamlit tab."
    )

    try:
        portfolio = pd.read_csv("portfolio_forecast.csv")
    except FileNotFoundError:
        st.warning("`portfolio_forecast.csv` not found. Run `generate_portfolio_forecast.py` first.")
        st.stop()

    st.warning(
        "⚠️ **Read this before trusting the % change column**: the forecast assumes no "
        "`SchoolHoliday` on any future day (it isn't knowable in advance), but the prior-period "
        "actuals it's compared against had a 58-84% SchoolHoliday rate for this specific date "
        "range (German summer school holidays). That mismatch alone plausibly explains most of "
        "the uniform decline you'll see below — treat `pct_change` as a rough signal, not a "
        "clean apples-to-apples comparison, until a real forward holiday calendar is added."
    )

    # --- Chain-wide summary ---
    total_forecast = portfolio["forecast_7d_total"].sum()
    tier_counts_p = portfolio["health_tier"].value_counts()
    best_store = portfolio.loc[portfolio["pct_change"].idxmax()]
    worst_store = portfolio.loc[portfolio["pct_change"].idxmin()]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Chain-wide forecast (7d)", f"{total_forecast:,.0f}")
    p2.metric("Needs Attention stores", int(tier_counts_p.get("Needs Attention", 0)))
    p3.metric("Best this week", f"Store {int(best_store['Store'])}", f"{best_store['pct_change']:+.1f}%")
    p4.metric("Worst this week", f"Store {int(worst_store['Store'])}", f"{worst_store['pct_change']:+.1f}%")

    st.divider()

    # --- Filters ---
    st.markdown("#### Filters")
    f1, f2 = st.columns(2)
    with f1:
        store_types = sorted(portfolio["StoreType"].dropna().unique())
        selected_types = st.multiselect("StoreType", store_types, default=store_types)
    with f2:
        min_dist, max_dist = int(portfolio["CompetitionDistance"].min()), int(portfolio["CompetitionDistance"].max())
        dist_range = st.slider("CompetitionDistance range (meters)", min_dist, max_dist, (min_dist, max_dist))

    filtered = portfolio[
        portfolio["StoreType"].isin(selected_types)
        & portfolio["CompetitionDistance"].between(*dist_range)
    ]
    st.caption(f"Showing {len(filtered)} of {len(portfolio)} stores")

    # --- Top/bottom 10 by forecasted growth ---
    st.markdown("#### Top 10 / Bottom 10 by forecasted growth")
    top10 = filtered.nlargest(10, "pct_change")
    bottom10 = filtered.nsmallest(10, "pct_change")

    fig_p, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(14, 5))
    ax_top.barh(top10["Store"].astype(str), top10["pct_change"], color=COLOR_ACTUAL)
    ax_top.set_title("Top 10 (best % change)", color=TEXT_LIGHT)
    ax_top.set_xlabel("% change vs prior 7d")
    ax_top.invert_yaxis()
    ax_top.grid(alpha=0.25, axis="x")

    ax_bot.barh(bottom10["Store"].astype(str), bottom10["pct_change"], color=COLOR_CLOSED)
    ax_bot.set_title("Bottom 10 (worst % change)", color=TEXT_LIGHT)
    ax_bot.set_xlabel("% change vs prior 7d")
    ax_bot.invert_yaxis()
    ax_bot.grid(alpha=0.25, axis="x")

    plt.tight_layout()
    st.pyplot(fig_p)
    plt.close(fig_p)

    # --- Sortable full table ---
    st.markdown("#### All stores")
    table_display = filtered[[
        "Store", "health_tier", "StoreType", "CompetitionDistance",
        "forecast_7d_total", "prev_7d_actual", "pct_change",
    ]].rename(columns={
        "health_tier": "Health Tier", "StoreType": "Store Type",
        "CompetitionDistance": "Competition Dist. (m)",
        "forecast_7d_total": "Forecast (7d)", "prev_7d_actual": "Prior Actual (7d)",
        "pct_change": "% Change",
    }).sort_values("% Change", ascending=False)

    st.dataframe(
        table_display.style.format({
            "Forecast (7d)": "{:,.0f}", "Prior Actual (7d)": "{:,.0f}",
            "% Change": "{:+.1f}%", "Competition Dist. (m)": "{:,.0f}",
        }),
        use_container_width=True,
        height=400,
    )
