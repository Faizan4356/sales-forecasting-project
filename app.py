"""
Phase 6 - Deployment
Streamlit app: pick a store + forecast horizon (7/14/30 days),
see historical sales + XGBoost forecast on one chart.

Run with:  streamlit run app.py

Requires train.csv (raw Kaggle file) in the same folder. The app rebuilds
cleaning + features + model itself so it's self-contained and doesn't
depend on having run Phases 1-5 as separate scripts first. (store.csv is
kept in the repo for the phase2/phase4 scripts but isn't read by this app
-- none of its columns are used in FEATURE_COLS.)
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


@st.cache_resource
def train_model(df):
    feat = add_calendar_features(df)
    feat = add_lag_roll_features(feat)
    feat = feat[feat["Open"] == 1].dropna(subset=FEATURE_COLS + ["Sales"])

    # Downcast the feature matrix to float32 -- XGBoost trains just as well
    # on it and it's half the memory of pandas' default float64, which
    # matters directly against the platform's ~1GB cap.
    X = feat[FEATURE_COLS].astype("float32")
    y = feat["Sales"].astype("float32")
    del feat  # drop the larger intermediate frame before fit() allocates its own copies

    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist",  # histogram-binned splits: much lower memory than the exact method
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
""")

st.divider()

tab_forecast, tab_upload, tab_health = st.tabs(
    ["🏬 Rossmann Store Forecast", "📁 Upload Your Own Data", "📊 Store Health"]
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
    health_forecast_df = recursive_forecast(model, health_store_history, 14)
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
