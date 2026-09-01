"""
Value Booster - Portfolio Overview data
Sales Intelligence Platform extension

Precomputes a 7-day forecast total per store, chain-wide, offline. A live
in-app recursive forecast across all 1,115 stores was measured at ~580ms/
store (~11 minutes total) -- far too slow for a Streamlit tab to compute
on demand, so this is generated as a batch script instead, mirroring how
store_health_scores.csv is produced by combine_store_health.py.

Requires: train_cleaned.csv, store.csv, store_health_scores.csv
Outputs:  portfolio_forecast.csv
          (Store, StoreType, CompetitionDistance, health_tier,
           prev_7d_actual, forecast_7d_total, pct_change)
"""

import pandas as pd
import numpy as np
import xgboost as xgb

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday", "Promo", "SchoolHoliday",
]
HORIZON_DAYS = 7


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


def recursive_forecast(model, store_history, horizon_days):
    history = store_history.sort_values("Date").copy()
    last_date = history["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    open_rate_by_dow = history.assign(dow=history["Date"].dt.dayofweek).groupby("dow")["Open"].mean()

    working = history[["Store", "Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
    predictions = []
    for future_date in future_dates:
        dow = future_date.dayofweek
        is_open = int(open_rate_by_dow.get(dow, 1.0) >= 0.5)
        new_row = pd.DataFrame([{
            "Store": working["Store"].iloc[0], "Date": future_date,
            "Sales": 0.0 if not is_open else np.nan, "Open": is_open,
            "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0,
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
        predictions.append({"Date": future_date, "Predicted_Sales": pred})
    return pd.DataFrame(predictions)


# =================================================================
# Train the same global model app.py uses
# =================================================================
print("Loading + cleaning data...")
df = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
df = df.sort_values(["Store", "Date"])

feat = add_calendar_features(df)
feat = add_lag_roll_features(feat)
feat = feat[feat["Open"] == 1].dropna(subset=FEATURE_COLS + ["Sales"])

print(f"Training model on {len(feat):,} rows...")
model = xgb.XGBRegressor(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist",
)
model.fit(feat[FEATURE_COLS].astype("float32"), feat["Sales"].astype("float32"))

# =================================================================
# Forecast HORIZON_DAYS ahead for every store
# =================================================================
store_ids = sorted(df["Store"].unique())
rows = []
print(f"Forecasting {HORIZON_DAYS} days ahead for {len(store_ids)} stores (this takes a few minutes)...")
for i, store_id in enumerate(store_ids):
    store_history = df[df["Store"] == store_id].sort_values("Date")
    fc = recursive_forecast(model, store_history, HORIZON_DAYS)
    forecast_total = fc["Predicted_Sales"].sum()

    prev_window = store_history.tail(HORIZON_DAYS)
    prev_actual_total = prev_window["Sales"].sum()

    pct_change = (
        (forecast_total - prev_actual_total) / prev_actual_total * 100
        if prev_actual_total > 0 else np.nan
    )
    rows.append({
        "Store": store_id,
        "prev_7d_actual": prev_actual_total,
        "forecast_7d_total": forecast_total,
        "pct_change": pct_change,
    })
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(store_ids)} stores done")

portfolio = pd.DataFrame(rows)

# =================================================================
# Merge health tier + store metadata for the Portfolio tab's filters
# =================================================================
store_meta = pd.read_csv("store.csv", usecols=["Store", "StoreType", "CompetitionDistance"])
portfolio = portfolio.merge(store_meta, on="Store", how="left")

try:
    health = pd.read_csv("store_health_scores.csv", usecols=["Store", "tier"])
    portfolio = portfolio.merge(health.rename(columns={"tier": "health_tier"}), on="Store", how="left")
except FileNotFoundError:
    print("store_health_scores.csv not found -- health_tier column will be empty. "
          "Run combine_store_health.py first for a complete Portfolio tab.")
    portfolio["health_tier"] = "Unknown"

portfolio.to_csv("portfolio_forecast.csv", index=False)
print("\nSaved portfolio_forecast.csv")
print(f"\nChain-wide forecasted total (next {HORIZON_DAYS} days): {portfolio['forecast_7d_total'].sum():,.0f}")
print("\n--- Top 5 by forecasted growth ---")
print(portfolio.sort_values("pct_change", ascending=False).head(5)[["Store", "pct_change", "forecast_7d_total"]])
print("\n--- Bottom 5 by forecasted growth ---")
print(portfolio.sort_values("pct_change").head(5)[["Store", "pct_change", "forecast_7d_total"]])
