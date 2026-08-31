"""
Phase 4 - Feature Engineering (Time-Series Specific)
Sales Forecasting Project (Rossmann Store Sales)

Requires: train_cleaned.csv (output of Phase 2)
"""

import pandas as pd
import numpy as np

df = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# IMPORTANT: every feature below is computed PER STORE via groupby("Store").
# Sales series are independent across stores — a lag/rolling window that
# crossed store boundaries would silently blend one store's sales history
# into another's features (e.g. "yesterday" for Store 2's first row would
# wrongly become Store 1's last day). groupby+transform/shift keeps each
# store's time series isolated.

# =================================================================
# STEP 1 — Lag features
# =================================================================
# WHY LAGS MATTER:
# Sales forecasting models (whether classical ML like XGBoost/LightGBM, or
# even as engineered inputs to SARIMA/Prophet) have no innate concept of
# "time" or memory — each row is just a feature vector. Lag features are
# how you hand the model its own recent history directly as input columns,
# so it can learn autocorrelation patterns (e.g. "sales tend to be similar
# to 7 days ago because of the weekly cycle we saw in the ACF plot").
# Without lags, a tree-based model literally cannot know Tuesday followed
# a big Monday.
#
# WHY THESE SPECIFIC LAGS (1, 7, 14, 30):
#   - lag_1  : yesterday's sales -> captures short-term momentum / day-to-day
#              carryover (e.g. a promo that ran yesterday still drawing
#              customers today).
#   - lag_7  : same day last week -> directly encodes the weekly seasonality
#              our ACF plot showed peaking at lag 7, 14, 21, 28.
#   - lag_14 : two weeks ago -> reinforces the weekly pattern with a second
#              observation, helps the model average out one-off noise from
#              a single lag_7 value.
#   - lag_30 : roughly a month ago -> approximates monthly/paycheck-cycle
#              effects and gives a longer-horizon reference point.
for lag in [1, 7, 14, 30]:
    df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

# =================================================================
# STEP 2 — Rolling averages (mean & std)
# =================================================================
# WHY ROLLING FEATURES MATTER:
# A single lag value is noisy (one closed store, one freak promo day, one
# stockout can throw it off). Rolling statistics summarize a WINDOW of
# recent history into a smoothed signal, giving the model:
#   - rolling MEAN: the current local "baseline" level of demand, letting
#     the model see whether today is above/below the store's recent normal
#     (useful since we saw in Phase 3 the trend itself drifts over time —
#     a static average across all history would miss that).
#   - rolling STD: the current local volatility. A store with high recent
#     variance (e.g. mid-promo, mid-holiday-season) should get different,
#     less confident predictions than a stable one. This directly feeds
#     uncertainty-aware models and helps flag anomalous regimes.
#
# CRITICAL: shift(1) before rolling. Without the shift, the rolling window
# for row t would include Sales on day t itself — i.e. the label leaking
# into its own feature. Shifting by 1 first ensures every rolling feature
# only uses information available BEFORE the day being predicted.
for window in [7, 30]:
    shifted = df.groupby("Store")["Sales"].shift(1)
    df[f"sales_roll_mean_{window}"] = (
        shifted.groupby(df["Store"]).transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    df[f"sales_roll_std_{window}"] = (
        shifted.groupby(df["Store"]).transform(lambda s: s.rolling(window, min_periods=1).std())
    )

# =================================================================
# STEP 3 — Date-based features
# =================================================================
# WHY: these give the model direct, explicit access to calendar structure
# (weekly/monthly/yearly seasonality) instead of forcing it to infer that
# structure purely from lag values. Tree models in particular benefit from
# categorical calendar features since they can split on them directly.
df["day_of_week"] = df["Date"].dt.dayofweek          # Mon=0 .. Sun=6
df["day_of_month"] = df["Date"].dt.day
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["month"] = df["Date"].dt.month
df["quarter"] = df["Date"].dt.quarter
df["year"] = df["Date"].dt.year
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# is_holiday: Rossmann's own StateHoliday column already flags public/
# Easter/Christmas holidays ("a","b","c" vs "0" = none). We fold that into
# a clean binary flag rather than re-deriving holidays from a calendar
# library, since the dataset's own ground truth is more reliable for this
# specific market (Germany) than a generic holiday package would be.
df["is_holiday"] = (df["StateHoliday"].astype(str) != "0").astype(int)

# =================================================================
# STEP 4 — Promotional / event flags
# =================================================================
# Promo: already present (0/1) — whether a promo ran that day. Kept as-is,
# but we also add "days since last promo start" style context: whether a
# promo just started, since the first day or two of a promo often behaves
# differently (spike) than sustained mid-promo days.
df["promo_started"] = (
    (df["Promo"] == 1) & (df.groupby("Store")["Promo"].shift(1) == 0)
).astype(int)

# Promo2 (the recurring/ongoing promotion program) is store-level metadata
# from store.csv — SchoolHoliday is already a clean 0/1 flag in the raw
# data. Both are carried through unchanged; no re-engineering needed since
# they arrive as already-clean binary flags.

# =================================================================
# Save
# =================================================================
feature_cols = [
    "Store", "Date", "Sales",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday",
    "Promo", "promo_started", "Promo2", "SchoolHoliday",
]
feature_cols = [c for c in feature_cols if c in df.columns]

out = df[feature_cols]
print("Feature matrix shape:", out.shape)
print("\nSample (Store 1, after enough history for all lags to populate):")
print(out[out["Store"] == 1].iloc[30:35])

print("\nNaN counts (expected at the start of each store's history, where lags/rolls have no prior data):")
print(out.isna().sum())

out.to_csv("train_features.csv", index=False)
print("\nSaved feature matrix to train_features.csv")
