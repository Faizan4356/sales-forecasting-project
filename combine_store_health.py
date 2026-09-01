"""
Step 3 - Combined Store Health Score
Sales Intelligence Platform extension

Merges three independent signals into one 0-100 Store Health Score per
store:
  (a) forecast reliability  -- MAPE of the XGBoost model on that store's
      last 30 days (held out, not trained on) -- how predictable the
      store's demand pattern is
  (b) anomaly rate           -- % of the store's last 90 evaluated days
      flagged by the LSTM Autoencoder (Step 1)
  (c) review sentiment       -- avg_sentiment from the synthetic review
      layer (Step 2, SYNTHETIC data -- see that script's header)

Requires: train_cleaned.csv, anomaly_flags.csv, store_review_scores.csv
Outputs:  store_health_scores.csv
          (Store, forecast_mape, anomaly_rate, avg_sentiment,
           dominant_theme, health_score, tier)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

HOLDOUT_DAYS = 30
ANOMALY_LOOKBACK_DAYS = 90

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday", "Promo", "SchoolHoliday",
]


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


def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# =================================================================
# (a) Forecast reliability: per-store MAPE on a chronological holdout
# =================================================================
# WHY a holdout, not in-sample error: scoring the model on data it was
# trained on would make every store look artificially predictable. A
# genuine "how reliable is this store's forecast" signal requires
# evaluating on days the model never saw during training -- the same
# methodology used in Phase 5's Prophet vs XGBoost comparison.
df = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
df = df.sort_values(["Store", "Date"])

feat = add_calendar_features(df)
feat = add_lag_roll_features(feat)
feat = feat[feat["Open"] == 1].dropna(subset=FEATURE_COLS + ["Sales"])

cutoff = feat["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
train_feat = feat[feat["Date"] <= cutoff]
test_feat = feat[feat["Date"] > cutoff]

print(f"Training forecast model on {len(train_feat):,} rows, "
      f"evaluating on {len(test_feat):,} rows ({test_feat['Date'].min().date()} to {test_feat['Date'].max().date()})")

model = xgb.XGBRegressor(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist",
)
model.fit(train_feat[FEATURE_COLS].astype("float32"), train_feat["Sales"].astype("float32"))

test_feat = test_feat.copy()
test_feat["predicted"] = model.predict(test_feat[FEATURE_COLS].astype("float32"))

forecast_mape = (
    test_feat.groupby("Store")
    .apply(lambda g: mape(g["Sales"].values, g["predicted"].values), include_groups=False)
    .rename("forecast_mape")
)

# =================================================================
# (b) Anomaly rate: % of each store's last N evaluated days flagged
# =================================================================
anomaly_flags = pd.read_csv("anomaly_flags.csv", parse_dates=["Date"])
anomaly_cutoff = anomaly_flags["Date"].max() - pd.Timedelta(days=ANOMALY_LOOKBACK_DAYS)
recent_anomalies = anomaly_flags[anomaly_flags["Date"] > anomaly_cutoff]
anomaly_rate = recent_anomalies.groupby("Store")["is_anomaly"].mean().rename("anomaly_rate")
anomaly_count = recent_anomalies.groupby("Store")["is_anomaly"].sum().rename("anomaly_count_90d")

# =================================================================
# (c) Review sentiment (SYNTHETIC -- see generate_and_analyze_reviews.py)
# =================================================================
reviews = pd.read_csv("store_review_scores.csv").set_index("Store")

# =================================================================
# Merge all three signals
# =================================================================
health = (
    forecast_mape.to_frame()
    .join(anomaly_rate, how="left")
    .join(anomaly_count, how="left")
    .join(reviews[["avg_sentiment", "dominant_theme"]], how="left")
    .reset_index()
)
health["anomaly_rate"] = health["anomaly_rate"].fillna(0)
health["anomaly_count_90d"] = health["anomaly_count_90d"].fillna(0).astype(int)
health["avg_sentiment"] = health["avg_sentiment"].fillna(health["avg_sentiment"].mean())
health["dominant_theme"] = health["dominant_theme"].fillna("Other")


# =================================================================
# Combined 0-100 Store Health Score
# =================================================================
def combine_store_score(forecast_mape, anomaly_rate, sentiment_score):
    """
    Weights forecast reliability highest: an unpredictable store is an
    operational risk (bad for staffing/inventory planning) regardless of
    WHY it's unpredictable, whereas an isolated anomaly or a lukewarm
    review is a narrower, more explainable problem.

    Weights: 50% forecast reliability, 30% anomaly rate, 20% sentiment.
    """
    if pd.isna(forecast_mape):
        forecast_mape = 50.0  # no evaluable holdout data -> treat as average/unknown, not perfect

    # MAPE -> reliability: 0% MAPE = 100, 50%+ MAPE = 0 (linear).
    reliability = np.clip(100 - forecast_mape * 2, 0, 100)
    # anomaly_rate is a 0-1 fraction; 0% flagged = 100, 25%+ flagged = 0.
    anomaly_component = np.clip(100 - anomaly_rate * 400, 0, 100)
    sentiment_component = np.clip(sentiment_score * 100, 0, 100)

    return round(0.5 * reliability + 0.3 * anomaly_component + 0.2 * sentiment_component, 1)


health["health_score"] = health.apply(
    lambda r: combine_store_score(r["forecast_mape"], r["anomaly_rate"], r["avg_sentiment"]), axis=1
)


def assign_tier(score):
    if score <= 40:
        return "Needs Attention"
    elif score <= 70:
        return "Monitor"
    return "Performing Well"


health["tier"] = health["health_score"].apply(assign_tier)


# =================================================================
# Plain-language explanation per store
# =================================================================
def generate_store_explanation(row):
    parts = []

    if row["forecast_mape"] <= 10:
        parts.append("Forecasts are reliable")
    elif row["forecast_mape"] <= 20:
        parts.append("Forecasts are moderately reliable")
    else:
        parts.append("Forecasts are unreliable, suggesting unpredictable demand patterns")

    if row["anomaly_count_90d"] >= 3:
        parts.append(f"{int(row['anomaly_count_90d'])} anomalous sales days in the last quarter")
    elif row["anomaly_count_90d"] >= 1:
        parts.append(f"{int(row['anomaly_count_90d'])} anomalous sales day(s) recently")

    if row["avg_sentiment"] < 0.4:
        parts.append(f"negative feedback about {row['dominant_theme'].lower()} "
                      "suggests an issue worth investigating")
    elif row["avg_sentiment"] > 0.7:
        parts.append(f"reviews are positive, largely about {row['dominant_theme'].lower()}")

    if len(parts) == 1:
        return parts[0] + "."
    return parts[0] + ", but " + "; ".join(parts[1:]) + "." if len(parts) > 1 else parts[0] + "."


health["explanation"] = health.apply(generate_store_explanation, axis=1)

out_cols = ["Store", "forecast_mape", "anomaly_rate", "anomaly_count_90d",
            "avg_sentiment", "dominant_theme", "health_score", "tier", "explanation"]
health[out_cols].to_csv("store_health_scores.csv", index=False)

print("\nSaved store_health_scores.csv")
print("\n--- Tier distribution ---")
print(health["tier"].value_counts())

print("\n--- 5 example stores ---")
for _, row in health.sample(5, random_state=1).iterrows():
    print(f"Store {int(row['Store']):>4} | score={row['health_score']:>5.1f} | {row['tier']:<16} | {row['explanation']}")
