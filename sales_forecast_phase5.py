"""
Phase 5 - Model Building & Comparison: Prophet vs XGBoost
Sales Forecasting Project (Rossmann Store Sales)

Requires: train_features.csv (Phase 4) and train_cleaned.csv (Phase 2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 120)

STORE_ID = 1          # Prophet fits per-series, so we compare on one store
HOLDOUT_DAYS = 30

def mape(y_true, y_pred):
    # Mean Absolute Percentage Error. Like RMSPE (Rossmann's own metric),
    # this normalizes error by the actual sales value so stores/days of
    # very different sizes are comparable on the same scale — a raw MAE
    # of 300 means very different things for a 500-sales day vs a 8000-
    # sales day, MAPE expresses both as "% off".
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =================================================================
# Load & prepare a single store's continuous, open-only series
# =================================================================
raw = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
feat = pd.read_csv("train_features.csv", parse_dates=["Date"])

store_raw = raw[raw["Store"] == STORE_ID].sort_values("Date")
store_feat = feat[feat["Store"] == STORE_ID].sort_values("Date")

store = store_feat.merge(store_raw[["Date", "Open"]], on="Date", how="left")
store = store[store["Open"] == 1].copy()          # forecast only "open" days
store = store.dropna(subset=[
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7", "sales_roll_mean_30", "sales_roll_std_30",
])

# WHY A CHRONOLOGICAL HOLDOUT (last 30 days), NOT RANDOM:
# A forecasting model must only ever see the past when predicting the
# future. A random split would let the model train on days after the ones
# it's being tested on, leaking future information and producing an
# unrealistically optimistic score. Holding out the last 30 calendar days
# mirrors exactly how the model will be used in production.
cutoff = store["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
train_df = store[store["Date"] <= cutoff].copy()
test_df = store[store["Date"] > cutoff].copy()

print(f"Store {STORE_ID}")
print(f"Train: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} rows)")
print(f"Test:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} rows)")

# =================================================================
# MODEL 1 — Prophet
# =================================================================
# Prophet wants columns named exactly "ds" (date) and "y" (target). It
# automatically models trend + weekly/yearly seasonality and can accept
# extra regressors, but for a fair baseline comparison here we let it
# work purely off the date signal (its core selling point: automatic
# seasonality with minimal setup), and add Promo as a regressor since
# it's a known-in-advance event, not something we're forecasting.
prophet_train = train_df.rename(columns={"Date": "ds", "Sales": "y"})[["ds", "y", "Promo"]]
prophet_test = test_df.rename(columns={"Date": "ds"})[["ds", "Promo"]]

prophet_model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
)
prophet_model.add_regressor("Promo")
prophet_model.fit(prophet_train)

forecast = prophet_model.predict(prophet_test)
pred_prophet = forecast["yhat"].values

# =================================================================
# MODEL 2 — XGBoost (using the lag/rolling/calendar features)
# =================================================================
feature_cols = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_30", "sales_roll_std_30",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_holiday", "Promo", "promo_started", "SchoolHoliday",
]
feature_cols = [c for c in feature_cols if c in train_df.columns]

X_train, y_train = train_df[feature_cols], train_df["Sales"]
X_test, y_test = test_df[feature_cols], test_df["Sales"]

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)

# =================================================================
# STEP 3 — Compare RMSE / MAE / MAPE
# =================================================================
y_true = test_df["Sales"].values

results = pd.DataFrame({
    "Model": ["Prophet", "XGBoost"],
    "RMSE": [
        np.sqrt(mean_squared_error(y_true, pred_prophet)),
        np.sqrt(mean_squared_error(y_true, pred_xgb)),
    ],
    "MAE": [
        mean_absolute_error(y_true, pred_prophet),
        mean_absolute_error(y_true, pred_xgb),
    ],
    "MAPE (%)": [
        mape(y_true, pred_prophet),
        mape(y_true, pred_xgb),
    ],
})

print("\n=== Model Comparison (last 30 days, Store", STORE_ID, ") ===")
print(results.to_string(index=False))

better = results.loc[results["RMSE"].idxmin(), "Model"]
print(f"\nLower RMSE: {better}")

# =================================================================
# STEP 4 — Plot actual vs predicted, both models, same chart
# =================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_df["Date"], y_true, label="Actual", color="black", marker="o", markersize=4, linewidth=1.5)
ax.plot(test_df["Date"], pred_prophet, label="Prophet", color="tab:blue", marker="s", markersize=4, linestyle="--")
ax.plot(test_df["Date"], pred_xgb, label="XGBoost", color="tab:orange", marker="^", markersize=4, linestyle="--")
ax.set_title(f"Store {STORE_ID} — Actual vs Predicted, Last {HOLDOUT_DAYS} Days")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"store_{STORE_ID}_prophet_vs_xgboost.png", dpi=150)
plt.close(fig)

print(f"\nSaved comparison plot: store_{STORE_ID}_prophet_vs_xgboost.png")
