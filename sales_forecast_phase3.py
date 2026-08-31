"""
Phase 3 - Exploratory Time-Series Analysis
Sales Forecasting Project (Rossmann Store Sales)

Requires: train_cleaned.csv (output of Phase 2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

STORE_ID = 1

# ---------------------------------------------------------------
# 0. Load cleaned data and isolate one store as a continuous series
# ---------------------------------------------------------------
df = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
store_df = df[df["Store"] == STORE_ID].sort_values("Date").set_index("Date")

# Decomposition and ACF/PACF require a series with NO missing values and a
# fixed frequency. We:
#   (a) explicitly set daily frequency,
#   (b) drop days the store was closed (Open == 0) — sales are structurally
#       zero on those days, which would distort seasonality/ACF as fake
#       "shocks" rather than real demand signal,
#   (c) forward-fill any remaining short NaN gaps from Phase 2 so the
#       algorithms have a complete series to work with.
sales = store_df["Sales"].asfreq("D")
open_flag = store_df["Open"].asfreq("D")

sales_open_only = sales.where(open_flag != 0)          # closed-day sales -> NaN
sales_filled = sales_open_only.interpolate(limit_direction="both")

print(f"Store {STORE_ID}: {sales_filled.isna().sum()} NaNs remaining after fill")

# =================================================================
# STEP 1 — Seasonal decomposition (trend / seasonality / residual)
# =================================================================
# WHY period=7: Rossmann sales are recorded daily and retail demand has a
# strong weekly cycle (weekday vs weekend, Sunday closures). An additive
# model is used because the seasonal swing looks roughly constant in
# absolute size across the trend level (not proportional to it) — if your
# own plot shows the seasonal amplitude growing with the trend, switch to
# model="multiplicative".
decomposition = seasonal_decompose(sales_filled, model="additive", period=7)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.suptitle(f"Store {STORE_ID} — Seasonal Decomposition (weekly period)", y=1.02)
plt.tight_layout()
plt.savefig(f"store_{STORE_ID}_decomposition.png", dpi=150)
plt.close(fig)

# =================================================================
# STEP 2 — Weekly and monthly seasonality patterns
# =================================================================
seasonal_df = sales_filled.reset_index()
seasonal_df.columns = ["Date", "Sales"]
seasonal_df["DayOfWeek"] = seasonal_df["Date"].dt.day_name()
seasonal_df["Month"] = seasonal_df["Date"].dt.month_name()

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

weekly_avg = seasonal_df.groupby("DayOfWeek")["Sales"].mean().reindex(weekday_order)
monthly_avg = seasonal_df.groupby("Month")["Sales"].mean().reindex(month_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
weekly_avg.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title(f"Store {STORE_ID} — Avg Sales by Day of Week")
axes[0].set_ylabel("Average Sales")

monthly_avg.plot(kind="bar", ax=axes[1], color="darkorange")
axes[1].set_title(f"Store {STORE_ID} — Avg Sales by Month")
axes[1].set_ylabel("Average Sales")

plt.tight_layout()
plt.savefig(f"store_{STORE_ID}_seasonality.png", dpi=150)
plt.close(fig)

# =================================================================
# STEP 3 — Stationarity check: Augmented Dickey-Fuller test
# =================================================================
# H0 (null): the series has a unit root -> it is NON-stationary.
# If p-value < 0.05, we reject H0 -> series IS stationary.
adf_result = adfuller(sales_filled.dropna())

print("\n--- Augmented Dickey-Fuller Test (raw sales) ---")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value:       {adf_result[1]:.6f}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

is_stationary = adf_result[1] < 0.05
print(f"=> Series is {'STATIONARY' if is_stationary else 'NON-STATIONARY'} at 5% significance")

# If non-stationary, also test the first-differenced series, since ARIMA-
# family models need to know the differencing order (d) that achieves
# stationarity.
sales_diff = sales_filled.diff().dropna()
adf_diff = adfuller(sales_diff)
print("\n--- ADF Test (first difference) ---")
print(f"ADF Statistic: {adf_diff[0]:.4f}")
print(f"p-value:       {adf_diff[1]:.6f}")
print(f"=> First-differenced series is "
      f"{'STATIONARY' if adf_diff[1] < 0.05 else 'NON-STATIONARY'} at 5% significance")

# =================================================================
# STEP 4 — ACF and PACF plots
# =================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(sales_filled.dropna(), lags=40, ax=axes[0])
axes[0].set_title(f"Store {STORE_ID} — Autocorrelation (ACF), raw sales")

plot_pacf(sales_filled.dropna(), lags=40, ax=axes[1], method="ywm")
axes[1].set_title(f"Store {STORE_ID} — Partial Autocorrelation (PACF), raw sales")

plt.tight_layout()
plt.savefig(f"store_{STORE_ID}_acf_pacf.png", dpi=150)
plt.close(fig)

print("\nSaved plots:")
print(f" - store_{STORE_ID}_decomposition.png")
print(f" - store_{STORE_ID}_seasonality.png")
print(f" - store_{STORE_ID}_acf_pacf.png")
