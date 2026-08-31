"""
Phase 1 - Data Loading & Understanding
Sales Forecasting Project (Rossmann Store Sales)

Download train.csv from: https://www.kaggle.com/c/rossmann-store-sales/data
and place it in the same folder as this script (or update DATA_PATH below).
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
DATA_PATH = "train.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst few rows:")
print(df.head())

# ---------------------------------------------------------------
# 2. Parse the date column and set it as a datetime index
# ---------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])

# Sort so time-based operations (plots, rolling windows, etc.) behave correctly
df = df.sort_values(["Store", "Date"])

# ---------------------------------------------------------------
# 3. Show the date range covered
# ---------------------------------------------------------------
print("\nDate range covered:")
print("Start:", df["Date"].min())
print("End:  ", df["Date"].max())
print("Total days:", (df["Date"].max() - df["Date"].min()).days)
print("Number of unique stores:", df["Store"].nunique())

# ---------------------------------------------------------------
# 4. Set Date as index (create a store-specific time series)
# ---------------------------------------------------------------
STORE_ID = 1  # change this to inspect a different store

store_df = df[df["Store"] == STORE_ID].copy()
store_df = store_df.set_index("Date")

print(f"\nStore {STORE_ID} data range:")
print(store_df.index.min(), "to", store_df.index.max())
print("Rows:", len(store_df))

# ---------------------------------------------------------------
# 5. Plot raw sales over time for that store
# ---------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(store_df.index, store_df["Sales"], linewidth=0.8)
plt.title(f"Daily Sales Over Time — Store {STORE_ID}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"store_{STORE_ID}_sales_over_time.png", dpi=150)
plt.show()

print(f"\nPlot saved as store_{STORE_ID}_sales_over_time.png")
