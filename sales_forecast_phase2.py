"""
Phase 2 - Data Cleaning
Sales Forecasting Project (Rossmann Store Sales)

Requires: train.csv and store.csv from
https://www.kaggle.com/c/rossmann-store-sales/data
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. Load raw data (same as Phase 1)
# ---------------------------------------------------------------
df = pd.read_csv("train.csv", parse_dates=["Date"])
store = pd.read_csv("store.csv")

df = df.sort_values(["Store", "Date"])

print("Raw shape:", df.shape)
print("Open value counts:\n", df["Open"].value_counts(dropna=False))
print("Sales == 0 count:", (df["Sales"] == 0).sum())

# =================================================================
# STEP 1 — Handle missing dates by reindexing to a continuous range
# =================================================================
#
# WHY: Rossmann's train.csv only has a row for each (Store, Date) pair
# that actually occurred in the raw export. In practice it usually IS
# already continuous per store, but you can't assume that for any
# real-world retail dataset — stores can have gaps from system outages,
# late reporting, or store closures that were dropped entirely upstream
# instead of logged as Open=0. If you skip this step and a model later
# assumes evenly-spaced daily observations (e.g. lag features, rolling
# means, seasonal decomposition), silent gaps will corrupt those
# calculations without raising an error.
#
# APPROACH: build the full calendar per store and reindex onto it,
# rather than reindexing the whole dataframe globally — different
# stores can open/close on different dates, so a single global date
# range would invent rows before a store existed.

def reindex_store(store_df: pd.DataFrame) -> pd.DataFrame:
    store_df = store_df.set_index("Date")
    full_range = pd.date_range(store_df.index.min(), store_df.index.max(), freq="D")
    store_df = store_df.reindex(full_range)
    store_df.index.name = "Date"
    return store_df

df_reindexed = (
    df.groupby("Store", group_keys=True)
    .apply(reindex_store, include_groups=False)
    .reset_index()
    .rename(columns={"level_1": "Date"})
)

# Store ID gets lost as a column during reindex (only survives in the groupby key)
if "Store" not in df_reindexed.columns:
    df_reindexed = df_reindexed.rename(columns={df_reindexed.columns[0]: "Store"})

n_new_rows = len(df_reindexed) - len(df)
print(f"\nRows added by reindexing (missing calendar days found): {n_new_rows}")

# =================================================================
# STEP 2 — Handle zero-sales / closed-store days
# =================================================================
#
# WHY NOT just drop rows where Sales == 0:
# Zero sales is not noise — it is almost always because Open == 0
# (the store was closed: Sundays, holidays, refurbishment). That's
# a real, structural, recurring signal. If you drop those rows:
#   - You break time continuity again (reintroducing the exact gap
#     problem Step 1 fixed), which corrupts lag/rolling features.
#   - You bias the model to never learn "this day is typically closed,"
#     so it may hallucinate positive sales forecasts for days the
#     store won't even be open.
#
# WHAT TO DO INSTEAD — separate the two possible causes:
#   (a) Open == 0 (store genuinely closed): keep the row, keep Sales
#       at 0 — this is the correct, informative value, not missing
#       data. Do NOT impute/interpolate over it.
#   (b) Open is NaN AND Sales is NaN (introduced by our reindex in
#       Step 1, i.e. a day that didn't exist in the raw export at
#       all): this is genuinely unknown/missing. Infer Open from
#       context (see below), and only impute Sales if the store was
#       inferred to be open.

# Rows created by reindexing will have NaN in all original columns.
is_new_gap_row = df_reindexed["Sales"].isna() & df_reindexed["Open"].isna()
print(f"Rows that are true gaps (unknown open/sales): {is_new_gap_row.sum()}")

# Infer Open for gap rows: assume closed on Sundays (DayOfWeek==7 in
# Rossmann's encoding) since most stores are closed then; otherwise
# assume open, matching the store's typical behavior. This is a
# reasonable heuristic, not certainty — flag it so downstream steps
# (and readers of the analysis) know these values were imputed.
df_reindexed["DayOfWeek"] = df_reindexed["Date"].dt.dayofweek + 1  # Mon=1..Sun=7
df_reindexed["Open_imputed"] = is_new_gap_row  # audit flag

df_reindexed.loc[is_new_gap_row & (df_reindexed["DayOfWeek"] == 7), "Open"] = 0
df_reindexed.loc[is_new_gap_row & (df_reindexed["DayOfWeek"] != 7), "Open"] = 1

# Where Open == 0 (known or inferred), Sales and Customers are correctly 0,
# not missing — fill them in rather than leaving NaN.
closed_mask = df_reindexed["Open"] == 0
df_reindexed.loc[closed_mask, "Sales"] = df_reindexed.loc[closed_mask, "Sales"].fillna(0)
df_reindexed.loc[closed_mask, "Customers"] = df_reindexed.loc[closed_mask, "Customers"].fillna(0)

# Any remaining NaN Sales are days inferred as "Open" but with a genuinely
# unknown sales figure. Interpolate using the store's own local trend rather
# than a global fill, and only across small gaps (limit=3) so we don't
# fabricate long stretches of fake sales.
df_reindexed["Sales"] = (
    df_reindexed.groupby("Store")["Sales"]
    .transform(lambda s: s.interpolate(method="linear", limit=3))
)

# Promo/StateHoliday/SchoolHoliday flags: for true gap rows, absence of
# information means "assume the default, non-promotional day" (0 / "0"),
# since promos are the exception, not the rule.
for col, default in [("Promo", 0), ("StateHoliday", "0"), ("SchoolHoliday", 0)]:
    if col in df_reindexed.columns:
        df_reindexed[col] = df_reindexed[col].fillna(default)

remaining_na_sales = df_reindexed["Sales"].isna().sum()
print(f"Remaining NaN Sales after cleaning: {remaining_na_sales}")
if remaining_na_sales:
    print("(Likely long closures — leave as NaN rather than guessing; "
          "handle explicitly at modeling time, e.g. exclude from loss.)")

# =================================================================
# STEP 3 — Merge in store metadata
# =================================================================
#
# WHY: store.csv carries static, per-store attributes (StoreType,
# Assortment, CompetitionDistance, Promo2, etc.) that explain
# *cross-store* variation a pure time-series model can't see from
# a single store's history alone. A left join on "Store" keeps every
# sales row and attaches metadata; using how="left" (not inner)
# ensures we never silently lose sales rows just because a store's
# metadata is incomplete.

df_clean = df_reindexed.merge(store, on="Store", how="left")

# CompetitionDistance: NaN plausibly means "no nearby competitor" per the
# dataset documentation, not "unknown." Fill with a large distance instead
# of the column mean so we don't imply a competitor exists nearby.
if "CompetitionDistance" in df_clean.columns:
    max_dist = df_clean["CompetitionDistance"].max()
    df_clean["CompetitionDistance"] = df_clean["CompetitionDistance"].fillna(max_dist * 2)

# CompetitionOpenSince* / Promo2Since* / PromoInterval NaNs mean the
# competition/promo2 program doesn't apply to that store — fill with 0 /
# "None" rather than dropping rows, so we don't lose otherwise-good data.
for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2SinceWeek", "Promo2SinceYear"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(0)
if "PromoInterval" in df_clean.columns:
    df_clean["PromoInterval"] = df_clean["PromoInterval"].fillna("None")

print("\nFinal cleaned shape:", df_clean.shape)
print(df_clean.head())

df_clean.to_csv("train_cleaned.csv", index=False)
print("\nSaved cleaned dataset to train_cleaned.csv")
