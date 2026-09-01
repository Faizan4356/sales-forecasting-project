"""
Step 1 - Anomaly Detection Layer (Deep Learning)
Sales Intelligence Platform extension

Trains an LSTM Autoencoder on 14-(open)-day sequences of each store's own
sales history to learn "normal" patterns, then flags days whose
reconstruction error is unusually high as anomalies -- a data error, a
one-off event, or something worth a manager's attention.

NOTE ON FRAMEWORK: the original spec called for TensorFlow/Keras, but
TensorFlow has no distribution available for this machine's Python version
(3.14). PyTorch (already installed) is used instead with the same LSTM
Autoencoder architecture and training behavior. This only affects this
one offline training script -- it changes nothing about the deployed app,
which never imports either framework, only reads this script's CSV output.

Requires: train_cleaned.csv (Phase 2 output)
Outputs:  anomaly_model.pt, anomaly_threshold.joblib, anomaly_flags.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib

SEQ_LEN = 14
HIDDEN_SIZE = 16
EPOCHS = 8
BATCH_SIZE = 256
LR = 1e-3

torch.manual_seed(42)
np.random.seed(42)

# =================================================================
# Load + prepare sequences
# =================================================================
df = pd.read_csv("train_cleaned.csv", parse_dates=["Date"])
df = df.sort_values(["Store", "Date"])

# WHY only Open==1 days: a closed store is a known, scheduled fact (many
# close every Sunday), not an anomaly. Training on closures would just
# teach the autoencoder that "sales = 0" is a normal pattern to reproduce,
# diluting its sensitivity to genuine irregularities on days the store
# WAS actually trading.
open_df = df[df["Open"] == 1][["Store", "Date", "Sales", "StateHoliday"]].copy()

# Phase 2's cleaning intentionally left ~13,860 rows with Open==1 but
# Sales==NaN (long closures where interpolation was capped at limit=3
# rather than guessing indefinitely -- see sales_forecast_phase2.py). A
# single NaN sequence element poisons an LSTM's gradients into NaN for
# the whole batch, so these must be dropped before building sequences.
n_before = len(open_df)
open_df = open_df.dropna(subset=["Sales"])
print(f"Dropped {n_before - len(open_df)} Open==1 rows with unresolved NaN Sales "
      f"(unfilled long closures from Phase 2 cleaning)")

# Per-store z-score normalization: store sizes vary by an order of
# magnitude (a small store vs. a flagship), so a raw-sales autoencoder
# would mostly just learn "big stores have big numbers" instead of each
# store's own shape. Normalizing per store puts every store on the same
# scale so the model learns SHAPE, not SIZE.
store_stats = open_df.groupby("Store")["Sales"].agg(["mean", "std"]).rename(
    columns={"mean": "store_mean", "std": "store_std"}
)
store_stats["store_std"] = store_stats["store_std"].replace(0, 1)  # guard: a store with constant sales
open_df = open_df.merge(store_stats, on="Store", how="left")
open_df["Sales_z"] = (open_df["Sales"] - open_df["store_mean"]) / open_df["store_std"]

# Sliding window of the last SEQ_LEN *open* days per store (not calendar
# days -- skipping closures means "14 trading days," which is the pattern
# that actually matters to a manager, not 14 raw calendar days padded
# with predictable zeros).
sequences, meta = [], []
for store_id, g in open_df.groupby("Store", sort=False):
    vals = g["Sales_z"].to_numpy()
    raw_sales = g["Sales"].to_numpy()
    dates = g["Date"].to_numpy()
    holidays = g["StateHoliday"].to_numpy()
    means = g["store_mean"].to_numpy()
    stds = g["store_std"].to_numpy()
    n = len(vals)
    if n < SEQ_LEN:
        continue
    for i in range(SEQ_LEN - 1, n):
        sequences.append(vals[i - SEQ_LEN + 1: i + 1])
        meta.append((store_id, dates[i], holidays[i], raw_sales[i], means[i], stds[i]))

X = np.asarray(sequences, dtype=np.float32)
meta_df = pd.DataFrame(meta, columns=["Store", "Date", "StateHoliday", "Sales", "store_mean", "store_std"])
print(f"Built {len(X):,} sequences of length {SEQ_LEN} across {open_df['Store'].nunique()} stores")

X_tensor = torch.tensor(X).unsqueeze(-1)  # (N, seq_len, 1)


# =================================================================
# LSTM Autoencoder
# =================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, hidden_size=16):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.encoder(x)                              # h: (1, batch, hidden)
        latent = h.repeat(self.seq_len, 1, 1).permute(1, 0, 2)    # (batch, seq_len, hidden)
        decoded, _ = self.decoder(latent)
        return self.output_layer(decoded)                        # (batch, seq_len, 1)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMAutoencoder(SEQ_LEN, HIDDEN_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)

print(f"Training on device: {device}")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    print(f"Epoch {epoch + 1}/{EPOCHS} - MSE: {total_loss / len(X_tensor):.5f}")

# =================================================================
# Reconstruction error per window, assigned to the window's LAST day
# =================================================================
model.eval()
errors, recon_last_z = [], []
with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch = X_tensor[i:i + BATCH_SIZE].to(device)
        recon = model(batch)
        err = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
        errors.append(err)
        recon_last_z.append(recon[:, -1, 0].cpu().numpy())

meta_df["reconstruction_error"] = np.concatenate(errors)
# Convert the model's reconstructed (normalized) value for the last day
# back to raw sales scale -- this is "what the model expected" for that day.
meta_df["expected_sales"] = (
    np.concatenate(recon_last_z) * meta_df["store_std"].to_numpy() + meta_df["store_mean"].to_numpy()
)

# =================================================================
# Threshold + anomaly flagging
# =================================================================
# 95th percentile of the training set's own reconstruction error: since
# the model was trained on (mostly) normal days, the top 5% of errors on
# that same set are, by construction, the days it struggled hardest to
# reproduce -- a reasonable cutoff for "unusual" without needing a
# separate labeled anomaly set (none exists for this data).
threshold = float(np.percentile(meta_df["reconstruction_error"], 95))
meta_df["is_anomaly"] = meta_df["reconstruction_error"] > threshold

n_anomalies = int(meta_df["is_anomaly"].sum())
print(f"\nThreshold (95th percentile of reconstruction error): {threshold:.5f}")
print(f"Flagged {n_anomalies:,} anomalous days out of {len(meta_df):,} evaluated ({n_anomalies/len(meta_df)*100:.2f}%)")

# =================================================================
# Sanity check: do StateHoliday days show higher reconstruction error?
# =================================================================
# If the model has learned something meaningful about "normal" daily
# patterns, days it was NOT specifically trained to expect (holidays are
# rare and irregular) should look more unusual to it than an ordinary
# Tuesday -- this doesn't validate genuine anomalies, but it's evidence
# the model isn't just outputting noise.
is_holiday = meta_df["StateHoliday"].astype(str) != "0"
holiday_err = meta_df.loc[is_holiday, "reconstruction_error"].mean()
regular_err = meta_df.loc[~is_holiday, "reconstruction_error"].mean()
print(f"\n--- Sanity check: StateHoliday vs regular days ---")
print(f"Mean reconstruction error on StateHoliday days: {holiday_err:.5f}  (n={is_holiday.sum()})")
print(f"Mean reconstruction error on regular days:       {regular_err:.5f}  (n={(~is_holiday).sum()})")
print(f"=> Holiday error is {'HIGHER' if holiday_err > regular_err else 'LOWER'} than regular days "
      f"({holiday_err / regular_err:.2f}x)" if regular_err else "")

# =================================================================
# Save model, threshold, and per-day flags
# =================================================================
torch.save(model.state_dict(), "anomaly_model.pt")
joblib.dump(threshold, "anomaly_threshold.joblib")

out_cols = ["Store", "Date", "Sales", "expected_sales", "reconstruction_error", "is_anomaly"]
meta_df[out_cols].to_csv("anomaly_flags.csv", index=False)

print("\nSaved: anomaly_model.pt, anomaly_threshold.joblib, anomaly_flags.csv")

# =================================================================
# 5 example anomalous days: actual vs expected
# =================================================================
print("\n--- 5 example anomalous days (actual vs. expected sales) ---")
examples = meta_df[meta_df["is_anomaly"]].sort_values("reconstruction_error", ascending=False).head(5)
for _, row in examples.iterrows():
    print(f"Store {int(row['Store']):>4} | {row['Date'].date()} | "
          f"actual={row['Sales']:>8.0f} | expected~={row['expected_sales']:>8.0f} | "
          f"error={row['reconstruction_error']:.4f}")
