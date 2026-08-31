# Sales / Demand Forecasting — Rossmann Store Sales

A time-series forecasting pipeline that predicts daily store-level sales using
historical patterns, engineered lag/rolling features, and a comparison of
Prophet vs. XGBoost. Deployed as an interactive Streamlit app.

## Business Problem

Retail chains need reliable short-term sales forecasts (1–4 weeks out) to plan
staffing, inventory, and promotions at the individual store level. Sales are
driven by a mix of predictable structure (weekly cycles, holidays, monthly
trends) and store-specific factors (competition, ongoing promotions, one-off
closures) — a naive "average of last month" forecast misses both. This
project builds a model that captures weekly/seasonal patterns per store and
evaluates whether a classical statistical approach (Prophet) or a
feature-driven ML approach (XGBoost) forecasts better in practice.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
(Kaggle) — daily sales for 1,115 Rossmann drug stores across Germany.

- **train.csv**: `Store`, `Date`, `Sales`, `Customers`, `Open`, `Promo`,
  `StateHoliday`, `SchoolHoliday` — 1,017,209 rows, 2013-01-01 to 2015-07-31
- **store.csv**: per-store metadata — `StoreType`, `Assortment`,
  `CompetitionDistance`, `Promo2` (recurring promotion) and its start
  date/interval

## Models Compared

Evaluated on Store 1's last 30 days as a chronological holdout (no random
splitting — a forecaster must only ever see the past when predicting the
future).

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Prophet | 402.8 | 313.2 | 7.08% |
| **XGBoost** | **314.4** | **243.7** | **5.54%** |

**XGBoost outperformed Prophet by ~22%** on every metric. Both models track
the broad weekly seasonality well, but XGBoost — with direct access to lag
(`sales_lag_1/7/14/30`) and rolling mean/std features — reacts faster to
short-term swings (sharp spikes/dips) that Prophet's smoother trend +
seasonality curve tends to lag behind. XGBoost was chosen as the production
model for the deployed app; it also scales to a single global model across
all 1,115 stores (with `Store` as a feature), instead of one model per store.

## Key Seasonal Insights

- **Strong weekly cycle**: seasonal decomposition (period=7) shows a stable,
  non-decaying weekly pattern across all 2.5 years — Monday is typically the
  highest-sales day, with a mid-week dip (Wed/Thu lowest) and a rise back up
  through the weekend.
- **December holiday surge**: average monthly sales peak sharply in December
  (~30% above the September low), visible as two isolated spikes in the
  trend component each year — a signal the plain weekly decomposition can't
  absorb on its own (it shows up in the residuals), so holiday/promo flags
  matter as explicit model inputs.
- **Series is stationary**: an Augmented Dickey-Fuller test on Store 1's raw
  sales rejected the unit-root null (p ≈ 0.0002) — no differencing is needed;
  the "non-stationarity" that matters here is deterministic seasonality, not
  a stochastic trend.
- **ACF/PACF confirm the weekly signal**: autocorrelation shows local peaks
  at lags 7, 14, 21, 28 on top of short-term decay, while PACF cuts off
  sharply after lag 1–2 — consistent with the lag-7/14 features and a
  SARIMA(p,0,q)(P,0,Q)₇-style seasonal structure.
- **Zero-sales days are structural, not noise**: ~17% of rows have zero
  sales, almost entirely explained by `Open == 0` (many stores close every
  Sunday). These days were kept (not dropped) during cleaning since they are
  a real, recurring pattern the model needs to learn — dropping them would
  reintroduce time gaps and bias the model toward never predicting closures.

## Project Structure

```
train.csv, store.csv          Raw Kaggle data
sales_forecast_phase1.py      Load data, parse dates, plot raw sales
sales_forecast_phase2.py      Clean: reindex dates, handle closures, merge store metadata
sales_forecast_phase3.py      EDA: decomposition, seasonality, ADF test, ACF/PACF
sales_forecast_phase4.py      Feature engineering: lags, rolling stats, calendar/promo flags
sales_forecast_phase5.py      Model comparison: Prophet vs XGBoost
app.py                        Streamlit app: interactive forecast by store + horizon
train_cleaned.csv             Output of Phase 2
train_features.csv            Output of Phase 4
```

## How to Run

### 1. Get the data
Download `train.csv` and `store.csv` from the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)
competition (requires accepting the competition rules on Kaggle) and place
them in the project root. Alternatively, a public dataset mirror avoids the
rules-acceptance step:

```bash
kaggle datasets download -d pratyushakar/rossmann-store-sales -p . --unzip
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn xgboost prophet streamlit
```

### 3. Run the pipeline (in order)

```bash
python sales_forecast_phase1.py   # load + plot raw sales
python sales_forecast_phase2.py   # clean -> train_cleaned.csv
python sales_forecast_phase3.py   # decomposition, ADF, ACF/PACF plots
python sales_forecast_phase4.py   # engineer features -> train_features.csv
python sales_forecast_phase5.py   # compare Prophet vs XGBoost
```

### 4. Launch the forecasting app

```bash
streamlit run app.py
```

Select a store and a forecast horizon (7 / 14 / 30 days) to see historical
sales alongside the XGBoost forecast, with closed-day predictions flagged
separately.

## Limitations / Next Steps

- Model comparison was run on a single store's 30-day holdout; results
  should be validated across more stores or with a rolling-origin backtest
  before treating the XGBoost advantage as fully general.
- The app's multi-step forecast is recursive (each day's prediction feeds
  the next day's lag features), so errors can compound over longer horizons
  — most reliable at 7 days, least at 30.
- Future `Promo`/`StateHoliday` values are unknown at forecast time and are
  assumed neutral (no promo, no holiday); feeding in a known promo calendar
  would likely improve accuracy further.
