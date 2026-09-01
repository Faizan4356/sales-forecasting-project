# Sales / Demand Forecasting — Rossmann Store Sales

**🔗 Live app: [sales-forecasting-project-espba4qtsytr6h4sjctpzr.streamlit.app](https://sales-forecasting-project-espba4qtsytr6h4sjctpzr.streamlit.app/)**

A time-series forecasting pipeline that predicts daily store-level sales using
historical patterns, engineered lag/rolling features, and a comparison of
Prophet vs. XGBoost. Deployed as an interactive Streamlit app.

> Retail chains lose money when forecasts miss — overstaffing quiet days,
> understocking busy ones. I built a sales intelligence platform on 3 years
> of Rossmann drugstore data, testing XGBoost against Prophet. XGBoost won
> by ~22% (RMSE 314 vs. 403) because it learns directly from a store's own
> recent sales trend, not just a generic seasonal curve. I extended it with
> LSTM anomaly detection, review sentiment analysis, and a combined Store
> Health Score, deployed as a live interactive app.

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

## Sales Intelligence Platform

The forecasting model above is the foundation of a broader **Sales
Intelligence Platform**, layering deep learning and NLP on top of it into
one combined per-store score:

```
Data Science          EDA, cleaning, seasonal decomposition (Phases 1-3)
      ↓
Machine Learning       XGBoost sales forecast (Phases 4-5)
      ↓
Deep Learning          LSTM Autoencoder — flags anomalous sales days
      ↓
NLP                    Review sentiment analysis (SYNTHETIC data)
      ↓
Combined Score          Store Health Score (0-100) + What-If Simulator
```

- **Anomaly detection** (`train_anomaly_model.py`): an LSTM Autoencoder
  learns each store's normal 14-(open)-day sales pattern from
  `train_cleaned.csv` and flags days with unusually high reconstruction
  error — a data error, a one-off event, or something worth a manager's
  attention. Only trades on `Open == 1` days (closures are a known,
  scheduled fact, not an anomaly). Validated with a sanity check:
  `StateHoliday` days show **2.73x higher** reconstruction error than
  regular days — evidence the model learned real structure, not noise.
  (Built in PyTorch, not TensorFlow/Keras — TensorFlow has no distribution
  available for this project's Python version.)
- **Review sentiment** (`generate_and_analyze_reviews.py`): **⚠️ the
  Rossmann dataset has no real customer review data.** This script
  generates synthetic review text with tone correlated to real store
  metadata (`CompetitionDistance`, `StoreType`), scores it with a
  HuggingFace sentiment pipeline, and tags a dominant theme (Staff, Stock
  Availability, Pricing, Cleanliness). Treat `avg_sentiment` as a
  demonstration of the pipeline, not ground truth about any real store.
- **Store Health Score** (`combine_store_health.py`): merges forecast
  reliability (XGBoost MAPE on a chronological holdout), anomaly rate, and
  review sentiment into one 0-100 score with a tier (`Needs Attention` /
  `Monitor` / `Performing Well`) and a plain-language explanation.
  Forecast reliability is weighted highest (50%, vs. 30% anomaly rate and
  20% sentiment) since an unpredictable store is an operational risk
  regardless of the underlying cause. Verified the score separates
  meaningfully: worst stores (score 11-40) combine 28-120% MAPE, 21-27%
  anomaly rates, and low sentiment; best stores (93-94.5) have ~5-6%
  MAPE, zero anomalies, and near-perfect sentiment.

## Value-Add Features

Beyond the core forecast and the four platform tabs above, the app and a
few extra scripts add business-facing polish:

- **Business Impact box** (Tab 1, above the chart): translates the raw
  forecast into decision language — total forecasted sales vs. the prior
  period, a suggested staffing level (`forecast / editable units-per-staff
  ratio`), a suggested inventory order quantity, and flags on any day whose
  forecast is 25%+ off that store's own typical pattern for that weekday,
  with an inferred reason (holiday, promo, or unexplained).
- **80% prediction interval**: two extra XGBoost quantile-regression models
  (10th/90th percentile) shade a confidence band around the forecast line,
  instead of a single number implying false precision.
- **"Why this forecast?" panel**: SHAP `TreeExplainer` on the point model
  shows the top 5 features driving the selected store's forecast (e.g.
  "sales_lag_14: -730", "Promo: -605"), averaged across the forecast window.
- **Signature interaction**: the Tab 1 chart is Plotly (not matplotlib),
  kept at a stable component `key` with `layout.transition` set, so
  changing the forecast horizon animates the line smoothly instead of an
  instant redraw.
- **🗺️ Portfolio Overview tab**: all 1,115 stores at once — chain-wide
  forecast total, tier counts, best/worst store this week, a top-10/
  bottom-10 growth chart, and a filterable/sortable table (by StoreType,
  CompetitionDistance). Precomputed offline (`generate_portfolio_forecast.py`)
  since a live forecast across all stores measured at ~11 minutes.
  **⚠️ Known caveat, shown in-app**: the comparison period (late July) had
  a 58-84% `SchoolHoliday` rate, but the forecast assumes 0% (future
  holidays aren't knowable) — this alone plausibly explains most of the
  uniform ~23% average "decline" the % change column shows. Treat it as a
  rough signal, not a clean apples-to-apples number, until a real forward
  holiday calendar is added.
- **Weekly alert report** (`generate_weekly_alert.py`): a proactive,
  plain-language Markdown summary — health tiers that dropped since last
  week (tracked via a saved snapshot, `store_health_scores_previous.csv`),
  stores with recent anomalous days, and stores with large forecasted
  swings (capped/summarized rather than dumping hundreds of rows when a
  systematic effect like the holiday mismatch above triggers many at
  once). See `.github/workflows/weekly_alert.yml` for a scheduled-run
  example (generates + uploads the report; emailing/Slack-posting it needs
  a credential added as a repo secret, deliberately not included here).

## Project Structure

```
train.csv, store.csv                    Raw Kaggle data
sales_forecast_phase1.py                Load data, parse dates, plot raw sales
sales_forecast_phase2.py                Clean: reindex dates, handle closures, merge store metadata
sales_forecast_phase3.py                EDA: decomposition, seasonality, ADF test, ACF/PACF
sales_forecast_phase4.py                Feature engineering: lags, rolling stats, calendar/promo flags
sales_forecast_phase5.py                Model comparison: Prophet vs XGBoost
train_anomaly_model.py                  LSTM Autoencoder anomaly detection -> anomaly_flags.csv
generate_and_analyze_reviews.py         Synthetic reviews + sentiment -> store_review_scores.csv
combine_store_health.py                 Combines everything -> store_health_scores.csv
generate_portfolio_forecast.py          Precomputes all-store 7-day forecasts -> portfolio_forecast.csv
generate_weekly_alert.py                Proactive weekly Markdown report -> weekly_alert_report.md
.github/workflows/weekly_alert.yml      Example scheduled run (GitHub Actions)
app.py                                  Streamlit app: 5 tabs (forecast, upload, health, what-if, portfolio)
requirements.txt                        Deployed app's dependencies (now incl. plotly, shap)
requirements-offline.txt                Extra deps for the offline Steps 1-2 scripts (torch, transformers)
train_cleaned.csv                       Output of Phase 2
train_features.csv                      Output of Phase 4
store_health_scores.csv                 Output of combine_store_health.py (committed -- app.py reads it)
portfolio_forecast.csv                  Output of generate_portfolio_forecast.py (committed -- app.py reads it)
store_health_scores_previous.csv        Snapshot for weekly tier-drop comparison (committed)
```

## How to Run

### 1. Get the data
`train.csv` and `store.csv` are committed in this repo, so no download is
needed to run locally. To refresh them from source instead:
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)
(competition, requires accepting the rules on Kaggle) or a public mirror
that skips that step:

```bash
kaggle datasets download -d pratyushakar/rossmann-store-sales -p . --unzip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# only needed for Steps 1-2 below (LSTM training, review generation) --
# NOT needed to run the app itself:
pip install -r requirements-offline.txt
```

### 3. Run the pipeline (in order)

```bash
python sales_forecast_phase1.py         # load + plot raw sales
python sales_forecast_phase2.py         # clean -> train_cleaned.csv
python sales_forecast_phase3.py         # decomposition, ADF, ACF/PACF plots
python sales_forecast_phase4.py         # engineer features -> train_features.csv
python sales_forecast_phase5.py         # compare Prophet vs XGBoost
python train_anomaly_model.py           # LSTM Autoencoder -> anomaly_flags.csv (needs requirements-offline.txt)
python generate_and_analyze_reviews.py  # synthetic reviews + sentiment -> store_review_scores.csv (needs requirements-offline.txt)
python combine_store_health.py          # merges everything -> store_health_scores.csv
python generate_portfolio_forecast.py   # all-store 7-day forecasts -> portfolio_forecast.csv (~11 min)
python generate_weekly_alert.py         # proactive summary -> weekly_alert_report.md
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app has five tabs:

- **🏬 Rossmann Store Forecast** — select a store and a forecast horizon
  (7 / 14 / 30 days) to see historical sales, an 80% prediction interval,
  and the XGBoost forecast, with closed-day predictions flagged separately.
  A Business Impact box above the chart translates the forecast into a
  suggested staffing level and inventory order, and a "Why this forecast?"
  panel shows the top SHAP feature contributions.
- **📁 Upload Your Own Data** — upload any CSV with a date column and a
  numeric value column (sales, revenue, orders, etc.). The app auto-detects
  likely date/value columns (editable via dropdown), then generates: the raw
  series with 7- and 30-period rolling-mean trend lines, day-of-week and
  monthly seasonality bar charts, a value distribution histogram, and
  summary stats (range, mean, min/max, total). Not specific to Rossmann or
  retail — works with any date + numeric time series.
- **📊 Store Health** — reads `store_health_scores.csv`. Shows a chain-wide
  breakdown of stores by health tier, then lets you pick a store to see its
  0-100 score, tier, plain-language explanation, the three contributing
  signals (forecast reliability, anomaly rate, review sentiment) as metric
  cards, and the same forecast chart as Tab 1 for context.
- **🔮 What-If: New Store Simulator** — projects a Store Health Score for a
  *hypothetical* store with no sales history. You enter StoreType,
  Assortment, CompetitionDistance, Promo2, and a sample review; forecast
  reliability and anomaly rate are estimated from existing stores with
  similar characteristics (shown as "estimated based on N existing
  stores"), and the sample review is scored by a lightweight keyword
  matcher — not the transformer model Tab 3's data uses — so the live
  simulator adds no memory overhead to the deployed app. Clearly labeled
  as a projection, not a guarantee.
- **🗺️ Portfolio Overview** — reads `portfolio_forecast.csv`. All stores at
  once: chain-wide forecast total, tier counts, best/worst store this week,
  a top-10/bottom-10 growth chart, and a filterable/sortable table
  (StoreType, CompetitionDistance). Carries a warning banner about the
  school-holiday comparison caveat described above.

## Deployment (Streamlit Community Cloud)

Already deployed at **https://sales-forecasting-project-espba4qtsytr6h4sjctpzr.streamlit.app/**
— it auto-redeploys on every push to `main`. To set it up from scratch on a
fork or a new app:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repo/branch, and set the main file path to
   `app.py`.
4. Deploy. The first load will take a minute or two (cleaning ~1M rows and
   training XGBoost on the fly) — after that, Streamlit's `@st.cache_data`
   / `@st.cache_resource` keep it fast for subsequent visits until the app
   restarts.

`train.csv` and `store.csv` are committed to the repo specifically so the
hosted app has data to read — no extra secrets or storage setup required.

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
- The anomaly detection and review sentiment layers are **demonstration
  additions**, not production-ready: the LSTM Autoencoder's anomaly
  threshold (95th percentile of training reconstruction error) has only
  been sanity-checked against StateHoliday days, not validated against
  known real-world incidents; and review sentiment is scored on
  **synthetically generated** text, not real customer feedback. Both
  would need real labeled anomalies and real review data respectively
  before the Store Health Score should inform actual business decisions.
- The 80% prediction interval scores the SAME feature row (built from the
  median forecast path) with the 10th/90th percentile quantile models,
  rather than running 3 fully independent recursive rollouts — a
  documented simplification that keeps the interval self-consistent but
  doesn't capture how uncertainty would compound differently at each
  quantile over a multi-day horizon.
- The Portfolio Overview's `pct_change` column has a known systematic bias
  from a school-holiday rate mismatch between the comparison period and
  the forecast — see the Value-Add Features section and the tab's own
  warning banner. A real forward holiday calendar would fix this.
