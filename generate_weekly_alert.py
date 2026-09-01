"""
Value Booster - Weekly Alert Summary
Sales Intelligence Platform extension

Generates a short, plain-language weekly report from the already-computed
pipeline outputs (store_health_scores.csv, anomaly_flags.csv,
portfolio_forecast.csv) -- this is the "proactive, not just reactive"
counterpart to the interactive app: instead of waiting for someone to open
the dashboard and click into a store, it surfaces what needs attention.

Run this AFTER the full pipeline is up to date:
  python train_anomaly_model.py
  python generate_and_analyze_reviews.py
  python combine_store_health.py
  python generate_portfolio_forecast.py
  python generate_weekly_alert.py

Outputs: weekly_alert_report.md
         store_health_scores_previous.csv (this week's snapshot, saved so
         NEXT week's run can detect tier changes -- the very first run has
         no prior snapshot to compare against, so tier-change detection is
         skipped with a note, not a fabricated result)
"""

import pandas as pd
from datetime import date

TIER_RANK = {"Needs Attention": 0, "Monitor": 1, "Performing Well": 2}
PCT_CHANGE_ALERT_THRESHOLD = 25  # flag forecasted swings beyond this, in either direction
ANOMALY_LOOKBACK_DAYS = 7

today = date.today().isoformat()

# =================================================================
# Load current pipeline outputs
# =================================================================
health = pd.read_csv("store_health_scores.csv")
anomaly_flags = pd.read_csv("anomaly_flags.csv", parse_dates=["Date"])
portfolio = pd.read_csv("portfolio_forecast.csv")

# =================================================================
# 1. Stores whose health tier dropped since last week
# =================================================================
PREV_SNAPSHOT_PATH = "store_health_scores_previous.csv"
tier_drops = pd.DataFrame(columns=["Store", "tier_prev", "tier_now"])
has_previous_snapshot = False

try:
    previous = pd.read_csv(PREV_SNAPSHOT_PATH)
    has_previous_snapshot = True
    merged = health[["Store", "tier"]].merge(
        previous[["Store", "tier"]], on="Store", suffixes=("_now", "_prev")
    )
    merged["rank_now"] = merged["tier_now"].map(TIER_RANK)
    merged["rank_prev"] = merged["tier_prev"].map(TIER_RANK)
    tier_drops = merged[merged["rank_now"] < merged["rank_prev"]][
        ["Store", "tier_prev", "tier_now"]
    ].rename(columns={"tier_prev": "tier_prev", "tier_now": "tier_now"})
except FileNotFoundError:
    pass  # first run -- nothing to compare against yet

# =================================================================
# 2. Stores with anomalous sales days in the past 7 days
# =================================================================
cutoff = anomaly_flags["Date"].max() - pd.Timedelta(days=ANOMALY_LOOKBACK_DAYS)
recent_anomalies = anomaly_flags[(anomaly_flags["Date"] > cutoff) & (anomaly_flags["is_anomaly"])]
anomaly_summary = (
    recent_anomalies.groupby("Store")
    .agg(anomaly_days=("Date", "count"), worst_day=("Date", "max"))
    .reset_index()
    .sort_values("anomaly_days", ascending=False)
)

# =================================================================
# 3. Stores forecasted for unusually high/low sales next week
# =================================================================
unusual_forecast = portfolio[portfolio["pct_change"].abs() >= PCT_CHANGE_ALERT_THRESHOLD].copy()
# Sorted by magnitude (largest swing first), not just direction -- with a
# systematic bias affecting most of the chain (see the note below), a
# plain ascending sort just dumps hundreds of similarly-sized rows in an
# arbitrary order. Magnitude-first surfaces the genuine outliers.
unusual_forecast = unusual_forecast.reindex(
    unusual_forecast["pct_change"].abs().sort_values(ascending=False).index
)

# =================================================================
# Executive summary
# =================================================================
n_needs_attention = int((health["tier"] == "Needs Attention").sum())
n_tier_drops = len(tier_drops)
n_anomaly_stores = anomaly_summary["Store"].nunique()
n_unusual_forecast = len(unusual_forecast)

summary_lines = [
    f"This week, **{n_needs_attention} store(s)** are in the \"Needs Attention\" health tier.",
]
if has_previous_snapshot:
    summary_lines.append(
        f"**{n_tier_drops} store(s)** dropped a health tier since last week's snapshot."
        if n_tier_drops else "No stores dropped a health tier since last week."
    )
else:
    summary_lines.append("(No prior snapshot yet -- tier-change tracking starts from this run.)")
summary_lines.append(
    f"**{n_anomaly_stores} store(s)** had at least one anomalous sales day flagged in the last "
    f"{ANOMALY_LOOKBACK_DAYS} days."
)
summary_lines.append(
    f"**{n_unusual_forecast} store(s)** are forecasted for a swing of {PCT_CHANGE_ALERT_THRESHOLD}%+ "
    "vs. their prior 7-day period next week."
)
executive_summary = " ".join(summary_lines)

# =================================================================
# Build the Markdown report
# =================================================================
lines = [
    f"# Weekly Sales Intelligence Report — {today}",
    "",
    "## Executive Summary",
    "",
    executive_summary,
    "",
    "## 1. Health Tier Drops Since Last Week",
    "",
]

if not has_previous_snapshot:
    lines.append("_No previous snapshot found — this is the first run. "
                  "A baseline snapshot has been saved for next week's comparison._")
elif tier_drops.empty:
    lines.append("No stores dropped a health tier this week.")
else:
    lines.append("| Store | Previous Tier | Current Tier |")
    lines.append("|---|---|---|")
    for _, row in tier_drops.iterrows():
        lines.append(f"| {int(row['Store'])} | {row['tier_prev']} | {row['tier_now']} |")

lines += ["", "## 2. Anomalous Sales Days (Last 7 Days)", ""]
if anomaly_summary.empty:
    lines.append("No anomalous sales days flagged in the last 7 days.")
else:
    lines.append("| Store | Anomalous Days | Most Recent |")
    lines.append("|---|---|---|")
    for _, row in anomaly_summary.head(20).iterrows():
        lines.append(f"| {int(row['Store'])} | {int(row['anomaly_days'])} | {row['worst_day'].date()} |")
    if len(anomaly_summary) > 20:
        lines.append(f"\n_...and {len(anomaly_summary) - 20} more store(s)._")

lines += ["", f"## 3. Unusual Sales Forecast Next Week (±{PCT_CHANGE_ALERT_THRESHOLD}%+)", ""]
if unusual_forecast.empty:
    lines.append(f"No stores forecasted for a swing of {PCT_CHANGE_ALERT_THRESHOLD}%+ next week.")
elif n_unusual_forecast > len(portfolio) * 0.15:
    # A large fraction of the whole chain crossing the threshold at once
    # is a sign of a systematic issue (e.g. a holiday-calendar mismatch
    # between the comparison period and the forecast), not 400 individual
    # stories worth alerting on -- surface it as a single flag instead of
    # burying the real outliers in a huge table.
    lines.append(
        f"**{n_unusual_forecast} of {len(portfolio)} stores** ({n_unusual_forecast / len(portfolio) * 100:.0f}%) "
        f"crossed the {PCT_CHANGE_ALERT_THRESHOLD}% threshold this week — too many at once to be "
        "individually meaningful, and likely reflects a systematic mismatch (see note below) rather "
        "than genuine store-level news. Showing the 15 largest swings only:"
    )
    lines.append("")
    lines.append("| Store | % Change | Forecast (7d) |")
    lines.append("|---|---|---|")
    for _, row in unusual_forecast.head(15).iterrows():
        lines.append(f"| {int(row['Store'])} | {row['pct_change']:+.1f}% | {row['forecast_7d_total']:,.0f} |")
else:
    lines.append("| Store | % Change | Forecast (7d) |")
    lines.append("|---|---|---|")
    for _, row in unusual_forecast.head(20).iterrows():
        lines.append(f"| {int(row['Store'])} | {row['pct_change']:+.1f}% | {row['forecast_7d_total']:,.0f} |")
    if len(unusual_forecast) > 20:
        lines.append(f"\n_...and {len(unusual_forecast) - 20} more store(s)._")

lines.append(
    "\n_Note: `pct_change` compares against a period with a different school-holiday rate "
    "than the forecast assumes -- see the Portfolio Overview tab's warning banner. Treat as "
    "a rough signal, not a precise number._"
)

report = "\n".join(lines)
with open("weekly_alert_report.md", "w", encoding="utf-8") as f:
    f.write(report)

# Save this week's health snapshot as next week's comparison baseline
health.to_csv(PREV_SNAPSHOT_PATH, index=False)

print(executive_summary)
print(f"\nSaved weekly_alert_report.md ({len(lines)} lines)")
print(f"Saved {PREV_SNAPSHOT_PATH} as next week's comparison baseline")

# =================================================================
# How to schedule this automatically
# =================================================================
# See .github/workflows/weekly_alert.yml for a GitHub Actions example that
# runs this every Monday morning. To actually deliver the report instead
# of just generating the file, add a step after this script that either:
#   - emails weekly_alert_report.md's contents (e.g. via a mail action or
#     an API like SendGrid/SES, using a repo secret for credentials), or
#   - posts it to Slack via `curl -X POST -H 'Content-type: application/json'
#     --data "{\"text\": ...}" $SLACK_WEBHOOK_URL` with the webhook URL
#     stored as a GitHub Actions secret, never hardcoded in the workflow.
# Both require credentials this repo doesn't have configured -- the
# workflow file below only handles generating and uploading the report as
# a build artifact, which is the safe, credential-free part to ship.
