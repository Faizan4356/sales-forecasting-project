"""
Step 2 - Store Review Sentiment Layer (NLP)
Sales Intelligence Platform extension

*** IMPORTANT: the Rossmann dataset does not include customer review data.
*** All review text in this script is SYNTHETICALLY GENERATED for
*** demonstration purposes -- it is templated text with tone correlated to
*** each store's real metadata (StoreType, CompetitionDistance), not real
*** customer feedback. Do not treat sentiment_score or dominant_theme as
*** ground truth about any actual store.

Requires: store.csv
Outputs:  store_review_scores.csv (Store, avg_sentiment, dominant_theme, review_count)
"""

import random
import numpy as np
import pandas as pd
from collections import Counter
from transformers import pipeline

random.seed(42)
np.random.seed(42)

# =================================================================
# STEP 1 — Generate synthetic reviews correlated with store metadata
# =================================================================
store = pd.read_csv("store.csv")

# WHY this correlation design: CompetitionDistance is a real signal for
# customer convenience (less nearby competition -> customers have fewer
# alternatives -> less friction/complaint about going elsewhere), and
# StoreType is a real segment marker Rossmann already uses to distinguish
# store formats. Tying synthetic tone to these REAL columns means the
# generated data has *some* structure to recover during analysis, rather
# than being pure noise -- useful for demonstrating the pipeline even
# though the text itself isn't genuine customer feedback.
THEMES = ["Staff", "Stock Availability", "Pricing", "Cleanliness"]

TEMPLATES = {
    "Staff": {
        "positive": [
            "The staff here are always friendly and quick to help.",
            "Really knowledgeable team, they helped me find everything I needed.",
            "Staff went out of their way to be helpful today.",
        ],
        "negative": [
            "Staff seemed uninterested and hard to find when I needed help.",
            "Long wait at checkout, not enough staff on the floor.",
            "Some of the employees were rude to customers.",
        ],
    },
    "Stock Availability": {
        "positive": [
            "Shelves were fully stocked, found everything on my list.",
            "Great selection, they always seem to have what I need.",
            "Never had an issue finding items here, well stocked.",
        ],
        "negative": [
            "Several items I wanted were out of stock again.",
            "Shelves looked pretty empty this week.",
            "Frustrating that popular items are always sold out.",
        ],
    },
    "Pricing": {
        "positive": [
            "Prices are fair and there are good deals on promo items.",
            "Good value for money compared to other stores nearby.",
            "Loved the promo discounts this visit, saved a lot.",
        ],
        "negative": [
            "Prices felt higher than other stores in the area.",
            "Not much of a discount even during the promo.",
            "A bit overpriced for what you get here.",
        ],
    },
    "Cleanliness": {
        "positive": [
            "Store was clean and well organized, easy to shop.",
            "Aisles were tidy and everything was easy to find.",
            "Always a pleasant, clean shopping environment.",
        ],
        "negative": [
            "Store looked a bit messy and could use some cleaning.",
            "Floors were dirty near the entrance.",
            "Aisles were cluttered and hard to navigate.",
        ],
    },
    "Other": {
        "positive": [
            "Convenient location, easy in and out.",
            "This is my go-to store, never disappoints.",
            "Quick, easy shopping trip overall.",
        ],
        "negative": [
            "Parking was a hassle and the store felt cramped.",
            "Nothing special about this store, quite average.",
            "Wouldn't go out of my way to shop here again.",
        ],
    },
}


def competition_percentile(series):
    # Higher CompetitionDistance = less nearby competition = more
    # convenience-driven positivity bias. Missing values (no competitor
    # reported) are treated as maximally convenient, consistent with how
    # Phase 2 imputed them (a large distance, not "unknown").
    filled = series.fillna(series.max())
    return filled.rank(pct=True)


store["competition_pct"] = competition_percentile(store["CompetitionDistance"])

reviews = []
for _, row in store.iterrows():
    store_id = row["Store"]
    n_reviews = random.randint(3, 5)

    # Base positivity probability: convenience (competition distance) is
    # the dominant driver; StoreType nudges it further so store formats
    # aren't all identical, purely to give the synthetic data some
    # store-to-store texture to analyze.
    base_positivity = 0.35 + 0.4 * row["competition_pct"]
    store_type_adj = {"a": 0.05, "b": -0.05, "c": 0.0, "d": 0.02}.get(str(row["StoreType"]), 0.0)
    positivity_prob = float(np.clip(base_positivity + store_type_adj, 0.05, 0.95))

    for _ in range(n_reviews):
        theme = random.choice(THEMES if pd.notna(row["StoreType"]) else ["Other"])
        polarity = "positive" if random.random() < positivity_prob else "negative"
        text = random.choice(TEMPLATES[theme][polarity])
        reviews.append({"Store": store_id, "theme": theme, "text": text})

reviews_df = pd.DataFrame(reviews)
print(f"Generated {len(reviews_df):,} synthetic reviews across {store['Store'].nunique()} stores")

# =================================================================
# STEP 2 — Score sentiment with a pretrained HuggingFace pipeline
# =================================================================
print("Loading sentiment pipeline (distilbert-base-uncased-finetuned-sst-2-english)...")
sentiment_pipe = pipeline("sentiment-analysis")

results = sentiment_pipe(reviews_df["text"].tolist(), batch_size=32)
# Map to a continuous 0-1 "probability the review is positive" score,
# rather than keeping the raw label, so aggregation (mean per store) is
# meaningful instead of averaging categorical labels.
reviews_df["sentiment_score"] = [
    r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] for r in results
]

# =================================================================
# STEP 3 — Simple keyword/topic theme tagging (independent check on
# the theme we generated with, to mimic what real extraction would do)
# =================================================================
THEME_KEYWORDS = {
    "Staff": ["staff", "employee", "team", "checkout"],
    "Stock Availability": ["stock", "shelves", "shelf", "sold out", "selection"],
    "Pricing": ["price", "prices", "discount", "value", "promo", "overpriced"],
    "Cleanliness": ["clean", "messy", "tidy", "dirty", "aisles", "cluttered"],
}


def extract_theme(text):
    text_lower = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return theme
    return "Other"


reviews_df["extracted_theme"] = reviews_df["text"].apply(extract_theme)

# =================================================================
# STEP 4 — Aggregate to one row per store
# =================================================================
agg = reviews_df.groupby("Store").agg(
    avg_sentiment=("sentiment_score", "mean"),
    review_count=("text", "count"),
)
dominant_theme = (
    reviews_df.groupby("Store")["extracted_theme"]
    .agg(lambda s: Counter(s).most_common(1)[0][0])
    .rename("dominant_theme")
)
store_review_scores = agg.join(dominant_theme).reset_index()

store_review_scores.to_csv("store_review_scores.csv", index=False)
print("\nSaved store_review_scores.csv")

print("\n--- Example stores ---")
print(store_review_scores.sample(5, random_state=1).to_string(index=False))

print("\n--- Sentiment distribution ---")
print(store_review_scores["avg_sentiment"].describe())
