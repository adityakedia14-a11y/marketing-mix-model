"""
Step 1 of the pipeline: turn the raw Conjura daily data into a clean WEEKLY table
for one brand, ready for the Marketing Mix Model.

Run:  python src/data_prep.py
Output: data/processed/weekly.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "conjura_mmm_data.csv"
OUT = ROOT / "data" / "processed" / "weekly.csv"

# The brand we model: a US Food & Drink (CPG) eCommerce brand, all territories.
# Chosen because its channels move fairly independently (max pairwise corr ~0.5),
# it has 2+ years of history and normal (not explosive) growth.
TIMESERIES_ID = "ac9965a4cc73306fa9e4fcc0ac581eb5"

# Raw platform columns -> the 4 channels we model.
CHANNEL_MAP = {
    "search": ["google_paid_search_spend"],
    # Google migrated Shopping campaigns into Performance Max in 2022,
    # so they are the same "product-listing ads" channel over time.
    "shopping_pmax": ["google_shopping_spend", "google_pmax_spend"],
    "meta": ["meta_facebook_spend", "meta_instagram_spend", "meta_other_spend"],
    "video": ["tiktok_spend", "google_video_spend"],
}
CHANNELS = list(CHANNEL_MAP)

# US Black Friday dates in the data window (the week containing BF + Cyber week).
BLACK_FRIDAYS = ["2021-11-26", "2022-11-25", "2023-11-24"]


def load_brand() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df.columns = df.columns.str.lower()
    df = df[df["mmm_timeseries_id"] == TIMESERIES_ID].copy()
    df["date"] = pd.to_datetime(df["date_day"])
    return df.sort_values("date")


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.fillna({c: 0 for c in sum(CHANNEL_MAP.values(), [])})
    for ch, cols in CHANNEL_MAP.items():
        daily[ch] = daily[cols].sum(axis=1)

    # Net revenue = merchandise value - discounts given.
    daily["revenue"] = daily["all_purchases_original_price"] - daily["all_purchases_gross_discount"]

    agg = daily.set_index("date").resample("W-MON", label="left", closed="left").agg(
        {**{c: "sum" for c in CHANNELS},
         "revenue": "sum",
         "all_purchases": "sum",
         "all_purchases_gross_discount": "sum",
         "all_purchases_original_price": "sum",
         "date_day": "count"}
    ).rename(columns={"date_day": "n_days", "all_purchases": "orders"})

    # Drop partial weeks at the start/end (fewer than 7 days of data).
    agg = agg[agg["n_days"] == 7].drop(columns="n_days")

    # --- Control variables (things that move sales that are NOT paid media) ---
    # Average discount rate = promotion intensity.
    agg["discount_rate"] = agg["all_purchases_gross_discount"] / agg["all_purchases_original_price"]
    agg = agg.drop(columns=["all_purchases_gross_discount", "all_purchases_original_price"])

    # Black Friday / Cyber Monday: week containing BF and the following week.
    bf_weeks = set()
    for d in pd.to_datetime(BLACK_FRIDAYS):
        wk_start = d - pd.Timedelta(days=d.weekday())  # Monday of BF week
        bf_weeks |= {wk_start, wk_start + pd.Timedelta(days=7)}
    agg["bfcm"] = agg.index.isin(bf_weeks).astype(int)

    # December gift season (weeks starting 1-24 Dec, excluding BFCM weeks).
    idx = agg.index
    agg["gift_season"] = ((idx.month == 12) & (idx.day <= 24) & (agg["bfcm"] == 0)).astype(int)

    agg.index.name = "week"
    return agg.reset_index()


def main():
    weekly = to_weekly(load_brand())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT, index=False)

    print(f"Saved {len(weekly)} weeks -> {OUT.relative_to(ROOT)}")
    print(f"Period: {weekly.week.min().date()} to {weekly.week.max().date()}")
    tot = weekly[CHANNELS].sum()
    print("\nTotal spend by channel (USD):")
    print((tot.round(0)).to_string())
    print(f"\nTotal revenue: {weekly.revenue.sum():,.0f}   Overall ROAS: {weekly.revenue.sum() / tot.sum():.2f}")
    print("\nCorrelation between channels:")
    print(weekly[CHANNELS].corr().round(2).to_string())


if __name__ == "__main__":
    main()
