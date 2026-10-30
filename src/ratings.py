"""
ratings.py — Good / Fair / Poor classroom ratings from daily numbers.

Bands, units, sources, and advice ALL live in src/rating_bands.csv:
edit that one file to change any threshold or recommendation. This code
only reads it. A missing column or a missing value rates as "no data",
never as a guess.

Run it (after preprocess.py has built merged_daily_deduped.csv):
  source .venv/bin/activate
  python src/ratings.py
  python src/ratings.py --output /tmp/ratings_test   # try-out folder

Outputs (git-ignored): rated_daily_deduped.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BANDS_FILE = Path(__file__).resolve().parent / "rating_bands.csv"


def load_bands() -> pd.DataFrame:
    """Read the editable bands table (one row per rated metric)."""
    return pd.read_csv(BANDS_FILE)


def rate_value(band, value: float) -> tuple[str, str]:
    """Rate one measurement: Good inside the good band, Fair inside the
    fair band, Poor outside both. Below good gets advice_low, above good
    gets advice_high; "no data" carries no advice."""
    if pd.isna(value):
        return "no data", ""
    if band.good_lo <= value <= band.good_hi:
        return "Good", ""
    if band.fair_lo <= value <= band.fair_hi:
        return "Fair", (band.advice_low if value < band.good_lo
                        else band.advice_high)
    return ("Poor", band.advice_low if value < band.good_lo
            else band.advice_high)


def rate_frame(merged: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Add {metric}_rating + {metric}_advice columns plus a worst-rating.

    worst_rating is Poor if any metric is Poor, else Fair if any is Fair,
    else Good ("no data" only when every metric is missing).
    """
    rated = merged.copy()
    rating_cols = []
    for row in bands.itertuples():
        column = row.column
        values = (merged[column] if column in merged.columns
                  else pd.Series([float("nan")] * len(merged)))
        rated[f"{row.metric}_rating"], rated[f"{row.metric}_advice"] = zip(
            *(rate_value(row, value) for value in values))
        rating_cols.append(f"{row.metric}_rating")
    for column in rated.columns:
        if column.endswith("_advice"):
            rated[column] = rated[column].fillna("")
    order = {"Poor": 0, "Fair": 1, "Good": 2, "no data": 3}
    worst = rated[rating_cols].apply(
        lambda day: min(day, key=lambda rating: order[rating]), axis=1)
    rated["worst_rating"] = worst.where(
        (rated[rating_cols] != "no data").any(axis=1), "no data")
    rated["n_poor"] = (rated[rating_cols] == "Poor").sum(axis=1)
    return rated


def main() -> None:
    """Rate every classroom-day and save the table + a short summary."""
    parser = argparse.ArgumentParser(description="Rate classroom-days.")
    parser.add_argument("--input", type=Path, default=Path("processed"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    bands = load_bands()
    merged = pd.read_csv(args.input / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    rated = rate_frame(merged, bands)
    rated.to_csv(output / "rated_daily_deduped.csv", index=False)
    print(f"Rated {len(rated)} classroom-days "
          f"({rated['worst_rating'].value_counts().to_dict()})")
    print(f"Wrote rated_daily_deduped.csv to {output}/")


if __name__ == "__main__":
    main()
