"""
dashboard.py — district dashboard for classroom air quality (Streamlit).

Another district uses it on their own schools: run preprocess.py on a
Batch1-style folder of sensor CSVs, then point the sidebar at that
processed folder. No code changes needed.

Views: per-classroom Good/Fair/Poor ratings with recommendations, the
ExampleGraph-style monthly PAC grid, trial-day predictions, and the
model comparison report.

Run it:
  source .venv/bin/activate
  streamlit run src/dashboard.py
  streamlit run src/dashboard.py -- --processed /path/to/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monthly_grids import TRIAL_OUTLINES, fit_full_model, school_grid
from predict_pac import FEATURES, load_labelled
from ratings import load_bands, rate_frame

RATING_COLOUR = {"Good": "#3cb371", "Fair": "#ffd966",
                 "Poor": "#e06666", "no data": "#d3d3d3"}

REQUIRED = {"merged_daily_deduped.csv": ["date", "school", "classroom",
                                        "sensor_id"]}


def check_folder(folder: Path) -> list[str]:
    """Plain-language problems with a processed folder (empty = usable)."""
    problems = []
    if not folder.is_dir():
        return [f"{folder} is not a folder."]
    for name, columns in REQUIRED.items():
        path = folder / name
        if not path.exists():
            problems.append(f"Missing {name} — run preprocess.py first.")
            continue
        missing = [c for c in columns if c not in
                   pd.read_csv(path, nrows=0).columns]
        if missing:
            problems.append(f"{name} lacks columns: {', '.join(missing)}.")
    return problems


@st.cache_data
def load_all(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Rated classroom-days, labelled trial days, a scorer, report text."""
    root = Path(folder)
    bands = load_bands()
    merged = pd.read_csv(root / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    rated = rate_frame(merged, bands)
    labelled = load_labelled(root)
    model, scaler = fit_full_model(labelled)
    report_path = root / "pac_model_report.txt"
    report = report_path.read_text() if report_path.exists() else ""
    return rated, labelled, {"model": model, "scaler": scaler}, report


def rating_cards(day: pd.Series, bands: pd.DataFrame) -> None:
    """One coloured card per metric: rating, value, and any advice."""
    st.subheader(f"{day['classroom']} — {day['date']} "
                 f"(overall: {day['worst_rating']})")
    columns = st.columns(len(bands))
    for column, row in zip(columns, bands.itertuples()):
        rating = day.get(f"{row.metric}_rating", "no data")
        value = day.get(row.column, float("nan"))
        text = (f"{value:.1f} {row.unit}" if pd.notna(value) else "no data")
        column.markdown(
            f"<div style='background:{RATING_COLOUR[rating]};"
            f"padding:8px;border-radius:6px;text-align:center'>"
            f"<b>{row.label}</b><br>{text}<br>{rating}</div>",
            unsafe_allow_html=True)
    advice = [(row.label, day.get(f"{row.metric}_advice", ""))
              for row in bands.itertuples()
              if day.get(f"{row.metric}_advice", "")]
    if advice:
        st.subheader("Recommendations")
        for label, tip in advice:
            st.write(f"**{label}:** {tip}")
    else:
        st.write("No recommendations — every rated metric looks good.")


def main() -> None:
    """Sidebar picks the data folder; views share the cached tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="processed")
    args, _ = parser.parse_known_args()
    folder = Path(args.processed)

    st.title("Classroom air — district dashboard")
    st.sidebar.header("Data folder")
    choice = st.sidebar.text_input("Processed folder", str(folder))
    problems = check_folder(Path(choice))
    if problems:
        st.error(" ".join(problems))
        st.info("New district? Run: python src/preprocess.py "
                "--input <your Batch1-style folder> --output <folder>, "
                "then enter that folder above.")
        return
    rated, labelled, scorer, report = load_all(choice)
    bands = load_bands()

    schools = sorted(rated["school"].unique())
    school = st.sidebar.selectbox("School", schools)
    rooms = sorted(rated[rated["school"] == school]["classroom"].unique())
    room = st.sidebar.selectbox("Classroom", rooms)
    days = rated[(rated["school"] == school)
                 & (rated["classroom"] == room)].sort_values("date")
    day = st.sidebar.selectbox("Day", days["date"].tolist(),
                               index=len(days) - 1)
    rating_cards(days[days["date"] == day].iloc[0], bands)

    month = st.sidebar.selectbox(
        "Grid month", sorted(rated["date"].str.slice(0, 7).unique()))
    month_days = rated[rated["date"].str.startswith(month)].copy()
    complete = month_days[FEATURES].notna().all(axis=1)
    month_days["p_on"] = float("nan")
    month_days.loc[complete, "p_on"] = scorer["model"].predict_proba(
        scorer["scaler"].transform(
            month_days.loc[complete, FEATURES].to_numpy(dtype=float)))[:, 1]
    st.subheader(f"{school} — {month} PAC grid")
    st.markdown(
        school_grid(school, month, month_days[month_days["school"] == school],
                    TRIAL_OUTLINES.get((school, month), {})),
        unsafe_allow_html=True)

    st.subheader("Trial-day predictions (labelled days only)")
    predictions_path = Path(choice) / "pac_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path,
                                  dtype={"sensor_id": str})
        st.dataframe(predictions[predictions["school"] == school])
    else:
        st.info("No pac_predictions.csv here yet — run predict_pac.py.")
    if report:
        st.subheader("Model comparison report")
        st.text(report)


if __name__ == "__main__":
    main()
