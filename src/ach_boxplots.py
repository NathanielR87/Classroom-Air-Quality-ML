"""
ach_boxplots.py — ON-vs-OFF box-and-whisker plots of ACH on the known trial days.

Trial days only: one OFF day + one ON day per school (same schedule as
predict_pac.py). One graph per school for the median view, and one per
school for the mean view (12 PNGs total).

Each box holds one room's individual ACH decay events from that trial day:
red = OFF day, green = ON day (same colours as the monthly grids). Daily
single values cannot form boxes (one OFF day + one ON day per room), so
the boxes use the underlying events instead. The median view is the
classic box; the mean view adds a diamond marking each box's mean.

The events come from the same sensor that the deduped daily tables use
for that room-day, so the boxes match the published daily numbers.

Run it (after preprocess.py has built the trial months):
  source .venv/bin/activate
  python src/ach_boxplots.py
  python src/ach_boxplots.py --output /tmp/box_test   # try-out folder

Outputs (git-ignored): <out>/figures/ach_boxplots/*.png + *.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_pac import load_labelled

OFF_RED = "#e06666"    # trial OFF day (matches the monthly grids)
ON_GREEN = "#3cb371"   # trial ON day (matches the monthly grids)
LABEL_OF = {0: "OFF", 1: "ON"}


def trial_events(labelled: pd.DataFrame, processed: Path) -> pd.DataFrame:
    """ACH decay events from each trial room-day, OFF/ON attached.

    Keeps only the sensor the deduped daily table chose for that room-day,
    so the boxes match the published daily numbers.
    """
    keys = labelled[["school", "classroom", "sensor_id", "date",
                      "pac_on"]].copy()
    for col in ("sensor_id",):
        keys[col] = keys[col].astype(str)
    events = pd.read_csv(processed / "ach_events.csv",
                         dtype={"sensor_id": str},
                         parse_dates=["time", "decay_start", "decay_end"])
    events["date"] = events["time"].dt.strftime("%Y-%m-%d")
    return events.merge(keys, on=["school", "classroom", "sensor_id", "date"],
                        how="inner")


def describe(values: pd.Series) -> dict:
    """Quartiles + mean for one box of ACH events (for the CSV)."""
    clean = values.dropna()
    if clean.empty:
        return {"n": 0, "min": np.nan, "q25": np.nan, "median": np.nan,
                "mean": np.nan, "q75": np.nan, "max": np.nan}
    return {"n": int(clean.size), "min": float(clean.min()),
            "q25": float(clean.quantile(0.25)),
            "median": float(clean.median()),
            "mean": float(clean.mean()),
            "q75": float(clean.quantile(0.75)),
            "max": float(clean.max())}


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per school x room x OFF/ON: the numbers behind every box."""
    rows = []
    for (school, room, label), group in frame.groupby(
            ["school", "classroom", "pac_on"]):
        rows.append({"school": school, "classroom": room,
                     "pac": LABEL_OF[label],
                     **describe(group["decay_rate"])})
    return pd.DataFrame(rows).sort_values(
        ["school", "classroom", "pac"]).reset_index(drop=True)


def kruskal_off_on(frame: pd.DataFrame) -> dict:
    """Kruskal-Wallis test: OFF-day vs ON-day ACH events (two groups).

    Returns the H statistic, p-value, and group sizes. Non-parametric, so
    no normality assumption about the ACH values is needed.
    """
    from scipy.stats import kruskal

    off = frame.loc[frame["pac_on"] == 0, "decay_rate"].dropna()
    on = frame.loc[frame["pac_on"] == 1, "decay_rate"].dropna()
    stat, p_value = kruskal(off, on)
    return {"n_off": int(off.size), "n_on": int(on.size),
            "H": float(stat), "p": float(p_value),
            "significant": bool(p_value < 0.05)}


def plot_school(school: str, frame: pd.DataFrame, view: str,
                out_dir: Path) -> dict:
    """One graph: every room's OFF (red) vs ON (green) event boxes.

    view is "median" (classic box) or "mean" (diamonds mark each mean).
    The Kruskal-Wallis OFF-vs-ON verdict is printed in bold under the
    x-axis, green-backed when ACH looks the same, red-backed when the
    days differ. Returns the test result for the CSV.
    """
    import matplotlib.pyplot as plt

    test = kruskal_off_on(frame)
    if test["significant"]:
        verdict = (f"Kruskal-Wallis H={test['H']:.2f}, p={test['p']:.3g}: "
                   f"OFF vs ON DIFFER significantly (n={test['n_off']}/"
                   f"{test['n_on']})")
        box = {"facecolor": "#f4cccc", "edgecolor": "#cc0000", "pad": 4.0}
    else:
        verdict = (f"Kruskal-Wallis H={test['H']:.2f}, p={test['p']:.3g}: "
                   f"NO significant OFF-vs-ON difference — ACH looks the "
                   f"same (n={test['n_off']}/{test['n_on']})")
        box = {"facecolor": "#d9ead3", "edgecolor": "#38761d", "pad": 4.0}
    rooms = sorted(frame["classroom"].unique())
    data, positions, colours = [], [], []
    for i, room in enumerate(rooms):
        for j, label in enumerate((0, 1)):
            vals = frame[(frame["classroom"] == room)
                         & (frame["pac_on"] == label)]["decay_rate"].dropna()
            if vals.empty:
                continue  # that room has no events that trial day: gap left
            data.append(vals.to_numpy(dtype=float))
            positions.append(2 * i + j + 1)
            colours.append(OFF_RED if label == 0 else ON_GREEN)
    fig, axis = plt.subplots(figsize=(max(8, len(rooms) * 0.9), 4.5))
    boxes = axis.boxplot(data, positions=positions, patch_artist=True,
                         showmeans=(view == "mean"),
                         meanprops={"marker": "D", "markerfacecolor": "black",
                                    "markeredgecolor": "black"})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    axis.set_xticks([2 * i + 1.5 for i in range(len(rooms))])
    axis.set_xticklabels(rooms, rotation=45, ha="right")
    axis.set_ylabel("ACH (1/hour)")
    extra = ("Boxes span that room's decay events; "
             "black diamonds mark means." if view == "mean" else
             "Boxes span that room's decay events that day.")
    axis.set_title(f"{school} — ACH per room, trial OFF (red) vs ON (green) "
                   f"[{view} view]\n{extra}", fontsize=10)
    axis.set_xlabel(verdict, fontsize=10, fontweight="bold", bbox=box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{school}_{view}_on_off.png", dpi=100)
    plt.close(fig)
    print(f"Wrote {school}_{view}_on_off.png")
    return test


def main() -> None:
    """Build the CSV always; the PNGs need matplotlib (approved 2026-10-29)."""
    parser = argparse.ArgumentParser(description="Trial-day ACH boxplots.")
    parser.add_argument("--input", type=Path, default=Path("processed"),
                        help="Folder with merged_daily_deduped.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write plots (default: <input>/figures/"
                             "ach_boxplots)")
    args = parser.parse_args()
    # Headless backend: this script only saves PNG files, and the default
    # macOS display backend crashes (Abort trap 6) without a screen.
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = args.output or args.input / "figures" / "ach_boxplots"
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled = load_labelled(args.input)
    print(f"Trial classroom-days with full data: {len(labelled)}")
    if labelled.empty:
        raise SystemExit("No trial days found — run preprocess.py "
                         "for the trial months first.")
    frame = trial_events(labelled, args.input)
    print(f"Trial-day ACH events in boxes: {len(frame)}")
    if frame.empty:
        raise SystemExit("No trial-day events found in ach_events.csv.")
    summary_table(frame).to_csv(out_dir / "event_box_summary.csv", index=False)
    print(f"Wrote event_box_summary.csv to {out_dir}/")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed, so no PNGs were drawn. "
              "The CSV above already holds every box's numbers.")
        return
    results = []
    for school, school_frame in frame.groupby("school"):
        for view in ("median", "mean"):
            test = plot_school(school, school_frame, view, out_dir)
        results.append({"school": school, **test})
    pd.DataFrame(results).to_csv(out_dir / "kruskal_results.csv", index=False)
    print(f"Wrote kruskal_results.csv to {out_dir}/")
    print(f"Done. Plots + CSVs in {out_dir}/")


if __name__ == "__main__":
    main()
