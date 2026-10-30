"""
monthly_grids.py — one ExampleGraph-style HTML grid per school for one month.

Each cell is one classroom-day, coloured by the supervised model's
P(PAC on): green >= 0.6, red <= 0.4, yellow too close to call, grey no
data. Hover a cell for the numbers. "Month call" averages P(on) over
the month. Trial days (known OFF/ON) get an outlined column.

Differences from exampleGraph.png, stated plainly: there is no survey
data in this workspace, so there are no Survey/Check columns — only
the model side. Real school names never appear; titles use School_XX.

Run it (after preprocess.py + labels exist in processed/):
  source .venv/bin/activate
  python src/monthly_grids.py                      # Dec 2025, all schools
  python src/monthly_grids.py --month 2025-10      # a different month

Outputs (git-ignored, shareable): processed/figures/School_XX_YYYY-MM.html
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_pac import FEATURES, load_labelled

MONTH = "2025-12"  # same month for every school: best shared coverage,
                   # and it holds 4 schools' trial days for outlining.

# (school, month) -> {day-of-month: label} for trial months.
# Every school with known ON/OFF days gets its trial month highlighted:
# December 2025 for the December trials, February 2026 for the February
# trials. Dates mirror src/trial_labels.csv; keys stay anonymized.
TRIAL_OUTLINES = {
    ("School_06", "2025-12"): {2: "OFF", 3: "ON"},
    ("School_07", "2025-12"): {4: "OFF", 5: "ON"},
    ("School_01", "2025-12"): {11: "OFF", 12: "ON"},
    ("School_05", "2025-12"): {15: "OFF", 16: "ON"},
    ("School_04", "2026-02"): {3: "OFF", 2: "ON"},
    ("School_03", "2026-02"): {2: "OFF", 3: "ON"}
}

# Grids to build by default: December for everyone, plus February
# for the two schools whose trial days fall in it.
GRID_PLAN = [("2025-12", None),
             ("2026-02", ["School_03", "School_04"])]


def fit_full_model(labelled: pd.DataFrame):
    """Supervised equation fit on ALL trial days (display scoring only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    matrix = scaler.fit_transform(labelled[FEATURES].to_numpy(dtype=float))
    model = LogisticRegression(max_iter=5000).fit(
        matrix, labelled["pac_on"].to_numpy())
    return model, scaler


def cell_colour(prob: float | None) -> tuple[str, str]:
    """Background colour + short call for one P(ON), mirroring the example."""
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "#d3d3d3", "no data"
    if prob >= 0.6:
        return "#3cb371", f"ON {prob:.2f}"
    if prob <= 0.4:
        return "#e06666", f"OFF {prob:.2f}"
    return "#ffd966", f"unsure {prob:.2f}"


def month_call(probs: list[float]) -> tuple[str, str]:
    """Average P(on) over the month, binned like the cells."""
    if not probs:
        return "#d3d3d3", "no data"
    mean = float(np.mean(probs))
    colour, _ = cell_colour(mean)
    word = "ON" if mean >= 0.6 else ("OFF" if mean <= 0.4 else "UNSURE")
    return colour, f"{word} {mean:.2f}"


def school_grid(school: str, month: str, days: pd.DataFrame,
                outlines: dict[int, str]) -> str:
    """One standalone HTML grid for a school-month of scored room-days."""
    year, mon = int(month[:4]), int(month[5:7])
    month_name = f"{calendar.month_name[mon]} {year}"
    n_days = calendar.monthrange(year, mon)[1]
    # Weekdays only: no school, no PAC question, and the table fits.
    show_days = [d for d in range(1, n_days + 1)
                 if calendar.weekday(year, mon, d) < 5]
    rooms = sorted(days["classroom"].unique())
    by_day = {(row.classroom, int(row.date[8:10])): row for row in days.itertuples()}

    # Trial days keep their data colour; the border tells OFF vs ON.
    border = {"OFF": "#1565c0", "ON": "#6a1b9a"}
    head = [f"<th>{d}</th>" for d in show_days]
    rows_html = []
    month_probs: dict[str, list[float]] = {room: [] for room in rooms}
    for room in rooms:
        cells = []
        for day in show_days:
            row = by_day.get((room, day))
            prob = None if row is None else float(row.p_on)
            colour, _ = cell_colour(prob)
            outline = (f"outline:3px solid {border[outlines[day]]};"
                       f"outline-offset:-3px;" if day in outlines else "")
            if row is None or prob is None or np.isnan(prob):
                tip = f"{room} · {month}-{day:02d} · no IAQ+ACH data"
            else:
                tip = (f"{room} · {row.date} · P(ON)={prob:.2f} · "
                       f"PM2.5 tot {row.Cumulative_Indoor_PM2_5:.0f} · "
                       f"max ACH {row.daily_max_ach:.2f}")
                month_probs[room].append(prob)
            if day in outlines:
                tip += f" · trial day: PAC {outlines[day]}"
            cells.append(f'<td title="{tip}" style="background:{colour};'
                         f'width:22px;height:22px;{outline}"></td>')
        colour, call = month_call(month_probs[room])
        rows_html.append(
            f"<tr><td><b>{room}</b></td>{''.join(cells)}"
            f'<td style="background:{colour};padding:2px 10px;white-space:nowrap">'
            f"{call}</td></tr>")

    outlined = ", ".join(f"{month}-{d:02d} {label}"
                         for d, label in sorted(outlines.items()))
    subtitle = (f"{month_name} (weekdays only) · month call averages P(on)"
                " over scored days"
                + (f" · trial days: blue border = OFF, purple border = ON ({outlined})"
                   if outlined else "")
                + " · no survey data in this workspace, so no Survey/Check columns")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{school} — {month_name} PAC grid</title></head>
<body style="font-family:sans-serif">
<h2>{school} — {month_name} PAC grid</h2>
<p>{subtitle}. Hover a cell for numbers.</p>
<p><span style="background:#3cb371;padding:2px 8px">Predicted ON (P&ge;0.6)</span>
<span style="background:#e06666;padding:2px 8px">Predicted OFF (P&le;0.4)</span>
<span style="background:#ffd966;padding:2px 8px">Too close to call</span>
<span style="background:#d3d3d3;padding:2px 8px">No IAQ+ACH data</span>
<span style="border:3px solid #1565c0;padding:0 6px">Trial day: PAC OFF</span>
<span style="border:3px solid #6a1b9a;padding:0 6px">Trial day: PAC ON</span></p>
<table cellspacing="1" cellpadding="0">
<tr><th>Room</th>{''.join(head)}<th>Month call</th></tr>
{''.join(rows_html)}
</table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly PAC grids, one per school.")
    parser.add_argument("--input", type=Path, default=Path("processed"))
    parser.add_argument("--month", default=None,
                        help="One month only (default: the GRID_PLAN set)")
    args = parser.parse_args()

    labelled = load_labelled(args.input)
    model, scaler = fit_full_model(labelled)
    print(f"Scoring model fit on {len(labelled)} trial days.")

    merged = pd.read_csv(args.input / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    plan = [(args.month, None)] if args.month else GRID_PLAN
    out_dir = args.input / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for month, only_schools in plan:
        month_days = merged[merged["date"].str.startswith(month)].copy()
        # Incomplete days cannot be scored (same rule as the model step);
        # they stay blank and render grey.
        complete = month_days[FEATURES].notna().all(axis=1)
        month_days["p_on"] = np.nan
        month_days.loc[complete, "p_on"] = model.predict_proba(
            scaler.transform(month_days.loc[complete, FEATURES].to_numpy(
                dtype=float)))[:, 1]
        schools = sorted(month_days["school"].unique())
        if only_schools:
            schools = [s for s in schools if s in only_schools]
        for school in schools:
            html = school_grid(school, month,
                               month_days[month_days["school"] == school],
                               TRIAL_OUTLINES.get((school, month), {}))
            path = out_dir / f"{school}_{month}.html"
            path.write_text(html)
            n_rooms = month_days[month_days["school"] == school][
                "classroom"].nunique()
            print(f"Wrote {path} ({n_rooms} rooms)")


if __name__ == "__main__":
    main()
