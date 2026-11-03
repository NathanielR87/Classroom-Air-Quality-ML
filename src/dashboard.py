"""
dashboard.py — district dashboard for classroom air quality (Streamlit).

Another district uses it on their own schools: run preprocess.py on a
Batch1-style folder of sensor CSVs, then point the sidebar at that
processed folder. No code changes needed.

Views: clickable school map + calendar day picker, per-classroom
Good/Fair/Poor ratings (peaks, typical, or both) with recommendations,
the condensed monthly PAC grid, condensed trial predictions, and the
model report summary.

Run it:
  source .venv/bin/activate
  streamlit run src/dashboard.py
  streamlit run src/dashboard.py -- --processed /path/to/processed
"""

from __future__ import annotations

import argparse
import calendar as calmod
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monthly_grids import TRIAL_OUTLINES, fit_full_model, school_grid
from predict_pac import FEATURES, load_labelled
from ratings import load_bands, rate_frame

RATING_COLOUR = {"Good": "#3cb371", "Fair": "#ffd966",
                 "Watch": "#ffd966", "High": "#ff9800",
                 "Poor": "#e06666", "no data": "#d3d3d3"}


def pretty(room: str) -> str:
    """Display name only: Room_105 -> Room 105. Data keys keep the _."""
    return room.replace("_", " ").replace(" m1", " M1")


def pretty_report(text: str) -> str:
    """Display-only: School_03 -> School 03. Column names keep the _."""
    import re

    return re.sub(r"(School|Room)_([A-Za-z0-9]+)", r"\1 \2", text)


def read_local_key() -> str:
    """Meta key from a local .env file or the environment.

    The engineer creates ONE file called .env next to src/ holding one
    line: META_MODEL_API_KEY=.... The dashboard only ever checks it
    exists; the key itself is never shown, saved, or committed (the
    repo ignores every non-src file by default).
    """
    import os

    for candidate in (Path.cwd() / ".env",
                      Path(__file__).resolve().parent.parent / ".env"):
        try:
            for line in candidate.read_text().splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "META_MODEL_API_KEY":
                    return value.strip().strip("\"'")
        except OSError:
            continue
    return os.environ.get("META_MODEL_API_KEY", "")


def run_engineer_pipeline(raw: str, out: Path, months: str,
                         schools: str) -> tuple[bool, str]:
    """Run preprocess.py in a subprocess; return (ok, log tail).

    Pure runner with no Streamlit calls, so it is testable headless.
    Numbering stays stable because the input folder keeps all its
    school subfolders (--schools only chooses work, never renames).
    """
    import subprocess
    import sys

    command = [sys.executable, str(Path(__file__).resolve().parent
                                   / "preprocess.py"),
               "--input", raw, "--output", str(out), "--workers", "4"]
    if months.strip():
        command += ["--months", months.strip()]
    if schools.strip():
        command += ["--schools", schools.strip()]
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=6 * 3600)
    except Exception as exc:
        return False, f"could not start the pipeline: {exc}"
    tail = "\n".join((finished.stdout + finished.stderr).splitlines()[-15:])
    return finished.returncode == 0, tail


def write_letter(processed: Path, school: str, month: str,
                key: str) -> Path:
    """Build the Meta principal letter for one school-month in-process."""
    import os

    from principal_report import (build_prompt, call_spark,
                                  school_month_context)

    facts = school_month_context(processed, school, month)
    os.environ["META_MODEL_API_KEY"] = key  # server memory only
    try:
        report = call_spark(build_prompt(facts, school, month))
    finally:
        os.environ.pop("META_MODEL_API_KEY", None)
    out = processed / f"principal_report_{school}_{month}.md"
    out.write_text(report)
    return out

REQUIRED = {"merged_daily_deduped.csv": ["date", "school", "classroom",
                                        "sensor_id"]}


def check_folder(folder: Path) -> list[str]:
    """Plain-language problems with a processed folder (empty = usable)."""
    if not folder.is_dir():
        return [f"{folder} is not a folder."]
    problems = []
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
def load_all(folder: str, view: str) -> tuple[pd.DataFrame, pd.DataFrame,
                                             dict, str]:
    """Rated days for one view, labelled trials, a scorer, report text."""
    root = Path(folder)
    bands = load_bands()
    merged = pd.read_csv(root / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    rated = rate_frame(merged, bands, view=view)
    labelled = load_labelled(root)
    model, scaler = fit_full_model(labelled)
    report_path = root / "pac_model_report.txt"
    report = report_path.read_text() if report_path.exists() else ""
    return rated, labelled, {"model": model, "scaler": scaler}, report


def room_number(name: str) -> int | None:
    """Room_106 -> 106; None when the name holds no number."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def plan_figure(plan: Path, dots: dict, colours: list[str],
                numbers: list[str], names: list[str]):
    """Plotly figure: floor plan with one colored dot per room."""
    import base64
    import io

    import plotly.graph_objects as go
    from PIL import Image

    img = Image.open(plan).convert("RGB")
    width, height = img.size
    if width > 1400:
        img = img.resize((1400, int(1400 * height / width)))
        width, height = img.size
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    uri = ("data:image/png;base64,"
           + base64.b64encode(buffer.getvalue()).decode())
    xs = [dots[name][0] * width for name in names]
    ys = [dots[name][1] * height for name in names]
    fig = go.Figure()
    fig.add_layout_image(dict(source=uri, xref="x", yref="y", x=0, y=0,
                              sizex=width, sizey=height, sizing="stretch",
                              layer="below"))
    fig.add_scatter(x=xs, y=ys, mode="markers+text", text=numbers,
                    textposition="top center",
                    textfont=dict(color="black", size=14),
                    marker=dict(color=colours, size=22,
                                line=dict(color="white", width=2)),
                    customdata=[pretty(name) for name in names],
                    hovertemplate="%{customdata}<extra></extra>")
    fig.update_xaxes(range=[0, width], visible=False)
    fig.update_yaxes(range=[height, 0], visible=False, scaleanchor="x")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      showlegend=False, clickmode="event+select",
                      dragmode=False)
    return fig


def popup_rows(day_row: pd.Series, bands: pd.DataFrame,
                 view: str) -> pd.DataFrame:
    """Status table for the popup: every visible metric, value, rating."""
    if view == "peaks":
        shown = bands[bands["view"].isin(["peak", "all"])]
    elif view == "typical":
        shown = bands[bands["view"].isin(["typical", "all"])]
    else:
        shown = bands
    rows = []
    for band in shown.itertuples():
        value = day_row.get(band.column, float("nan"))
        rows.append({"Metric": band.label,
                     "Value": (f"{value:.1f} {band.unit}"
                               if pd.notna(value) else "no data"),
                     "Rating": day_row.get(f"{band.metric}_rating",
                                           "no data")})
    return pd.DataFrame(rows)


@st.dialog("Classroom status", width="large")
def room_popup(title: str, table: pd.DataFrame, overall: str,
               advice: list) -> None:
    """Modal popup with the clicked room's current status."""
    st.write(f"**{title}** — overall: **{overall}**")
    st.dataframe(table, hide_index=True)
    if advice:
        st.subheader("Recommendations")
        for label, tip in advice:
            st.write(f"**{label}:** {tip}")
    if st.button("Close"):
        st.session_state.dismissed_room = st.session_state.popup_room
        st.session_state.popup_room = None
        st.rerun()


def plan_dots_map(rated: pd.DataFrame, school: str, day: str,
                    plan: Path) -> bool:
    """Clickable dots on the real floor plan, colored by overall rating.

    Clicking a dot selects that room (same as the buttons): the status
    panel below then shows the room's full ratings. Streamlit has no
    floating popups, so selection drives the detail view instead.
    Returns True (the button grid is skipped for mapped schools).
    """
    from school_maps import SCHOOL_DOTS

    day_rows = rated[(rated["school"] == school) & (rated["date"] == day)]
    have = {row.classroom: row.worst_rating for row in day_rows.itertuples()}
    dots = SCHOOL_DOTS[school]
    names = sorted(dots)
    colours = [RATING_COLOUR.get(have.get(name, "no data"), "#d3d3d3")
               for name in names]
    numbers = [name.replace("Room_", "").upper() for name in names]
    fig = plan_figure(plan, dots, colours, numbers, names)
    event = st.plotly_chart(fig, key=f"plan-{school}", on_select="rerun",
                                selection_mode="points",
                                use_container_width=True)
    points = (event.selection.get("points", []) if event else [])
    if points:
        picked = names[points[0]["point_index"]]
        if picked != st.session_state.get("room"):
            st.session_state.room = picked
            ever = rated[rated["classroom"] == picked]["date"]
            if (picked not in
                    set(rated[rated["school"] == school]["classroom"])
                    and not ever.empty):
                st.session_state.day = sorted(ever)[-1]
        if (picked != st.session_state.get("popup_room")
                and picked != st.session_state.get("dismissed_room")):
            st.session_state.popup_room = picked
            st.session_state.dismissed_room = None
    else:
        st.session_state.dismissed_room = None
    st.caption("Dot color = overall rating that day "
               "(green good, yellow watch, orange high, red poor).")
    popup = st.session_state.get("popup_room")
    if popup:
        from ratings import load_bands

        view = st.session_state.get("view", "both")
        bands = load_bands()
        rows = rated[(rated["school"] == school)
                     & (rated["classroom"] == popup)
                     & (rated["date"] == day)]
        if rows.empty:
            st.session_state.popup_room = None
        else:
            row = rows.iloc[0]
            shown = (bands if view == "both" else bands[bands["view"].isin(
                ["peak" if view == "peaks" else "typical", "all"])])
            advice = [(band.label, row.get(f"{band.metric}_advice", ""))
                      for band in shown.itertuples()
                      if row.get(f"{band.metric}_advice", "")]
            room_popup(f"{pretty(popup)} — {day}",
                       popup_rows(row, bands, view),
                       row["worst_rating"], advice)
    return True


def school_map(rated: pd.DataFrame, school: str, processed: Path,
                 workspace: Path) -> None:
    """Clickable map: real rooms colored by overall rating, gaps filled
    with grey pretend rooms that have no monitor.

    Schools with a real floor plan (see school_maps.py) show the
    anonymized plan on top; the buttons below stay the click mechanism.
    Other schools get a fully schematic button layout.
    """
    from school_maps import SCHOOL_DOTS, floor_plan_png

    plan = floor_plan_png(processed, workspace, school)
    day = st.session_state.day
    if plan is not None and school in SCHOOL_DOTS:
        if plan_dots_map(rated, school, day, plan):
            return
    else:
        st.caption("Made-up layout — positions are schematic, not the "
                   "building.")
    view = st.session_state.view
    rooms = rated[(rated["school"] == school)
                  & (rated["date"] == day)]
    by_number = {}
    for row in rooms.itertuples():
        number = room_number(row.classroom)
        if number is not None:
            by_number[number] = (row.classroom, row.worst_rating)
    if not by_number:
        st.info("No rated rooms for this day.")
        return
    # Fill only small gaps (101, 103 -> show 102): wide numeric spans stay
    # compact instead of drawing a hundred grey rooms. On the real plan
    # these sit between their neighbours; here they stay in order.
    ordered = sorted(by_number)
    numbers = []
    for i, number in enumerate(ordered):
        numbers.append(number)
        if i + 1 < len(ordered) and 1 < ordered[i + 1] - number <= 3:
            numbers.extend(range(number + 1, ordered[i + 1]))
    for start in range(0, len(numbers), 6):
        columns = st.columns(6)
        for column, number in zip(columns, numbers[start:start + 6]):
            if number in by_number:
                name, rating = by_number[number]
                column.button(
                    str(number), key=f"room-{school}-{number}",
                    help=f"{pretty(name)}: {rating} ({view})",
                    type=("primary" if name == st.session_state.room
                          else "secondary"),
                    on_click=_pick_room, args=(name,))
                column.caption(
                    f"<span style='color:{RATING_COLOUR[rating]}'>"
                    f"● {rating}</span>", unsafe_allow_html=True)
            else:
                column.button(str(number), key=f"gap-{school}-{number}",
                              disabled=True, help="No monitor in this room.")
                column.caption("no monitor")


def _pick_room(name: str) -> None:
    """Map click handler: switch room, keep the day if it has data."""
    st.session_state.room = name


def _pick_day(day: str) -> None:
    """Calendar click handler."""
    st.session_state.day = day


def calendar_picker(rated: pd.DataFrame, school: str, room: str) -> None:
    """A month calendar of buttons; only days with data can be picked."""
    have = set(rated[(rated["school"] == school)
                     & (rated["classroom"] == room)]["date"])
    if not have:
        st.info("No days for this room.")
        return
    months = sorted({day[:7] for day in have})
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = months[-1]
    if st.session_state.get("cal_month") not in months:
        st.session_state.cal_month = months[-1]
    month = st.selectbox("Month", months,
                         index=months.index(st.session_state.cal_month),
                         key="cal_month")
    year, mon = int(month[:4]), int(month[5:7])
    header = st.columns(5)
    for column, label in zip(header, ["Mo", "Tu", "We", "Th", "Fr"]):
        column.markdown(f"<div style='text-align:center'><b>{label}</b>"
                        "</div>", unsafe_allow_html=True)
    for week in calmod.monthcalendar(year, mon):
        weekdays = [day_number for day_number in week[:5]]
        columns = st.columns(5)
        for column, day_number in zip(columns, weekdays):
            if not day_number:
                column.write("")
                continue
            stamp = f"{month}-{day_number:02d}"
            column.button(
                str(day_number), key=f"cal-{stamp}",
                disabled=stamp not in have,
                type=("primary" if stamp == st.session_state.day
                      else "secondary"),
                use_container_width=True,
                on_click=_pick_day, args=(stamp,))


def rating_cards(day: pd.Series, bands: pd.DataFrame, view: str) -> None:
    """Rating cards for the visible metrics plus an overall rating."""
    if view == "peaks":
        shown = bands[bands["view"].isin(["peak", "all"])]
    elif view == "typical":
        shown = bands[bands["view"].isin(["typical", "all"])]
    else:
        shown = bands
    st.subheader(f"{pretty(day['classroom'])} — {day['date']} "
                 f"(overall {view}: {day['worst_rating']})")
    rows = [shown.iloc[i:i + 5] for i in range(0, len(shown), 5)]
    for row in rows:
        columns = st.columns(5)
        for column, band in zip(columns, row.itertuples()):
            rating = day.get(f"{band.metric}_rating", "no data")
            value = day.get(band.column, float("nan"))
            text = (f"{value:.1f} {band.unit}" if pd.notna(value)
                    else "no data")
            column.markdown(
                f"<div style='background:{RATING_COLOUR[rating]};"
                f"padding:8px;border-radius:6px;text-align:center'>"
                f"<b>{band.label}</b><br>{text}<br>{rating}</div>",
                unsafe_allow_html=True)
    advice = [(band.label, day.get(f"{band.metric}_advice", ""))
              for band in shown.itertuples()
              if day.get(f"{band.metric}_advice", "")]
    if advice:
        st.subheader("Recommendations")
        for label, tip in advice:
            st.write(f"**{label}:** {tip}")
    else:
        st.write("No recommendations — every rated metric looks good.")


def decay_expander(processed: Path, batch1: Path, day: pd.Series) -> None:
    """Slide-11 style chart: today's CO2 with the kept ACH events orange.

    Streamlit cannot hover a graph over a card, so this expander under the
    ACH ratings is the equivalent: one click shows exactly which CO2
    readings produced the day's ACH. Recomputed from raw data on demand;
    needs the Batch1 folder from the sidebar.
    """
    import datetime as dt

    from decay_chart import day_events, decay_figure

    with st.expander("Where this ACH comes from — today's CO2 decay"):
        key_path = (processed / "_private_do_not_share" / "school_key.csv")
        if not batch1.is_dir() or not key_path.exists():
            st.info("Point the sidebar at your raw Batch1 folder to draw "
                    "the decay chart.")
            return
        key = pd.read_csv(key_path)
        real_of = dict(zip(key["school_private_name"],
                           key["batch1_folder_name"]))
        real = real_of.get(day["school"])
        if real is None:
            st.info("This school is not in the private key.")
            return
        with st.spinner("Recomputing today's decays…"):
            try:
                minutes, events = day_events(
                    batch1, real, day["classroom"], str(day["sensor_id"]),
                    dt.date.fromisoformat(day["date"]))
            except Exception as exc:  # raw gaps must not break the page
                st.warning(f"Could not rebuild decays: {exc}")
                return
        if events.empty:
            st.info("No kept decay events that day — ACH is missing too.")
            return
        st.pyplot(decay_figure(
            minutes, events,
            f"{pretty(day['school'])} {pretty(day['classroom'])}"
            f" — {day['date']}"),
            use_container_width=True)


def report_summary(report: str) -> None:
    """The model report condensed to its verdict lines; full text folded."""
    if not report:
        return
    keep = [line for line in report.splitlines()
            if line.startswith(("Labeled", "Model B", "Model A",
                                "Baseline", "Contract bar"))]
    st.subheader("Model report (condensed)")
    for line in keep:
        st.write(pretty_report(line))
    with st.expander("What these numbers mean"):
        st.write("**125 labeled days** are the only days known for sure "
                 "(trial ON/OFF days): 83 train the models, 42 are held "
                 "out for testing. Two whole schools are never trained "
                 "on, so the test score measures new schools, not memory.")
        st.write("**Model B (supervised)** learns from the labels. Each "
                 "day gets a score from the equation: every feature is "
                 "first standardized ((value - mean) / scale) so big "
                 "units can't swamp small ones, then multiplied by its "
                 "weight. Positive weights push toward ON: daily max "
                 "ACH (+0.90) and humidity (+0.98) strongest. Negative "
                 "weights push toward OFF: cumulative PM2.5 (-0.50), so "
                 "cleaner air days read as PAC-on. PM2.5 max (+0.03) "
                 "barely matters. The score converts to a 0-1 "
                 "probability; 0.5 and up calls ON.")
        st.write("**Model A (unsupervised)** gets no labels — it just "
                 "clusters days and hopes one cluster is ON. Its 0.405 "
                 "test score (worse than guessing) shows labels matter.")
        st.write("**The four counts** are (actual, predicted): (0,0) "
                 "right OFF, (1,1) right ON, (0,1) false alarm, (1,0) "
                 "missed ON. Model B's test 0.690 = (17+12) right out "
                 "of 42, about 7 in 10 on unseen schools.")
        st.write("**Baseline 0.500** is always guessing the majority "
                 "class — the do-nothing bar. The contract is met "
                 "because B beats A by 0.286 and baseline by 0.190.")
    with st.expander("Full model report"):
        st.text(pretty_report(report))


def all_schools_grids(rated: pd.DataFrame, scorer: dict) -> None:
    """One page, one grid per trial school in its own trial month.

    Months come from trial_labels.csv (December for four schools,
    February for two); schools without trial days never appear here.
    Every grid outlines its OFF (blue) and ON (purple) trial days, so
    the ON-vs-OFF story reads across schools at a glance.
    """
    from predict_pac import load_trial_labels

    for row in load_trial_labels().itertuples():
        month = row.off_date[:7]
        month_days = rated[rated["date"].str.startswith(month)].copy()
        complete = month_days[FEATURES].notna().all(axis=1)
        month_days["p_on"] = float("nan")
        month_days.loc[complete, "p_on"] = scorer["model"].predict_proba(
            scorer["scaler"].transform(
                month_days.loc[complete, FEATURES].to_numpy(
                    dtype=float)))[:, 1]
        school_days = month_days[month_days["school"] == row.school]
        if school_days.empty:
            continue
        st.subheader(f"{pretty(row.school)} — {month} PAC grid")
        st.markdown(
            school_grid(row.school, month, school_days,
                        TRIAL_OUTLINES.get((row.school, month), {})),
            unsafe_allow_html=True)


def main() -> None:
    """Sidebar picks the folder; map + calendar pick the room-day."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="processed")
    args, _ = parser.parse_known_args()
    folder = Path(args.processed)

    st.title("Classroom air — district dashboard")
    st.sidebar.caption("The dashboard reads the processed folder below. "
                       "The raw folder is only used for the ACH decay "
                       "chart; everything else comes from processed data.")
    choice = st.sidebar.text_input(
        "Processed folder", str(folder), key="folder",
        help="Rated daily tables the dashboard reads. Another district "
             "runs preprocess.py on their own sensors and points this "
             "here — no code changes.")
    choice = st.session_state.folder
    raw_default = Path.home() / "Desktop" / "AdamData" / "Batch1"
    raw = st.sidebar.text_input(
        "Raw Batch1 folder (for decay charts)", str(raw_default),
        help="Raw sensor CSVs. Only the 'Where this ACH comes from' "
             "chart reads these; every other panel uses the processed "
             "folder above.")

    st.sidebar.header("Engineer run (new school)")
    st.sidebar.caption("Three steps, nothing to install. "
                       "1: raw data folder. 2: key file (below). 3: Run.")
    eng_months = st.sidebar.text_input(
        "Months", "2025-12", key="eng_months",
        help="Blank = every month (slow first run). One month keeps "
             "the demo fast; later runs fill in the gaps.")
    eng_schools = st.sidebar.text_input(
        "School filter", "", key="eng_schools",
        help="Optional: one folder name inside the raw folder to run a "
             "single school. Blank = all schools. Numbering stays "
             "stable either way.")
    eng_name = st.sidebar.text_input("Run name", "eng_demo",
                                     key="eng_name")
    key = read_local_key()
    if key:
        st.sidebar.caption("Meta key file: found.")
    else:
        st.sidebar.caption("Meta key file: not found.")
        with st.sidebar.expander("How to add the key (once)"):
            st.write("Create a file named .env in the AdamData folder "
                     "(next to src/) holding exactly one line:")
            st.code("META_MODEL_API_KEY=paste-it-here")
            st.write("Save it, then continue. The dashboard never "
                     "shows, saves, or uploads the key.")
    if st.sidebar.button("Run pipeline", key="eng_run"):
        out = Path(choice or "processed") / "runs" / eng_name.strip()
        with st.spinner("Running the pipeline — first months take a "
                        "few minutes, finished ones are skipped…"):
            ok, tail = run_engineer_pipeline(
                raw, out, eng_months, eng_schools)
        st.sidebar.text(tail or "(no output)")
        if ok:
            st.session_state.folder = str(out)
            st.sidebar.success(f"Done — now showing {out}.")
            st.rerun()
        else:
            st.sidebar.error("The run failed — log above.")
    problems = check_folder(Path(choice))
    if problems:
        st.error(" ".join(problems))
        st.info("New district? Run: python src/preprocess.py "
                "--input <your Batch1-style folder> --output <folder>, "
                "then enter that folder above.")
        return

    page = st.sidebar.radio("Page", ["Classroom", "All schools ON/OFF"],
                              key="page")
    view = st.sidebar.radio("Values", ["Peaks", "Typical", "Both"],
                            key="view_toggle").lower()
    st.sidebar.caption("Typical = medians and means, a normal school day. "
                       "Peaks = the day's worst moments. Both = all of it. "
                       "Weekends never appear: schools are closed.")
    st.sidebar.caption("Typical uses the median for ACH, means for PM2.5, "
                       "CO2 and VOC (temperature and humidity are always "
                       "averages).")
    st.session_state.view = view
    rated, _labelled, scorer, report = load_all(choice, view)
    bands = load_bands()

    schools = sorted(rated["school"].unique())
    if page == "All schools ON/OFF":
        all_schools_grids(rated, scorer)
        return

    if st.session_state.get("school") not in schools:
        st.session_state.school = schools[0]
    st.sidebar.selectbox("School", schools, key="school",
                           format_func=pretty)
    school = st.session_state.school

    if "room" not in st.session_state:
        first = rated[rated["school"] == school].sort_values(
            ["classroom", "date"]).iloc[0]
        st.session_state.room = first["classroom"]
        st.session_state.day = first["date"]
    room_days = rated[(rated["school"] == school)
                      & (rated["classroom"] == st.session_state.room)]
    if room_days.empty:  # room has no rated days: fall back to first room
        first = rated[rated["school"] == school].sort_values(
            ["classroom", "date"]).iloc[0]
        st.session_state.room = first["classroom"]
        st.session_state.day = first["date"]
        room_days = rated[(rated["school"] == school)
                          & (rated["classroom"] == st.session_state.room)]
    if st.session_state.get("day") not in set(room_days["date"]):
        st.session_state.day = sorted(room_days["date"])[-1]

    st.subheader("School map — click a classroom")
    school_map(rated, school, Path(choice), Path.cwd())
    st.subheader("Calendar — click a day")
    calendar_picker(rated, school, st.session_state.room)

    day_rows = rated[(rated["school"] == school)
                     & (rated["classroom"] == st.session_state.room)
                     & (rated["date"] == st.session_state.day)]
    if not day_rows.empty:
        rating_cards(day_rows.iloc[0], bands, view)
        decay_expander(Path(choice), Path(raw), day_rows.iloc[0])

    month = st.session_state.day[:7]
    month_days = rated[rated["date"].str.startswith(month)].copy()
    complete = month_days[FEATURES].notna().all(axis=1)
    month_days["p_on"] = float("nan")
    month_days.loc[complete, "p_on"] = scorer["model"].predict_proba(
        scorer["scaler"].transform(
            month_days.loc[complete, FEATURES].to_numpy(dtype=float)))[:, 1]
    st.subheader(f"{pretty(school)} — {month} PAC grid")
    st.markdown(
        school_grid(school, month, month_days[month_days["school"] == school],
                    TRIAL_OUTLINES.get((school, month), {})),
        unsafe_allow_html=True)

    st.subheader("Trial-day predictions (condensed)")
    predictions_path = Path(choice) / "pac_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path,
                                  dtype={"sensor_id": str})
        mine = predictions[predictions["school"] == school]
        show = mine[["classroom", "date", "pac_on", "predicted_on",
                     "p_on"]].copy()
        show["classroom"] = show["classroom"].map(pretty)
        show["p_on"] = show["p_on"].round(2)
        st.dataframe(show, hide_index=True)
    else:
        st.info("No pac_predictions.csv here yet — run predict_pac.py.")
    report_summary(report)
    letter_path = (Path(choice)
                   / f"principal_report_{school}_{month}.md")
    if letter_path.exists():
        with st.expander("Principal letter (Meta AI)"):
            st.markdown(letter_path.read_text())
    if key:
        if st.button("Write principal letter with Meta AI",
                     key="write_letter"):
            with st.spinner("Asking Meta AI to write the letter…"):
                try:
                    out_letter = write_letter(Path(choice), school,
                                              month, key)
                except Exception as exc:  # key/network issues stay here
                    st.error(f"Letter failed: {exc}")
                else:
                    st.success(f"Wrote {out_letter.name}.")
                    st.rerun()
    else:
        st.caption("Add the Meta key file (sidebar, Engineer run) to "
                   "enable the principal letter.")


if __name__ == "__main__":
    main()
