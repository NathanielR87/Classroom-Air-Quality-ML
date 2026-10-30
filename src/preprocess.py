"""
preprocess.py — ONE simple pipeline for Batch1 (replaces 3 old steps).

Old way (3 separate codes, 2 languages, HPC paths):
  1. IAQ_meta_processing_sep25.R          -> daily PM2.5 summaries
  2. rawsensor_preprocess_*.ipynb         -> monthly CO2+PM files
  3. process_data.py + postprocessing     -> ACH (ventilation) per day

New way: run this one Python file. It reads the raw Batch1 folders,
does all 3 steps, and writes clean tables to processed/.

What it does, in plain words:
  A. Read every classroom sensor's minute readings (CO2, PM2.5, temp,
     humidity, VOC). Sensors with "alex" in the folder name are SKIPPED
     (they only track power). "outside" sensors are skipped too.
  B. Make one row per classroom-day: average temp/humidity, highest and
     total (summed) CO2 and PM2.5 during school hours (Mon-Fri 08:00-14:30).
  C. From the CO2 signal, find moments where CO2 falls smoothly after
     people leave, and measure how fast it falls. That speed is ACH
     (air changes per hour = how ventilated the room is).
  D. Join B + C into one table. PAC ON/OFF labels come later (see
     predict_pac.py, which uses the known trial days).

Long runs are safe: each finished sensor is saved to
processed/_checkpoints/ straight away, so restarting the SAME command
skips finished sensors instead of starting over. Progress goes to
processed/progress.log.

Privacy: real school names NEVER appear in the outputs. Schools are
renamed School_01, School_02, ... The matching list is saved separately
in processed/_private_do_not_share/school_key.csv — keep it to yourself
and never share/commit it. (All of processed/ is ignored by git.)

Duplicates: if two sensors sit in the same room, the main tables keep
both (one row per sensor). Extra *_deduped.csv files keep only one row
per room-day (the sensor with the most readings that day wins).

Run it:
  source .venv/bin/activate
  python src/preprocess.py                      # full Batch1 (slow!)
  python src/preprocess.py --max-sensors 2 --months 2025-03   # quick test
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Settings you may change
# ----------------------------------------------------------------------
DENVER = "America/Denver"
SCHOOL_START = "08:00"   # daily-summary window (was in the R code)
SCHOOL_END = "14:30"
CO2_MIN, CO2_MAX = 400, 5000        # sane CO2 range in ppm (ACH step)
MIN_EVENT_POINTS = 15               # shortest trusted decay event
MIN_EVENT_R2 = 0.85                 # worst trusted straight-line fit
RANDOM_STATE = 42

# Raw file prefixes we read. Everything else (pm10, ct, input-1) is ignored.
CHANNELS = {
    "co2": "carbon-dioxide-co",
    "pm25": "pm2-5-mass-concentration",
    "temp": "temperature",
    "rh": "humidity",
    "voc": "volatile-organic-compounds-voc",
}
VALUE_COLUMNS = list(CHANNELS)  # co2, pm25, temp, rh, voc


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def log_line(output: Path, message: str) -> None:
    """Append one timestamped line to processed/progress.log."""
    output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output / "progress.log", "a") as handle:
        handle.write(f"{stamp} | {message}\n")


def _clock_to_minutes(clock: str) -> int:
    """'08:00' -> 480. Same window the old code checked as text.

    The old code compared zero-padded "HH:MM" strings, which sorts the
    same as minutes-since-midnight but costs a slow string format per
    row. Integer comparison gives identical masks, much faster.
    """
    hours, mins = clock.split(":")
    return int(hours) * 60 + int(mins)


def room_name(sensor_folder: str) -> str | None:
    """Get a plain room label from a sensor folder name.

    Example: "1086922-c101-cr-159" -> "Room_159". Returns None when the
    folder is not a classroom we want (alex / outside / weather).
    """
    name = sensor_folder.lower()
    if "alex" in name or "outside" in name or "observed-weather" in name:
        return None  # power-tracking or outdoor sensor: skip it
    match = re.search(r"-(?:cr|classroom)-(.+?)-*$", name)
    if not match:
        return None  # not labelled as a classroom: skip it
    room = match.group(1).strip("-").replace("-", "_")
    return f"Room_{room}"


def sensor_id(sensor_folder: str) -> str:
    """First number in the folder name, e.g. "1086922". Not a school name."""
    match = re.match(r"(\d+)-", sensor_folder)
    return match.group(1) if match else sensor_folder


def file_month(filename: str) -> str | None:
    """Pull YYYY-MM out of a raw filename like "...-202503.csv"."""
    match = re.search(r"-(\d{4})(\d{2})\.csv$", filename)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def read_one_file(path: Path) -> pd.DataFrame:
    """Read one raw CSV, keeping only the value + timestamp columns.

    Timestamps arrive as seconds since 1970 (UTC). We convert them to
    Denver local time right away, like the old code did.
    """
    try:
        data = pd.read_csv(
            path, usecols=["channel_converted_value", "channel_last_timestamp"]
        )
    except Exception:
        return pd.DataFrame(columns=["Time", "value"])  # empty/broken file
    if data.empty:
        return pd.DataFrame(columns=["Time", "value"])
    times = pd.to_datetime(data["channel_last_timestamp"], unit="s", utc=True)
    return pd.DataFrame(
        {"Time": times.dt.tz_convert(DENVER).dt.tz_localize(None),
         "value": pd.to_numeric(data["channel_converted_value"], errors="coerce")}
    )


_file_lists: dict[str, list[Path]] = {}


def sensor_files(csv_folder: Path) -> list[Path]:
    """Non-empty CSVs in one sensor folder, sorted by name.

    Listed once per sensor and remembered: the main loop asks again for
    every month, but the folder never changes mid-run, so re-scanning
    the disk each time is pure overhead. Same file set as before.
    """
    key = str(csv_folder)
    if key not in _file_lists:
        _file_lists[key] = sorted(
            (path for path in csv_folder.glob("*.csv") if path.stat().st_size > 0),
            key=lambda path: path.name)
    return _file_lists[key]


def read_sensor(csv_folder: Path, months: set[str] | None = None):
    """Read all channels for one sensor and line them up by minute.

    Returns (minute_table, co2_by_source_month).
    minute_table columns: Time, co2, pm25, temp, rh, voc.
    co2_by_source_month keeps the filename month, because the old ACH
    code worked one filename-month at a time (keeps results the same).
    """
    pieces: dict[str, list[pd.DataFrame]] = {key: [] for key in CHANNELS}
    co2_months: list[pd.DataFrame] = []
    for path in sensor_files(csv_folder):
        if months is not None and file_month(path.name) not in months:
            continue  # quick-test month filter: do not even open the file
        for key, prefix in CHANNELS.items():
            if path.name.startswith(prefix) or (
                key == "pm25" and path.name.startswith("pm-2-5-mass-concentration")
            ):
                frame = read_one_file(path)
                if key == "co2" and not frame.empty:
                    frame = frame.assign(source_month=file_month(path.name))
                    co2_months.append(frame)
                    frame = frame.drop(columns=["source_month"])
                pieces[key].append(frame.rename(columns={"value": key}))
                break
    minute = None
    for key in CHANNELS:
        table = (
            pd.concat(pieces[key], ignore_index=True)
            if pieces[key]
            else pd.DataFrame(columns=["Time", key])
        )
        # Same minute twice (re-download): keep the first reading.
        table = table.drop_duplicates(subset=["Time"]).sort_values("Time")
        minute = table if minute is None else minute.merge(table, on="Time", how="outer")
    minute = minute.sort_values("Time").reset_index(drop=True)
    co2_monthly = (
        pd.concat(co2_months, ignore_index=True)
        if co2_months
        else pd.DataFrame(columns=["Time", "value", "source_month"])
    )
    return minute, co2_monthly


# ----------------------------------------------------------------------
# Step B: one row per classroom-day (was the R script)
# ----------------------------------------------------------------------
def clean_minutes(minute: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible readings, then trim each day's extremes.

    Same rules as the old R code: negative values become blank, and for
    each sensor-day with 20+ readings, the lowest/highest 5% of each
    channel become blank (guards against glitches and spikes).
    """
    data = minute.copy()
    data["Date"] = data["Time"].dt.date
    for col in VALUE_COLUMNS:
        data.loc[data[col] < 0, col] = np.nan
        # Same trim as before, vectorised: per-day 5%/95% bounds and the
        # 20-reading minimum, computed once per channel instead of day by day.
        # groupby.quantile uses the same linear rule, and count skips NaN,
        # so days with fewer than 20 readings keep every value, as before.
        grouped = data.groupby("Date")[col]
        enough = grouped.transform("count") >= 20
        low = data["Date"].map(grouped.quantile(0.05))
        high = data["Date"].map(grouped.quantile(0.95))
        data[col] = data[col].where(~enough | data[col].between(low, high))
    return data


def daily_summary(minute: pd.DataFrame, school: str, room: str, sensor: str) -> pd.DataFrame:
    """Summarise school hours (Mon-Fri 08:00-14:30) into classroom-days.

    "Cumulative" means the readings are ADDED UP (old R behaviour), so
    a day with missing minutes gets a smaller total. minutes_used tells
    you how many PM2.5 readings went into that day's total. VOC keeps
    its school-day peak (VOC_max, in ppb as the sensors report it).
    """
    data = clean_minutes(minute)
    minutes = data["Time"].dt.hour * 60 + data["Time"].dt.minute
    in_school = (
        (data["Time"].dt.weekday < 5)
        & (minutes >= _clock_to_minutes(SCHOOL_START))
        & (minutes <= _clock_to_minutes(SCHOOL_END))
    )
    rows = []
    for date, day in data[in_school].groupby("Date"):
        rows.append({
            "date": date, "school": school, "classroom": room, "sensor_id": sensor,
            "Indoor_CO2_max": day["co2"].max(),
            "Cumulative_Indoor_CO2": day["co2"].sum(min_count=1),
            "Indoor_PM2_5_max": day["pm25"].max(),
            "Cumulative_Indoor_PM2_5": day["pm25"].sum(min_count=1),
            "Temp_avg": day["temp"].mean(),
            "RH_avg": day["rh"].mean(),
            "VOC_max": day["voc"].max(),
            "minutes_used": int(day["pm25"].notna().sum()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Step C: ACH from CO2 decay (was process_data.py + postprocessing)
# Kept as a few small steps so it stays readable. The maths is unchanged.
# ----------------------------------------------------------------------
def find_background(co2: pd.Series) -> pd.Series:
    """Smooth line hugging the LOW CO2 values = empty-room background.

    Uses asymmetric least squares (scipy): points above the line count
    almost nothing, so the line sinks to the empty-room lows.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    y = co2.interpolate(limit_direction="both").to_numpy(dtype=float)
    # Pad both ends so the line behaves at the edges (old code did this).
    y = np.hstack(([y[0]] * 100_000, y, [y[-1]] * 100_000))
    size = len(y)
    diff2 = sparse.diags([np.ones(size), -2 * np.ones(size), np.ones(size)],
                         [0, 1, 2], shape=(size - 2, size))
    weights = np.ones(size)
    line = y
    for _ in range(100):
        w_matrix = sparse.diags(weights)
        line = spsolve(w_matrix + 1e7 * diff2.T @ diff2, weights * y)
        weights = 0.0001 * (y > line) + 0.9999 * (y < line)
    return pd.Series(line[100_000:-100_000], index=co2.index)


def prepare_co2(frame: pd.DataFrame) -> pd.DataFrame:
    """Tidy one month of CO2: sane range, no spikes, background added."""
    from sklearn.neighbors import LocalOutlierFactor

    data = frame.rename(columns={"value": "co2"}).dropna(subset=["co2"])
    data = data[(data["co2"] >= CO2_MIN) & (data["co2"] <= CO2_MAX)]
    data = data.sort_values("Time").reset_index(drop=True)
    if len(data) < 20:
        return data.iloc[0:0]
    # Drop isolated spikes that do not match their 10 nearest neighbours.
    keep = LocalOutlierFactor(n_neighbors=10).fit_predict(
        data[["co2"]].assign(position=np.arange(len(data)))
    ) == 1
    data = data[keep].reset_index(drop=True)
    data["baseline"] = find_background(data["co2"])
    data["above_bg"] = data["co2"] - data["baseline"]  # height above background
    smooth = data["above_bg"].rolling(5, min_periods=1, center=True).mean()
    data["smooth_above"] = smooth
    minutes = data["Time"].diff().dt.total_seconds() / 60
    data["fall_speed"] = np.log(smooth.where(smooth > 0)).diff() / minutes
    data["fall_size"] = data["fall_speed"].abs()
    # Drop infinities (e.g. duplicate timestamps divide by zero minutes),
    # like the old code did before clustering.
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=["fall_speed"]).reset_index(drop=True)


def label_decays(data: pd.DataFrame) -> pd.DataFrame:
    """Split minutes into 'falling CO2' groups (one group = one decay)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

    def scale(frame: pd.DataFrame) -> np.ndarray:
        return MinMaxScaler().fit_transform(QuantileTransformer(
            output_distribution="uniform", n_quantiles=min(1000, len(frame)),
            random_state=RANDOM_STATE).fit_transform(frame))

    if len(data) < 20:
        data["decay_group"] = -1
        return data
    # Cluster 1: is CO2 clearly above background?
    data["elevated"] = KMeans(2, random_state=RANDOM_STATE, n_init=10).fit_predict(
        scale(data[["smooth_above", "fall_size"]]))
    if data.loc[data["elevated"] == 1, "above_bg"].mean() < \
            data.loc[data["elevated"] == 0, "above_bg"].mean():
        data["elevated"] = 1 - data["elevated"]
    # Cluster 2: among elevated minutes, is CO2 falling?
    high = data[data["elevated"] == 1].copy()
    data["falling"] = 0
    if len(high) >= 10:
        high["cluster"] = KMeans(2, random_state=RANDOM_STATE, n_init=10,
                                 max_iter=100000).fit_predict(
            scale(high[["fall_speed", "fall_size"]]))
        falling_cluster = high.groupby("cluster")["fall_speed"].mean().idxmin()
        data.loc[high.index, "falling"] = (high["cluster"] == falling_cluster).astype(int)
    # Group unbroken falling runs, separately for each day.
    data["decay_group"] = -1
    group_number = 0
    for _, day in data.groupby(data["Time"].dt.date):
        runs = (day["falling"] != 1).cumsum()
        for _, run in day[day["falling"] == 1].groupby(runs):
            if len(run) >= 5:  # need at least 5 falling readings in a row
                data.loc[run.index, "decay_group"] = group_number
                group_number += 1
    return data


def measure_decays(data: pd.DataFrame, school: str, room: str, sensor: str) -> pd.DataFrame:
    """Fit each decay group: its slope is the ACH for that event.

    CO2 falls like C = background + (start - background) * e^(-ACH * t).
    Taking logs turns that curve into a straight line whose slope is ACH.
    Only clean fits survive: 15+ points and R-squared >= 0.85.
    """
    from sklearn.linear_model import LinearRegression

    events = []
    for _, event in data[data["decay_group"] >= 0].groupby("decay_group"):
        if len(event) < 3:
            continue
        start, background = event["co2"].iloc[0], event["baseline"].iloc[0]
        if start - background <= 0:
            continue
        ratio = (event["co2"] - background) / (start - background)
        event = event[ratio > 0]
        if len(event) < 3:
            continue
        hours = (event["Time"] - event["Time"].iloc[0]).dt.total_seconds() / 3600
        line = LinearRegression().fit(hours.to_numpy().reshape(-1, 1),
                                      (-np.log((event["co2"] - background)
                                               / (start - background))).to_numpy())
        r2 = line.score(hours.to_numpy().reshape(-1, 1),
                        (-np.log((event["co2"] - background)
                                 / (start - background))).to_numpy())
        if len(event) >= MIN_EVENT_POINTS and r2 >= MIN_EVENT_R2:
            events.append({
                "school": school, "classroom": room, "sensor_id": sensor,
                "time": event["Time"].iloc[len(event) // 2],
                "decay_start": event["Time"].iloc[0],
                "decay_end": event["Time"].iloc[-1],
                "decay_rate": float(line.coef_[0]), "r2": float(r2),
                "num_of_point": int(len(event)),
            })
    return pd.DataFrame(events)


def ach_for_sensor(co2_monthly: pd.DataFrame, school: str, room: str, sensor: str) -> pd.DataFrame:
    """Run the ACH steps for one sensor, one filename-month at a time."""
    all_events = []
    for _, month_frame in co2_monthly.groupby("source_month"):
        ready = prepare_co2(month_frame)
        if ready.empty:
            continue
        all_events.append(measure_decays(label_decays(ready), school, room, sensor))
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if events.empty:
        return events
    # Postprocessing (old rules): school hours only, weekdays, ACH > 0,
    # then drop each room's lowest/highest 5% of events for the month.
    minutes = events["time"].dt.hour * 60 + events["time"].dt.minute
    events = events[(minutes >= _clock_to_minutes("06:00"))
                    & (minutes <= _clock_to_minutes("16:30"))
                    & (events["time"].dt.weekday < 5) & (events["decay_rate"] > 0)]
    kept = []
    for _, room_month in events.groupby(
            [events["time"].dt.to_period("M"), "classroom", "sensor_id"]):
        if len(room_month) >= 2:
            low, high = room_month["decay_rate"].quantile([0.05, 0.95])
            room_month = room_month[room_month["decay_rate"].between(low, high)]
            room_month = room_month.assign(low_q=low, high_q=high)
        kept.append(room_month)
    return pd.concat(kept, ignore_index=True) if kept else events.iloc[0:0]


def daily_ach(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events to one row per classroom-day.

    Keeps the highest event, the median event, AND the mean event, so the
    boxplots can use any of the three without another recalculation.
    """
    if events.empty:
        return pd.DataFrame(columns=["date", "school", "classroom", "sensor_id",
                                      "daily_max_ach", "daily_median_ach",
                                      "daily_mean_ach",
                                      "n_decay_events"])
    events = events.assign(date=events["time"].dt.date)
    return events.groupby(["date", "school", "classroom", "sensor_id"]).agg(
        daily_max_ach=("decay_rate", "max"),
        daily_median_ach=("decay_rate", "median"),
        daily_mean_ach=("decay_rate", "mean"),
        n_decay_events=("decay_rate", "size")).reset_index()


# ----------------------------------------------------------------------
# Checkpoints, duplicates, and the main driver
# ----------------------------------------------------------------------
def checkpoint_name(school: str, room: str, sensor: str, kind: str, month: str) -> str:
    """Safe filename for one sensor-month of saved work.

    Example: School_01__Room_159__1086922__2025-03__daily.csv.
    Splitting by month means a December-only run and a full run simply
    fill in each other's gaps — rerunning never repeats finished months.
    (Daily trims are per calendar day, so month-by-month daily results
    are exactly the same as computing the whole history at once.)
    """
    safe = f"{school}__{room}__{sensor}__{month}".replace("/", "_").replace(" ", "_")
    return f"{safe}__{kind}.csv"


def months_done(checkpoint_dir: Path, school: str, room: str, sensor: str) -> set[str]:
    """Which filename-months already have saved daily results for this sensor."""
    prefix = f"{school}__{room}__{sensor}__".replace("/", "_").replace(" ", "_")
    return {path.name[len(prefix):-len("__daily.csv")]
            for path in checkpoint_dir.glob(f"{prefix}*__daily.csv")}


def sensor_months(csv_folder: Path, wanted: set[str] | None) -> list[str]:
    """All filename-months present in one sensor's folder (or the wanted subset)."""
    found = set()
    for path in csv_folder.glob("*.csv"):
        month = file_month(path.name)
        if month and (wanted is None or month in wanted):
            found.add(month)
    return sorted(found)


def save_sensor_work(checkpoint_dir: Path, school: str, room: str, sensor: str,
                     month: str,
                     daily_frame: pd.DataFrame, events_frame: pd.DataFrame) -> None:
    """Save one sensor-month's results straight away (never lose finished work)."""
    daily_frame.to_csv(
        checkpoint_dir / checkpoint_name(school, room, sensor, "daily", month), index=False)
    if not events_frame.empty:
        events_frame.to_csv(
            checkpoint_dir / checkpoint_name(school, room, sensor, "events", month), index=False)


def combine_checkpoints(checkpoint_dir: Path, kind: str) -> pd.DataFrame:
    """Read every saved per-sensor file of one kind into one table."""
    frames = []
    for path in sorted(checkpoint_dir.glob(f"*__{kind}.csv")):
        try:
            frames.append(pd.read_csv(path, parse_dates=["time", "decay_start", "decay_end"]
                                      if kind == "events" else ["date"]))
        except Exception:
            continue  # a half-written file: the sensor will redo it next run
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dedupe(daily: pd.DataFrame, value_hint: str) -> pd.DataFrame:
    """One row per room-day: keep the sensor with the most readings.

    Uses minutes_used when present (daily summary), else event count.
    The full table is always saved too — this only ADDS a deduped copy.
    """
    if daily.empty or "sensor_id" not in daily:
        return daily
    rank = "minutes_used" if value_hint == "minutes" else "n_decay_events"
    if rank not in daily.columns:
        return daily
    ranked = daily.sort_values(rank, ascending=False)
    return ranked.drop_duplicates(subset=["date", "school", "classroom"],
                                  keep="first").sort_values(
                                      ["school", "classroom", "date"]).reset_index(drop=True)


def find_sensors(batch_folder: Path):
    """List (csv_folder, fake_school, room, sensor) for kept classrooms."""
    found = []
    real_schools = sorted(p.name for p in batch_folder.iterdir() if p.is_dir())
    fake_names = {real: f"School_{i:02d}" for i, real in enumerate(real_schools, 1)}
    for real in real_schools:
        for folder in sorted((batch_folder / real).iterdir()):
            if not folder.is_dir():
                continue
            room = room_name(folder.name)
            if room is None:
                continue  # alex / outside / non-classroom: skip
            found.append((folder / "csv", fake_names[real], room, sensor_id(folder.name)))
    return found, fake_names


def run_sensor(job: tuple) -> int:
    """Run one sensor's remaining months; safe in a worker process.

    Each sensor-month writes its OWN checkpoint files, so parallel
    sensors never share an output file and cannot overwrite each other.
    Same functions, same fixed random seed, same log lines as the
    serial run — only the wall time shrinks.
    """
    (csv_folder, school, room, sensor, number, total, todo,
     output, skip_ach) = job
    checkpoint_dir = output / "_checkpoints"
    new_count = 0
    for month in todo:
        try:
            minute, co2_monthly = read_sensor(csv_folder, months={month})
            daily_frame = (daily_summary(minute, school, room, sensor)
                           if not minute.empty else pd.DataFrame())
            events_frame = (ach_for_sensor(co2_monthly, school, room, sensor)
                            if (not skip_ach and not co2_monthly.empty)
                            else pd.DataFrame())
        except Exception as exc:  # one bad month must not kill the run
            log_line(output, f"month failed: {school} {room} {sensor} {month}: {exc}")
            print(f"  skipped (read error): {school} {room} {month}")
            continue
        save_sensor_work(checkpoint_dir, school, room, sensor, month,
                         daily_frame, events_frame)
        new_count += 1
        log_line(output,
                 f"sensor {number}/{total} month {month} saved: "
                 f"{school} {room} {sensor} "
                 f"({len(daily_frame)} days, {len(events_frame)} ACH events)")
        print(f"[{number}/{total}] {school} {room} {month}: "
              f"{len(daily_frame)} days, {len(events_frame)} ACH events")
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch1 preprocessing (one file).")
    parser.add_argument("--input", type=Path,
                        default=Path.home() / "Desktop/AdamData/Batch1",
                        help="Batch1 folder (default: ~/Desktop/AdamData/Batch1)")
    parser.add_argument("--output", type=Path, default=Path("processed"),
                        help="Where to write results (default: processed/)")
    parser.add_argument("--months", default=None,
                        help="Only these filename months, e.g. 2025-03,2025-04")
    parser.add_argument("--schools", default=None,
                        help="Only these Batch1 school folders (substring match, "
                             "e.g. c101,c103). Real names stay out of outputs.")
    parser.add_argument("--max-sensors", type=int, default=None,
                        help="Stop after N sensors (for a quick test)")
    parser.add_argument("--skip-ach", action="store_true",
                        help="Only make daily summaries (much faster test)")
    parser.add_argument("--identified-output", type=Path, default=None,
                        help="Also write school-named copies here for personal use "
                             "only (never share; git already ignores it)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Sensors to run at once (same math, faster wall time). "
                             "Try 4 on this Mac; 1 means today's serial run.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    all_sensors, fake_names = find_sensors(args.input)
    # --schools filters by REAL folder name, but only to choose work:
    # outputs still use only the School_XX names.
    real_of = {fake: real for real, fake in fake_names.items()}
    sensors = [
        entry for entry in all_sensors
        if args.schools is None
        or any(part in real_of[entry[1]].lower() for part in args.schools.lower().split(","))
    ]
    if args.max_sensors:
        sensors = sensors[: args.max_sensors]
    log_line(args.output, f"start: {len(sensors)} sensors, months={args.months or 'all'}")
    print(f"Found {len(sensors)} classroom sensors. Starting...")

    # Private school key: the ONLY place real names appear. Never share it.
    key_folder = args.output / "_private_do_not_share"
    key_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"school_private_name": fake, "batch1_folder_name": real}
         for real, fake in sorted(fake_names.items(), key=lambda item: item[1])]
    ).to_csv(key_folder / "school_key.csv", index=False)
    print(f"School key saved to {key_folder / 'school_key.csv'} (keep private!)")

    wanted = set(args.months.split(",")) if args.months else None
    jobs = []
    done_count = 0
    for number, (csv_folder, school, room, sensor) in enumerate(sensors, 1):
        if not csv_folder.exists():
            continue
        # Work month by month; finished months are skipped automatically.
        todo = [month for month in sensor_months(csv_folder, wanted)
                if month not in months_done(checkpoint_dir, school, room, sensor)]
        if not todo:
            done_count += 1
            continue
        jobs.append((csv_folder, school, room, sensor, number, len(sensors),
                     todo, args.output, args.skip_ach))
    # One worker = today's serial run. More workers split sensors across
    # processes; checkpoints and the combine step below are unchanged.
    new_count = 0
    if args.workers <= 1:
        for job in jobs:
            new_count += run_sensor(job)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers) as pool:
            for saved in pool.map(run_sensor, jobs):
                new_count += saved

    # Combine everything saved so far (this run + earlier runs) into final tables.
    daily = combine_checkpoints(checkpoint_dir, "daily")
    events = combine_checkpoints(checkpoint_dir, "events")
    # Checkpoint round-trips change the date type, so normalise to plain
    # YYYY-MM-DD text on both sides before joining them.
    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    ach_day = daily_ach(events)
    if "date" in ach_day.columns:
        ach_day["date"] = pd.to_datetime(ach_day["date"]).dt.strftime("%Y-%m-%d")

    daily_deduped = dedupe(daily, "minutes")
    daily.to_csv(args.output / "daily_summary.csv", index=False)
    daily_deduped.to_csv(args.output / "daily_summary_deduped.csv", index=False)
    events.to_csv(args.output / "ach_events.csv", index=False)
    ach_day.to_csv(args.output / "ach_daily.csv", index=False)

    # Step D: join daily air quality + daily ventilation (no PAC yet).
    if not daily.empty and not ach_day.empty:
        merged = daily.merge(ach_day, on=["date", "school", "classroom", "sensor_id"],
                             how="left")
    else:
        merged = daily
    merged_deduped = dedupe(merged, "minutes")
    merged.to_csv(args.output / "merged_daily.csv", index=False)
    merged_deduped.to_csv(args.output / "merged_daily_deduped.csv", index=False)

    # Personal identified copy: same tables with real school names.
    # The anonymized tables above are untouched; this folder is only for you.
    if args.identified_output is not None:
        names = pd.read_csv(key_folder / "school_key.csv").set_index(
            "school_private_name")["batch1_folder_name"]
        args.identified_output.mkdir(parents=True, exist_ok=True)
        for frame, name in [(daily, "daily_summary.csv"),
                            (daily_deduped, "daily_summary_deduped.csv"),
                            (events, "ach_events.csv"),
                            (ach_day, "ach_daily.csv"),
                            (merged, "merged_daily.csv"),
                            (merged_deduped, "merged_daily_deduped.csv")]:
            identified = frame.copy()
            if "school" in identified.columns:
                identified["school"] = identified["school"].map(names).fillna(
                    identified["school"])
            identified.to_csv(args.identified_output / name, index=False)
        log_line(args.output, f"identified copy written to {args.identified_output}")
    log_line(args.output,
             f"combine: {done_count} sensor-months already done, "
             f"{new_count} newly saved, "
             f"total {len(daily)} daily rows and {len(events)} ACH events")
    print(f"Done. {done_count} sensor-months already finished, {new_count} new. "
          f"Total: {len(daily)} daily rows and {len(events)} ACH events.")


if __name__ == "__main__":
    main()
