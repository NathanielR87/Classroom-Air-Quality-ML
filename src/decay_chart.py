"""
decay_chart.py — one day of CO2 with the kept decay events highlighted,
matching slide 11 of PAC_Pipeline_Walkthrough.pdf ("A whole day in one
classroom"): grey = readings not used for ACH, orange = readings inside
a kept decay event, dark line = empty-room baseline. Each event is
labelled with its time range, ACH, R² and reading count.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (ach_for_sensor, label_decays, measure_decays,
                        prepare_co2, read_sensor, room_name, sensor_id)


def find_sensor_folder(batch1: Path, real_school: str,
                       room: str, sensor: str) -> Path | None:
    """The raw csv/ folder for one classroom sensor, or None."""
    school_dir = batch1 / real_school
    if not school_dir.is_dir():
        return None
    for folder in sorted(school_dir.iterdir()):
        if (folder.is_dir() and sensor_id(folder.name) == sensor
                and room_name(folder.name) == room):
            csv_folder = folder / "csv"
            if csv_folder.is_dir():
                return csv_folder
    return None


def day_events(batch1: Path, real_school: str, room: str,
               sensor: str, day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minute CO2 (with baseline) and kept ACH events for one room-day."""
    csv_folder = find_sensor_folder(batch1, real_school, room, sensor)
    if csv_folder is None:
        return pd.DataFrame(), pd.DataFrame()
    _, co2_monthly = read_sensor(csv_folder, months={day.strftime("%Y-%m")})
    if co2_monthly.empty:
        return pd.DataFrame(), pd.DataFrame()
    month = co2_monthly[co2_monthly["Time"].dt.date == day]
    if month.empty:
        return pd.DataFrame(), pd.DataFrame()
    ready = prepare_co2(month)
    if ready.empty:
        return pd.DataFrame(), pd.DataFrame()
    events = measure_decays(label_decays(ready), "", "", "")
    events = events[events["time"].dt.date == day].reset_index(drop=True)
    return ready, events


def decay_figure(minutes: pd.DataFrame, events: pd.DataFrame,
                 title: str):
    """Slide-11 style figure: full-day CO2, baseline, kept events in orange."""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(minutes["Time"], minutes["co2"], color="#9e9e9e",
              linewidth=0.8, label="CO2 reading (not used for ACH)")
    axis.plot(minutes["Time"], minutes["baseline"], color="#333333",
              linewidth=1.2, label="Background (baseline)")
    for i, (_, event) in enumerate(events.iterrows()):
        span = minutes[(minutes["Time"] >= event["decay_start"])
                       & (minutes["Time"] <= event["decay_end"])]
        axis.plot(span["Time"], span["co2"], color="#ef6c00", linewidth=2.0)
        label = (f"{event['decay_start'].strftime('%H:%M')}–"
                 f"{event['decay_end'].strftime('%H:%M')}\n"
                 f"ACH {event['decay_rate']:.2f} · R² {event['r2']:.2f} "
                 f"· {event['num_of_point']} readings")
        # Alternate label heights so neighbouring events do not overlap.
        axis.annotate(label, xy=(span["Time"].iloc[len(span) // 2],
                                 span["co2"].max()),
                      xytext=(0, 14 if i % 2 == 0 else 34),
                      textcoords="offset points",
                      fontsize=7, ha="center", va="bottom",
                      bbox={"facecolor": "white", "edgecolor": "#ef6c00",
                            "pad": 2.0},
                      arrowprops={"arrowstyle": "-", "color": "#ef6c00",
                                  "linewidth": 0.8})
    axis.plot([], [], color="#ef6c00", linewidth=2.0,
              label="Used to calculate ACH (inside a kept decay event)")
    axis.set_xlabel("Time")
    axis.set_ylabel("CO2 (ppm)")
    axis.set_title(title)
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig
