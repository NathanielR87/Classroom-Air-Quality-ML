"""
principal_report.py — a plain-language principal report for one school-month.

Every number in the prompt comes from our verified tables (ratings,
Kruskal verdicts, model report). The model only turns those numbers
into readable prose; --dry-run prints and saves the exact prompt with
no network call, so you can review it before any key is used.

Run it:
  source .venv/bin/activate
  python src/principal_report.py --school School_01 --month 2025-12 --dry-run
  META_MODEL_API_KEY="..." python src/principal_report.py \
      --school School_01 --month 2025-12        # live call (Muse Spark)

Live output (git-ignored): principal_report_<school>_<month>.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ratings import load_bands, rate_frame

MODEL = "muse-spark-1.3"  # cheap text model, right-sized for reports


def school_month_context(processed: Path, school: str, month: str) -> str:
    """Assemble the grounded facts: ratings, trial verdict, model score."""
    bands = load_bands()
    merged = pd.read_csv(processed / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    days = rate_frame(merged, bands)
    days = days[(days["school"] == school)
                & (days["date"].str.startswith(month))]
    if days.empty:
        raise SystemExit(f"No classroom-days for {school} {month}.")

    lines = [f"School {school}, {month}: {len(days)} classroom-days, "
             f"{days['classroom'].nunique()} rooms.",
             f"Overall day ratings: {days['worst_rating'].value_counts().to_dict()}."]
    for row in bands.itertuples():
        ratings = days[f"{row.metric}_rating"].value_counts().to_dict()
        lines.append(f"{row.label} ({row.unit}): {ratings}.")
        poor_tip = days.loc[days[f"{row.metric}_rating"] == "Poor",
                            f"{row.metric}_advice"].mode()
        if not poor_tip.empty and poor_tip.iloc[0]:
            lines.append(f"  Top {row.label} recommendation: {poor_tip.iloc[0]}")
    kruskal = pd.read_csv(processed / "figures" / "ach_boxplots"
                          / "kruskal_results.csv")
    hit = kruskal[kruskal["school"] == school]
    if not hit.empty:
        row = hit.iloc[0]
        verdict = ("no significant OFF-vs-ON difference"
                   if not row["significant"] else "a significant difference")
        lines.append(f"Ventilation check (Kruskal-Wallis on trial days): "
                     f"H={row['H']:.2f}, p={row['p']:.3f} — {verdict}.")
    report = (processed / "pac_model_report.txt").read_text()
    supervised = [line for line in report.splitlines()
                  if line.startswith("Model B (supervised)   test:")]
    if supervised:
        lines.append(f"PAC prediction honesty check: {supervised[0].strip()}.")
    return "\n".join(lines)


def build_prompt(facts: str, school: str, month: str) -> str:
    """The exact prompt: plain words, numbers only from the facts block."""
    return f"""You write a short plain-language report for an elementary
school principal about classroom air quality. Use ONLY the numbers in
FACTS below — do not invent, estimate, or round beyond what is shown.
Keep it under 300 words with three sections: Overall, What stands out,
Recommended next steps.

School: {school}. Month: {month}.

FACTS:
{facts}
"""


def call_spark(prompt: str) -> str:
    """One Muse Spark call through the Meta Model API (needs the key)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("The 'openai' package is not installed — ask "
                         "before installing anything, then rerun.")
    key = os.environ.get("META_MODEL_API_KEY", "")
    if not key:
        raise SystemExit("META_MODEL_API_KEY is not set in this session.")
    client = OpenAI(api_key=key,
                    base_url="https://api.meta.ai/v1")
    reply = client.chat.completions.create(
        model=MODEL, temperature=0.0,
        messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content


def main() -> None:
    """Assemble facts; dry-run saves the prompt, live calls the model."""
    parser = argparse.ArgumentParser(description="Principal report.")
    parser.add_argument("--school", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--input", type=Path, default=Path("processed"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print + save the prompt, call nothing.")
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    facts = school_month_context(args.input, args.school, args.month)
    prompt = build_prompt(facts, args.school, args.month)
    stem = f"principal_report_{args.school}_{args.month}"
    (output / f"{stem}.prompt.txt").write_text(prompt)
    if args.dry_run:
        print(prompt)
        print(f"[dry-run] Saved prompt to {output / f'{stem}.prompt.txt'}; "
              f"no network call made.")
        return
    report = call_spark(prompt)
    (output / f"{stem}.md").write_text(report)
    print(report)
    print(f"Wrote {stem}.md to {output}/")


if __name__ == "__main__":
    main()
