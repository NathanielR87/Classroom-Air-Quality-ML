# HACKATHON_NEW.md — what was built during the hackathon (Oct–Nov 2026)

Judges: everything under "New" below was written during the event.
Everything under "Pre-existing" was the starting point and is NOT
claimed as hackathon work.

## New (hackathon-built, in `src/`)
- `preprocess.py` — one-file Python pipeline replacing 3 old steps
  (R daily summaries + monthly CO2/PM prep + SLURM ACH). Resumable
  per-sensor-month checkpoints, anonymized School_XX outputs, deduped
  copies, private school key. Daily table holds peaks AND school-day
  means (CO2, PM2.5, VOC) plus Temp/RH averages. ACH retuned during
  the event: 15+ points (was 10), R² ≥ 0.85 (was 0.9), CO2 400–5000 ppm
  (was 100–5000), plus a new `daily_mean_ach` column.
- `ratings.py` + `rating_bands.csv` — Good/Fair/Poor classroom ratings
  from web-researched bands (ASHRAE/WHO/EPA, noted per metric), editable
  without touching code. Peak / Typical / Both views, one overall
  rating per view, advice attached to every weak metric.
- `predict_pac.py` — supervised logistic PAC ON/OFF equation vs
  unsupervised KMeans vs majority baseline on 125 labeled
  classroom-days from the known trial ON/OFF days, graded on
  held-out schools. Labels read from anonymized
  `trial_labels.csv` (no real school names in tracked code).
- `train_compare.py` — leave-one-school-out rotation vs random-20
  iterative training, compared on the same labels; plus the
  extra-feature trial (base 6 won; extras overfit).
- `ach_boxplots.py` — per-school trial OFF-vs-ON boxplots (median + mean
  views) with Kruskal-Wallis verdicts annotated on each graph.
- `decay_chart.py` — slide-11 style one-day CO2 chart (grey readings,
  baseline, orange kept events with ACH/R²/counts), shown in the
  dashboard under the ACH ratings.
- `monthly_grids.py`, `dashboard.py` — condensed ExampleGraph-style
  monthly PAC grids; Streamlit district dashboard (drop in another
  district's processed folder): clickable school map (real floor plan
  where one exists, schematic with filler no-monitor rooms otherwise),
  weekday calendar day picker, peak/typical/both ratings +
  recommendations, slide-11 style ACH decay chart, condensed
  predictions, model report with a plain-language explainer.
- `principal_report.py` — grounded Muse Spark principal reports
  (live Meta Model API call verified: principal letter generated
  for School_01 2025-12 from the rated tables; the trial-verdict and
  honesty-check lines degrade gracefully for fresh single-school
  outputs that lack them).
- Dashboard "Engineer run": sidebar runs `preprocess.py` on a raw
  folder into `processed/runs/<name>` (months/school filters, log
  tail, auto-switch), then a "Write principal letter" button calls
  Muse Spark in-process. The Meta key comes from a local `.env`
  file (never shown, saved, or committed); verified headless for
  School_06 December-only (253 days, 11 rooms, all dots placed) and
  the full School_06 history in `processed/runs/eng_school06/`
  (2,935 classroom-days, 2024-10 to 2026-05, same 11 rooms backed by
  the real floor plan).
- `trial_labels.csv`, `rating_bands.csv` — anonymized configs.
- `school_maps.py` — real floor plan for School_06 shown with all
  identifying title-block text painted out (PDF itself never committed);
  clickable Plotly dots per monitored room, colored by overall rating;
  clicking a dot opens a modal popup with that room's full status.
  Overall rating now counts Poor metrics: 1 = Watch (yellow), 2 = High
  (orange), 3+ = Poor (red).
- This file.

## Pre-existing (starting point, NOT hackathon work)
- `Batch1/` raw sensor exports (read-only data).
- `Nates_copy_CDC_code/` — the original R/notebook/SLURM pipeline.
- `PAC_Pipeline_Walkthrough.pdf` — slides describing the old pipeline.
- The 12 known PAC trial ON/OFF days (user-supplied).
