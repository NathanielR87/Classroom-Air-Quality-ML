# HACKATHON_NEW.md — what was built during the hackathon (Oct–Nov 2026)

Judges: everything under "New" below was written during the event.
Everything under "Pre-existing" was the starting point and is NOT
claimed as hackathon work.

## New (hackathon-built, in `src/`)
- `preprocess.py` — one-file Python pipeline replacing 3 old steps
  (R daily summaries + monthly CO2/PM prep + SLURM ACH). Resumable
  per-sensor-month checkpoints, anonymized School_XX outputs, deduped
  copies, private school key. ACH retuned during the event: 15+ points
  (was 10), R² ≥ 0.85 (was 0.9), CO2 400–5000 ppm (was 100–5000), plus a
  new `daily_mean_ach` column.
- `predict_pac.py` — supervised logistic PAC ON/OFF equation vs
  unsupervised KMeans vs majority baseline on the 12 known trial days,
  graded on held-out schools.
- `train_compare.py` — leave-one-school-out rotation vs random-20
  iterative training, compared on the same labels; plus the
  extra-feature trial (base 6 won; extras overfit).
- `ach_boxplots.py` — per-school trial OFF-vs-ON boxplots (median + mean
  views) with Kruskal-Wallis verdicts annotated on each graph.
- `monthly_grids.py`, `dashboard.py` — ExampleGraph-style monthly PAC
  grids and the Streamlit dashboard shell.
- `trial_labels.csv` — anonymized trial OFF/ON dates per School_XX.
- This file.

## Pre-existing (starting point, NOT hackathon work)
- `Batch1/` raw sensor exports (read-only data).
- `Nates_copy_CDC_code/` — the original R/notebook/SLURM pipeline.
- `PAC_Pipeline_Walkthrough.pdf` — slides describing the old pipeline.
- The 12 known PAC trial ON/OFF days (user-supplied).
