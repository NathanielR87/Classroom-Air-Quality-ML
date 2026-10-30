# AGENTS.md

Muse Code reads this file as project rules when it runs in this directory.

## Project
- Name: Adam data
- Meta hackathon project: I already have working code that preprocesses indoor air quality data and predicts whether a classroom's HEPA PAC (portable air cleaner) is ON or OFF. The goal is to understand, clean up, and improve that pipeline: make preprocessing easier to run on large datasets, and improve the ML prediction.
- Deadline: November 4 2026

## Folders
- Batch1/ — RAW school air quality data, one subfolder per school. READ ONLY. Local copy has 7 schools (anonymized School_01–School_07; the private folder-name mapping lives in processed/_private_do_not_share/school_key.csv and is never committed), 85 sensor/room folders, ~11k CSVs (~28 GB).
- Nates_copy_CDC_code/ — my existing preprocessing + prediction code. This is the starting point do not change and just make copy if you are going to make any changes.
- PAC_Pipeline_Walkthrough.pdf — slides explaining what each script does and the order they run in. Read this first. NOTE: the PDF is at ~/Desktop/AdamData/PAC_Pipeline_Walkthrough.pdf, not inside Nates_copy_CDC_code/.
- processed/ — cleaned data output goes here (create it if needed; does not exist yet)
- src/ — new or refactored code goes here (exists; currently only src/README.md)

## Pipeline order (from README.ipynb + PAC_Pipeline_Walkthrough.pdf)
1. IAQ daily summaries (R): Nates_copy_CDC_code/IAQ_meta_processing_sep25.R (parallel variant: IAQ_meta_processing_sep25_parra.R)
2. ACH step A — monthly per-classroom CO2+PM files (Python): rawsensor_preprocess_mschools_batch.ipynb (older: rawsensor_preprocess_mschools.ipynb; backfill variant: rawsensor_preprocess_mschools_batch_backfill.ipynb)
3. ACH step B — CO2-decay ACH per classroom (Python): loadbalancer_workflow/process_data.py via SLURM (local/notebook twins: ACH_mschools_monthly.ipynb, ACH_mschools_monthly_rerun.ipynb, ACH_safeguards.ipynb, ACH_mschools_test_raw.py)
4. ACH step C — postprocess to daily-max ACH (Python): ACH_mschools_monthly_postprocessing.ipynb
5. Merge IAQ + ACH + PAC-assignment list (Python): ACH_IAQ_List_Merge.ipynb
6. PAC ON/OFF estimate: [TODO — user to confirm canonical file; see "Only you can answer" in chat 2026-10-27. Candidates seen: Untitled.ipynb (R, tidyLPA estimate_profiles, cited in PDF but NOT in this copy), ACH_Distributino.ipynb Script 3 (supervised PAC on/off trial days), pac_sample.py (cited in PDF, NOT in this copy)]

## Stack
- Language: Python 3, R
- Python packages used by existing code: pandas, numpy, scikit-learn (LinearRegression, KMeans, DBSCAN, LocalOutlierFactor, scalers), scipy (sparse), matplotlib, seaborn (ACH_Distributino.ipynb only)
- R packages used by existing code: pacman, dplyr, lubridate, readr, purrr, tibble, data.table; parallel variant adds future, future.apply
- Compute: original paths are CU Research Computing PetaLibrary (/pl/active/LAB/...) + SLURM Load Balancer (loadbalancer_workflow/run_processing.sh). [TODO — user: will refactored code run locally on this Mac, on the HPC, or both?]
- Local machine today (2026-10-27): /usr/bin/python3 is 3.9.6 with only pip/setuptools (no pandas); R/Rscript not installed. Do not install without asking (see Rules).

## Hackathon decisions — FINAL for Stage A scope (accepted 2026-10-27)
- Status: Final for the Stage A scope contract only. Stage C is a deferred proposal and returns for its own interview. Ending this interview does not authorize implementation; building starts only on a separate user "go."
- Settled by user 2026-10-27: (1) Missing files (iaq_daily_all_export.csv, Elementary School List_PACnoPAC_List.xlsx, Untitled.ipynb, pac_sample.py) do not exist and cannot be obtained — do not block on them. (2) PAC prediction is deferred; add later with an equation (no LPA rebuild now). (3) Ignore every `alex` sensor (power tracking only) and its ct/ct-bridge files. (4) Install Python packages + R locally on this Mac and run tests here. Python done: .venv with pandas 2.3.3 / numpy 2.0.2 / scikit-learn 1.6.1 / scipy 1.13.1. R install still pending. (5) Scope is Batch1 only (7 schools). (6) PDF review questions were for understanding only — not decisions to make now. (7) School names must be removed from copied/output data (School_01, School_02, ...) with a separate private mapping table for the user's own use; duplicates stay in the main table plus a separate deduped table.
- In progress when paused: src/preprocess.py (single Python pipeline draft) exists; its small test run was stopped by the user — do not restart it unless asked.
- Settled by user 2026-10-27 (grill Q1): priority is A first, then C — i.e. (A) cleaner preprocessing that actually runs, then (C) better PAC prediction. User states they already have data for days with PAC on vs off; that will be the learning (training/validation) data for C. Note: the PAC on/off days found in the repo so far are the 6-school trial schedule (per-school dates in src/trial_labels.csv) — to confirm whether that is the full learning set.
- Settled by user 2026-10-27 (grill Q2 = A): Stage A is done only when one Python file runs end-to-end on this laptop for all of Batch1 (7 schools), writes the anonymized daily + ACH + merged tables, and checked classroom-days still agree with the original pipeline's math (e.g. daily PM2.5 total, an ACH event). Any intentional number change must be written down with its reason before Stage C uses the tables.
- Settled by user 2026-10-27 (grill Q3 = Option 1, with a tweak): keep the original ACH math, save each sensor-month as soon as it finishes (processed/_checkpoints/), and skip finished sensor-months on restart, so a long run never starts over. Narrow month runs (e.g. trial months) and full runs fill in each other's gaps. User tweak to decide: run this with an agent working 24/7, if possible and if it makes sense.
- Settled by user 2026-10-27 (grill Q4 = Option 1 + weekly check): Stage A runs as an unattended resumable background job on this Mac (plugged in, sleep pauses it), not an always-on agent. A weekly check reads the progress log and reports status instead of babysitting the run.
- Settled by user 2026-10-27 (grill Q5 = Option 1): the weekly check reports sensors finished of total, new outputs, log errors, and rough remaining time. It may restart ONLY a dead pipeline with the same approved command (finished sensors are skipped automatically). It never deletes outputs, edits code, or starts a different job. Cadence: every 7 days from Stage A start (no weekday named).
- Scope contract (accepted 2026-10-27): goal, boundary, done checklist, staged design, and execution rules are as stated in the chat message of 2026-10-27. Stage A is approved to be built when the user says "go"; Stage C stays deferred with its own interview.
- Stage C contract (Final, accepted 2026-10-28): two-model comparison — (A) unsupervised, given no ON/OFF day labels, vs (B) supervised, trained on the user's known HEPA on/off days. No school data serves as ON/OFF examples until the user supplies labels after preprocessing is optimized. Input contract: current merged_daily.csv columns as-is; labels join on date + school + classroom. Scoring is relative: supervised must clearly beat unsupervised AND a naive majority-guess baseline, no fixed threshold. Prediction method stays editable later without re-grilling while the comparison and scoring rule hold. Labels supplied 2026-10-28 (user-provided R tibble, translated): one OFF day + one ON day each for School_01, School_05, School_06, School_07 (Dec 2025) and School_03, School_04 (Feb 2026); see src/trial_labels.csv. Still pending: whether held-out validation splits by school, by day, or both.
- Stage A amendment (Final, accepted 2026-10-28): pipeline also writes a second identified copy of processed data with real school names, for personal use only. Never committed or uploaded; anonymized tables remain the only shareable output.

## Rules
- I'm learning: explain each change in 1-2 sentences before making it
- NEVER edit, move, or delete anything in Batch1/
- Read my existing code and the PDF before writing new code; reuse my logic where it works
- Don't edit files in Nates_copy_CDC_code/ in place; put improved versions in src/
- If a change would alter the model's results, tell me what changes and why before doing it
- Keep functions small and commented
- Don't install new packages without asking me first
- Don't commit data files to git
- All new code goes in src/. Nothing outside src/ gets committed to git.
- Raw data and my original code are NOT in git. Always read them by full path:
  ~/Desktop/AdamData/Batch1/ and ~/Desktop/AdamData/Nates_copy_CDC_code/
- Never upload or commit CSV or data files.

## Common Commands
- Original code hard-codes HPC paths (/pl/active/LAB/...); edit the CONFIG/path block at the top of each file before running. There is no single CLI entry point yet.
- Step 1 (R, daily summaries): open Nates_copy_CDC_code/IAQ_meta_processing_sep25.R, set main_school_dir + raw_out_dir, then run the whole script in RStudio/Rscript. Outputs: per_school_raw/, per_school_clean/, per_school_daily_summary/, daily_summary_all/iaq_daily_summary_all.csv (+ .rds twins).
- Step 2 (ACH input prep): run Nates_copy_CDC_code/rawsensor_preprocess_mschools_batch.ipynb; set base_input_path, base_output_path, START_YYYYMM, END_YYYYMM. Output: <out>/YYYY-MM/<school>/<classroom>.csv with columns Time, co2, pm25.
- Step 3 (ACH, HPC only as written): cd Nates_copy_CDC_code/loadbalancer_workflow/; edit input_base/output_base in process_data.py and the active `source ./create_lb_cmd_file.sh ...` line + #SBATCH --time in run_processing.sh; then `sbatch run_processing.sh`. Single-unit form is `python process_data.py "<YYYY-MM>" "<school-folder>"`. Output: <out>/YYYY-MM/<school>/<classroom>_ACH.csv (one row per kept decay event).
- Step 4 (ACH postprocess): run Nates_copy_CDC_code/ACH_mschools_monthly_postprocessing.ipynb (set INPUT_BASES/OUTPUT_BASE). Outputs per month: ACH_postprocessed_combined.csv, ACH_postprocessed_daily_max.csv, ACH_postprocessed_log.csv; cell 2 also builds ALL_MONTHS_*.csv.
- Step 5 (merge): run Nates_copy_CDC_code/ACH_IAQ_List_Merge.ipynb (set ACH_POSTPROCESSED_BASE, TARGET_MONTHS, IAQ_PATH, SCHOOL_LIST_PATH). [TODO — user: local locations of iaq_daily_all_export.csv and "Elementary School List_PACnoPAC_List.xlsx"; neither is in this workspace.]
- Step 6 (PAC ON/OFF): [TODO — user to confirm canonical command/file; not runnable from this copy alone.]
- Helpers/audits (not in main order): test_run.ipynb, Batch21_data_inventory.ipynb, Batch21_month_grid.ipynb (coverage audits), dailySumCode.ipynb (combine daily summaries), ACH_Distributino.ipynb (PM-vs-ACH plots + PAC trial-day plots).

## Project Layout
- Batch1/<school>/<sensor-folder>/csv/*.csv — raw Attune exports; sensor-folder names look like 1086922-c101-cr-159 (cr/classroom/alex variants; *-outside and observed-weather are non-classroom). READ ONLY.
- Nates_copy_CDC_code/ — original notebooks/scripts (do not edit in place). Key files: IAQ_meta_processing_sep25.R (+ _parra.R), rawsensor_preprocess_mschools_batch.ipynb, loadbalancer_workflow/process_data.py, ACH_mschools_monthly_postprocessing.ipynb, ACH_IAQ_List_Merge.ipynb, README.ipynb (run instructions), ACH_Distributino.ipynb (exploration + PAC trial plots).
- src/ — all new/refactored code (only src/README.md so far).
- processed/ — [planned] cleaned outputs; create when first refactored step runs.
- Git: only .gitignore, AGENTS.md, README.md, src/ are tracked (.gitignore ignores everything else, plus *.csv/*.xlsx everywhere). Raw data + Nates_copy_CDC_code/ are untracked by design.