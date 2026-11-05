# Classroom Air Quality ML

Predict whether a classroom's portable air cleaner (HEPA PAC) is ON or OFF
from indoor air quality sensors, and show the result on a dashboard school
districts can run themselves.

**Meta Hackathon (Nov 2026), Build for Your World.** Everything in `src/`
was built during the event. [HACKATHON_NEW.md](HACKATHON_NEW.md) lists what
is new versus the pre-existing starting point.

**Last updated: 2026-11-03.**

## What it does

1. **Pipeline** (`src/preprocess.py`). One file replaces the old R +
   notebook + HPC chain: daily classroom summaries (peaks and school-day
   means for CO2, PM2.5, VOC) plus CO2-decay ventilation (ACH). A decay
   event counts only with 15+ readings, R² ≥ 0.85, CO2 400–5000 ppm.
   Resumable per-sensor-month checkpoints: finished work is never redone.
2. **Ratings** (`src/ratings.py`, `src/rating_bands.csv`). Good/Fair/Poor
   bands from public health guidance, editable without touching code.
   Peak views use maxima; typical views use the median for ACH and means
   for PM2.5, CO2, VOC (temperature and humidity are always averages).
   One overall rating per room-day, plus a recommendation per weak metric.
3. **PAC model** (`src/predict_pac.py`). A supervised logistic equation
   trained on known trial ON/OFF days (125 labeled classroom-days) is
   graded against an unsupervised cluster model and a majority-guess
   baseline on held-out schools. The supervised model scores 0.690 on
   unseen schools; the unsupervised 0.405; guessing 0.500.
4. **Dashboard** (`src/dashboard.py`). Streamlit. Click a classroom on the
   school map (real floor plan where one exists, schematic otherwise),
   pick a weekday on the calendar, see every metric's rating and the ACH
   decay chart behind it. An "Engineer run" sidebar section processes a
   new school's raw folder in the browser and populates everything.
5. **VR classroom** (`src/vr_classroom.html`). A single web page: walk
   inside a typical classroom and see the air. CO2 particles use the
   school's real numbers (one dot ≈ 1 ppm of the school-day mean):
   Room M1 at 1,275 ppm (2025-11-05) versus Room 105 at 386 ppm
   (2024-12-30). Drag to look, WASD/arrows to move. Open the file in a
   browser; no server needed. The 3D library is inlined in the page,
   so it works fully offline — as does the whole dashboard demo. Only
   writing a NEW principal letter or running SAM needs the internet.
6. **Principal letter** (`src/principal_report.py`). A district engineer
   presses one button and Muse Spark, via the Meta Model API, writes a
   plain-language letter from that month's rated tables. No key is typed
   into the page; it is read from a local `.env` file and never displayed,
   saved, or committed.
7. **Room finder** (`src/sam_rooms.py`). SAM 3.1, via the same Meta Model
   API, segments "room" boxes on a new school's floor plan and suggests
   dashboard dot positions (box centers as image fractions), so a new
   school's map needs checking, not hand-measuring. Dry-run verified;
   live call uses the same key as the letter.

## Quickstart

```bash
source .venv/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib streamlit pypdf plotly openai

python src/preprocess.py --input /path/to/Batch1-style-folder --output processed
python src/ratings.py
python src/predict_pac.py
streamlit run src/dashboard.py   # then open the local URL in a browser
```

School names are anonymized (`School_01`...) in every output; a private
mapping file stays off Git by default. This public repo contains no raw
sensor data, no student data, and no real school names.

## No powerful computer?

The Meta Model API does not run your code; it returns model output
(letters, masks). Processing still needs Python somewhere — but not
much of it: the full pipeline above ran end-to-end on a MacBook Air.
For a click-only demo, deploy this repo on Streamlit Community Cloud
(share.streamlit.io → New app → this repo → `src/dashboard.py`).
Caveat: the cloud cannot see folders on your machine, so the hosted
copy displays data you give it; the local dashboard above remains
the way to process a folder of raw sensor CSVs.

## Layout

- `src/` — all pipeline, rating, prediction, dashboard, and report code.
- `HACKATHON_NEW.md` — hackathon work log (new versus pre-existing).
- `AGENTS.md` — project rules for contributors and coding agents.
- Raw data, original notebooks, and the private school key are NOT in this
  repository (see `.gitignore`).

## Changelog

- **2026-11-03** — git history scrubbed (old HPC lab path removed from all
  commits); engineer-run dashboard section; live Meta letter verified;
  loose reference files grouped under `docs/reference/`; this README.
- **2026-11-02** — School 06 real floor-plan map with re-seated room dots;
  eng_demo + full School 06 runs; dot-center fixes (rooms 105/106/161
  were on dividers); model-report plain-language explainer.
- **2026-10-29** — ratings engine, dashboard, PAC model, trial boxplots
  with Kruskal-Wallis verdicts, principal-report dry run.
