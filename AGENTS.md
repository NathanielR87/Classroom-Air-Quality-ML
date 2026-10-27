# AGENTS.md

Muse Code reads this file as project rules when it runs in this directory.

## Project
- Name: Adam data
- Meta hackathon project: I already have working code that preprocesses indoor air quality data and predicts whether a classroom's HEPA PAC (portable air cleaner) is ON or OFF. The goal is to understand, clean up, and improve that pipeline: make preprocessing easier to run on large datasets, and improve the ML prediction.
- Deadline: November 4 2026

## Folders
- Batch1/ — RAW school air quality data, one subfolder per school. READ ONLY.
- Nates_copy_CDC_code/ — my existing preprocessing + prediction code. This is the starting point do not change and just make copy if you are going to make any changes.
- Nates_copy_CDC_code/PAC_Pipeline_Walkthrough.pdf — slides explaining what each script does and the order they run in. Read this first.
- processed/ — cleaned data output goes here (create it if needed)
- src/ — new or refactored code goes here


## Stack
- Language: Python 3, R

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
- Not set up yet; fill in once we know how the scripts run

## Project Layout
- See Folders above; update once src/ exists