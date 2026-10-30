"""
predict_pac.py — predict HEPA PAC ON/OFF from one classroom-day of data,
comparing TWO models on the known trial days (one OFF day + one ON day
per school, schedule below, same dates as the old ACH_Distributino.ipynb
analysis and the user's 2026-10-28 label drop).

Model A (unsupervised) sees NO labels: it clusters classroom-days in
two groups from the measurements alone. Labels are used ONLY afterwards
to check whether the groups line up with OFF vs ON.

Model B (supervised) is plain logistic regression, which is just one
equation: turn each day's measurements into a score, then a probability:

    score = w1 * PM_total + w2 * PM_max + w3 * ACH_max + w4 * ACH_med
            + w5 * Temp + w6 * RH + b
    P(ON) = 1 / (1 + e^-score)     ->  ON if P >= 0.5, else OFF

We test honestly: schools are split so both models are graded on schools
they never trained on (held-out test set). The contract's success bar:
supervised must clearly beat unsupervised AND a naive majority guess.

Run it (after preprocess.py has built processed/merged_daily_deduped.csv):
  source .venv/bin/activate
  python src/predict_pac.py
  python src/predict_pac.py --output /tmp/model_test   # try-out folder

Outputs (in the output folder):
  pac_predictions.csv   one row per labelled classroom-day, both models' calls
  pac_model_report.txt  the equation (weights) + the A-vs-B comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Known trial days live in src/trial_labels.csv, keyed by anonymized
# School_XX (public-safe: no real school names appear in tracked code).
# The School_XX assignment is alphabetical by Batch1 folder and never
# changes because Batch1 is read-only. One December trial school and one
# February trial school carry role=test, so the grade covers schools AND
# months the model never trained on.
def load_trial_labels() -> pd.DataFrame:
    """Trial OFF/ON dates + train/test role per anonymized school."""
    here = Path(__file__).resolve().parent
    return pd.read_csv(here / "trial_labels.csv", dtype={"school": str})

# One feature = one measurement about that classroom-day.
FEATURES = ["Cumulative_Indoor_PM2_5", "Indoor_PM2_5_max",
            "daily_max_ach", "daily_median_ach", "Temp_avg", "RH_avg"]

RANDOM_STATE = 42


def load_labelled(processed: Path) -> pd.DataFrame:
    """Keep trial days only, tagged with OFF/ON labels and train/test role.

    Trial dates come from src/trial_labels.csv keyed by anonymized
    School_XX, so no real school names are needed — or appear — anywhere
    in this step. The role column marks the held-out test schools.
    """
    merged = pd.read_csv(processed / "merged_daily_deduped.csv",
                         dtype={"sensor_id": str})
    labels = load_trial_labels()
    rows = []
    for row in labels.itertuples():
        for day, label in ((row.off_date, 0), (row.on_date, 1)):
            hit = merged[(merged["school"] == row.school)
                         & (merged["date"] == day)].copy()
            hit["pac_on"] = label
            hit["role"] = row.role
            rows.append(hit)
    labelled = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # Keep only days where every feature exists (a day with no ACH event
    # has no ventilation measurement and cannot be scored fairly).
    return labelled.dropna(subset=FEATURES).reset_index(drop=True)


def train_and_test(labelled: pd.DataFrame):
    """Fit on training schools, grade on held-out test schools."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labelled = labelled.assign(in_test=labelled["role"].eq("test"))
    # From here on, only anonymized School_XX names travel with the rows,
    # so no report or prediction file can leak a real school name.
    train = labelled[~labelled["in_test"]].reset_index(drop=True)
    test = labelled[labelled["in_test"]].reset_index(drop=True)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURES].to_numpy(dtype=float))
    test_x = scaler.transform(test[FEATURES].to_numpy(dtype=float))
    model = LogisticRegression(max_iter=5000).fit(train_x, train["pac_on"].to_numpy())
    for frame, matrix in ((train, train_x), (test, test_x)):
        frame["p_on"] = model.predict_proba(matrix)[:, 1]
        frame["predicted_on"] = (frame["p_on"] >= 0.5).astype(int)
    return model, scaler, train, test


def unsupervised_scores(train: pd.DataFrame, test: pd.DataFrame):
    """Model A: cluster days with NO labels, then check the groups.

    KMeans sees only the measurements (fit on training schools' rows).
    Cluster numbers are meaningless, so the OFF/ON meaning of each
    group is picked to best match the TRAINING labels only — that one
    yes/no choice is the only label information Model A ever gets.
    The same choice is then applied unchanged to the held-out schools.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURES].to_numpy(dtype=float))
    test_x = scaler.transform(test[FEATURES].to_numpy(dtype=float))
    model = KMeans(2, random_state=RANDOM_STATE, n_init=10).fit(train_x)
    actual = train["pac_on"].to_numpy()
    train_clusters = model.labels_
    # Cluster numbers are meaningless: cluster 1 might be the OFF group.
    # Pick the meaning that best matches the TRAINING labels only (one
    # yes/no choice — the only label information Model A ever gets),
    # then apply it unchanged to the held-out schools.
    flip = (((train_clusters == 1).astype(int) == actual).mean()
            >= ((train_clusters == 0).astype(int) == actual).mean())
    for frame, clusters in ((train, train_clusters),
                            (test, model.predict(test_x))):
        frame["cluster"] = clusters
        frame["unsupervised_on"] = (
            (clusters == 1).astype(int) if flip else clusters)
    return train, test


def accuracy(frame: pd.DataFrame, call_col: str = "predicted_on") -> tuple[float, dict]:
    """Fraction right, plus a small confusion table."""
    right = (frame[call_col] == frame["pac_on"]).mean()
    table = {(int(actual), int(pred)): int(((frame["pac_on"] == actual)
                                            & (frame[call_col] == pred)).sum())
             for actual in (0, 1) for pred in (0, 1)}
    return float(right), table


def baseline_accuracy(train: pd.DataFrame, frame: pd.DataFrame) -> float:
    """Naive guess: always call the training set's more common answer."""
    majority = int(train["pac_on"].mean() >= 0.5)
    return float((frame["pac_on"] == majority).mean())
    """Fraction right, plus a small confusion table."""
    right = (frame["predicted_on"] == frame["pac_on"]).mean()
    table = {(int(actual), int(pred)): int(((frame["pac_on"] == actual)
                                            & (frame["predicted_on"] == pred)).sum())
             for actual in (0, 1) for pred in (0, 1)}
    return float(right), table


def fake_names_for(output_key_path: Path, real_names: list[str]) -> list[str]:
    """Map real Batch1 folder names back to School_XX labels (for reports)."""
    key = pd.read_csv(output_key_path)
    fake_of = dict(zip(key["batch1_folder_name"], key["school_private_name"]))
    return [fake_of[real] for real in real_names]


def write_report(output: Path, model, scaler, train, test,
                 test_fake_names: list[str],
                 sup: tuple, uns: tuple, base: tuple) -> None:
    """Save the equation plus the A-vs-B comparison and contract verdict.

    sup/uns hold (train_acc, train_table, test_acc, test_table);
    base holds (train_acc, test_acc) for the naive majority guess.
    """
    sup_train_acc, sup_train_table, sup_test_acc, sup_test_table = sup
    uns_train_acc, uns_train_table, uns_test_acc, uns_test_table = uns
    base_train_acc, base_test_acc = base
    bar_met = sup_test_acc > uns_test_acc and sup_test_acc > base_test_acc
    lines = ["PAC ON/OFF model comparison report", "==================================",
             f"Labeled classroom-days: {len(train) + len(test)} "
             f"(train {len(train)}, held-out test {len(test)})",
             f"Test schools (never trained on): {', '.join(test_fake_names)}", "",
             "Model B (supervised) equation: score = b + sum(w * (feature - mean) / scale):"]
    for name, weight, mean, scale in zip(FEATURES, model.coef_[0],
                                         scaler.mean_, scaler.scale_):
        lines.append(f"  w[{name}] = {weight:+.4f}   (mean {mean:.3g}, scale {scale:.3g})")
    lines.append(f"  b = {model.intercept_[0]:+.4f}")
    lines += ["", "P(ON) = 1 / (1 + e^-score); Model B calls ON when P >= 0.5.",
              "Model A calls ON for the cluster that matched ON on training rows.", "",
              f"Model B (supervised)   train: {sup_train_acc:.3f}  {sup_train_table}",
              f"Model B (supervised)   test:  {sup_test_acc:.3f}  {sup_test_table}",
              f"Model A (unsupervised) train: {uns_train_acc:.3f}  {uns_train_table}",
              f"Model A (unsupervised) test:  {uns_test_acc:.3f}  {uns_test_table}",
              f"Baseline (majority)    train: {base_train_acc:.3f}",
              f"Baseline (majority)    test:  {base_test_acc:.3f}", "",
              f"Contract bar (B test strictly above A test AND baseline): "
              f"{'MET' if bar_met else 'NOT MET'} — "
              f"margins: B-A {sup_test_acc - uns_test_acc:+.3f}, "
              f"B-baseline {sup_test_acc - base_test_acc:+.3f}.",
              "Table key: (actual, predicted) -> count; 1 = ON, 0 = OFF."]
    (output / "pac_model_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PAC ON/OFF prediction.")
    parser.add_argument("--input", type=Path, default=Path("processed"),
                        help="Folder with merged_daily_deduped.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write predictions (default: same as input)")
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    labelled = load_labelled(args.input)
    print(f"Labeled classroom-days: {len(labelled)}")
    if labelled.empty:
        raise SystemExit("No labeled trial days found — run preprocess.py "
                         "for the trial schools/months first.")
    model, scaler, train, test = train_and_test(labelled)
    train, test = unsupervised_scores(train, test)
    sup = accuracy(train) + accuracy(test)
    uns = accuracy(train, "unsupervised_on") + accuracy(test, "unsupervised_on")
    base = (baseline_accuracy(train, train), baseline_accuracy(train, test))
    print(f"Supervised   train {sup[0]:.3f}, test {sup[2]:.3f}; "
          f"unsupervised train {uns[0]:.3f}, test {uns[2]:.3f}; "
          f"baseline train {base[0]:.3f}, test {base[1]:.3f}")

    both = pd.concat([train, test], ignore_index=True)
    both.to_csv(output / "pac_predictions.csv", index=False)
    write_report(output, model, scaler, train, test,
                 sorted(test["school"].unique()), sup, uns, base)
    print(f"Wrote pac_predictions.csv + pac_model_report.txt to {output}/")


if __name__ == "__main__":
    main()
