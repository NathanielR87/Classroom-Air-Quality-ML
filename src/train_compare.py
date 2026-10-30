"""
train_compare.py — two ways to train PAC ON/OFF, compared side by side.

Both methods learn from the SAME known trial days (one OFF + one ON day
per school) and the SAME features, so any score gap comes from the
training method, not from different data.

Method A (school rotation): leave-one-school-out. Train on 5 schools,
test on the 6th unseen school, rotate through all 6. Three learners are
compared on each rotation: logistic regression, random forest, and the
unsupervised KMeans check, plus a naive majority-guess baseline.

Method B (random-20 iterative): 30 rounds; each round trains on 20 random
days and checks the rest. The best-scoring round's model is kept
("adjusting" by keeping winners). Two numbers are reported: the MEAN
round score (honest estimate of 20-day training) and the BEST round score
(optimistic — the winner was picked by looking at the check).

Run it (after preprocess.py + predict_pac.py outputs exist in processed/):
  source .venv/bin/activate
  python src/train_compare.py
  python src/train_compare.py --output /tmp/train_test   # try-out folder

Outputs (git-ignored): train_compare_report.txt, loso_folds.csv,
random20_rounds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_pac import FEATURES, RANDOM_STATE, load_labelled

N_ROUNDS = 30   # random-20 rounds (enough to see the spread, fast to run)
TRAIN_N = 20    # days per random-20 training draw


def fit_logreg(train: pd.DataFrame):
    """Scaled logistic regression: the same equation as predict_pac.py."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    matrix = scaler.fit_transform(train[FEATURES].to_numpy(dtype=float))
    model = LogisticRegression(max_iter=5000).fit(
        matrix, train["pac_on"].to_numpy())
    return model, scaler


def fit_forest(train: pd.DataFrame):
    """Random forest: 200 plain yes/no votes, no scaling needed."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=200,
                                   random_state=RANDOM_STATE)
    model.fit(train[FEATURES].to_numpy(dtype=float),
              train["pac_on"].to_numpy())
    return model, None


def predict(model, scaler, frame: pd.DataFrame) -> np.ndarray:
    """ON/OFF calls (1/0) for each row, through an optional scaler."""
    matrix = frame[FEATURES].to_numpy(dtype=float)
    if scaler is not None:
        matrix = scaler.transform(matrix)
    if hasattr(model, "predict_proba"):
        return (model.predict_proba(matrix)[:, 1] >= 0.5).astype(int)
    return model.predict(matrix).astype(int)


def unsupervised_calls(train: pd.DataFrame, test: pd.DataFrame
                       ) -> tuple[np.ndarray, np.ndarray]:
    """KMeans calls with NO labels (labels only name the clusters after)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURES].to_numpy(dtype=float))
    test_x = scaler.transform(test[FEATURES].to_numpy(dtype=float))
    clusters = KMeans(2, random_state=RANDOM_STATE, n_init=10).fit(train_x)
    actual = train["pac_on"].to_numpy()
    # Cluster 1 might be the OFF group: pick the meaning matching TRAINING
    # rows only, then apply it unchanged to the test rows.
    flip = (((clusters.labels_ == 1).astype(int) == actual).mean()
            >= ((clusters.labels_ == 0).astype(int) == actual).mean())
    to_call = np.vectorize(lambda c: (c == 1) if flip else (c == 0))
    return (to_call(clusters.labels_).astype(int),
            to_call(clusters.predict(test_x)).astype(int))


def method_a_rotation(labelled: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-school-out: each school takes a turn as the unseen test."""
    schools = sorted(labelled["school"].unique())
    rows = []
    for held_out in schools:
        train = labelled[labelled["school"] != held_out]
        test = labelled[labelled["school"] == held_out]
        actual = test["pac_on"].to_numpy()
        majority = int(train["pac_on"].mean() >= 0.5)
        logreg, scaler = fit_logreg(train)
        forest, _ = fit_forest(train)
        uns_train, uns_test = unsupervised_calls(train, test)
        rows.append({
            "held_out_school": held_out,
            "n_train": len(train), "n_test": len(test),
            "logreg": float((predict(logreg, scaler, test) == actual).mean()),
            "forest": float((predict(forest, None, test) == actual).mean()),
            "unsupervised": float((uns_test == actual).mean()),
            "baseline": float((np.full(len(test), majority) == actual).mean()),
        })
    return pd.DataFrame(rows)


def method_b_random20(labelled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """30 rounds of train-on-20 / check-the-rest, keeping the winner."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(labelled))
    rounds, best = [], {"score": -1.0, "round": -1}
    for round_no in range(N_ROUNDS):
        rng.shuffle(idx)
        train = labelled.iloc[idx[:TRAIN_N]].reset_index(drop=True)
        test = labelled.iloc[idx[TRAIN_N:]].reset_index(drop=True)
        model, scaler = fit_logreg(train)
        score = float((predict(model, scaler, test)
                       == test["pac_on"].to_numpy()).mean())
        rounds.append({"round": round_no, "n_train": len(train),
                       "n_test": len(test), "held_out_score": score})
        if score > best["score"]:  # the "adjust": keep winners only
            best = {"score": score, "round": round_no}
    rounds_frame = pd.DataFrame(rounds)
    summary = {"mean_score": float(rounds_frame["held_out_score"].mean()),
               "best_score": float(best["score"]),
               "best_round": int(best["round"])}
    return rounds_frame, summary


FEATURE_SETS = {
    "base (6)": FEATURES,
    "base + mean ACH": FEATURES + ["daily_mean_ach"],
    "base + mean ACH + events": FEATURES + ["daily_mean_ach",
                                            "n_decay_events"],
    "base + mean ACH + events + CO2": FEATURES + ["daily_mean_ach",
                                                  "n_decay_events",
                                                  "Indoor_CO2_max",
                                                  "Cumulative_Indoor_CO2"],
}


def rotation_score(labelled: pd.DataFrame, features: list[str]
                   ) -> pd.DataFrame:
    """LOSO accuracy per held-out school for logreg on one feature set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = []
    for held_out in sorted(labelled["school"].unique()):
        train = labelled[labelled["school"] != held_out]
        test = labelled[labelled["school"] == held_out]
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=5000).fit(
            scaler.fit_transform(train[features].to_numpy(dtype=float)),
            train["pac_on"].to_numpy())
        calls = (model.predict_proba(scaler.transform(
            test[features].to_numpy(dtype=float)))[:, 1] >= 0.5).astype(int)
        rows.append({"held_out_school": held_out,
                     "accuracy": float(
                         (calls == test["pac_on"].to_numpy()).mean())})
    return pd.DataFrame(rows)


def write_report(output: Path, loso: pd.DataFrame, rounds: pd.DataFrame,
                 summary: dict) -> None:
    """Plain-words comparison of the two training methods."""
    lines = ["PAC training comparison: school rotation vs random-20",
             "=====================================================",
             f"Labeled classroom-days: {int(loso['n_train'].iloc[0] + loso['n_test'].iloc[0])} "
             f"({len(loso)} schools rotated)", "",
             "Method A — leave-one-school-out (each score is on a school",
             "the model never trained on):"]
    for row in loso.itertuples():
        lines.append(
            f"  held out {row.held_out_school}: "
            f"logreg {row.logreg:.3f}  forest {row.forest:.3f}  "
            f"unsup {row.unsupervised:.3f}  baseline {row.baseline:.3f}")
    lines += ["  MEAN: "
              f"logreg {loso['logreg'].mean():.3f}  "
              f"forest {loso['forest'].mean():.3f}  "
              f"unsup {loso['unsupervised'].mean():.3f}  "
              f"baseline {loso['baseline'].mean():.3f}", "",
              "Method B — random-20 iterative (30 rounds of train-on-20,",
              "check-the-rest, keep the winner):",
              f"  mean round score (honest): {summary['mean_score']:.3f}",
              f"  best round score (optimistic): {summary['best_score']:.3f} "
              f"(round {summary['best_round']})", "",
              "Reading the two together: the BEST-round number is the one "
              "the naive version of method B would quote, but it was picked "
              "by looking at the check — the MEAN round score is the fair "
              "version. Method A's scores are all on unseen schools, so if "
              "A's mean beats B's mean AND the baselines, rotation training "
              "is the one to trust for new days."]
    (output / "train_compare_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run both training methods on the same labels and write the verdict."""
    parser = argparse.ArgumentParser(description="Compare PAC training ways.")
    parser.add_argument("--input", type=Path, default=Path("processed"),
                        help="Folder with merged_daily_deduped.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write (default: same as input)")
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    labelled = load_labelled(args.input)
    print(f"Labeled classroom-days: {len(labelled)}")
    if labelled.empty:
        raise SystemExit("No labeled trial days — run preprocess.py first.")
    loso = method_a_rotation(labelled)
    rounds, summary = method_b_random20(labelled)
    loso.to_csv(output / "loso_folds.csv", index=False)
    rounds.to_csv(output / "random20_rounds.csv", index=False)
    write_report(output, loso, rounds, summary)
    sets, means = [], {}
    for name, features in FEATURE_SETS.items():
        scores = rotation_score(labelled, features)
        scores.insert(0, "feature_set", name)
        sets.append(scores)
        means[name] = float(scores["accuracy"].mean())
        print(f"Feature set '{name}': rotation mean {means[name]:.3f}")
    pd.concat(sets, ignore_index=True).to_csv(
        output / "feature_sets.csv", index=False)
    best = max(means, key=means.get)
    with open(output / "train_compare_report.txt", "a") as handle:
        handle.write("\nExtra-feature trial (rotation logreg means):\n")
        for name, mean in means.items():
            handle.write(f"  {name}: {mean:.3f}\n")
        handle.write(f"Winner: {best} ({means[best]:.3f}).\n")
    print(f"Method A means: logreg {loso['logreg'].mean():.3f}, "
          f"forest {loso['forest'].mean():.3f}, "
          f"unsup {loso['unsupervised'].mean():.3f}, "
          f"base {loso['baseline'].mean():.3f}")
    print(f"Method B: mean round {summary['mean_score']:.3f}, "
          f"best round {summary['best_score']:.3f}")
    print(f"Wrote train_compare_report.txt + CSVs to {output}/")


if __name__ == "__main__":
    main()
