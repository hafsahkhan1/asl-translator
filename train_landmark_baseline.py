"""
Quick baseline landmark classifier (random forest) so the live app and
evaluation can be tested end to end. Hafsah's final landmark model replaces
models/landmark_model.joblib using the same format.

Usage:
    python train_landmark_baseline.py --csv asl_landmarks.csv
    # also train on our own recordings (first run extract_landmarks.py on them):
    python train_landmark_baseline.py --csv asl_landmarks.csv --extra jack_landmarks.csv

Rows from --extra files count --extra-weight times as much as dataset rows,
since there are far fewer of them (~30 per letter vs ~2,500).
Note: anything passed to --extra must NOT also be used as the test set.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("n") and c[1] in "xyz"]
    return df[cols].to_numpy(dtype=np.float32), df["label"].astype(str).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="asl_landmarks.csv", help="main dataset landmarks")
    ap.add_argument("--extra", nargs="*", default=[], help="landmark CSVs from our own recordings")
    ap.add_argument("--extra-weight", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.join("models", "landmark_model.joblib"))
    args = ap.parse_args()

    X, y = load(args.csv)
    print(f"{args.csv}: {len(y)} samples, {X.shape[1]} features, {len(set(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    w_train = np.ones(len(y_train))

    for path in args.extra:
        Xe, ye = load(path)
        print(f"{path}: {len(ye)} extra samples (weight {args.extra_weight:g})")
        X_train = np.vstack([X_train, Xe])
        y_train = np.concatenate([y_train, ye])
        w_train = np.concatenate([w_train, np.full(len(ye), args.extra_weight)])

    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)
    print(f"Held-out dataset accuracy: {accuracy_score(y_test, model.predict(X_test)):.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
