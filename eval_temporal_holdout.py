"""
eval_temporal_holdout.py - Evaluate CompactIQ's real-world generalization,
as opposed to the in-distribution accuracy reported in train_model.py / pycompat_model.py.

The single random 80/20 row split used elsewhere in this repo lets the SAME package
appear in both train and test (just at a different version), so the model can partly
"memorize" a package's typical behavior instead of predicting from first principles.
That number answers "how well does this model fit packages it has already studied,"
not "how well will it perform on a version or package it hasn't seen yet" -- which is
the actual deployment scenario (predicting compatibility for package releases that
come out after the model was trained).

This script reports three splits side by side, all trained with the same
hyperparameters PyCompatModel._train() uses in production:

  1. RANDOM (baseline)   - reproduces the existing row-level 80/20 split, for reference.
  2. VERSION HOLDOUT     - per package, the newest ~20% of versions are held out as
                           test. This is the "timestamp-based" split, but built on
                           true version-release order rather than data.json's raw
                           `timestamp` field.

                           IMPORTANT: data.json's `timestamp` field is *scrape order*,
                           not release order -- every package was scraped
                           newest-version-first within an ~11 hour window on
                           2026-02-24/25 (verified below). A literal cut on that field
                           would put OLDER versions in the test set and put NEWER ones
                           in train -- backwards from what a temporal holdout is for.
                           Version number order is monotonic with real PyPI release
                           order, so it's the correct proxy for "time" here.
  3. PACKAGE HOLDOUT     - entire packages held out, never seen in any form during
                           training. This measures the cost of the "unknown package"
                           fallback in predict.py / pycompat_model.py, which silently
                           encodes an unseen package as the median index of the known
                           package list.

Usage:
    python eval_temporal_holdout.py [data.json]
"""

import sys
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from pycompat_model import PyCompatModel

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.json"
RANDOM_STATE = 42


def parse_version_tuple(version_str):
    return PyCompatModel._parse_version(version_str)


def verify_timestamp_is_scrape_order(raw_data):
    """Confirm (and print evidence) that data.json's timestamp is scrape order,
    not release order, so the reader can see why this script doesn't split on it."""
    by_pkg = defaultdict(list)
    for r in raw_data:
        by_pkg[r["package"]].append(r)

    inverted = 0
    checked = 0
    for pkg, rows in by_pkg.items():
        rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
        first_v = parse_version_tuple(rows_sorted[0]["version"])
        last_v = parse_version_tuple(rows_sorted[-1]["version"])
        checked += 1
        if first_v > last_v:
            inverted += 1

    print(f"Timestamp-order check: {inverted}/{checked} packages were scraped "
          f"newest-version-first (i.e. raw timestamp order is INVERTED relative to "
          f"release order).")
    print("-> Using version-release order as the time axis instead of raw timestamp.\n")


def build_dataset(raw_data):
    model = PyCompatModel()
    df = pd.DataFrame(raw_data)
    df = model._engineer_features(df)
    feature_cols = model._feature_columns()
    return df, feature_cols, model.mappings


def train_and_eval(df, feature_cols, train_idx, test_idx, label):
    X = df[feature_cols].values
    y_compat = df["is_compatible"].values
    y_error = df["error_type_encoded"].values

    X_train, X_test = X[train_idx], X[test_idx]
    yc_train, yc_test = y_compat[train_idx], y_compat[test_idx]
    ye_train, ye_test = y_error[train_idx], y_error[test_idx]

    compat_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        min_samples_leaf=1, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
    )
    compat_model.fit(X_train, yc_train)
    yc_pred = compat_model.predict(X_test)
    compat_acc = accuracy_score(yc_test, yc_pred)
    compat_f1 = f1_score(yc_test, yc_pred, average="weighted")

    error_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.1,
        min_samples_split=5, random_state=RANDOM_STATE,
    )
    error_model.fit(X_train, ye_train)
    ye_pred = error_model.predict(X_test)
    error_acc = accuracy_score(ye_test, ye_pred)
    error_f1 = f1_score(ye_test, ye_pred, average="weighted")

    print("=" * 70)
    print(f"{label}  (train={len(train_idx)}, test={len(test_idx)})")
    print("=" * 70)
    print(f"Compatibility -> accuracy: {compat_acc:.4f}  f1: {compat_f1:.4f}")
    print(f"Error type    -> accuracy: {error_acc:.4f}  f1: {error_f1:.4f}")
    print(classification_report(yc_test, yc_pred, target_names=["incompatible", "compatible"], zero_division=0))
    print()

    return {
        "split": label,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "compatibility": {"accuracy": round(float(compat_acc), 4), "f1_score": round(float(compat_f1), 4)},
        "error_type": {"accuracy": round(float(error_acc), 4), "f1_score": round(float(error_f1), 4)},
    }


def random_split_indices(df):
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_compatible"].values
    )
    return train_idx, test_idx


def version_holdout_indices(df, holdout_frac=0.2):
    """Per package, hold out the newest ~holdout_frac of *distinct versions* (not rows)."""
    train_idx, test_idx = [], []
    for pkg, group in df.groupby("package"):
        versions = sorted(
            group["version"].unique(),
            key=lambda v: parse_version_tuple(v),
        )
        n_holdout = max(1, math.ceil(len(versions) * holdout_frac))
        holdout_versions = set(versions[-n_holdout:])
        mask_test = group["version"].isin(holdout_versions)
        test_idx.extend(group.index[mask_test].tolist())
        train_idx.extend(group.index[~mask_test].tolist())
    return np.array(train_idx), np.array(test_idx)


def package_holdout_indices(df, holdout_frac=0.2):
    """Hold out entire packages -- never seen in training in any version/row."""
    packages = sorted(df["package"].unique())
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(packages)
    n_holdout = max(1, math.ceil(len(packages) * holdout_frac))
    holdout_packages = set(packages[:n_holdout])
    mask_test = df["package"].isin(holdout_packages)
    test_idx = df.index[mask_test].to_numpy()
    train_idx = df.index[~mask_test].to_numpy()
    return train_idx, test_idx, sorted(holdout_packages)


def main():
    with open(DATA_PATH) as f:
        raw_data = json.load(f)

    verify_timestamp_is_scrape_order(raw_data)

    df, feature_cols, mappings = build_dataset(raw_data)
    print(f"Loaded {len(df)} rows, {df['package'].nunique()} packages\n")

    results = []

    train_idx, test_idx = random_split_indices(df)
    results.append(train_and_eval(df, feature_cols, train_idx, test_idx,
                                   "1. RANDOM ROW SPLIT (existing baseline, in-distribution)"))

    train_idx, test_idx = version_holdout_indices(df)
    results.append(train_and_eval(df, feature_cols, train_idx, test_idx,
                                   "2. VERSION HOLDOUT (newest ~20% of versions per package)"))

    train_idx, test_idx, held_out_pkgs = package_holdout_indices(df)
    r = train_and_eval(df, feature_cols, train_idx, test_idx,
                        "3. PACKAGE HOLDOUT (entire packages never seen in training)")
    r["held_out_packages_sample"] = held_out_pkgs[:10]
    results.append(r)

    with open("model_metrics_temporal_holdout.json", "w") as f:
        json.dump({"splits": results}, f, indent=2)
    print("Saved model_metrics_temporal_holdout.json")

    print("\nSummary (compatibility model):")
    print(f"{'split':<55}{'accuracy':>10}{'f1':>10}")
    for r in results:
        print(f"{r['split']:<55}{r['compatibility']['accuracy']:>10.4f}{r['compatibility']['f1_score']:>10.4f}")


if __name__ == "__main__":
    main()
