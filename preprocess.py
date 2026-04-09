"""
preprocess.py - Data preprocessing and feature engineering for package compatibility prediction.
Loads data.json, engineers features, and saves processed data.
"""

import json
import pandas as pd
import numpy as np
import re
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "processed_data.csv")


def parse_version(version_str):
    """Parse version string into major, minor, patch numeric components."""
    parts = re.split(r'[.\-]', str(version_str))
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


def load_and_preprocess(data_file=DATA_FILE):
    """Load raw JSON data and perform feature engineering."""
    with open(data_file, "r") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    print(f"Loaded {len(df)} records from {data_file}")

    # --- Feature Engineering ---

    # 1. Parse package version into numeric components
    version_parts = df["version"].apply(parse_version)
    df["version_major"] = version_parts.apply(lambda x: x[0])
    df["version_minor"] = version_parts.apply(lambda x: x[1])
    df["version_patch"] = version_parts.apply(lambda x: x[2])

    # 2. Python version as float
    df["python_version_num"] = df["python_version"].astype(float)

    # 3. Encode package name
    package_map = {pkg: idx for idx, pkg in enumerate(sorted(df["package"].unique()))}
    df["package_encoded"] = df["package"].map(package_map)

    # 4. Encode platform (supports future multi-platform)
    platform_map = {p: idx for idx, p in enumerate(sorted(df["platform"].unique()))}
    df["platform_encoded"] = df["platform"].map(platform_map)

    # 5. Encode error type
    error_map = {e: idx for idx, e in enumerate(sorted(df["error_type"].unique()))}
    df["error_type_encoded"] = df["error_type"].map(error_map)

    # 6. Derive compatibility score (target variable)
    # 1.0 = both install + import succeed
    # 0.5 = install succeeds but import fails
    # 0.0 = install fails
    df["compatibility"] = 0.0
    df.loc[df["install_success"] & df["import_success"], "compatibility"] = 1.0
    df.loc[df["install_success"] & ~df["import_success"], "compatibility"] = 0.5

    # 7. Binary target: fully compatible (1) or not (0)
    df["is_compatible"] = (df["install_success"] & df["import_success"]).astype(int)

    # 8. Version recency score (higher = newer within same package)
    for pkg in df["package"].unique():
        mask = df["package"] == pkg
        versions = df.loc[mask, ["version_major", "version_minor", "version_patch"]].values
        # Rank versions by numeric value
        version_nums = versions[:, 0] * 10000 + versions[:, 1] * 100 + versions[:, 2]
        unique_sorted = sorted(set(version_nums))
        rank_map = {v: i / max(len(unique_sorted) - 1, 1) for i, v in enumerate(unique_sorted)}
        df.loc[mask, "version_recency"] = [rank_map[v] for v in version_nums]

    # 9. Package name length and has-hyphen flags (can correlate with namespace packages)
    df["pkg_name_len"] = df["package"].apply(len)
    df["pkg_has_hyphen"] = df["package"].apply(lambda x: 1 if "-" in x else 0)

    # Save mappings for inference
    mappings = {
        "package_map": package_map,
        "platform_map": platform_map,
        "error_map": error_map,
        "reverse_error_map": {v: k for k, v in error_map.items()},
    }

    return df, mappings


def save_processed(df, output_file=OUTPUT_FILE):
    """Save processed dataframe to CSV."""
    df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file} ({len(df)} rows)")


def get_feature_columns():
    """Return the list of feature columns used for training."""
    return [
        "package_encoded",
        "version_major",
        "version_minor",
        "version_patch",
        "python_version_num",
        "platform_encoded",
        "version_recency",
        "pkg_name_len",
        "pkg_has_hyphen",
    ]


if __name__ == "__main__":
    df, mappings = load_and_preprocess()
    save_processed(df)

    print(f"\n--- Dataset Summary ---")
    print(f"Total records: {len(df)}")
    print(f"Unique packages: {df['package'].nunique()}")
    print(f"Compatible: {df['is_compatible'].sum()} ({100*df['is_compatible'].mean():.1f}%)")
    print(f"Incompatible: {(~df['is_compatible'].astype(bool)).sum()} ({100*(1-df['is_compatible'].mean()):.1f}%)")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nPackage mapping (first 10): {dict(list(mappings['package_map'].items())[:10])}")
    print(f"Error mapping: {mappings['error_map']}")

    # Save mappings
    import joblib
    mappings_file = os.path.join(os.path.dirname(__file__), "mappings.joblib")
    joblib.dump(mappings, mappings_file)
    print(f"Saved mappings to {mappings_file}")
