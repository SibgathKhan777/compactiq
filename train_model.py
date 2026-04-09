"""
train_model.py - Train prediction models for package compatibility.
Trains Random Forest for compatibility + Gradient Boosting for error type prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from preprocess import load_and_preprocess, get_feature_columns

MODEL_DIR = os.path.dirname(__file__)
COMPAT_MODEL_FILE = os.path.join(MODEL_DIR, "compat_model.joblib")
ERROR_MODEL_FILE = os.path.join(MODEL_DIR, "error_model.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
MAPPINGS_FILE = os.path.join(MODEL_DIR, "mappings.joblib")


def train_compatibility_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for binary compatibility prediction."""
    print("\n" + "=" * 60)
    print("TRAINING COMPATIBILITY MODEL (Random Forest)")
    print("=" * 60)

    # Hyperparameter search
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    # Quick grid search with CV
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1")
    print(f"CV F1 Scores: {cv_scores}")
    print(f"CV Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance
    feature_cols = get_feature_columns()
    importances = best_model.feature_importances_
    print(f"\nFeature Importances:")
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    metrics = {
        "compatibility": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "cv_mean_f1": round(cv_scores.mean(), 4),
            "cv_std_f1": round(cv_scores.std(), 4),
            "best_params": grid_search.best_params_,
            "feature_importances": {
                feat: round(imp, 4) for feat, imp in zip(feature_cols, importances)
            },
        }
    }

    return best_model, metrics


def train_error_type_model(X_train, X_test, y_train, y_test, error_map):
    """Train Gradient Boosting for error type prediction."""
    print("\n" + "=" * 60)
    print("TRAINING ERROR TYPE MODEL (Gradient Boosting)")
    print("=" * 60)

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    reverse_map = {v: k for k, v in error_map.items()}
    target_names = [reverse_map[i] for i in sorted(reverse_map.keys())]

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(
        f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names, zero_division=0)}"
    )

    metrics = {
        "error_type": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        }
    }

    return gb, metrics


def train_all():
    """Main training pipeline."""
    # Load and preprocess
    df, mappings = load_and_preprocess()

    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y_compat = df["is_compatible"].values
    y_error = df["error_type_encoded"].values

    # Split data
    X_train, X_test, y_compat_train, y_compat_test, y_err_train, y_err_test = (
        train_test_split(
            X, y_compat, y_error, test_size=0.2, random_state=42, stratify=y_compat
        )
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train models
    compat_model, compat_metrics = train_compatibility_model(
        X_train, X_test, y_compat_train, y_compat_test
    )

    error_model, error_metrics = train_error_type_model(
        X_train, X_test, y_err_train, y_err_test, mappings["error_map"]
    )

    # Save models
    joblib.dump(compat_model, COMPAT_MODEL_FILE)
    print(f"\nSaved compatibility model to {COMPAT_MODEL_FILE}")

    joblib.dump(error_model, ERROR_MODEL_FILE)
    print(f"Saved error type model to {ERROR_MODEL_FILE}")

    # Save mappings
    joblib.dump(mappings, MAPPINGS_FILE)
    print(f"Saved mappings to {MAPPINGS_FILE}")

    # Save combined metrics
    all_metrics = {**compat_metrics, **error_metrics}
    all_metrics["training_info"] = {
        "total_records": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_packages": df["package"].nunique(),
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Saved metrics to {METRICS_FILE}")

    return compat_model, error_model, mappings, all_metrics


if __name__ == "__main__":
    train_all()
