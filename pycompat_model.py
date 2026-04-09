"""
PyCompat — Python Package Compatibility Prediction Model
=========================================================
Standalone model package for Hugging Face and project integration.

Usage:
    from pycompat_model import PyCompatModel
    
    model = PyCompatModel.load("./model")
    result = model.predict("boto3", "1.42.49", "3.12", "darwin_x86_64")
    recommendations = model.recommend("alembic", "3.9")
"""

import os
import json
import re
import pickle
import numpy as np
import joblib


class PyCompatModel:
    """
    Self-contained package compatibility prediction model.
    Can be saved/loaded as a single directory for Hugging Face Hub or local use.
    """

    MODEL_VERSION = "1.0.0"
    MODEL_NAME = "pycompat-predictor"

    def __init__(self):
        self.compat_model = None
        self.error_model = None
        self.mappings = None
        self.metadata = {}
        self.package_versions = {}  # package -> list of known versions

    # ─── Training ───────────────────────────────────────────────

    @classmethod
    def train_from_data(cls, data_path):
        """Train a new model from a data.json file."""
        instance = cls()
        instance._train(data_path)
        return instance

    def _train(self, data_path):
        """Full training pipeline."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        # Load data
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        df = pd.DataFrame(raw_data)
        print(f"📦 Loaded {len(df)} records, {df['package'].nunique()} packages")

        # Store known package versions for recommendations
        for pkg in df["package"].unique():
            self.package_versions[pkg] = sorted(
                df[df["package"] == pkg]["version"].unique().tolist()
            )

        # Feature engineering
        df = self._engineer_features(df)

        # Prepare data
        feature_cols = self._feature_columns()
        X = df[feature_cols].values
        y_compat = df["is_compatible"].values
        y_error = df["error_type_encoded"].values

        X_train, X_test, yc_train, yc_test, ye_train, ye_test = train_test_split(
            X, y_compat, y_error, test_size=0.2, random_state=42, stratify=y_compat
        )

        # Train compatibility model
        print("🔧 Training compatibility model...")
        self.compat_model = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5,
            min_samples_leaf=1, random_state=42, class_weight="balanced", n_jobs=-1
        )
        self.compat_model.fit(X_train, yc_train)
        yc_pred = self.compat_model.predict(X_test)
        compat_acc = accuracy_score(yc_test, yc_pred)
        compat_f1 = f1_score(yc_test, yc_pred, average="weighted")
        print(f"   Accuracy: {compat_acc:.4f} | F1: {compat_f1:.4f}")

        # Train error type model
        print("🔧 Training error type model...")
        self.error_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            min_samples_split=5, random_state=42
        )
        self.error_model.fit(X_train, ye_train)
        ye_pred = self.error_model.predict(X_test)
        error_acc = accuracy_score(ye_test, ye_pred)
        error_f1 = f1_score(ye_test, ye_pred, average="weighted")
        print(f"   Accuracy: {error_acc:.4f} | F1: {error_f1:.4f}")

        # Store metadata
        self.metadata = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "total_records": len(df),
            "total_packages": df["package"].nunique(),
            "python_versions": sorted(df["python_version"].unique().tolist()),
            "platforms": sorted(df["platform"].unique().tolist()),
            "feature_columns": feature_cols,
            "metrics": {
                "compatibility": {"accuracy": round(compat_acc, 4), "f1_score": round(compat_f1, 4)},
                "error_type": {"accuracy": round(error_acc, 4), "f1_score": round(error_f1, 4)},
            },
            "feature_importances": {
                feat: round(imp, 4)
                for feat, imp in zip(feature_cols, self.compat_model.feature_importances_)
            },
        }

        print(f"✅ Training complete!")
        print(f"   Compat accuracy: {compat_acc:.1%} | Error accuracy: {error_acc:.1%}")

    def _engineer_features(self, df):
        """Apply feature engineering to a DataFrame."""
        import pandas as pd

        # Parse version
        vparts = df["version"].apply(self._parse_version)
        df["version_major"] = vparts.apply(lambda x: x[0])
        df["version_minor"] = vparts.apply(lambda x: x[1])
        df["version_patch"] = vparts.apply(lambda x: x[2])

        # Python version as float
        df["python_version_num"] = df["python_version"].astype(float)

        # Encode categoricals
        self.mappings = {
            "package_map": {pkg: i for i, pkg in enumerate(sorted(df["package"].unique()))},
            "platform_map": {p: i for i, p in enumerate(sorted(df["platform"].unique()))},
            "error_map": {e: i for i, e in enumerate(sorted(df["error_type"].unique()))},
        }
        self.mappings["reverse_error_map"] = {v: k for k, v in self.mappings["error_map"].items()}

        df["package_encoded"] = df["package"].map(self.mappings["package_map"])
        df["platform_encoded"] = df["platform"].map(self.mappings["platform_map"])
        df["error_type_encoded"] = df["error_type"].map(self.mappings["error_map"])

        # Target
        df["is_compatible"] = (df["install_success"] & df["import_success"]).astype(int)

        # Version recency
        df["version_recency"] = 0.5
        for pkg in df["package"].unique():
            mask = df["package"] == pkg
            v = df.loc[mask, ["version_major", "version_minor", "version_patch"]].values
            vnums = v[:, 0] * 10000 + v[:, 1] * 100 + v[:, 2]
            usorted = sorted(set(vnums))
            rmap = {val: i / max(len(usorted) - 1, 1) for i, val in enumerate(usorted)}
            df.loc[mask, "version_recency"] = [rmap[val] for val in vnums]

        # Name features
        df["pkg_name_len"] = df["package"].apply(len)
        df["pkg_has_hyphen"] = df["package"].apply(lambda x: 1 if "-" in x else 0)

        return df

    @staticmethod
    def _parse_version(version_str):
        parts = re.split(r'[.\-]', str(version_str))
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        return major, minor, patch

    @staticmethod
    def _feature_columns():
        return [
            "package_encoded", "version_major", "version_minor", "version_patch",
            "python_version_num", "platform_encoded", "version_recency",
            "pkg_name_len", "pkg_has_hyphen",
        ]

    # ─── Prediction ─────────────────────────────────────────────

    def predict(self, package, version, python_version, platform="darwin_x86_64"):
        """
        Predict compatibility for a package+version on a given system.

        Args:
            package: Package name (e.g. "boto3")
            version: Version string (e.g. "1.42.49")
            python_version: Python version (e.g. "3.12")
            platform: Platform string (e.g. "darwin_x86_64")

        Returns:
            dict with is_compatible, confidence, predicted_error_type, etc.
        """
        if self.compat_model is None:
            raise RuntimeError("Model not loaded. Call load() or train_from_data() first.")

        features = self._build_features(package, version, python_version, platform)

        compat_pred = self.compat_model.predict(features)[0]
        compat_proba = self.compat_model.predict_proba(features)[0]
        confidence = float(max(compat_proba))

        error_pred = "unknown"
        if self.error_model is not None:
            err_enc = self.error_model.predict(features)[0]
            rev_map = self.mappings.get("reverse_error_map", {})
            # JSON converts int keys to strings, so check both
            error_pred = rev_map.get(err_enc, rev_map.get(str(err_enc), "unknown"))

        return {
            "package": package,
            "version": version,
            "python_version": python_version,
            "platform": platform,
            "is_compatible": bool(compat_pred),
            "confidence": round(confidence, 4),
            "compatibility_probability": round(
                float(compat_proba[1]) if len(compat_proba) > 1 else float(compat_proba[0]), 4
            ),
            "predicted_error_type": error_pred if not compat_pred else "none",
        }

    def recommend(self, package, python_version, platform="darwin_x86_64", top_n=5):
        """
        Recommend best compatible versions for a package.

        Args:
            package: Package name
            python_version: Python version
            platform: Platform string
            top_n: Number of recommendations to return

        Returns:
            list of dicts sorted by compatibility probability (descending)
        """
        versions = self.package_versions.get(package, [])
        if not versions:
            return []

        results = []
        for v in versions:
            pred = self.predict(package, v, python_version, platform)
            results.append(pred)

        results.sort(key=lambda x: (x["is_compatible"], x["compatibility_probability"]), reverse=True)
        return results[:top_n]

    def predict_batch(self, queries):
        """
        Batch prediction for multiple queries.

        Args:
            queries: list of dicts with keys: package, version, python_version, platform

        Returns:
            list of prediction dicts
        """
        return [
            self.predict(
                q["package"], q["version"],
                q["python_version"], q.get("platform", "darwin_x86_64")
            )
            for q in queries
        ]

    def _build_features(self, package, version, python_version, platform):
        pkg_enc = self.mappings["package_map"].get(package, len(self.mappings["package_map"]) // 2)
        plat_enc = self.mappings["platform_map"].get(platform, 0)
        major, minor, patch = self._parse_version(version)
        py_ver = float(python_version)

        # Version recency
        recency = 0.5
        versions = self.package_versions.get(package, [])
        if versions and version in versions:
            idx = versions.index(version)
            recency = idx / max(len(versions) - 1, 1)

        return np.array([[
            pkg_enc, major, minor, patch, py_ver, plat_enc,
            recency, len(package), 1 if "-" in package else 0
        ]])

    # ─── Save / Load ────────────────────────────────────────────

    def save(self, path):
        """
        Save model to a directory (compatible with Hugging Face Hub).

        Creates:
            path/
              config.json          — Model metadata and mappings
              compat_model.joblib  — Compatibility classifier
              error_model.joblib   — Error type classifier
              README.md            — Hugging Face model card
        """
        os.makedirs(path, exist_ok=True)

        # Save models
        joblib.dump(self.compat_model, os.path.join(path, "compat_model.joblib"))
        joblib.dump(self.error_model, os.path.join(path, "error_model.joblib"))

        # Save config (mappings + metadata + package_versions)
        config = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "mappings": self.mappings,
            "metadata": self.metadata,
            "package_versions": self.package_versions,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Generate model card
        self._write_model_card(path)

        print(f"✅ Model saved to {path}/")
        print(f"   Files: config.json, compat_model.joblib, error_model.joblib, README.md")

    @classmethod
    def load(cls, path):
        """
        Load model from a directory.

        Args:
            path: Directory containing config.json and .joblib files

        Returns:
            PyCompatModel instance ready for predictions
        """
        instance = cls()

        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        instance.mappings = config["mappings"]
        instance.metadata = config.get("metadata", {})
        instance.package_versions = config.get("package_versions", {})
        instance.compat_model = joblib.load(os.path.join(path, "compat_model.joblib"))
        instance.error_model = joblib.load(os.path.join(path, "error_model.joblib"))

        print(f"✅ Model loaded from {path}/")
        return instance

    def _write_model_card(self, path):
        """Generate Hugging Face model card README."""
        metrics = self.metadata.get("metrics", {})
        compat_m = metrics.get("compatibility", {})
        error_m = metrics.get("error_type", {})

        card = f"""---
language: en
license: mit
library_name: scikit-learn
tags:
  - python
  - package-compatibility
  - prediction
  - scikit-learn
  - tabular-classification
metrics:
  - accuracy
  - f1
model-index:
  - name: {self.MODEL_NAME}
    results:
      - task:
          type: tabular-classification
          name: Package Compatibility Prediction
        metrics:
          - name: Accuracy
            type: accuracy
            value: {compat_m.get('accuracy', 'N/A')}
          - name: F1 Score
            type: f1
            value: {compat_m.get('f1_score', 'N/A')}
---

# PyCompat — Python Package Compatibility Predictor

AI model that predicts whether a Python package version is compatible with a given system
(OS, Python version, platform) and recommends the best compatible versions.

## Model Details

- **Model Type:** Random Forest (compatibility) + Gradient Boosting (error type)
- **Training Data:** {self.metadata.get('total_records', 'N/A')} compatibility test records
- **Packages:** {self.metadata.get('total_packages', 'N/A')} unique packages
- **Python Versions:** {', '.join(self.metadata.get('python_versions', []))}
- **Platforms:** {', '.join(self.metadata.get('platforms', []))}

## Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Compatibility | {compat_m.get('accuracy', 'N/A')} | {compat_m.get('f1_score', 'N/A')} |
| Error Type | {error_m.get('accuracy', 'N/A')} | {error_m.get('f1_score', 'N/A')} |

## Usage

```python
from pycompat_model import PyCompatModel

# Load model
model = PyCompatModel.load("./model")

# Single prediction
result = model.predict("boto3", "1.42.49", "3.12", "darwin_x86_64")
print(result)
# {{'is_compatible': True, 'confidence': 0.9977, 'predicted_error_type': 'none', ...}}

# Get recommendations
recs = model.recommend("alembic", "3.9")
for r in recs:
    status = "✅" if r["is_compatible"] else "❌"
    print(f"  v{{r['version']}} {{status}} ({{r['confidence']:.0%}})")

# Batch prediction
results = model.predict_batch([
    {{"package": "boto3", "version": "1.42.49", "python_version": "3.12"}},
    {{"package": "alembic", "version": "1.18.4", "python_version": "3.9"}},
])
```

## Error Types Predicted

| Error Type | Description |
|-----------|-------------|
| `none` | Fully compatible |
| `no_wheel` | No compatible wheel/distribution found |
| `import_error` | Installs but fails to import |
| `abi_mismatch` | ABI incompatibility with dependencies |
| `build_error` | Failed to build from source |
| `timeout` | Network timeout during install |

## Training

```python
from pycompat_model import PyCompatModel

model = PyCompatModel.train_from_data("data.json")
model.save("./model")
```
"""
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(card)

    # ─── Hugging Face Hub ───────────────────────────────────────

    def push_to_hub(self, repo_id, token=None):
        """
        Push model to Hugging Face Hub.

        Args:
            repo_id: e.g. "username/pycompat-model"
            token: Hugging Face API token (or set HF_TOKEN env var)

        Requires: pip install huggingface_hub
        """
        from huggingface_hub import HfApi, create_repo

        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Provide a token or set HF_TOKEN environment variable")

        # Save to temp dir
        tmp_dir = "/tmp/pycompat_hf_upload"
        self.save(tmp_dir)

        # Create repo and upload
        api = HfApi(token=token)
        try:
            create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        except Exception:
            pass

        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"🚀 Model pushed to https://huggingface.co/{repo_id}")

    @classmethod
    def from_hub(cls, repo_id, token=None):
        """
        Load model from Hugging Face Hub.

        Args:
            repo_id: e.g. "username/pycompat-model"

        Returns:
            PyCompatModel instance
        """
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id, token=token)
        return cls.load(local_dir)


# ─── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
PyCompat Model CLI
==================
  Train:     python pycompat_model.py train data.json ./model
  Predict:   python pycompat_model.py predict ./model boto3 1.42.49 3.12
  Recommend: python pycompat_model.py recommend ./model alembic 3.9
  Push:      python pycompat_model.py push ./model username/pycompat-model
        """)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "train":
        data_path = sys.argv[2] if len(sys.argv) > 2 else "data.json"
        save_path = sys.argv[3] if len(sys.argv) > 3 else "./model"
        model = PyCompatModel.train_from_data(data_path)
        model.save(save_path)

    elif cmd == "predict":
        model_path = sys.argv[2]
        pkg = sys.argv[3]
        ver = sys.argv[4]
        pyver = sys.argv[5]
        plat = sys.argv[6] if len(sys.argv) > 6 else "darwin_x86_64"
        model = PyCompatModel.load(model_path)
        result = model.predict(pkg, ver, pyver, plat)
        print(json.dumps(result, indent=2))

    elif cmd == "recommend":
        model_path = sys.argv[2]
        pkg = sys.argv[3]
        pyver = sys.argv[4]
        plat = sys.argv[5] if len(sys.argv) > 5 else "darwin_x86_64"
        model = PyCompatModel.load(model_path)
        recs = model.recommend(pkg, pyver, plat, top_n=10)
        print(f"\n🔍 Top recommendations for {pkg} on Python {pyver}:\n")
        for i, r in enumerate(recs, 1):
            s = "✅" if r["is_compatible"] else "❌"
            print(f"  {i}. v{r['version']} {s}  confidence: {r['confidence']:.0%}  error: {r['predicted_error_type']}")

    elif cmd == "push":
        model_path = sys.argv[2]
        repo_id = sys.argv[3]
        model = PyCompatModel.load(model_path)
        model.push_to_hub(repo_id)

    else:
        print(f"Unknown command: {cmd}")
