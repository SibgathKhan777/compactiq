"""
predict.py - Prediction and recommendation engine.
Uses trained models to predict compatibility and recommend best versions.
"""

import os
import json
import re
import joblib
import numpy as np

MODEL_DIR = os.path.dirname(__file__)
COMPAT_MODEL_FILE = os.path.join(MODEL_DIR, "compat_model.joblib")
ERROR_MODEL_FILE = os.path.join(MODEL_DIR, "error_model.joblib")
MAPPINGS_FILE = os.path.join(MODEL_DIR, "mappings.joblib")
DATA_FILE = os.path.join(MODEL_DIR, "data.json")


def parse_version(version_str):
    """Parse version string into major, minor, patch."""
    parts = re.split(r'[.\-]', str(version_str))
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


class CompatibilityPredictor:
    """Main prediction engine for package compatibility."""

    def __init__(self):
        self.compat_model = None
        self.error_model = None
        self.mappings = None
        self.data = None
        self._load_models()
        self._load_data()

    def _load_models(self):
        """Load trained models and mappings."""
        if os.path.exists(COMPAT_MODEL_FILE):
            self.compat_model = joblib.load(COMPAT_MODEL_FILE)
        if os.path.exists(ERROR_MODEL_FILE):
            self.error_model = joblib.load(ERROR_MODEL_FILE)
        if os.path.exists(MAPPINGS_FILE):
            self.mappings = joblib.load(MAPPINGS_FILE)
        if self.compat_model is None:
            raise FileNotFoundError(
                "Models not found. Run train_model.py first."
            )

    def _load_data(self):
        """Load raw data for recommendation lookups."""
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                self.data = json.load(f)

    def _build_features(self, package, version, python_version, platform):
        """Build feature vector for a single prediction."""
        pkg_map = self.mappings["package_map"]
        plat_map = self.mappings["platform_map"]

        # Handle unknown packages/platforms
        pkg_encoded = pkg_map.get(package, -1)
        plat_encoded = plat_map.get(platform, -1)

        if pkg_encoded == -1:
            # For unknown packages, use median encoding
            pkg_encoded = len(pkg_map) // 2

        if plat_encoded == -1:
            plat_encoded = 0

        major, minor, patch = parse_version(version)
        py_ver = float(python_version)

        # Version recency (approximate for new queries)
        version_recency = 0.5  # default mid-range
        if self.data:
            pkg_versions = [
                r["version"]
                for r in self.data
                if r["package"] == package
            ]
            if pkg_versions:
                unique_versions = sorted(set(pkg_versions))
                if version in unique_versions:
                    idx = unique_versions.index(version)
                    version_recency = idx / max(len(unique_versions) - 1, 1)

        pkg_name_len = len(package)
        pkg_has_hyphen = 1 if "-" in package else 0

        return np.array(
            [
                [
                    pkg_encoded,
                    major,
                    minor,
                    patch,
                    py_ver,
                    plat_encoded,
                    version_recency,
                    pkg_name_len,
                    pkg_has_hyphen,
                ]
            ]
        )

    def predict_compatibility(self, package, version, python_version, platform="darwin_x86_64"):
        """
        Predict compatibility for a specific package+version+system.
        Returns dict with prediction, confidence, and predicted error type.
        """
        features = self._build_features(package, version, python_version, platform)

        # Compatibility prediction
        compat_pred = self.compat_model.predict(features)[0]
        compat_proba = self.compat_model.predict_proba(features)[0]
        confidence = float(max(compat_proba))

        # Error type prediction
        error_pred = None
        if self.error_model is not None:
            error_encoded = self.error_model.predict(features)[0]
            reverse_error = self.mappings.get("reverse_error_map", {})
            error_pred = reverse_error.get(error_encoded, "unknown")

        return {
            "package": package,
            "version": version,
            "python_version": python_version,
            "platform": platform,
            "is_compatible": bool(compat_pred),
            "confidence": round(confidence, 4),
            "compatibility_probability": round(float(compat_proba[1]) if len(compat_proba) > 1 else float(compat_proba[0]), 4),
            "predicted_error_type": error_pred if not compat_pred else "none",
        }

    def recommend_best_versions(self, package, python_version, platform="darwin_x86_64", top_n=5):
        """
        Recommend the best compatible versions for a package on a given system.
        Returns ranked list of versions with confidence scores.
        """
        if not self.data:
            return []

        # Get all known versions for this package
        versions = sorted(
            set(r["version"] for r in self.data if r["package"] == package)
        )

        if not versions:
            return {"error": f"Unknown package: {package}", "recommendations": []}

        # Predict compatibility for each version
        results = []
        for version in versions:
            pred = self.predict_compatibility(package, version, python_version, platform)
            results.append(
                {
                    "version": version,
                    "is_compatible": pred["is_compatible"],
                    "confidence": pred["confidence"],
                    "compatibility_probability": pred["compatibility_probability"],
                    "predicted_error_type": pred["predicted_error_type"],
                }
            )

        # Sort: compatible first, then by confidence, then by version (newest first)
        results.sort(
            key=lambda x: (
                x["is_compatible"],
                x["compatibility_probability"],
                parse_version(x["version"]),
            ),
            reverse=True,
        )

        return {
            "package": package,
            "python_version": python_version,
            "platform": platform,
            "total_versions": len(versions),
            "compatible_count": sum(1 for r in results if r["is_compatible"]),
            "recommendations": results[:top_n],
            "all_results": results,
        }

    def get_package_stats(self):
        """Get compatibility statistics for all packages."""
        if not self.data:
            return {}

        from collections import defaultdict

        stats = defaultdict(lambda: {"total": 0, "compatible": 0, "versions": set(), "errors": defaultdict(int)})

        for r in self.data:
            pkg = r["package"]
            stats[pkg]["total"] += 1
            stats[pkg]["versions"].add(r["version"])
            if r["install_success"] and r["import_success"]:
                stats[pkg]["compatible"] += 1
            stats[pkg]["errors"][r["error_type"]] += 1

        result = {}
        for pkg, s in stats.items():
            result[pkg] = {
                "total_tests": s["total"],
                "compatible": s["compatible"],
                "compatibility_rate": round(s["compatible"] / s["total"] * 100, 1),
                "num_versions": len(s["versions"]),
                "error_breakdown": dict(s["errors"]),
            }

        return dict(sorted(result.items(), key=lambda x: x[1]["compatibility_rate"]))

    def get_all_packages(self):
        """Return list of all known packages."""
        if self.mappings:
            return sorted(self.mappings["package_map"].keys())
        return []

    def get_versions_for_package(self, package):
        """Return all known versions for a package."""
        if self.data:
            return sorted(set(r["version"] for r in self.data if r["package"] == package))
        return []


# CLI interface
if __name__ == "__main__":
    import sys

    predictor = CompatibilityPredictor()

    if len(sys.argv) >= 4:
        pkg = sys.argv[1]
        ver = sys.argv[2]
        pyver = sys.argv[3]
        platform = sys.argv[4] if len(sys.argv) > 4 else "darwin_x86_64"

        result = predictor.predict_compatibility(pkg, ver, pyver, platform)
        print(json.dumps(result, indent=2))
    elif len(sys.argv) >= 3:
        pkg = sys.argv[1]
        pyver = sys.argv[2]

        result = predictor.recommend_best_versions(pkg, pyver)
        print(f"\nTop recommendations for {pkg} on Python {pyver}:")
        for i, rec in enumerate(result["recommendations"], 1):
            status = "✅" if rec["is_compatible"] else "❌"
            print(
                f"  {i}. v{rec['version']} {status} "
                f"(confidence: {rec['confidence']:.1%}, "
                f"compat prob: {rec['compatibility_probability']:.1%})"
            )
    else:
        print("Usage:")
        print("  python predict.py <package> <version> <python_version> [platform]")
        print("  python predict.py <package> <python_version>  (for recommendations)")
        print("\nExample:")
        print("  python predict.py boto3 1.42.55 3.12")
        print("  python predict.py alembic 3.9")
