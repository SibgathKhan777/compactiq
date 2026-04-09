"""
app.py - Flask API server with auto-retrain support.
Serves predictions, recommendations, and dataset analytics.
"""

import os
import json
import hashlib
import threading
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(MODEL_DIR, "data.json")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
COMPAT_MODEL_FILE = os.path.join(MODEL_DIR, "compat_model.joblib")

# Global predictor instance
predictor = None
data_hash = None
retrain_lock = threading.Lock()


def get_file_hash(filepath):
    """Get MD5 hash of a file for change detection."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def init_predictor():
    """Initialize or reload the predictor."""
    global predictor
    from predict import CompatibilityPredictor
    try:
        predictor = CompatibilityPredictor()
        print("✅ Predictor loaded successfully")
    except FileNotFoundError:
        print("⚠️ Models not found. Training...")
        auto_retrain()


def auto_retrain():
    """Retrain models from current data."""
    global predictor, data_hash
    with retrain_lock:
        print("🔄 Retraining models...")
        try:
            from train_model import train_all
            train_all()
            from predict import CompatibilityPredictor
            predictor = CompatibilityPredictor()
            data_hash = get_file_hash(DATA_FILE)
            print("✅ Retrain complete")
            return True
        except Exception as e:
            print(f"❌ Retrain failed: {e}")
            return False


def check_data_changes():
    """Background thread to watch for data.json changes and auto-retrain."""
    global data_hash
    data_hash = get_file_hash(DATA_FILE)
    while True:
        time.sleep(30)  # Check every 30 seconds
        new_hash = get_file_hash(DATA_FILE)
        if new_hash and new_hash != data_hash:
            print("📦 Data change detected, auto-retraining...")
            auto_retrain()


# --- Routes ---

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/index.css")
def serve_css():
    return send_from_directory(".", "index.css")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict compatibility for a specific package+version+system."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    package = data.get("package")
    version = data.get("version")
    python_version = data.get("python_version")
    platform = data.get("platform", "darwin_x86_64")

    if not all([package, version, python_version]):
        return jsonify({"error": "package, version, and python_version are required"}), 400

    if predictor is None:
        return jsonify({"error": "Model not loaded. Try /retrain first."}), 503

    try:
        result = predictor.predict_compatibility(package, version, python_version, platform)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """Get best compatible versions for a package on a system."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    package = data.get("package")
    python_version = data.get("python_version")
    platform = data.get("platform", "darwin_x86_64")
    top_n = data.get("top_n", 10)

    if not all([package, python_version]):
        return jsonify({"error": "package and python_version are required"}), 400

    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        result = predictor.recommend_best_versions(package, python_version, platform, top_n)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/packages", methods=["GET"])
def get_packages():
    """List all known packages."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    packages = predictor.get_all_packages()
    # Also return versions per package
    package_info = {}
    for pkg in packages:
        versions = predictor.get_versions_for_package(pkg)
        package_info[pkg] = versions

    return jsonify({"packages": packages, "package_versions": package_info})


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get dataset statistics and model metrics."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    pkg_stats = predictor.get_package_stats()

    # Load model metrics
    model_metrics = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            model_metrics = json.load(f)

    # Overall stats
    total_packages = len(pkg_stats)
    total_tests = sum(s["total_tests"] for s in pkg_stats.values())
    total_compatible = sum(s["compatible"] for s in pkg_stats.values())
    overall_compat_rate = round(total_compatible / total_tests * 100, 1) if total_tests > 0 else 0

    # Error breakdown
    error_totals = {}
    for s in pkg_stats.values():
        for err, count in s["error_breakdown"].items():
            error_totals[err] = error_totals.get(err, 0) + count

    # Python version stats
    py_stats = {}
    if predictor.data:
        for r in predictor.data:
            pv = r["python_version"]
            if pv not in py_stats:
                py_stats[pv] = {"total": 0, "compatible": 0}
            py_stats[pv]["total"] += 1
            if r["install_success"] and r["import_success"]:
                py_stats[pv]["compatible"] += 1
        for pv in py_stats:
            py_stats[pv]["rate"] = round(
                py_stats[pv]["compatible"] / py_stats[pv]["total"] * 100, 1
            )

    return jsonify({
        "overview": {
            "total_packages": total_packages,
            "total_tests": total_tests,
            "total_compatible": total_compatible,
            "overall_compatibility_rate": overall_compat_rate,
        },
        "error_breakdown": error_totals,
        "python_version_stats": py_stats,
        "package_stats": pkg_stats,
        "model_metrics": model_metrics,
    })


@app.route("/retrain", methods=["POST"])
def retrain():
    """Manually trigger model retraining."""
    success = auto_retrain()
    if success:
        return jsonify({"status": "success", "message": "Models retrained successfully"})
    return jsonify({"status": "error", "message": "Retraining failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
        "data_file": os.path.exists(DATA_FILE),
    })


if __name__ == "__main__":
    # Initialize
    init_predictor()

    # Start background watcher for auto-retrain
    watcher = threading.Thread(target=check_data_changes, daemon=True)
    watcher.start()

    print("\n🚀 Package Compatibility API running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
