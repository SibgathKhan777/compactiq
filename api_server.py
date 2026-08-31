"""
api_server.py — Production-ready standalone API for PyCompat model
===================================================================

Start:
    python api_server.py                     # defaults to ./model on port 8080
    python api_server.py --port 9000         # custom port
    python api_server.py --model ./my_model  # custom model path

Use from any project:
    import requests
    r = requests.post("http://localhost:8080/api/predict", json={
        "package": "boto3", "version": "1.42.49",
        "python_version": "3.12", "platform": "darwin_x86_64"
    })
    print(r.json())
"""

import os
import sys
import json
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from pycompat_model import PyCompatModel
from validate import validate_install_code
from deploy_check import compare_local_vs_deploy

app = Flask(__name__)
CORS(app)

model = None


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict compatibility for a single package.

    Request body:
        { "package": "boto3", "version": "1.42.49", "python_version": "3.12", "platform": "darwin_x86_64" }

    Response:
        { "is_compatible": true, "confidence": 0.997, "predicted_error_type": "none", ... }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = ["package", "version", "python_version"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = model.predict(
        data["package"], data["version"],
        data["python_version"], data.get("platform", "darwin_x86_64")
    )
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction.

    Request body:
        { "queries": [
            { "package": "boto3", "version": "1.42.49", "python_version": "3.12" },
            { "package": "alembic", "version": "1.18.4", "python_version": "3.9" }
        ]}

    Response:
        { "results": [...], "count": 2 }
    """
    data = request.get_json()
    queries = data.get("queries", [])
    if not queries:
        return jsonify({"error": "queries array required"}), 400

    results = model.predict_batch(queries)
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Get version recommendations.

    Request body:
        { "package": "alembic", "python_version": "3.9", "top_n": 5 }

    Response:
        { "recommendations": [...], "package": "alembic" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    pkg = data.get("package")
    pyver = data.get("python_version")
    if not pkg or not pyver:
        return jsonify({"error": "package and python_version required"}), 400

    recs = model.recommend(
        pkg, pyver,
        data.get("platform", "darwin_x86_64"),
        data.get("top_n", 5)
    )
    return jsonify({"recommendations": recs, "package": pkg, "python_version": pyver})


@app.route("/api/validate", methods=["POST"])
def validate():
    """
    Validate (and if needed, correct) LLM-generated `pip install` code -- the
    same contract as the middleware repo's /api/validate-llm-code, built from
    this repo's own ML model + live PyPI verification + joint dependency
    checking, with no Docker and no LLM call involved.

    Request body:
        { "code": "pip install torch==2.8.0 torchvision==0.17.0",
          "python_version": "3.12", "platform": "darwin_x86_64",
          "live": true }

    Response:
        { "original_code": ..., "corrected_code": ..., "risk_score": ...,
          "joint_dependency_conflicts": [...], "packages": [...] }
    """
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "JSON body with 'code' required"}), 400

    result = validate_install_code(
        data["code"],
        python_version=data.get("python_version", "3.12"),
        platform=data.get("platform", "darwin_x86_64"),
        model=model,
        live=data.get("live", True),
    )
    return jsonify(result)


@app.route("/api/deploy-check", methods=["POST"])
def deploy_check():
    """
    Compares dependencies that work on your local machine against a deployment
    target (e.g. a Linux Docker container) and flags anything that would break
    only after deploying -- the "works on my machine" gap.

    Request body:
        { "code": "pip install torch==2.8.0", "python_version": "3.12",
          "local_platform": "darwin_arm64", "deploy_platform": "linux_x86_64" }
        Optional: "deploy_python_version" (if the deploy target pins a different
        Python version than local), "live": false (ML-only).

    Response:
        { "safe_to_deploy": bool, "regressions": [...], "local": {...}, "deploy": {...} }
    """
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "JSON body with 'code' required"}), 400

    result = compare_local_vs_deploy(
        data["code"],
        python_version=data.get("python_version", "3.12"),
        local_platform=data.get("local_platform", "darwin_x86_64"),
        deploy_platform=data.get("deploy_platform", "linux_x86_64"),
        deploy_python_version=data.get("deploy_python_version"),
        model=model,
        live=data.get("live", True),
    )
    return jsonify(result)


@app.route("/api/packages", methods=["GET"])
def packages():
    """List all known packages and their versions."""
    return jsonify({
        "packages": sorted(model.package_versions.keys()),
        "count": len(model.package_versions)
    })


@app.route("/api/info", methods=["GET"])
def info():
    """Model information and metadata."""
    return jsonify({
        "model_name": model.MODEL_NAME,
        "model_version": model.MODEL_VERSION,
        "metadata": model.metadata,
        "total_packages": len(model.package_versions),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


def main():
    global model

    parser = argparse.ArgumentParser(description="PyCompat API Server")
    parser.add_argument("--model", default="./model", help="Path to model directory")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"📦 Loading model from {args.model}...")
    model = PyCompatModel.load(args.model)

    print(f"\n🚀 PyCompat API running at http://localhost:{args.port}")
    print(f"\n   Endpoints:")
    print(f"   POST /api/predict        — Single prediction")
    print(f"   POST /api/predict/batch   — Batch predictions")
    print(f"   POST /api/recommend       — Version recommendations")
    print(f"   POST /api/validate        — Validate + auto-correct pip install code")
    print(f"   POST /api/deploy-check    — Compare local vs. deployment-target compatibility")
    print(f"   GET  /api/packages        — List packages")
    print(f"   GET  /api/info            — Model info")
    print(f"   GET  /api/health          — Health check\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
