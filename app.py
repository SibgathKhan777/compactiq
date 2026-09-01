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

from hardware_profile import detect_host, describe_host
from validate import validate_install_code, parse_pip_install
from stack_suggest import suggest_stack, known_categories
from deploy_check import compare_local_vs_deploy, DEPLOY_PRESETS

DEPLOY_KEYWORDS = ["deploy", "deploying", "deployment", "production", " prod ", "docker",
                    "in the cloud", "on the server", "on a server", "on ec2", "on aws",
                    "on render", "on heroku", "on railway"]

# Phrase -> preset key, checked longest-first so "aws graviton" wins over the
# more general "aws ec2" / "aws" when both could plausibly match.
DEPLOY_PRESET_PHRASES = [
    ("graviton", "aws_graviton"),
    ("arm ec2", "aws_graviton"),
    ("aws lambda", "aws_lambda"),
    ("lambda", "aws_lambda"),
    ("ec2", "aws_ec2"),
    ("aws", "aws_ec2"),
    ("google compute", "gcp_compute"),
    ("gcp", "gcp_compute"),
    ("azure", "azure_vm"),
    ("windows server", "windows_server"),
    ("docker arm", "docker_linux_arm"),
    ("docker", "docker_linux"),
]


def _detect_deploy_preset(message):
    """Returns (platform_key, label) for the first matching named deploy target, or (None, None)."""
    lowered = message.lower()
    for phrase, preset_key in DEPLOY_PRESET_PHRASES:
        if phrase in lowered:
            preset = DEPLOY_PRESETS[preset_key]
            return preset["platform"], preset["label"]
    return None, None


# What the user SAYS their own machine is, e.g. "I'm on Windows and did pip
# freeze" or "I develop on an M1 Mac". Without this, the chat's "local"
# platform always defaults to whatever machine happens to be running this
# Flask server (via hardware_profile.detect_host()) -- correct when a
# developer runs the dashboard on their own laptop, but wrong the moment
# someone describes a DIFFERENT machine than the one actually running the
# server, which is exactly how a "will my Windows requirements.txt work on
# AWS EC2" question gets phrased. Checked longest/most-specific phrase first.
LOCAL_PLATFORM_PHRASES = [
    ("apple silicon", "darwin_arm64"),
    ("m1 mac", "darwin_arm64"), ("m2 mac", "darwin_arm64"), ("m3 mac", "darwin_arm64"), ("m4 mac", "darwin_arm64"),
    ("arm mac", "darwin_arm64"),
    ("intel mac", "darwin_x86_64"),
    ("on windows", "win_amd64"), ("windows machine", "win_amd64"), ("windows laptop", "win_amd64"),
    ("windows pc", "win_amd64"), ("i'm on windows", "win_amd64"), ("i am on windows", "win_amd64"),
    ("on linux", "linux_x86_64"), ("ubuntu", "linux_x86_64"), ("debian", "linux_x86_64"),
    ("on mac", "darwin_arm64"), ("macos", "darwin_arm64"), ("macbook", "darwin_arm64"),
]


def _detect_stated_local_platform(message):
    """Returns platform_key if the message explicitly states the user's own machine, else None."""
    lowered = message.lower()
    for phrase, platform_key in LOCAL_PLATFORM_PHRASES:
        if phrase in lowered:
            return platform_key
    return None

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(MODEL_DIR, "data.json")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")  # legacy path from train_model.py, usually absent
COMPAT_MODEL_FILE = os.path.join(MODEL_DIR, "compat_model.joblib")
PYCOMPAT_MODEL_DIR = os.path.join(MODEL_DIR, "model")
PYCOMPAT_CONFIG_FILE = os.path.join(PYCOMPAT_MODEL_DIR, "config.json")  # where PyCompatModel.save() actually writes metrics

# Global predictor instance
predictor = None
data_hash = None
retrain_lock = threading.Lock()

# Separate PyCompatModel instance for the chat/validate feature -- validate.py's
# pipeline (live PyPI checks + joint dependency conflicts) is built against
# PyCompatModel's predict()/recommend() interface, which is not the same method
# names as CompatibilityPredictor above (predict_compatibility/
# recommend_best_versions). Loading it once at startup, same model files, no
# retraining involved.
chat_model = None


def init_chat_model():
    global chat_model
    from pycompat_model import PyCompatModel
    try:
        chat_model = PyCompatModel.load(PYCOMPAT_MODEL_DIR)
    except Exception as e:
        print(f"⚠️ Chat assistant model failed to load: {e}")
        chat_model = None


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

    # Load model metrics. The real numbers are written by PyCompatModel.save()
    # into model/config.json's metadata.metrics (that's what api_server.py and
    # the chat assistant actually serve); METRICS_FILE is a legacy path from
    # the old train_model.py script and is usually absent, which is why this
    # used to always show "--" here even though the model has real numbers.
    model_metrics = {}
    if os.path.exists(PYCOMPAT_CONFIG_FILE):
        with open(PYCOMPAT_CONFIG_FILE, "r") as f:
            config = json.load(f)
        metadata = config.get("metadata", {})
        model_metrics = metadata.get("metrics", {})
        # PyCompatModel stores feature_importances as a separate top-level
        # metadata key, but the dashboard's Analytics tab reads it nested at
        # compatibility.feature_importances (the legacy train_model.py shape)
        # -- reshape so the existing frontend code doesn't need to change.
        if "feature_importances" in metadata and "compatibility" in model_metrics:
            model_metrics["compatibility"]["feature_importances"] = metadata["feature_importances"]
    elif os.path.exists(METRICS_FILE):
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


def _format_deploy_reply(comparison, host_profile, stack_label=None, deploy_label=None, stated_local_platform=None):
    """
    Turns a compare_local_vs_deploy() result into a natural-language reply
    focused on what's DIFFERENT between the local dev machine and the
    deployment target -- the "works on my machine" gap -- rather than a full
    per-platform breakdown.
    """
    target_desc = f"**{deploy_label}**" if deploy_label else f"a **{comparison['deploy_platform']}** deploy target"
    if stated_local_platform:
        local_desc = f"the **{comparison['local_platform']}** machine you described"
    else:
        local_desc = f"your local machine ({describe_host(host_profile)})"
    lines = [f"Comparing {local_desc} against {target_desc} (Python {comparison['deploy_python_version']})."]
    if stack_label:
        lines.append(f"That sounds like a **{stack_label}** project.")

    already_broken = comparison.get("already_broken", [])

    if comparison["safe_to_deploy"]:
        lines.append(
            f"✅ No regressions found -- everything you pinned works locally exactly as-is and also "
            f"checks out on {comparison['deploy_platform']}. Safe to deploy as-is:"
        )
        lines.append(f"\n```\n{comparison['local']['corrected_code'].strip()}\n```")
        return "\n".join(lines)

    if comparison["regressions"]:
        lines.append(
            f"⚠️ Found {len(comparison['regressions'])} package(s) that work locally but would "
            f"**break on {comparison['deploy_platform']}** -- this is exactly the kind of gap "
            f"that only shows up after you deploy:"
        )
        for r in comparison["regressions"]:
            lines.append(f"- **{r['package']}=={r['version']}**: {r['reason']}")

    if already_broken:
        lines.append(
            f"\n⚠️ {len(already_broken)} package(s) weren't actually safe on **either** platform as "
            f"pinned -- this isn't a deploy-specific gap, the version itself needs fixing regardless "
            f"of where you run it:"
        )
        for b in already_broken:
            lines.append(f"- **{b['package']}=={b['requested_version']}** → {b['deploy_corrected_to']}: {b['reason']}")

    deploy_code = comparison["deploy"]["corrected_code"]
    still_unresolved = [p["package"] for p in comparison["deploy"]["packages"] if not p["is_clean"]]
    if still_unresolved:
        lines.append(
            f"\n⚠️ Best attempt -- **not fully resolved** ({', '.join(still_unresolved)} still won't work, "
            f"no fix exists on {comparison['deploy_platform']}):\n```\n{deploy_code.strip()}\n```"
        )
    else:
        lines.append(f"\nWhat would actually work on {comparison['deploy_platform']}:\n```\n{deploy_code.strip()}\n```")

    return "\n".join(lines)


def _format_chat_reply(code, host_profile, result, stack_label=None):
    """
    Turns a validate_install_code() result into the natural-language reply
    contract: parse -> profile hardware -> correct -> explain -> return
    unchanged if valid, corrected + explanation if not.
    """
    lines = [f"Detected host: {describe_host(host_profile)}."]

    if stack_label:
        lines.append(f"That sounds like a **{stack_label}** project -- here's a typical stack from packages "
                      f"this model has real training data for, checked against your host:")

    if not result["packages"]:
        example_cats = ", ".join(known_categories())
        lines.append("I couldn't find any `package==version` pins in that message, and it didn't match a "
                      "project type I know a stack for. Paste a `pip install ...` line (like an LLM just handed "
                      f"you), or describe a project I can suggest a stack for: {example_cats}.")
        return "\n".join(lines)

    if not result["changed"]:
        lines.append(f"✅ Checked {len(result['packages'])} package(s) against this host, PyPI, "
                      f"and each other's declared dependencies -- everything looks compatible, "
                      f"no changes needed.")
        lines.append(f"\n```\n{code.strip()}\n```")
        return "\n".join(lines)

    changed_count = sum(1 for p in result["packages"] if p["changed"])
    if result.get("fully_resolved", True):
        lines.append(f"⚠️ Corrected {changed_count} of {len(result['packages'])} package(s) for this host "
                     f"(settled after {result.get('passes', 1)} pass(es) -- corrections were re-checked "
                     f"against each other, not just applied blind).")
    else:
        lines.append(f"⚠️ Found issues, and correcting one package kept introducing conflicts with another "
                     f"across {result.get('passes', 1)} pass(es) -- this is my best attempt, but it's "
                     f"**not fully clean**. Please review before using it.")
    lines.append(f"\nOriginal:\n```\n{result['original_code'].strip()}\n```")
    lines.append(f"\nCorrected:\n```\n{result['corrected_code'].strip()}\n```")
    lines.append("\nWhy:")
    for p in result["packages"]:
        if p["changed"] or not p["is_clean"]:
            lines.append(f"- **{p['package']}** ({p['requested_version']} → {p['corrected_version']}): {p['explanation']}")
    return "\n".join(lines)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat-style dependency verification. Send a message containing one or more
    `package==version` pins (e.g. code an LLM just generated); this parses them,
    profiles this host's OS/CPU architecture/GPU, checks each pin against the ML
    model + live PyPI + each other's declared dependencies, and replies in
    natural language -- unchanged if everything's fine, corrected + explained
    if not.

    Request body:
        { "message": "pip install torch==2.8.0 torchvision==0.17.0" }
        Optional overrides: "python_version", "platform" (skips auto-detection),
        "live": false (ML-only, skips PyPI calls).

    Response:
        { "reply": "...", "host_profile": {...}, "validation": {...} | null,
          "parsed_pins_found": bool }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "JSON body with 'message' required"}), 400

    message = data["message"]
    host_profile = detect_host()
    python_version = data.get("python_version", host_profile["python_version"])
    platform_key = data.get("platform", host_profile["platform_key"])

    pins = parse_pip_install(message)
    stack_label = None

    if not pins:
        # No explicit pins -- see if the message describes a project type we
        # have a known-good stack for (only packages the model has real data on).
        if chat_model is not None:
            stack_label, stack_packages = suggest_stack(message)
            if stack_label:
                built_pins = []
                for pkg in stack_packages:
                    rec = chat_model.recommend(pkg, python_version, platform_key, top_n=1)
                    if rec:
                        built_pins.append(f"{pkg}=={rec[0]['version']}")
                if built_pins:
                    message = "pip install " + " ".join(built_pins)
                    pins = parse_pip_install(message)

        if not pins:
            reply = _format_chat_reply(message, host_profile, {"packages": [], "changed": False, "original_code": message, "corrected_code": message}, stack_label=stack_label)
            return jsonify({"reply": reply, "host_profile": host_profile, "validation": None, "parsed_pins_found": False, "suggested_stack": stack_label})

    if chat_model is None:
        return jsonify({"error": "Chat assistant model not loaded"}), 503

    live = data.get("live", True)
    lowered_message = f" {message.lower()} "
    preset_platform, preset_label = _detect_deploy_preset(message)
    wants_deploy_check = preset_platform is not None or any(kw in lowered_message for kw in DEPLOY_KEYWORDS)

    if wants_deploy_check:
        deploy_platform = data.get("deploy_platform") or preset_platform or "linux_x86_64"
        # Prefer what the user SAYS their machine is over the server's actual
        # host -- "I'm on Windows, will this work on EC2" must compare against
        # Windows, not whatever machine happens to be running this dashboard.
        stated_local = data.get("local_platform") or _detect_stated_local_platform(message)
        local_platform_for_check = stated_local or platform_key
        wants_docker_verify = data.get("docker_verify", False) or any(
            kw in lowered_message for kw in ("verify with docker", "docker verify", "really test", "actually test", "docker test")
        )
        try:
            comparison = compare_local_vs_deploy(
                message, python_version, local_platform_for_check,
                deploy_platform=deploy_platform, model=chat_model, live=live,
                docker_verify=wants_docker_verify,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        reply = _format_deploy_reply(
            comparison, host_profile, stack_label=stack_label, deploy_label=preset_label,
            stated_local_platform=stated_local,
        )
        return jsonify({
            "reply": reply, "host_profile": host_profile,
            "deploy_comparison": comparison, "validation": comparison["local"],
            "parsed_pins_found": True, "suggested_stack": stack_label,
        })

    try:
        result = validate_install_code(message, python_version=python_version, platform=platform_key, model=chat_model, live=live)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    reply = _format_chat_reply(message, host_profile, result, stack_label=stack_label)
    return jsonify({"reply": reply, "host_profile": host_profile, "validation": result, "parsed_pins_found": True, "suggested_stack": stack_label})


@app.route("/deploy-check", methods=["POST"])
def deploy_check_route():
    """
    Compares packages that work on a local machine against a deployment
    target (AWS EC2, Graviton, Azure, Windows Server, Docker, etc.) and flags
    anything that would break only after deploying.

    Request body:
        { "code": "boto3==1.42.49\\npywin32==308", "python_version": "3.12",
          "local_platform": "win_amd64", "deploy_platform": "linux_x86_64",
          "docker_verify": false }

    docker_verify: when true, runs real `pip install` + `import` tests in
        Docker containers for the deploy side (linux targets only), catching
        failures no metadata check can see -- e.g. a package with a
        platform-universal wheel that transitively depends on something
        OS-locked. Slow (5-15s per package); off by default.

    Response:
        { "safe_to_deploy": bool, "regressions": [...], "local": {...}, "deploy": {...} }
    """
    data = request.get_json()
    if not data or not data.get("code", "").strip():
        return jsonify({"error": "code is required"}), 400

    if chat_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        result = compare_local_vs_deploy(
            data["code"],
            python_version=data.get("python_version", "3.12"),
            local_platform=data.get("local_platform", "darwin_x86_64"),
            deploy_platform=data.get("deploy_platform", "linux_x86_64"),
            deploy_python_version=data.get("deploy_python_version"),
            model=chat_model,
            live=data.get("live", True),
            docker_verify=data.get("docker_verify", False),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """Manually trigger model retraining."""
    success = auto_retrain()
    if success:
        return jsonify({"status": "success", "message": "Models retrained successfully"})
    return jsonify({"status": "error", "message": "Retraining failed"}), 500


@app.route("/host", methods=["GET"])
def host():
    """Detected host hardware profile (OS, CPU architecture, Python version, GPU)."""
    return jsonify(detect_host())


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
    init_chat_model()

    # Start background watcher for auto-retrain
    watcher = threading.Thread(target=check_data_changes, daemon=True)
    watcher.start()

    print("\n🚀 Package Compatibility API running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
