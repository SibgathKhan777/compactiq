"""
validate.py - CompactIQ's equivalent of the middleware repo's
`POST /api/validate-llm-code`: given LLM-generated `pip install ...` code,
validate every pin against the ML model, live PyPI, AND each other (joint
dependency conflicts), then return a risk-scored explanation with a corrected
install line.

Unlike the middleware, this never spins up Docker and never calls an LLM for the
explanation. Everything here is either a trained classifier already in this repo
(pycompat_model.py) or a live, cheap PyPI metadata lookup (live_verify.py,
dependency_conflicts.py) -- consistent with compactiq's "lightweight, no queue, no
containers" design instead of duplicating the middleware's service stack. The
tradeoff: this can't catch an install-succeeds-but-import-fails-at-runtime bug the
way the middleware's Docker install+import test can. It catches everything that's
knowable from PyPI's published metadata plus the trained model's per-package signal.
"""

import re
import sys
import json

from pycompat_model import PyCompatModel
import live_verify
from dependency_conflicts import check_batch
from explain import explain_package_result, suggest_correction

PIN_RE = re.compile(r"([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)")


def parse_pip_install(code):
    """Extract (package, version) pins from a `pip install ...` line or block."""
    return PIN_RE.findall(code)


def validate_install_code(code, python_version="3.12", platform="darwin_x86_64", model=None, live=True):
    """
    Args:
        code: raw text containing one or more `package==version` pins
              (e.g. "pip install torch==2.8.0 torchvision==0.17.0")
        python_version: target Python version, e.g. "3.12"
        platform: target platform key, e.g. "darwin_x86_64"
        model: a loaded PyCompatModel, or None to load "./model"
        live: whether to hit PyPI for live verification + dependency conflicts.
              Set False to run ML-only (useful offline, or in tests).

    Returns a dict matching the shape of the middleware repo's
    /api/validate-llm-code response: original code, corrected code, an overall
    risk score, and a per-package breakdown with explanations.
    """
    model = model or PyCompatModel.load("./model")
    pins = parse_pip_install(code)

    if not pins:
        return {
            "original_code": code, "corrected_code": code, "changed": False,
            "risk_score": 0.0, "packages": [],
            "explanation": "No `package==version` pins found to validate.",
        }

    pins_dict = {pkg: ver for pkg, ver in pins}

    if live:
        conflict_report = check_batch(pins_dict, python_version)
    else:
        conflict_report = {"conflicts": [], "unresolved": [], "is_jointly_compatible": True}

    conflicts_by_pkg = {}
    for c in conflict_report["conflicts"]:
        conflicts_by_pkg.setdefault(c["depends_on"], []).append(c)

    per_package = []
    corrected_pins = []
    max_risk = 0.0

    for pkg, version in pins:
        ml_result = model.predict(pkg, version, python_version, platform)

        live_result = None
        if live:
            live_result = live_verify.check_platform_wheel(pkg, version, platform, python_version)

        pkg_conflicts = conflicts_by_pkg.get(pkg, [])
        verdict = explain_package_result(ml_result, live_result, pkg_conflicts)
        max_risk = max(max_risk, verdict["risk_score"])

        chosen_version = version
        if not verdict["is_clean"]:
            live_check_fn = live_verify.check_platform_wheel if live else None
            correction = suggest_correction(model, pkg, python_version, platform, live_check_fn=live_check_fn)
            if correction and correction["version"] != version:
                chosen_version = correction["version"]

        corrected_pins.append((pkg, chosen_version))
        per_package.append({
            "package": pkg,
            "requested_version": version,
            "corrected_version": chosen_version,
            "changed": chosen_version != version,
            "ml_result": ml_result,
            "live_result": live_result,
            "conflicts": pkg_conflicts,
            **verdict,
        })

    corrected_code = code
    for (pkg, orig_ver), (_, new_ver) in zip(pins, corrected_pins):
        if new_ver != orig_ver:
            corrected_code = corrected_code.replace(f"{pkg}=={orig_ver}", f"{pkg}=={new_ver}")

    return {
        "original_code": code,
        "corrected_code": corrected_code,
        "changed": corrected_code != code,
        "risk_score": round(max_risk, 3),
        "joint_dependency_conflicts": conflict_report["conflicts"],
        "unresolved_packages": conflict_report["unresolved"],
        "packages": per_package,
    }


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "pip install boto3==1.42.49"
    pyver = sys.argv[2] if len(sys.argv) > 2 else "3.12"
    plat = sys.argv[3] if len(sys.argv) > 3 else "darwin_x86_64"
    result = validate_install_code(code, pyver, plat)
    print(json.dumps(result, indent=2, default=str))
