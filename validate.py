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


def _evaluate_pins(pins_dict, python_version, platform, model, live):
    """
    One evaluation pass over a fixed set of {package: version} pins: joint
    conflict check + per-package ML/live verdict + a proposed correction for
    anything not clean. Does NOT re-check corrections against each other --
    that's what the caller's settle loop is for. Returns
    (per_package_list, conflict_report, next_pins_dict, any_change).
    """
    if live:
        conflict_report = check_batch(pins_dict, python_version, platform)
    else:
        conflict_report = {"conflicts": [], "unresolved": [], "is_jointly_compatible": True}

    conflicts_by_pkg = {}
    for c in conflict_report["conflicts"]:
        conflicts_by_pkg.setdefault(c["depends_on"], []).append(c)

    per_package = []
    next_pins = {}
    marker_exclusions = {}  # pkg -> {"version": ..., "marker": ...}; removed from next_pins entirely
    any_change = False

    for pkg, version in pins_dict.items():
        ml_result = model.predict(pkg, version, python_version, platform)

        live_result = None
        if live:
            live_result = live_verify.check_platform_wheel(pkg, version, platform, python_version)

        pkg_conflicts = conflicts_by_pkg.get(pkg, [])
        verdict = explain_package_result(ml_result, live_result, pkg_conflicts)

        chosen_version = version
        if not verdict["is_clean"]:
            live_check_fn = live_verify.check_platform_wheel if live else None
            # Combine every declared constraint another package in this batch
            # placed on `pkg` (e.g. "keyring requires pywin32-ctypes>=0.2.0")
            # into one specifier, so the correction can't pick a version that's
            # "compatible" per the model but still violates a real PyPI-declared
            # requirement.
            specifiers = [c["required_specifier"] for c in pkg_conflicts if c.get("required_specifier") not in (None, "any")]
            combined_specifier = ",".join(specifiers) if specifiers else None
            correction = suggest_correction(
                model, pkg, python_version, platform,
                live_check_fn=live_check_fn, required_specifier=combined_specifier,
            )
            if correction:
                if correction.get("marker_exclude"):
                    # Feeding a marker-annotated string back into the settle
                    # loop's version parsing would just break things (it's
                    # not a real version). Lock this package's fate in now and
                    # drop it from next_pins entirely -- future passes never
                    # touch it again; the marker gets applied directly to
                    # corrected_code once the loop finishes.
                    marker_exclusions[pkg] = {"version": version, "marker": correction["marker"]}
                    any_change = True
                elif correction["version"] != version:
                    chosen_version = correction["version"]
                    any_change = True

        if pkg not in marker_exclusions:
            next_pins[pkg] = chosen_version
        per_package.append({
            "package": pkg,
            "version_this_pass": version,
            "corrected_version": chosen_version,
            "ml_result": ml_result,
            "live_result": live_result,
            "conflicts": pkg_conflicts,
            **verdict,
        })

    return per_package, conflict_report, next_pins, any_change, marker_exclusions


def validate_install_code(code, python_version="3.12", platform="darwin_x86_64", model=None, live=True, max_passes=3):
    """
    Args:
        code: raw text containing one or more `package==version` pins
              (e.g. "pip install torch==2.8.0 torchvision==0.17.0")
        python_version: target Python version, e.g. "3.12"
        platform: target platform key, e.g. "darwin_x86_64"
        model: a loaded PyCompatModel, or None to load "./model"
        live: whether to hit PyPI for live verification + dependency conflicts.
              Set False to run ML-only (useful offline, or in tests).
        max_passes: correcting one package can introduce a NEW conflict with
              another (e.g. bumping A to satisfy B's requirement now violates
              C's pin on A) -- a single correction pass doesn't catch that, since
              suggest_correction() only sees one package at a time. This re-runs
              the full joint check on the corrected set, up to max_passes times,
              until nothing changes or the cap is hit.

    Returns a dict matching the shape of the middleware repo's
    /api/validate-llm-code response: original code, corrected code, an overall
    risk score, a per-package breakdown with explanations, and whether it
    actually settled into a fully clean state (`fully_resolved`) or gave up
    at the pass cap still carrying issues -- reported honestly rather than
    silently returning a "corrected" line that isn't actually clean.
    """
    model = model or PyCompatModel.load("./model")
    pins = parse_pip_install(code)

    if not pins:
        return {
            "original_code": code, "corrected_code": code, "changed": False,
            "risk_score": 0.0, "packages": [], "passes": 0, "fully_resolved": True,
            "explanation": "No `package==version` pins found to validate.",
        }

    original_pins = {pkg: ver for pkg, ver in pins}
    current_pins = dict(original_pins)

    # Tracks, per package, the explanation from whichever pass most recently
    # found it NOT clean -- i.e. the actual reason it got corrected. Without
    # this, a package that settles clean after correction would show its final
    # (now-clean) "looks compatible" explanation instead of why it changed,
    # making the "Why" section empty for every package the loop just fixed.
    issue_log = {}

    all_marker_exclusions = {}  # pkg -> {"version": ..., "marker": ...}, accumulated across passes

    per_package, conflict_report, next_pins, any_change = None, None, None, True
    passes_run = 0
    for passes_run in range(1, max_passes + 1):
        per_package, conflict_report, next_pins, any_change, marker_exclusions = _evaluate_pins(
            current_pins, python_version, platform, model, live
        )
        all_marker_exclusions.update(marker_exclusions)
        for p in per_package:
            if not p["is_clean"]:
                issue_log[p["package"]] = {"explanation": p["explanation"], "risk_score": p["risk_score"]}
        if not any_change:
            break
        current_pins = next_pins

    # A marker-excluded package is a legitimate resolution (we determined it
    # genuinely doesn't belong on this platform and handled that correctly),
    # not an unresolved issue -- don't let its last-seen is_clean=False (from
    # the pass where it got flagged, before exclusion) count against
    # fully_resolved.
    fully_resolved = len(conflict_report["conflicts"]) == 0 and all(
        p["is_clean"] or p["package"] in all_marker_exclusions for p in per_package
    )
    max_risk = max((p["risk_score"] for p in per_package), default=0.0)

    corrected_code = code
    final_report = []
    for pkg, orig_ver in original_pins.items():
        if pkg in all_marker_exclusions:
            marker = all_marker_exclusions[pkg]["marker"]
            marker_display = f"{orig_ver}; {marker}"
            # corrected_code is meant to be copy-pasted as a real shell command
            # or requirements.txt line. An unquoted `;` inside it would make a
            # shell split "pip install ... pkg==X; sys_platform == ..." into
            # multiple broken commands at every semicolon -- wrap the whole
            # marker-conditioned requirement in single quotes so it's one token.
            corrected_code = corrected_code.replace(f"{pkg}=={orig_ver}", f"'{pkg}=={marker_display}'")
            issue = issue_log.get(pkg)
            final_report.append({
                "package": pkg,
                "requested_version": orig_ver,
                "corrected_version": marker_display,
                "changed": True,
                "ml_result": None,
                "live_result": None,
                "conflicts": [],
                "risk_score": 0.0,
                "explanation": (
                    f"{pkg} only ever publishes artifacts for one OS family ({marker.split(chr(34))[1]}) -- "
                    f"no version bump fixes this, it's not built for other platforms at all. Marked with "
                    f"an environment marker so `pip install -r requirements.txt` skips it here instead of "
                    f"failing. Previously: {issue['explanation'] if issue else 'no wheel or source distribution for this platform.'}"
                ),
                "is_clean": True,
            })
            continue

        final_ver = current_pins[pkg]
        changed = final_ver != orig_ver
        if changed:
            corrected_code = corrected_code.replace(f"{pkg}=={orig_ver}", f"{pkg}=={final_ver}")
        matching = next((p for p in per_package if p["package"] == pkg), None)
        issue = issue_log.get(pkg)
        final_report.append({
            "package": pkg,
            "requested_version": orig_ver,
            "corrected_version": final_ver,
            "changed": changed,
            "ml_result": matching["ml_result"] if matching else None,
            "live_result": matching["live_result"] if matching else None,
            "conflicts": matching["conflicts"] if matching else [],
            "risk_score": issue["risk_score"] if issue else (matching["risk_score"] if matching else 0.0),
            "explanation": issue["explanation"] if issue else (matching["explanation"] if matching else ""),
            "is_clean": matching["is_clean"] if matching else True,
        })

    result = {
        "original_code": code,
        "corrected_code": corrected_code,
        "changed": corrected_code != code,
        "risk_score": round(max_risk, 3),
        "joint_dependency_conflicts": conflict_report["conflicts"],
        "unresolved_packages": conflict_report["unresolved"],
        "packages": final_report,
        "passes": passes_run,
        "fully_resolved": fully_resolved,
    }
    if not fully_resolved:
        if passes_run >= max_passes:
            reason = f"corrections kept introducing new conflicts across all {passes_run} pass(es)"
        else:
            reason = "no compatible alternative version was found for at least one package (often because it's outside the trained catalog)"
        result["explanation"] = (
            f"Could not fully resolve -- {reason}. Returning the best attempt; "
            f"review the remaining issues below before using this."
        )
    return result


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "pip install boto3==1.42.49"
    pyver = sys.argv[2] if len(sys.argv) > 2 else "3.12"
    plat = sys.argv[3] if len(sys.argv) > 3 else "darwin_x86_64"
    result = validate_install_code(code, pyver, plat)
    print(json.dumps(result, indent=2, default=str))
