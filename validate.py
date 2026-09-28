"""
validate.py - CompactIQ's equivalent of the middleware repo's
`POST /api/validate-llm-code`: given LLM-generated `pip install ...` code,
validate every pin against the ML model, live PyPI, AND each other (joint
dependency conflicts), then return a risk-scored explanation with a corrected
install line.

By default this never spins up Docker and never calls an LLM for the explanation
-- everything is either a trained classifier already in this repo
(pycompat_model.py) or a live, cheap PyPI metadata lookup (live_verify.py,
dependency_conflicts.py), consistent with compactiq's "lightweight, no queue, no
containers" design instead of duplicating the middleware's service stack. Metadata
alone can still be wrong, though -- a package can publish a platform-universal
wheel while transitively depending on something OS-locked (confirmed real case:
`wmi`), which no static check catches. Pass docker_verify=True to get real ground
truth via docker_verify.py: it actually runs `pip install` + `import` in a
container for anything that looks clean, opt-in because each container run costs
5-15+ seconds.

Pass use_llm=True to also consult an LLM (llm_fallback.py, via Groq) for
packages outside the trained catalog (is_known_package=False) -- exactly the
case where pycompat_model.py's own prediction is documented as "an
extrapolation, not a real signal." The LLM's opinion is surfaced labeled and
UNVERIFIED, never as fact (it can hallucinate a package's existence just as
easily as the classifier can guess wrong), and enabling it automatically
forces real Docker verification for that specific package regardless of the
docker_verify flag -- an unknown-to-ML package is exactly the case that most
needs real ground truth, LLM opinion or not. No-ops safely (available=False)
if GROQ_API_KEY isn't set.
"""

import re
import sys
import json

from pycompat_model import PyCompatModel
import live_verify
from dependency_conflicts import check_batch
from explain import explain_package_result, suggest_correction
from docker_verify import docker_verify_install
from llm_fallback import ask_llm_about_package

PIN_RE = re.compile(r"([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-]+)")


def parse_pip_install(code):
    """Extract (package, version) pins from a `pip install ...` line or block."""
    return PIN_RE.findall(code)


def _evaluate_pins(pins_dict, python_version, platform, model, live, docker_verify=False, use_llm=False):
    """
    One evaluation pass over a fixed set of {package: version} pins: joint
    conflict check + per-package ML/live verdict + a proposed correction for
    anything not clean. Does NOT re-check corrections against each other --
    that's what the caller's settle loop is for. Returns
    (per_package_list, conflict_report, next_pins_dict, any_change).

    docker_verify: when True and the package LOOKS clean from metadata alone,
        actually run a real `pip install` + `import` test in a container
        before trusting that verdict (see docker_verify.py). This catches
        failures metadata can't see -- e.g. a package with a platform-
        universal wheel that transitively depends on something OS-locked.
        Only runs for packages that pass the cheap metadata check first, to
        keep the (slow, 5-15s/container) Docker calls bounded.

    use_llm: when True, consults an LLM for any package outside the trained
        catalog (see llm_fallback.py). Its opinion is folded into the
        explanation labeled as unverified, and -- regardless of the
        docker_verify flag -- forces a real Docker verification for that
        package too, since an unknown-to-ML package is exactly the case that
        most needs real ground truth.
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
    package_renames = {}  # pkg -> {"new_name": ..., "version": ...}; removed from next_pins entirely
    any_change = False

    for pkg, version in pins_dict.items():
        ml_result = model.predict(pkg, version, python_version, platform)

        live_result = None
        if live:
            live_result = live_verify.check_platform_wheel(pkg, version, platform, python_version)

        pkg_conflicts = conflicts_by_pkg.get(pkg, [])
        verdict = explain_package_result(ml_result, live_result, pkg_conflicts)

        llm_result = None
        force_docker_for_unknown = False
        if use_llm and ml_result.get("is_known_package") is False:
            llm_result = ask_llm_about_package(pkg, version, python_version, platform)
            force_docker_for_unknown = True
            if llm_result.get("available"):
                llm_label = (llm_result["verdict"] or "unknown").upper()
                verdict = {**verdict, "explanation": (
                    f"{verdict['explanation']} LLM second opinion (unverified, NOT ground truth -- "
                    f"an LLM can hallucinate a package's existence just as easily as the classifier "
                    f"can guess wrong): {llm_label} -- {llm_result['opinion']}"
                )}
            else:
                # Surfaced explicitly rather than silently doing nothing --
                # the same transparency fix applied earlier for
                # docker_verify_ineffective. Requesting an LLM opinion and
                # getting a normal result back should never look identical to
                # it actually running when GROQ_API_KEY is missing/invalid or
                # the request failed.
                verdict = {**verdict, "explanation": (
                    f"{verdict['explanation']} (LLM second opinion was requested but unavailable: "
                    f"{llm_result.get('error')})"
                )}

        docker_result = None
        docker_confirmed_failure = False
        if docker_verify or force_docker_for_unknown:
            # Metadata can be DEFINITIVELY certain something is impossible
            # (no wheel, no sdist at all -- literally nothing to install) --
            # no point spending 5-15s confirming the obvious. Anything short
            # of that (including "flagged only because it's outside the
            # trained catalog", which is a statistical hedge, not a real
            # problem) is worth actually testing: that hedge is exactly the
            # uncertainty a real install result can settle either way.
            live_says_impossible = (
                live_result is not None
                and live_result.get("exists") is True
                and not live_result.get("wheel_available", True)
                and not live_result.get("sdist_available", True)
            )
            if not live_says_impossible:
                # No wheel but an sdist exists -- this is exactly the
                # ambiguous case a bare test container gets wrong (no
                # compiler by default), so provision it like a real EC2
                # instance would be (build-essential etc.) before testing.
                needs_build_tools = (
                    live_result is not None
                    and live_result.get("exists") is True
                    and not live_result.get("wheel_available", True)
                    and live_result.get("sdist_available", False)
                )
                docker_result = docker_verify_install(
                    pkg, version, python_version, platform, install_build_tools=needs_build_tools
                )

            if docker_result is not None and docker_result.get("install_success") is not None:
                actually_works = docker_result["install_success"] and docker_result["import_success"]
                docker_confirmed_failure = not actually_works
                # A real container result is stronger evidence than any
                # static signal (ML novelty hedge, "might need a build
                # toolchain" caution, etc.) -- always lead with it, whichever
                # way it goes, rather than only overriding when it happens to
                # flip the prior is_clean value. Otherwise a Docker-confirmed
                # failure for something ALREADY flagged unclean for an
                # unrelated, vaguer reason (e.g. wmi's "novel package" hedge)
                # never surfaces the actual, specific, correct root cause
                # (wmi transitively depends on pywin32, which doesn't exist
                # on Linux) in the visible explanation.
                if actually_works:
                    verdict = {
                        "risk_score": 0.0,
                        "is_clean": True,
                        "explanation": (
                            f"Confirmed via a real Docker install+import test on {platform}/py{python_version}: "
                            f"`pip install {pkg}=={version}` succeeds. (Static checks alone said: "
                            f"{verdict['explanation']})"
                        ),
                    }
                else:
                    verdict = {
                        "risk_score": max(verdict["risk_score"], 0.7),
                        "is_clean": False,
                        "explanation": (
                            f"Confirmed via a real Docker install test on {platform}/py{python_version}: "
                            f"`pip install {pkg}=={version}` actually FAILS "
                            f"({docker_result.get('error_type', 'unknown')}) -- "
                            f"{(docker_result.get('error_log_snippet') or '').strip()[-250:]} "
                            f"(Static checks alone said: {verdict['explanation']})"
                        ),
                    }

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
                trust_marker_exclusion=docker_confirmed_failure,
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
                elif correction.get("package_rename"):
                    # A real -binary sibling package works (e.g. psycopg2 ->
                    # psycopg2-binary) -- this changes the PACKAGE NAME, not
                    # just the version, so like marker exclusions it can't
                    # flow back through next_pins under the old name.
                    package_renames[pkg] = {"new_name": correction["package_rename"], "version": correction["version"]}
                    any_change = True
                elif correction["version"] != version:
                    chosen_version = correction["version"]
                    any_change = True

        if pkg not in marker_exclusions and pkg not in package_renames:
            next_pins[pkg] = chosen_version
        per_package.append({
            "package": pkg,
            "version_this_pass": version,
            "corrected_version": chosen_version,
            "ml_result": ml_result,
            "live_result": live_result,
            "docker_result": docker_result,
            "llm_result": llm_result,
            "conflicts": pkg_conflicts,
            **verdict,
        })

    return per_package, conflict_report, next_pins, any_change, marker_exclusions, package_renames


def validate_install_code(code, python_version="3.12", platform="darwin_x86_64", model=None, live=True,
                           max_passes=3, docker_verify=False, use_llm=False):
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
        docker_verify: when True, actually runs `pip install` + `import` in a
              real Docker container for anything that looks clean from
              metadata alone, catching failures metadata can't see (e.g. a
              transitive dependency that's OS-locked). Only works for
              linux_x86_64/linux_aarch64 targets and requires a running Docker
              daemon; slow (5-15s per container) and OFF by default -- opt in
              when you want real ground truth, not just PyPI metadata.
        use_llm: when True, consults an LLM (llm_fallback.py) for any package
              outside the trained catalog -- surfaced labeled as an unverified
              second opinion, never as fact -- and forces real Docker
              verification for that package regardless of docker_verify.
              No-ops safely if GROQ_API_KEY isn't set.

    Returns a dict matching the shape of the middleware repo's
    /api/validate-llm-code response: original code, corrected code, an overall
    risk score, a per-package breakdown with explanations, and whether it
    actually settled into a fully clean state (`fully_resolved`) or gave up
    at the pass cap still carrying issues -- reported honestly rather than
    silently returning a "corrected" line that isn't actually clean.
    """
    model = model or PyCompatModel.load("./model")
    pins = parse_pip_install(code)

    # Checked once up front (not the per-package silent None docker_verify_install()
    # already returns when unavailable) so the caller can tell "Docker verification
    # was requested and genuinely ran" apart from "was requested but silently did
    # nothing" -- e.g. the Docker daemon/binary isn't present on this host at all.
    # Without this, checking "verify with Docker" in the UI and getting a normal
    # result back looks identical whether or not any container ever actually ran.
    docker_verify_available = None
    if docker_verify or use_llm:
        from docker_verify import docker_available
        docker_verify_available = docker_available()

    if not pins:
        return {
            "original_code": code, "corrected_code": code, "changed": False,
            "risk_score": 0.0, "packages": [], "passes": 0, "fully_resolved": True,
            "explanation": "No `package==version` pins found to validate.",
            "docker_verify_requested": docker_verify,
            "docker_verify_available": docker_verify_available,
            "use_llm": use_llm,
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
    all_package_renames = {}  # pkg -> {"new_name": ..., "version": ...}, accumulated across passes

    per_package, conflict_report, next_pins, any_change = None, None, None, True
    passes_run = 0
    for passes_run in range(1, max_passes + 1):
        per_package, conflict_report, next_pins, any_change, marker_exclusions, package_renames = _evaluate_pins(
            current_pins, python_version, platform, model, live, docker_verify=docker_verify, use_llm=use_llm
        )
        all_marker_exclusions.update(marker_exclusions)
        all_package_renames.update(package_renames)
        for p in per_package:
            if not p["is_clean"]:
                issue_log[p["package"]] = {"explanation": p["explanation"], "risk_score": p["risk_score"]}
        if not any_change:
            break
        current_pins = next_pins

    # A marker-excluded or renamed package is a legitimate resolution (we
    # determined it genuinely doesn't belong here, or found a real working
    # sibling package), not an unresolved issue -- don't let its last-seen
    # is_clean=False (from the pass where it got flagged, before resolution)
    # count against fully_resolved.
    fully_resolved = len(conflict_report["conflicts"]) == 0 and all(
        p["is_clean"] or p["package"] in all_marker_exclusions or p["package"] in all_package_renames
        for p in per_package
    )
    max_risk = max((p["risk_score"] for p in per_package), default=0.0)

    # Marker-conditioned requirements ("pkg==X; sys_platform == ...") need
    # DIFFERENT quoting depending on what `code` actually is:
    #  - a shell command line ("pip install a b c") needs the whole
    #    requirement wrapped in quotes, or the shell would split it into
    #    broken commands at the unquoted semicolon.
    #  - real requirements.txt file content (one requirement per line, no
    #    "pip install" prefix -- e.g. what fix_project_requirements.py reads
    #    off disk) must NOT be quoted: pip's requirements.txt parser follows
    #    PEP 508 grammar directly, no shell involved, and a literal leading
    #    quote character breaks parsing entirely ("Expected package name").
    # "pip install" only appears when this is shell-command-style input, so
    # it's a reliable signal for which context we're producing output for.
    is_shell_command = "pip install" in code.lower()

    corrected_code = code
    final_report = []
    for pkg, orig_ver in original_pins.items():
        if pkg in all_marker_exclusions:
            marker = all_marker_exclusions[pkg]["marker"]
            marker_display = f"{orig_ver}; {marker}"
            replacement = f"'{pkg}=={marker_display}'" if is_shell_command else f"{pkg}=={marker_display}"
            corrected_code = corrected_code.replace(f"{pkg}=={orig_ver}", replacement)
            issue = issue_log.get(pkg)
            final_report.append({
                "package": pkg,
                "requested_version": orig_ver,
                "corrected_version": marker_display,
                "changed": True,
                "ml_result": None,
                "live_result": None,
                "docker_result": None,
                "llm_result": None,
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

        if pkg in all_package_renames:
            new_name = all_package_renames[pkg]["new_name"]
            new_version = all_package_renames[pkg]["version"]
            corrected_code = corrected_code.replace(f"{pkg}=={orig_ver}", f"{new_name}=={new_version}")
            issue = issue_log.get(pkg)
            final_report.append({
                "package": pkg,
                "requested_version": orig_ver,
                "corrected_version": f"{new_name}=={new_version}",
                "changed": True,
                "ml_result": None,
                "live_result": None,
                "docker_result": None,
                "llm_result": None,
                "conflicts": [],
                "risk_score": 0.0,
                "explanation": (
                    f"{pkg} has no working wheel here, but its prebuilt sibling package {new_name} does "
                    f"(the standard real-world fix -- e.g. psycopg2 -> psycopg2-binary). Switched to "
                    f"{new_name}=={new_version}. Previously: "
                    f"{issue['explanation'] if issue else 'no wheel for this platform.'}"
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
            "docker_result": matching["docker_result"] if matching else None,
            "llm_result": matching["llm_result"] if matching else None,
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
        "docker_verify_requested": docker_verify,
        "docker_verify_available": docker_verify_available,
        "use_llm": use_llm,
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
