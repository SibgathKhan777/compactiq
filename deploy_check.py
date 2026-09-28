"""
deploy_check.py - Compares dependency compatibility between where you're
developing (usually your laptop) and where you're actually going to deploy
(usually a Linux container/VM) -- the classic "works on my machine, breaks in
prod" gap.

Reuses validate.py's full pipeline (ML + live PyPI + joint dependency
conflicts) once per platform, and diffs the results per package so the report
surfaces only what actually CHANGES between the two environments -- not
everything, just the deltas that would bite you at deploy time. This is
exactly the scenario the live PyPI layer exists for: the training data has
zero platform diversity beyond a small linux_x86_64 sample (see
eval_temporal_holdout.py), so a local-vs-deploy comparison leans almost
entirely on live wheel-availability checks, not the ML model's platform
signal, which is honest -- there isn't one yet.
"""

from validate import validate_install_code

# Common deployment targets. linux_x86_64 covers the overwhelming majority of
# cloud/Docker/CI environments (Vercel, Render, Railway, most GitHub Actions
# runners, most AWS/GCP VM images); linux_aarch64 covers Graviton/Apple-Silicon
# cloud instances and Apple Silicon Docker Desktop's default build target.
COMMON_DEPLOY_TARGETS = ["linux_x86_64", "linux_aarch64", "win_amd64"]

# Named presets for common real-world deployment targets, so callers can say
# "AWS EC2" instead of needing to already know the right platform_key string.
DEPLOY_PRESETS = {
    "aws_ec2": {"platform": "linux_x86_64", "label": "AWS EC2 (x86_64, standard/Amazon Linux)"},
    "aws_ec2_x86": {"platform": "linux_x86_64", "label": "AWS EC2 (x86_64, standard/Amazon Linux)"},
    "aws_graviton": {"platform": "linux_aarch64", "label": "AWS EC2 Graviton (ARM64)"},
    "aws_lambda": {"platform": "linux_x86_64", "label": "AWS Lambda (x86_64 default runtime)"},
    "gcp_compute": {"platform": "linux_x86_64", "label": "Google Compute Engine (x86_64)"},
    "azure_vm": {"platform": "linux_x86_64", "label": "Azure VM (x86_64)"},
    "docker_linux": {"platform": "linux_x86_64", "label": "Docker (linux/amd64)"},
    "docker_linux_arm": {"platform": "linux_aarch64", "label": "Docker (linux/arm64)"},
    "windows_server": {"platform": "win_amd64", "label": "Windows Server"},
}


def _is_installable(live_result):
    """
    True/False if the live PyPI check has an opinion, None if live checking
    was skipped (live=False) or the check itself failed to reach PyPI.
    """
    if live_result is None:
        return None
    if live_result.get("exists") is not True:
        return None  # version doesn't exist at all -- not a platform-specific signal
    return bool(live_result.get("wheel_available")) or bool(live_result.get("sdist_available"))


def compare_local_vs_deploy(code, python_version, local_platform, deploy_platform="linux_x86_64",
                             deploy_python_version=None, model=None, live=True, docker_verify=False, use_llm=False):
    """
    Args:
        code: pip install code that's known to work locally
        python_version: local Python version
        local_platform: platform_key of the local dev machine (see hardware_profile.detect_host())
        deploy_platform: target deployment platform_key (default "linux_x86_64" -- the most
            common Docker/cloud target)
        deploy_python_version: Python version on the deploy target, if it differs from local
            (e.g. a Docker base image pinned to a specific version). Defaults to python_version.
        model: a loaded PyCompatModel, or None to load "./model"
        live: whether to hit PyPI live -- recommended, this check specifically needs ground
            truth for a platform the model has essentially no training signal for.
        docker_verify: when True, actually runs `pip install` + `import` in a real
            container for the DEPLOY side (only linux_x86_64/linux_aarch64 targets are
            Docker-testable; ignored for other targets and for the local side, which the
            caller has already told us works). Catches failures no metadata check can see
            -- e.g. a package with a platform-universal wheel that transitively depends
            on something OS-locked. Slow (5-15s per package) and off by default.
        use_llm: when True, consults an LLM for any package outside the trained
            catalog on the DEPLOY side, surfaced as a labeled, unverified second
            opinion -- and forces real Docker verification for that package too,
            regardless of docker_verify. No-ops safely if GROQ_API_KEY isn't set.

    Returns:
        {
          "local_platform", "deploy_platform": the two platform_keys compared,
          "local", "deploy": full validate_install_code() results for each,
          "regressions": [ {package, version, reason}, ... ]  -- clean locally, broken on deploy,
          "safe_to_deploy": bool,
        }
    """
    deploy_python_version = deploy_python_version or python_version

    local_result = validate_install_code(code, python_version, local_platform, model=model, live=live)
    deploy_result = validate_install_code(
        code, deploy_python_version, deploy_platform, model=model, live=live,
        docker_verify=docker_verify, use_llm=use_llm
    )

    local_by_pkg = {p["package"]: p for p in local_result["packages"]}
    deploy_by_pkg = {p["package"]: p for p in deploy_result["packages"]}

    regressions = []       # fine as originally pinned locally, broken specifically on deploy
    already_broken = []    # the pin you gave wasn't safe on EITHER platform -- not a deploy-specific
                            # regression at all, just an invalid pin. validate_install_code() auto-
                            # corrects broken pins internally (same behavior /api/validate uses), so if
                            # a package needed correcting on BOTH local and deploy, there's no visible
                            # difference between the two runs for the regression check above to catch --
                            # without this bucket, a package that was never installable anywhere (a
                            # hallucinated/nonexistent package, or one so old it can't build) silently
                            # gets corrected to a working version on both sides and reports as "safe to
                            # deploy", even though what you actually asked about was never viable at all.
    for pkg, local_p in local_by_pkg.items():
        deploy_p = deploy_by_pkg.get(pkg)
        if deploy_p is None:
            continue

        local_installable = _is_installable(local_p["live_result"])
        deploy_installable = _is_installable(deploy_p["live_result"])

        # Hard-fail regression: PyPI ground truth says it installs locally but
        # has NEITHER a wheel NOR an sdist on the deploy platform (e.g. pywin32
        # on Linux). This is driven by the live wheel-availability check, not
        # the ML model's is_clean flag -- a package outside the trained
        # catalog gets flagged "not clean" on EVERY platform for the same
        # generic "novel package" reason, which would otherwise mask exactly
        # this case: is_clean never flips from True to False because it was
        # already False locally, even though locally it actually installs
        # fine and on deploy it categorically cannot.
        if local_installable is True and deploy_installable is False:
            regressions.append({
                "package": pkg,
                "version": local_p["requested_version"],
                "severity": "hard_fail",
                "reason": deploy_p["explanation"],
            })
        # Softer regression: was clean locally, isn't clean on deploy (conflicts,
        # newly-surfaced platform markers, etc.) -- the original flip check.
        elif local_p["is_clean"] and not deploy_p["is_clean"]:
            regressions.append({
                "package": pkg,
                "version": local_p["requested_version"],
                "severity": "flagged",
                "reason": deploy_p["explanation"],
            })
        # Neither -- but if it's unclean on BOTH sides, the original pin
        # wasn't safe regardless of platform. Deliberately NOT gated on
        # "changed": a totally novel package with no catalog entry at all
        # gets no correction candidate either (suggest_correction() returns
        # None), so corrected_version never differs from requested_version --
        # but it's still broken everywhere, and that's the case this bucket
        # exists to catch.
        elif not local_p["is_clean"] and not deploy_p["is_clean"]:
            already_broken.append({
                "package": pkg,
                "requested_version": local_p["requested_version"],
                "local_corrected_to": local_p["corrected_version"],
                "deploy_corrected_to": deploy_p["corrected_version"],
                "reason": deploy_p["explanation"],
            })

    # Surfaced at the top level (mirroring deploy_result's own flags) so a
    # caller doesn't have to know to dig into "deploy" to tell "Docker
    # verification was requested and actually ran" apart from "was requested
    # but the Docker daemon/binary isn't present on this host, so every
    # docker_result silently stayed null" -- those two cases produce an
    # otherwise-identical response, which is exactly what made this look like
    # a broken feature rather than an unmet prerequisite.
    docker_verify_ineffective = (
        docker_verify and deploy_result.get("docker_verify_available") is False
    )

    return {
        "local_platform": local_platform,
        "local_python_version": python_version,
        "deploy_platform": deploy_platform,
        "deploy_python_version": deploy_python_version,
        "local": local_result,
        "deploy": deploy_result,
        "docker_verify_requested": docker_verify,
        "docker_verify_available": deploy_result.get("docker_verify_available"),
        "docker_verify_ineffective": docker_verify_ineffective,
        "regressions": regressions,
        "already_broken": already_broken,
        # "Safe to deploy" means what you actually pinned works on the target,
        # exactly as-is -- NOT "some version of this exists that works
        # everywhere". A package that needed correcting on both platforms
        # (already_broken) is just as much a reason to say "not safe" as a
        # deploy-specific regression.
        "safe_to_deploy": len(regressions) == 0 and len(already_broken) == 0,
    }


if __name__ == "__main__":
    import sys
    import json

    code = sys.argv[1] if len(sys.argv) > 1 else "pip install boto3==1.42.49"
    py_version = sys.argv[2] if len(sys.argv) > 2 else "3.12"
    local_plat = sys.argv[3] if len(sys.argv) > 3 else "darwin_x86_64"
    deploy_plat = sys.argv[4] if len(sys.argv) > 4 else "linux_x86_64"

    result = compare_local_vs_deploy(code, py_version, local_plat, deploy_plat)
    print(json.dumps({
        "local_platform": result["local_platform"],
        "deploy_platform": result["deploy_platform"],
        "safe_to_deploy": result["safe_to_deploy"],
        "regressions": result["regressions"],
    }, indent=2, default=str))
