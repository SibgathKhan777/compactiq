"""
explain.py - Natural-language explanation and corrected-version selection.

Turns the raw signals from pycompat_model.py (ML prediction + novelty flags),
live_verify.py (PyPI ground truth) and dependency_conflicts.py (joint pin checking)
into the same contract the middleware repo's `POST /api/validate-llm-code` returns:
an explanation of what's wrong, plus a corrected version choice.

Built from templates, no LLM call required -- this matches the middleware's own
documented fallback behavior when no OPENAI_API_KEY is set ("a template-based
fallback is used"). CompactIQ never calls an LLM anywhere in this repo, by design.
"""


def explain_package_result(ml_result, live_result, conflicts_for_package):
    """
    ml_result: dict from PyCompatModel.predict() (includes is_known_package,
               is_known_platform, reliability, etc.)
    live_result: dict from live_verify.check_platform_wheel(), or None if live
                 checking was skipped/unreachable.
    conflicts_for_package: list of conflict dicts from dependency_conflicts.check_batch()
                            where depends_on == this package.
    """
    reasons = []
    risk = 0.0

    # Live PyPI evidence of "neither a wheel nor a source distribution exists
    # for this platform" is a definitive, 100%-certain fact -- not a
    # prediction. When we have it, the ML model's "never seen this package
    # during training" hedge is just noise stacked on top of a hard fact, and
    # reads as if the tool is LESS certain than it actually is. Only surface
    # the novelty hedge when live evidence doesn't already settle the question.
    definitive_live_failure = (
        live_result is not None
        and live_result.get("exists") is True
        and not live_result.get("wheel_available", True)
        and not live_result.get("sdist_available", True)
    )

    if not ml_result["is_known_package"] and not definitive_live_failure:
        reasons.append(
            f"'{ml_result['package']}' was never seen during training, so the ML "
            f"prediction ({'compatible' if ml_result['is_compatible'] else 'incompatible'}, "
            f"{ml_result['confidence']:.0%} confidence) is an extrapolation, not a real signal "
            f"-- package-holdout evaluation shows accuracy on unseen packages is below "
            f"the majority-class baseline, i.e. worse than a coin flip weighted by base rate."
        )
        risk += 0.3

    if not ml_result["is_known_platform"] and not definitive_live_failure:
        reasons.append(
            f"platform '{ml_result['platform']}' was never seen during training "
            f"(the training set only contains darwin_x86_64) -- the ML prediction for "
            f"this platform carries no real signal."
        )
        risk += 0.2

    if live_result is not None:
        if live_result.get("exists") is False:
            reasons.append(
                f"{ml_result['package']}=={ml_result['version']} does not exist on PyPI "
                f"right now (checked live) -- this version was likely hallucinated, "
                f"mistyped, or has been yanked since training."
            )
            risk += 0.6
        elif live_result.get("exists") is True and not live_result.get("wheel_available", True):
            if live_result.get("sdist_available"):
                reasons.append(
                    f"PyPI has no prebuilt wheel for {ml_result['package']}=={ml_result['version']} "
                    f"on {ml_result['platform']}/py{ml_result['python_version']} -- only a source "
                    f"distribution, so install requires a working local build toolchain."
                )
                risk += 0.3
            else:
                reasons.append(
                    f"Confirmed live against PyPI: {ml_result['package']}=={ml_result['version']} "
                    f"publishes neither a wheel nor a source distribution for "
                    f"{ml_result['platform']}/py{ml_result['python_version']} -- this is a hard fact, "
                    f"not a model guess. Install will fail, full stop."
                )
                risk += 0.5
        elif live_result.get("exists") is None:
            reasons.append(
                f"could not reach PyPI to verify {ml_result['package']}=={ml_result['version']} "
                f"live (network error) -- falling back to the ML prediction alone."
            )
            risk += 0.1

    if not ml_result["is_compatible"] and ml_result["is_known_package"]:
        reasons.append(
            f"model predicts incompatible for a package it has seen before "
            f"(predicted error type: {ml_result['predicted_error_type']}, "
            f"{ml_result['confidence']:.0%} confidence)."
        )
        risk += 0.4 * ml_result["confidence"]

    for c in conflicts_for_package:
        reasons.append(
            f"{c['from_package']}=={c['from_version']} declares a dependency on "
            f"{c['depends_on']}{c['required_specifier']}, but this install pins "
            f"{c['depends_on']}=={c['pinned_version']}, which violates that range."
        )
        risk += 0.5

    risk = min(round(risk, 3), 1.0)
    is_clean = len(reasons) == 0

    if is_clean:
        explanation = (
            f"{ml_result['package']}=={ml_result['version']} looks compatible: known "
            f"package, known platform, verified on PyPI, no dependency conflicts in "
            f"this install."
        )
    else:
        explanation = " ".join(reasons)

    return {"risk_score": risk, "explanation": explanation, "is_clean": is_clean}


def suggest_correction(model, package, python_version, platform, live_check_fn=None, top_n=5, required_specifier=None):
    """
    Recommend a replacement version, preferring one that is BOTH ML-predicted
    compatible AND (if a live checker is supplied) actually has a wheel on PyPI.
    The ML-only `recommend()` can rank a version highly that PyPI can't even serve
    for a given platform, since the model was never trained on platform diversity --
    live-checking the top candidates catches that before it's suggested as a "fix".

    required_specifier: a version specifier string (e.g. ">=0.2.0") that another
        package in the same batch declared as a real PyPI dependency (from
        dependency_conflicts.py). Candidates that don't satisfy it are skipped --
        recommending a version that's "compatible" but still violates a real
        declared constraint isn't actually a fix.

    If the package is outside the trained catalog entirely (model.recommend()
    has nothing), falls back to picking a version directly from PyPI's real
    release list -- this is what lets a package the model has never heard of
    (e.g. an indirect dependency like pywin32-ctypes) still get a genuine
    correction instead of silently keeping the broken pin.
    """
    spec = None
    if required_specifier and required_specifier != "any":
        try:
            from packaging.specifiers import SpecifierSet
            spec = SpecifierSet(required_specifier)
        except Exception:
            spec = None

    def _satisfies(version):
        if spec is None:
            return True
        try:
            return spec.contains(version, prereleases=True)
        except Exception:
            return True

    candidates = [c for c in model.recommend(package, python_version, platform, top_n=top_n) if _satisfies(c["version"])]

    if candidates:
        if live_check_fn is None:
            for c in candidates:
                if c["is_compatible"]:
                    return c
            return candidates[0]

        for c in candidates:
            if not c["is_compatible"]:
                continue
            live = live_check_fn(package, c["version"], platform, python_version)
            c["live_verified"] = live.get("wheel_available")
            if live.get("wheel_available"):
                return c

        for c in candidates:
            if c["is_compatible"]:
                return c

    # Check whether the package is unambiguously single-platform by design
    # (e.g. pywin32 publishes only win32/win_amd64/win_arm64 wheels; pyobjc-*
    # publishes only macosx wheels) BEFORE trusting the sdist-based fallback
    # below. "An sdist exists" is NOT reliable evidence a package will
    # actually work cross-platform -- pyobjc-framework-Cocoa ships a source
    # dist alongside its macOS-only wheels, but that sdist needs macOS
    # frameworks/headers and will never successfully build on Linux or
    # Windows. Wheel-tag evidence of "every release only targets one OS" is a
    # much stronger signal than "a source archive happens to exist", so check
    # it first: if a version bump doesn't fix this, no amount of local build
    # tooling will either, and the real answer is the standard requirements.txt
    # environment-marker pattern (skip it on platforms it was never meant for),
    # not a false "sdist available, might work" fallback.
    marker_suggestion = _suggest_platform_marker_exclusion(package, python_version, platform, live_check_fn)
    if marker_suggestion is not None:
        return marker_suggestion

    return _live_pypi_fallback(package, python_version, platform, live_check_fn, spec)


_PLATFORM_KEY_TO_SYS_PLATFORM = {
    "darwin_x86_64": "darwin", "darwin_arm64": "darwin",
    "linux_x86_64": "linux", "linux_aarch64": "linux",
    "win_amd64": "win32",
}


def _infer_native_platform_marker(filenames):
    """
    Looking at a package's actual published wheel filenames, guess which OS
    it's exclusively built for. Returns a `sys_platform` marker value
    ("win32" / "darwin" / "linux"), or None if the package publishes for
    more than one OS family (so it's not "single-platform by design" --
    something else is actually wrong with this specific version/pin).
    """
    tags = " ".join(filenames).lower()
    has_win = any(t in tags for t in ("win32", "win_amd64", "win_arm64"))
    has_mac = "macosx" in tags
    has_linux = "manylinux" in tags or "linux_" in tags
    families = [f for f, present in (("win32", has_win), ("darwin", has_mac), ("linux", has_linux)) if present]
    return families[0] if len(families) == 1 else None


def _suggest_platform_marker_exclusion(package, python_version, platform_key, live_check_fn):
    if live_check_fn is None:
        return None

    import live_verify

    info = live_verify.check_package_exists(package)
    if not info.get("exists"):
        return None

    latest = info["latest_version"]
    version_info = live_verify.check_version_exists(package, latest)
    filenames = [f["filename"] for f in version_info.get("files", [])]
    marker_platform = _infer_native_platform_marker(filenames)
    if marker_platform is None:
        return None

    this_platform_sys = _PLATFORM_KEY_TO_SYS_PLATFORM.get(platform_key)
    if marker_platform == this_platform_sys:
        return None  # it DOES support this platform -- something else is the real issue

    return {
        "version": latest,
        "is_compatible": False,
        "marker_exclude": True,
        "marker": f'sys_platform == "{marker_platform}"',
        "source": "platform_marker_suggestion",
    }


def _live_pypi_fallback(package, python_version, platform, live_check_fn, spec):
    """Pick a version straight from PyPI's real release list, no ML involved."""
    if live_check_fn is None:
        return None

    import live_verify
    from packaging.version import Version, InvalidVersion

    info = live_verify.check_package_exists(package)
    if not info.get("exists"):
        return None

    def _parse(v):
        try:
            return Version(v)
        except InvalidVersion:
            return None

    ranked = []
    for v in info.get("all_versions", []):
        pv = _parse(v)
        if pv is None or pv.is_prerelease or pv.is_devrelease:
            continue
        if spec is not None:
            try:
                if not spec.contains(v, prereleases=True):
                    continue
            except Exception:
                continue
        ranked.append((pv, v))
    ranked.sort(key=lambda x: x[0], reverse=True)

    for _, v in ranked[:8]:  # only probe live wheel availability for the newest handful
        live = live_check_fn(package, v, platform, python_version)
        if live.get("wheel_available") or live.get("sdist_available"):
            return {"version": v, "is_compatible": True, "live_verified": True, "source": "live_pypi_fallback"}
    return None
