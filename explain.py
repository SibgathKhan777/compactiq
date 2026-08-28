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

    if not ml_result["is_known_package"]:
        reasons.append(
            f"'{ml_result['package']}' was never seen during training, so the ML "
            f"prediction ({'compatible' if ml_result['is_compatible'] else 'incompatible'}, "
            f"{ml_result['confidence']:.0%} confidence) is an extrapolation, not a real signal "
            f"-- package-holdout evaluation shows accuracy on unseen packages is below "
            f"the majority-class baseline, i.e. worse than a coin flip weighted by base rate."
        )
        risk += 0.3

    if not ml_result["is_known_platform"]:
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
                    f"PyPI has neither a wheel nor a source distribution for "
                    f"{ml_result['package']}=={ml_result['version']} on "
                    f"{ml_result['platform']}/py{ml_result['python_version']} -- install will fail."
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


def suggest_correction(model, package, python_version, platform, live_check_fn=None, top_n=5):
    """
    Recommend a replacement version, preferring one that is BOTH ML-predicted
    compatible AND (if a live checker is supplied) actually has a wheel on PyPI.
    The ML-only `recommend()` can rank a version highly that PyPI can't even serve
    for a given platform, since the model was never trained on platform diversity --
    live-checking the top candidates catches that before it's suggested as a "fix".
    """
    candidates = model.recommend(package, python_version, platform, top_n=top_n)
    if not candidates:
        return None

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
    return candidates[0]
