"""
live_verify.py - Live ground-truth verification against PyPI, independent of the
trained ML model's static training snapshot.

Fills the dataset-freshness gap: the ML model in pycompat_model.py can only ever
answer questions about package/version/platform combinations shaped like what was
in data.json's 2026-02-24/25 snapshot. This module answers two questions the model
structurally cannot, live, at request time, with no retraining involved:

  1. Does this package version actually exist on PyPI right now? Catches releases
     that happened after training, or a version string an LLM hallucinated.
  2. Does PyPI actually publish a wheel (or usable sdist) for the requested platform
     and Python version? Catches platform/ABI gaps the classifier can only guess at
     via `platform_encoded` -- which has zero variance in the training data, since
     every training row is darwin_x86_64.

No Docker, no install, no import test: this only inspects PyPI's published metadata
and file listing, so it stays cheap enough to call on every request. It cannot catch
what the middleware repo's Docker-based install+import test catches (e.g. a package
that installs fine but raises on import due to a runtime environment quirk) -- that
would require actually running the code, which is a deliberate scope line, not an
oversight: compactiq stays dependency-light, and its live-verification layer only
checks what's true from PyPI's metadata alone.
"""

import json
import re
import urllib.request
import urllib.error

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
PYPI_VERSION_JSON_URL = "https://pypi.org/pypi/{package}/{version}/json"

# Maps our simplified platform keys to the wheel-filename tag patterns that would
# satisfy them. Extend this as real non-Mac platform data ever gets collected.
_PLATFORM_TAG_PATTERNS = {
    "darwin_x86_64": [r"macosx[\w.]*x86_64", r"macosx[\w.]*universal2"],
    "darwin_arm64": [r"macosx[\w.]*arm64", r"macosx[\w.]*universal2"],
    "linux_x86_64": [r"manylinux[\w.]*x86_64", r"linux_x86_64"],
    "linux_aarch64": [r"manylinux[\w.]*aarch64", r"linux_aarch64"],
    "win_amd64": [r"win_amd64"],
}


def _fetch_json(url, timeout=6):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "compactiq-live-verify/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, "not_found"
        return None, f"http_error_{e.code}"
    except Exception as e:
        return None, f"network_error:{type(e).__name__}:{e}"


def check_package_exists(package):
    """Does this package exist on PyPI at all, and what's actually latest right now?"""
    data, err = _fetch_json(PYPI_JSON_URL.format(package=package))
    if err == "not_found":
        return {"exists": False, "reason": "package_not_on_pypi"}
    if err:
        return {"exists": None, "reason": err}
    return {
        "exists": True,
        "latest_version": data["info"]["version"],
        "all_versions": sorted(data["releases"].keys()),
    }


def check_version_exists(package, version):
    """Does this exact version exist on PyPI? Returns its published files if so."""
    data, err = _fetch_json(PYPI_VERSION_JSON_URL.format(package=package, version=version))
    if err == "not_found":
        return {"exists": False, "reason": "version_not_found", "files": [], "requires_dist": []}
    if err:
        return {"exists": None, "reason": err, "files": [], "requires_dist": []}
    files = [
        {"filename": f["filename"], "packagetype": f["packagetype"]}
        for f in data.get("urls", [])
    ]
    return {"exists": True, "files": files, "requires_dist": data["info"].get("requires_dist") or []}


def check_platform_wheel(package, version, platform_key, python_version):
    """
    Does PyPI publish a wheel matching this platform/python combo, or at least an
    sdist that could theoretically be built from source? This is the live check that
    substitutes for the model's zero-variance `platform_encoded` feature.
    """
    version_info = check_version_exists(package, version)
    if not version_info["exists"]:
        return {"wheel_available": False, "sdist_available": False, **version_info}

    py_tag_short = "cp" + python_version.replace(".", "")
    platform_patterns = _PLATFORM_TAG_PATTERNS.get(platform_key, [])

    matching_wheels = []
    has_sdist = False
    for f in version_info["files"]:
        name = f["filename"]
        if f["packagetype"] == "sdist":
            has_sdist = True
            continue
        if not name.endswith(".whl"):
            continue
        py_ok = (py_tag_short in name) or ("py3-none" in name) or ("py2.py3-none" in name)
        # A "-any.whl" wheel is platform-independent (pure Python) -- it satisfies
        # every platform_key, not just ones matching a specific OS/arch pattern.
        is_universal = name.endswith("-any.whl")
        plat_ok = is_universal or (any(re.search(p, name) for p in platform_patterns) if platform_patterns else True)
        if py_ok and plat_ok:
            matching_wheels.append(name)

    return {
        "exists": True,
        "wheel_available": len(matching_wheels) > 0,
        "matching_wheels": matching_wheels,
        "sdist_available": has_sdist,
        "requires_dist": version_info["requires_dist"],
    }


if __name__ == "__main__":
    import sys

    pkg = sys.argv[1] if len(sys.argv) > 1 else "boto3"
    ver = sys.argv[2] if len(sys.argv) > 2 else "1.42.49"
    pyver = sys.argv[3] if len(sys.argv) > 3 else "3.12"
    plat = sys.argv[4] if len(sys.argv) > 4 else "darwin_x86_64"

    print(json.dumps(check_platform_wheel(pkg, ver, plat, pyver), indent=2))
