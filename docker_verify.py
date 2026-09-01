"""
docker_verify.py - Wires real Docker-based install+import testing into the LIVE
validate/deploy-check request path, not just the offline data-collection script.

collect_platform_data.py already proved this mechanism works (it produced this
session's first real linux_x86_64 training rows). This module reuses the exact same
test_one() logic as an opt-in verification layer: when metadata-based checks
(live_verify.py's wheel/sdist listing, dependency_conflicts.py's requires_dist
checking) say something looks fine, Docker can still catch it being wrong.

Proven case: `wmi` publishes a platform-universal `py2.py3-none-any` wheel and passes
every metadata check this repo runs -- but a real `pip install wmi` on Linux fails,
because wmi transitively depends on `pywin32`, which doesn't exist there. No pin the
user gave us mentions pywin32 at all, so dependency_conflicts.py's direct
requires_dist checking never sees it either -- it's a TRANSITIVE failure several
levels deep. No amount of static metadata analysis catches that; actually attempting
the install is the only way.

Deliberately opt-in (docker_verify=True), not the default: each container run takes
5-15+ seconds, so running it for every package on every interactive chat/API request
would make the tool unusably slow. Only Docker-testable for linux_x86_64/linux_aarch64
targets (no Windows container support from a non-Windows Docker host) -- gracefully
returns None (falls back to metadata-only) otherwise.
"""

import subprocess

from collect_platform_data import test_one, PLATFORM_TO_DOCKER


def docker_available():
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def docker_verify_install(package, version, python_version, platform_key, timeout=90, install_build_tools=False):
    """
    Runs a REAL `pip install` + `import` test in a fresh container.

    install_build_tools: install build-essential/python3-dev/libpq-dev/
        libssl-dev/libffi-dev before attempting the install -- pass True when
        metadata already shows no wheel but an sdist exists, so the test
        reflects a properly provisioned server (the real EC2 fix) rather than
        the bare python:*-slim image, which has no compiler at all and would
        otherwise report a false failure for anything needing to build from
        source.

    Returns None if this platform isn't Docker-testable or the Docker daemon
    isn't reachable (caller should fall back to metadata-only checks).
    Otherwise returns a dict: {install_success, import_success, error_type,
    error_log_snippet} -- the same shape collect_platform_data.py produces.
    """
    if platform_key not in PLATFORM_TO_DOCKER:
        return None
    if not docker_available():
        return None
    try:
        return test_one(
            package, version, python_version, platform_key,
            timeout=timeout, install_build_tools=install_build_tools,
        )
    except Exception as e:
        return {
            "install_success": None, "import_success": None,
            "error_type": "docker_error", "error_log_snippet": str(e),
        }


if __name__ == "__main__":
    import sys
    import json

    pkg = sys.argv[1] if len(sys.argv) > 1 else "wmi"
    ver = sys.argv[2] if len(sys.argv) > 2 else "1.5.1"
    pyver = sys.argv[3] if len(sys.argv) > 3 else "3.12"
    plat = sys.argv[4] if len(sys.argv) > 4 else "linux_x86_64"

    result = docker_verify_install(pkg, ver, pyver, plat)
    print(json.dumps(result, indent=2))
