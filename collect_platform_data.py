"""
collect_platform_data.py - Collects REAL install+import test results on a target
platform via Docker, in the exact schema data.json already uses, so the results can
be appended and the model retrained with actual (not simulated) platform diversity.

This is the one piece worth borrowing from the middleware repo's Docker-based
install+import test -- without adopting its Celery/Postgres/Redis service stack.

Deliberately a standalone script, not a service: run it, inspect the JSON it prints,
and decide whether to merge it into data.json. It never touches data.json itself
(no silent writes) -- pipe its output through merge_collected_data.py, or eyeball it
first. A bad batch of container results should never be able to corrupt the training
set without a human looking at it.

Usage:
    python collect_platform_data.py --platform linux_x86_64 --python 3.12 \\
        boto3==1.42.49 requests==2.32.0
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

PLATFORM_TO_DOCKER = {
    "linux_x86_64": "linux/amd64",
    "linux_aarch64": "linux/arm64",
}

# Packages whose distributed/import name differs from their PyPI name.
IMPORT_NAME_OVERRIDES = {
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    "python-dotenv": "dotenv",
    "scikit-learn": "sklearn",
    "google-cloud-storage": "google.cloud.storage",
    "google-cloud-bigquery": "google.cloud.bigquery",
    "google-cloud-core": "google.cloud",
    "google-auth": "google.auth",
    "google-auth-oauthlib": "google_auth_oauthlib",
    "google-api-python-client": "googleapiclient",
    "google-api-core": "google.api_core",
    "opentelemetry-api": "opentelemetry",
    "opentelemetry-sdk": "opentelemetry.sdk",
    "protobuf": "google.protobuf",
    "pyjwt": "jwt",
    "pynacl": "nacl",
    "pyopenssl": "OpenSSL",
    "psycopg2-binary": "psycopg2",
    "azure-core": "azure.core",
    "azure-identity": "azure.identity",
    "grpcio": "grpc",
    "grpcio-status": "grpc_status",
    "grpcio-tools": "grpc_tools",
    "pillow": "PIL",
}


def import_name_for(package):
    if package in IMPORT_NAME_OVERRIDES:
        return IMPORT_NAME_OVERRIDES[package]
    return package.replace("-", "_")


# The standard EC2 provisioning fix for "package X needs to compile from
# source" (build-essential + python3-dev + libpq-dev + libssl-dev + libffi-dev
# covers the overwhelming majority of real packages: psycopg2, cryptography,
# and most other C-extension libraries). Only installed when actually needed
# (see install_build_tools below) -- it costs real time (~15-30s in a fresh
# container) and a bare wheel install never needs it.
_APT_BUILD_TOOLS_CMD = (
    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "build-essential python3-dev libpq-dev libssl-dev libffi-dev >/tmp/apt.log 2>&1"
)


def test_one(package, version, python_version, platform_key, timeout=120, install_build_tools=False):
    """
    Runs one install+import test in a fresh container. Returns a
    data.json-shaped record.

    install_build_tools: when True, installs build-essential/python3-dev/
        libpq-dev/libssl-dev/libffi-dev before attempting the install -- the
        real-world EC2 fix for "package needs to compile from source". The
        python:*-slim base image has NO compiler by default, so without this,
        every sdist-only package looks like a hard failure here even when it
        would install fine on a properly provisioned server. Costs real time,
        so only pass True when metadata already suggests it's needed (no
        wheel, but an sdist exists).
    """
    docker_platform = PLATFORM_TO_DOCKER.get(platform_key)
    if not docker_platform:
        raise ValueError(f"No docker platform mapping for {platform_key}")

    image = f"python:{python_version}-slim"
    import_name = import_name_for(package)
    setup = f"{_APT_BUILD_TOOLS_CMD}; " if install_build_tools else ""
    script = (
        f"{setup}"
        f"pip install --no-cache-dir --disable-pip-version-check '{package}=={version}' "
        f"> /tmp/install.log 2>&1; echo INSTALL_RC=$?; "
        f"python -c 'import {import_name}' > /tmp/import.log 2>&1; echo IMPORT_RC=$?; "
        f"echo ---INSTALL_LOG---; tail -c 400 /tmp/install.log; "
        f"echo ---IMPORT_LOG---; tail -c 400 /tmp/import.log"
    )
    cmd = ["docker", "run", "--rm", "--platform", docker_platform, image, "bash", "-c", script]

    if install_build_tools:
        timeout = max(timeout, 180)  # apt-get + a from-source build can genuinely take a while

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return _record(package, version, python_version, platform_key, False, False, "timeout", "docker run timed out")
    except FileNotFoundError:
        raise RuntimeError("docker CLI not found -- is Docker installed?")

    install_rc = _extract_rc(out, "INSTALL_RC")
    import_rc = _extract_rc(out, "IMPORT_RC")

    install_success = install_rc == 0
    import_success = install_success and import_rc == 0

    error_type = "none"
    snippet = ""
    if not install_success:
        low = out.lower()
        if "no matching distribution" in low or "could not find a version" in low:
            error_type = "no_wheel"
        elif "connection" in low or "timed out" in low or "temporary failure" in low:
            error_type = "timeout"
        else:
            error_type = "build_error"
        snippet = out[-500:]
    elif not import_success:
        error_type = "import_error"
        snippet = out[-500:]

    return _record(package, version, python_version, platform_key, install_success, import_success, error_type, snippet)


def _extract_rc(text, marker):
    m = re.search(rf"{marker}=(-?\d+)", text)
    return int(m.group(1)) if m else -1


def _record(package, version, python_version, platform_key, install_success, import_success, error_type, snippet):
    return {
        "package": package,
        "version": version,
        "python_version": python_version,
        "platform": platform_key,
        "install_success": install_success,
        "import_success": import_success,
        "error_type": error_type,
        "error_log_snippet": snippet,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser(description="Collect real install+import test results via Docker")
    ap.add_argument("pins", nargs="+", help="package==version, e.g. boto3==1.42.49")
    ap.add_argument("--platform", default="linux_x86_64", choices=list(PLATFORM_TO_DOCKER.keys()))
    ap.add_argument("--python", default="3.12", help="Python version, e.g. 3.12")
    ap.add_argument("--out", default=None, help="Write results to this file instead of stdout")
    args = ap.parse_args()

    results = []
    for pin in args.pins:
        if "==" not in pin:
            print(f"Skipping '{pin}' -- expected package==version", file=sys.stderr)
            continue
        package, version = pin.split("==", 1)
        print(f"Testing {package}=={version} on {args.platform}, Python {args.python}...", file=sys.stderr)
        record = test_one(package, version, args.python, args.platform)
        status = "OK" if record["import_success"] else f"FAIL ({record['error_type']})"
        print(f"  -> {status}", file=sys.stderr)
        results.append(record)

    output = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote {len(results)} record(s) to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
