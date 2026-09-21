"""
hardware_profile.py - Best-effort detection of the host machine's OS, CPU
architecture, Python version, and GPU/accelerator (NVIDIA CUDA / Apple Metal).

This is the piece the middleware repo calls "hardware profiling" -- here it's
implemented with only the standard library plus one optional subprocess probe
(`nvidia-smi`), no GPU vendor SDKs, consistent with compactiq staying
dependency-light. It reports the hardware of the machine actually running this
process (the dashboard server) -- the natural default when the dashboard is run
locally, which is the primary way this repo is used.
"""

import platform
import subprocess
import sys


def _detect_platform_key(system, machine):
    system = system.lower()
    machine = machine.lower()

    if system == "darwin":
        return "darwin_arm64" if machine in ("arm64", "aarch64") else "darwin_x86_64"
    if system == "linux":
        return "linux_aarch64" if machine in ("arm64", "aarch64") else "linux_x86_64"
    if system == "windows":
        return "win_amd64"
    return f"{system}_{machine}"


KNOWN_PLATFORM_KEYS = ("darwin_x86_64", "darwin_arm64", "linux_x86_64", "linux_aarch64", "win_amd64")

_OS_ALIASES = {"darwin": "darwin", "macos": "darwin", "mac": "darwin", "osx": "darwin",
               "linux": "linux",
               "windows": "windows", "win32": "windows", "win": "windows"}
_ARCH_ALIASES = {"x86_64": "x86_64", "amd64": "x86_64",
                 "arm64": "arm64", "aarch64": "arm64"}


def platform_key_from_os_arch(os_name, arch):
    """
    Maps a submitted (os, arch) pair -- e.g. from a crowd-sourced observation
    that reports "platform": "linux", "arch": "x86_64" as separate fields,
    unlike data.json's single combined platform_key -- onto one of the model's
    known platform_key strings.

    Returns None (not a guess, not a newly-invented key) if the combo doesn't
    resolve to one of KNOWN_PLATFORM_KEYS. Deliberately stricter than
    _detect_platform_key() above, which falls back to inventing an
    "{system}_{machine}" string for local host-detection purposes where that's
    harmless -- a value never used to encode the model's package_map/platform_map,
    doesn't get near training data, and can't corrupt it silently.
    """
    os_key = _OS_ALIASES.get(str(os_name).strip().lower())
    arch_key = _ARCH_ALIASES.get(str(arch).strip().lower())
    if os_key is None or arch_key is None:
        return None

    if os_key == "windows":
        return "win_amd64" if arch_key == "x86_64" else None
    candidate = f"{os_key}_{arch_key}"
    return candidate if candidate in KNOWN_PLATFORM_KEYS else None


def _detect_nvidia_gpu():
    """Best-effort NVIDIA GPU/CUDA probe via `nvidia-smi`, if present on PATH."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            first_line = out.stdout.strip().splitlines()[0]
            return {"available": True, "detail": first_line}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return {"available": False, "detail": None}


def detect_host():
    """
    Returns a dict describing the host this process is running on:
        {
          "os": "Darwin" | "Linux" | "Windows",
          "arch": "arm64" | "x86_64" | ...,
          "platform_key": "darwin_arm64" | "darwin_x86_64" | "linux_x86_64" | ...,
          "python_version": "3.12",
          "cuda": {"available": bool, "detail": str|None},
          "apple_metal": {"available": bool, "detail": str|None},
        }
    """
    system = platform.system()
    machine = platform.machine()
    platform_key = _detect_platform_key(system, machine)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    cuda = _detect_nvidia_gpu()

    apple_metal = {"available": False, "detail": None}
    if system == "Darwin" and machine in ("arm64",):
        apple_metal = {"available": True, "detail": "Apple Silicon (Metal / MPS backend available)"}

    return {
        "os": system,
        "arch": machine,
        "platform_key": platform_key,
        "python_version": py_version,
        "cuda": cuda,
        "apple_metal": apple_metal,
    }


def describe_host(profile=None):
    """One-line human-readable summary of a detect_host() profile."""
    profile = profile or detect_host()
    parts = [f"{profile['os']} ({profile['arch']})", f"Python {profile['python_version']}"]
    if profile["cuda"]["available"]:
        parts.append(f"NVIDIA GPU: {profile['cuda']['detail']}")
    elif profile["apple_metal"]["available"]:
        parts.append(profile["apple_metal"]["detail"])
    else:
        parts.append("no dedicated GPU detected")
    return " • ".join(parts)


if __name__ == "__main__":
    import json
    profile = detect_host()
    print(json.dumps(profile, indent=2))
    print()
    print(describe_host(profile))
