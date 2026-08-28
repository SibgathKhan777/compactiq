"""
dependency_conflicts.py - Joint compatibility check across a *set* of pinned
packages, using each package's real PyPI-declared dependencies.

pycompat_model.py's ML model predicts one (package, version, python_version,
platform) tuple at a time -- it has no way to know that pinning torch==2.8.0
together with torchvision==0.17.0 in the same install might violate torchvision's
own declared `torch` requirement. That's a genuinely different problem: joint
constraint satisfaction over a set of pins, not a per-row classification, and no
amount of retraining the classifier fixes it.

This is NOT a general SAT/dependency solver (the middleware repo uses Z3 for that,
searching the space of possible versions to find one that satisfies everyone). This
checks the *specific* pins the caller already proposed against each package's
declared `requires_dist` metadata from PyPI -- enough to catch the common real-world
failure (two packages pinned in the same install that violate each other's declared
version range) without pulling in a constraint solver or a full resolver.
"""

from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name
from packaging.markers import UndefinedEnvironmentName
from packaging.version import InvalidVersion

from live_verify import check_version_exists


def _canon(name):
    return canonicalize_name(name)


def check_batch(pins, python_version):
    """
    pins: dict of {package_name: version_string}, e.g.
          {"torch": "2.8.0", "torchvision": "0.17.0"}

    Returns:
        {
          "conflicts": [ {from_package, from_version, depends_on,
                           required_specifier, pinned_version, satisfied: False}, ... ],
          "unresolved": [ {package, version, reason}, ... ],  # couldn't check live
          "is_jointly_compatible": bool,
        }
    """
    canon_pins = {_canon(k): v for k, v in pins.items()}
    conflicts = []
    unresolved = []

    for pkg, version in pins.items():
        info = check_version_exists(pkg, version)
        if not info.get("exists"):
            unresolved.append({"package": pkg, "version": version, "reason": info.get("reason", "unknown")})
            continue

        for req_str in info.get("requires_dist", []):
            try:
                req = Requirement(req_str)
            except InvalidRequirement:
                continue

            if req.marker is not None:
                try:
                    applies = req.marker.evaluate({"python_version": python_version, "extra": ""})
                except UndefinedEnvironmentName:
                    applies = True
                if not applies:
                    continue

            dep_name = _canon(req.name)
            if dep_name not in canon_pins:
                continue  # dependency isn't part of this batch -- nothing to cross-check

            pinned_version = canon_pins[dep_name]
            try:
                spec = str(req.specifier)
                satisfied = req.specifier.contains(pinned_version, prereleases=True) if spec else True
            except InvalidVersion:
                satisfied = None

            if satisfied is False:
                conflicts.append({
                    "from_package": pkg,
                    "from_version": version,
                    "depends_on": req.name,
                    "required_specifier": str(req.specifier) or "any",
                    "pinned_version": pinned_version,
                    "satisfied": False,
                })

    return {
        "conflicts": conflicts,
        "unresolved": unresolved,
        "is_jointly_compatible": len(conflicts) == 0,
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
        print("Usage: python dependency_conflicts.py <python_version> <pkg1> <ver1> [<pkg2> <ver2> ...]")
        print('Example: python dependency_conflicts.py 3.12 torch 2.8.0 torchvision 0.17.0')
        sys.exit(1)

    py_version = sys.argv[1]
    rest = sys.argv[2:]
    pin_dict = {rest[i]: rest[i + 1] for i in range(0, len(rest), 2)}
    print(json.dumps(check_batch(pin_dict, py_version), indent=2))
