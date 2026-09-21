"""
observation_store.py - Shared schema, validation, and file I/O for crowd-submitted
compatibility observations, used by both the /api/observations write path
(api_server.py) and the promote_observations.py read/promote path.

Deliberately append-only and separate from data.json: an observation is a raw,
unverified, self-reported claim from an arbitrary caller (device_id_hash is
whatever the caller sends -- no auth, no proof of identity). data.json is the
model's actual training set, and nothing here writes to it directly. Only
promote_observations.py, after applying corroboration checks, may append to it.

Record shape (one JSON object per line in observations.jsonl):
    {
      "id": "<uuid4>",
      "package": str, "version": str, "python_version": str,
      "os": str, "arch": str,
      "platform_key": str | None,   # computed via hardware_profile's known-platform
                                     # mapping; None if the (os, arch) combo doesn't
                                     # resolve to one of the model's known platforms --
                                     # such rows are kept (append-only) but can never
                                     # be promoted, since data.json has no slot for
                                     # a platform the model was never trained on.
      "outcome": "success" | "fail",
      "error_type": str | None,     # as submitted; normalized only at promotion time
      "device_id_hash": str,
      "observed_via": str,
      "received_at": "<iso timestamp>",
      "promoted": bool,
      "promoted_at": "<iso timestamp>" | None,
    }
"""

import json
import os
import uuid
from datetime import datetime, timezone

from hardware_profile import platform_key_from_os_arch

OBSERVATIONS_FILE = os.path.join(os.path.dirname(__file__), "observations.jsonl")

REQUIRED_FIELDS = ["package", "version", "python_version", "platform", "arch",
                   "outcome", "device_id_hash", "observed_via"]

# This endpoint exists to feed the training set with results an agent actually
# captured by running `pip install` and observing the real exit code/import
# result -- never a human's typed-in opinion ("yeah that failed for me"), which
# is a categorically weaker signal (no proof it was ever tried, subject to
# misremembering/misattribution). "chat" deliberately isn't in this set even
# though it's a valid `observed_via` value used elsewhere in this repo (the
# Dependency Assistant) -- that surface captures a human's stated intent, not
# a captured install result, and has no business feeding this pipeline.
ALLOWED_OBSERVED_VIA = {"auto"}


class ObservationValidationError(ValueError):
    pass


def normalize_observation(payload):
    """
    Validates a raw request body and returns a clean record ready to append.
    Raises ObservationValidationError with a human-readable message on bad input.

    Note: "platform" in the request body means an OS-like string ("linux",
    "darwin", "windows") -- NOT data.json's combined platform_key. It's renamed
    to "os" in the stored record to keep that distinction visible on disk.
    """
    if not isinstance(payload, dict):
        raise ObservationValidationError("JSON body must be an object")

    missing = [f for f in REQUIRED_FIELDS if not payload.get(f)]
    if missing:
        raise ObservationValidationError(f"Missing required fields: {missing}")

    outcome = payload["outcome"]
    if outcome not in ("success", "fail"):
        raise ObservationValidationError('"outcome" must be "success" or "fail"')

    observed_via = payload["observed_via"]
    if observed_via not in ALLOWED_OBSERVED_VIA:
        raise ObservationValidationError(
            f'"observed_via" must be one of {sorted(ALLOWED_OBSERVED_VIA)} -- '
            f"this endpoint only accepts results an agent actually captured by "
            f"running the install, not a self-reported opinion"
        )

    error_type = payload.get("error_type")
    if error_type is not None and not isinstance(error_type, str):
        raise ObservationValidationError('"error_type" must be a string or null')
    if outcome == "success" and error_type not in (None, "none"):
        # Contradictory input (reported success but also gave an error_type) --
        # kept as submitted rather than silently dropped, so promote_observations.py
        # can see the discrepancy; it forces error_type back to "none" for
        # success rows at promotion time regardless of what's stored here.
        pass

    platform_key = platform_key_from_os_arch(payload["platform"], payload["arch"])

    return {
        "id": str(uuid.uuid4()),
        "package": str(payload["package"]),
        "version": str(payload["version"]),
        "python_version": str(payload["python_version"]),
        "os": str(payload["platform"]),
        "arch": str(payload["arch"]),
        "platform_key": platform_key,
        "outcome": outcome,
        "error_type": error_type,
        "device_id_hash": str(payload["device_id_hash"]),
        "observed_via": str(payload["observed_via"]),
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted": False,
        "promoted_at": None,
    }


def append_observation(record, path=OBSERVATIONS_FILE):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_observations(path=OBSERVATIONS_FILE):
    """Returns the full list of records in file order. Empty list if the file
    doesn't exist yet (nothing submitted so far)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rewrite_observations(records, path=OBSERVATIONS_FILE):
    """Overwrites the file with the given records, one JSON object per line, in
    order. Used after marking a batch of rows as promoted. Not safe against
    concurrent writers -- this repo has no lock/queue around observations.jsonl,
    matching its existing dependency-light, single-writer-at-a-time assumption
    (promote_observations.py is meant to be run as an offline batch job, not
    called concurrently with itself or while /api/observations is mid-append)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, path)
