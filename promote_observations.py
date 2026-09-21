"""
promote_observations.py - Batch job that reviews observations.jsonl (raw,
unverified, crowd-submitted compatibility reports collected by /api/observations)
and promotes the trustworthy ones into data.json, the model's actual training set.

Nothing here is real-time: run this on a schedule (cron, manual, whatever) --
it is NOT wired into request handling. Deliberately conservative: a row is
promoted only if BOTH hold:

  (a) LIVE CORROBORATION -- live_verify.check_platform_wheel() confirms the
      claimed package==version genuinely exists on PyPI right now, AND (for a
      "success" report specifically) that PyPI actually publishes a wheel or
      sdist for the claimed platform_key. A report about a package/version
      that doesn't exist is either a typo, a stale claim, or fabricated --
      promoting it would poison training data with a row no one could ever
      reproduce. A "success" report for a platform PyPI has no build for at
      all is a direct conflict with ground truth (there is nothing it could
      have successfully installed), so it's rejected too. This asymmetry is
      deliberate: absence of a wheel is strong evidence against a claimed
      success, but says nothing against a claimed *failure* (plenty of real
      failures happen for reasons having nothing to do with wheel
      availability -- disk space, network, a broken local venv -- so a "fail"
      report is only checked for the package/version existing at all).
  (b) TRUST -- either:
        (i)  the reporting device's reputation score is >= REPUTATION_THRESHOLD, and
             the device is not currently rate-limited (see VELOCITY_* below), or
        (ii) at least 2 independent, non-rate-limited device_id_hashes reported
             the SAME outcome for the SAME (package, version, python_version, platform_key)

Before writing anything to data.json, the current file is snapshotted to
data.json.bak.<unix_timestamp> -- promotion is the one path that mutates the
model's real training data, so a rollback point always exists.

ASSUMPTIONS made explicit here (flagged per the request that spawned this
script -- confirm/revisit before relying on this in place of real moderation):

  - Reputation store is a plain dict persisted to reputation_store.json,
    loaded fully into memory and rewritten wholesale each run. This is
    explicitly a placeholder ("assume a simple in-memory/dict-based reputation
    store for now") -- it has no decay and no real identity verification.
  - Velocity check (VELOCITY_WINDOW_SECONDS / VELOCITY_MAX_PER_WINDOW) is a
    blunt per-device rate limit computed directly from observations.jsonl's
    own received_at timestamps -- no separate store. It mitigates, but does
    NOT solve, the sybil risk noted above: nothing stops a bad actor from
    minting many distinct device_id_hashes (each individually under the
    velocity limit) to fabricate "2 independent devices agreeing" or to farm
    reputation slowly. A real system needs actual device attestation or
    account-level identity before either the reputation or agreement checks
    mean much against a motivated attacker; this is a placeholder against
    the cheap version of the attack (one device spamming), not the
    sophisticated one.
  - outcome="fail" without install/import-level detail: the submitted schema
    only has one "outcome" enum (success|fail), not data.json's separate
    install_success/import_success. This script infers which stage failed
    from error_type (see _infer_success_flags) using the same error_type
    vocabulary data.json already uses. An error_type of "unknown_error" (the
    agreed default for a fail report that didn't specify one) is treated as a
    total failure (both install_success and import_success False) since
    there's no signal to say otherwise -- this is a conservative guess, not a
    verified fact.
  - error_log_snippet: the submitted schema has no log field at all (a crowd
    reporter isn't asked to paste one). Promoted rows get a synthesized
    placeholder string tagged "[crowd-reported]" rather than "", so they stay
    visually distinguishable from data.json's originally Docker/scrape-
    collected real log snippets.
  - Package name casing is stored exactly as submitted, matching how
    data.json's existing rows are inconsistently cased too (canonicalization
    already happens at lookup time in pycompat_model.py, not at storage time).

Usage:
    python promote_observations.py                # apply for real
    python promote_observations.py --dry-run       # report what would be promoted, write nothing
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone

from live_verify import check_platform_wheel
from observation_store import read_observations, rewrite_observations, OBSERVATIONS_FILE

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")
REPUTATION_FILE = os.path.join(os.path.dirname(__file__), "reputation_store.json")

REPUTATION_THRESHOLD = 3          # placeholder constant -- see module docstring
REPUTATION_INCREMENT_ON_PROMOTE = 1
MIN_INDEPENDENT_DEVICES = 2

VELOCITY_WINDOW_SECONDS = 3600     # placeholder constants -- see module docstring
VELOCITY_MAX_PER_WINDOW = 10

# error_type -> (install_success, import_success). Mirrors data.json's existing
# error_type vocabulary (see data.json rows: none/import_error/abi_mismatch/
# no_wheel/build_error/timeout). "unknown_error" is this script's own addition
# for a fail report that didn't specify a reason -- treated conservatively as a
# total failure, per the confirmed default.
_ERROR_TYPE_TO_FLAGS = {
    "none": (True, True),
    "import_error": (True, False),
    "abi_mismatch": (True, False),
    "no_wheel": (False, False),
    "build_error": (False, False),
    "timeout": (False, False),
    "unknown_error": (False, False),
}


def _load_reputation():
    if not os.path.exists(REPUTATION_FILE):
        return {}
    with open(REPUTATION_FILE) as f:
        return json.load(f)


def _save_reputation(reputation):
    with open(REPUTATION_FILE, "w") as f:
        json.dump(reputation, f, indent=2)


def _infer_success_flags(outcome, error_type):
    if outcome == "success":
        return True, True
    # outcome == "fail"
    normalized = error_type if error_type in _ERROR_TYPE_TO_FLAGS else "unknown_error"
    return _ERROR_TYPE_TO_FLAGS[normalized]


def _normalized_error_type(outcome, error_type):
    if outcome == "success":
        return "none"
    return error_type if error_type in _ERROR_TYPE_TO_FLAGS else "unknown_error"


def _group_key(rec):
    return (rec["package"], rec["version"], rec["python_version"], rec["platform_key"], rec["outcome"])


def _parse_received_at(rec):
    try:
        return datetime.fromisoformat(rec["received_at"])
    except (KeyError, ValueError, TypeError):
        return None


def compute_rate_limited_devices(records, now=None, window_seconds=VELOCITY_WINDOW_SECONDS,
                                  max_per_window=VELOCITY_MAX_PER_WINDOW):
    """
    Returns the set of device_id_hashes that have submitted more than
    `max_per_window` observations (any promoted state -- this is about
    submission rate, not promotion outcome) within the trailing
    `window_seconds`. Computed directly from observations.jsonl's own
    received_at timestamps, no separate store needed.
    """
    now = now or datetime.now(timezone.utc)
    counts = {}
    for r in records:
        ts = _parse_received_at(r)
        if ts is None:
            continue
        age = (now - ts).total_seconds()
        if 0 <= age <= window_seconds:
            counts[r["device_id_hash"]] = counts.get(r["device_id_hash"], 0) + 1
    return {device for device, count in counts.items() if count > max_per_window}, counts


def find_promotable(records, reputation, live_check=check_platform_wheel, log=print, now=None):
    """
    Returns (promotable, skip_reasons) where `promotable` is a list of
    (record, live_info) pairs cleared for promotion, and skip_reasons is a
    dict record_id -> reason string, for reporting.
    """
    unpromoted = [r for r in records if not r.get("promoted")]
    rate_limited_devices, velocity_counts = compute_rate_limited_devices(records, now=now)

    # Build device-agreement groups across ALL unpromoted rows up front, so
    # e.g. two devices reporting the same combo in the same batch corroborate
    # each other even though neither individually met the reputation bar.
    # Rate-limited devices are excluded here too -- their reports shouldn't
    # count as independent corroboration for anyone, not just be blocked from
    # promoting themselves.
    groups = {}
    for r in unpromoted:
        if r["platform_key"] is None or r["device_id_hash"] in rate_limited_devices:
            continue
        groups.setdefault(_group_key(r), set()).add(r["device_id_hash"])

    promotable = []
    skip_reasons = {}
    live_cache = {}

    for r in unpromoted:
        if r["platform_key"] is None:
            skip_reasons[r["id"]] = "platform_key unresolved (unknown os/arch combo)"
            continue

        if r["device_id_hash"] in rate_limited_devices:
            skip_reasons[r["id"]] = (
                f"device rate-limited (velocity check): "
                f"{velocity_counts.get(r['device_id_hash'], 0)} observations in trailing "
                f"{VELOCITY_WINDOW_SECONDS}s > limit {VELOCITY_MAX_PER_WINDOW}"
            )
            continue

        cache_key = (r["package"], r["version"], r["platform_key"], r["python_version"])
        if cache_key not in live_cache:
            live_cache[cache_key] = live_check(r["package"], r["version"], r["platform_key"], r["python_version"])
        live_info = live_cache[cache_key]

        if live_info.get("exists") is not True:
            skip_reasons[r["id"]] = f"live PyPI check did not confirm existence: {live_info.get('reason')}"
            continue

        if r["outcome"] == "success" and not (live_info.get("wheel_available") or live_info.get("sdist_available")):
            skip_reasons[r["id"]] = (
                f"conflicts with live ground truth: PyPI publishes no wheel or sdist "
                f"for {r['platform_key']} -- a claimed success has nothing it could have installed"
            )
            continue

        reputation_ok = reputation.get(r["device_id_hash"], 0) >= REPUTATION_THRESHOLD
        agreement_ok = len(groups.get(_group_key(r), set())) >= MIN_INDEPENDENT_DEVICES

        if not (reputation_ok or agreement_ok):
            skip_reasons[r["id"]] = (
                f"insufficient trust (reputation={reputation.get(r['device_id_hash'], 0)}"
                f"<{REPUTATION_THRESHOLD}, independent_devices={len(groups.get(_group_key(r), set()))}"
                f"<{MIN_INDEPENDENT_DEVICES})"
            )
            continue

        promotable.append((r, live_info))

    return promotable, skip_reasons


def promote(dry_run=False, data_file=DATA_FILE, observations_file=OBSERVATIONS_FILE, log=print):
    records = read_observations(observations_file)
    if not records:
        log("No observations on file. Nothing to do.")
        return {"promoted": 0, "skipped": 0}

    reputation = _load_reputation()
    promotable, skip_reasons = find_promotable(records, reputation, log=log)

    log(f"Scanned {sum(1 for r in records if not r.get('promoted'))} unpromoted observation(s): "
        f"{len(promotable)} promotable, {len(skip_reasons)} skipped.")
    for reason in sorted(set(skip_reasons.values())):
        count = sum(1 for v in skip_reasons.values() if v == reason)
        log(f"  skipped ({count}): {reason}")

    if not promotable:
        return {"promoted": 0, "skipped": len(skip_reasons)}

    if dry_run:
        log(f"[dry-run] Would promote {len(promotable)} row(s); no files written.")
        for r, _ in promotable:
            log(f"  would promote: {r['package']}=={r['version']} py{r['python_version']} "
                f"{r['platform_key']} outcome={r['outcome']}")
        return {"promoted": 0, "skipped": len(skip_reasons), "dry_run_would_promote": len(promotable)}

    # Snapshot before mutating the real training set.
    if os.path.exists(data_file):
        backup_path = f"{data_file}.bak.{int(time.time())}"
        shutil.copy2(data_file, backup_path)
        log(f"Backed up {data_file} -> {backup_path}")

    with open(data_file, "r") as f:
        data = json.load(f)

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    promoted_ids = set()
    for r, _live_info in promotable:
        install_success, import_success = _infer_success_flags(r["outcome"], r["error_type"])
        error_type = _normalized_error_type(r["outcome"], r["error_type"])
        data.append({
            "package": r["package"],
            "version": r["version"],
            "python_version": r["python_version"],
            "platform": r["platform_key"],
            "install_success": install_success,
            "import_success": import_success,
            "error_type": error_type,
            "error_log_snippet": "" if r["outcome"] == "success" else f"[crowd-reported] error_type={error_type}",
            "timestamp": now_iso,
        })
        promoted_ids.add(r["id"])
        reputation[r["device_id_hash"]] = reputation.get(r["device_id_hash"], 0) + REPUTATION_INCREMENT_ON_PROMOTE

    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Appended {len(promoted_ids)} row(s) to {data_file} ({len(data)} total rows now).")

    _save_reputation(reputation)

    for r in records:
        if r["id"] in promoted_ids:
            r["promoted"] = True
            r["promoted_at"] = now_iso
    rewrite_observations(records, observations_file)
    log(f"Marked {len(promoted_ids)} observation(s) as promoted in {observations_file}.")

    return {"promoted": len(promoted_ids), "skipped": len(skip_reasons)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Promote corroborated crowd observations into data.json")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be promoted without writing anything")
    ap.add_argument("--data-file", default=DATA_FILE)
    ap.add_argument("--observations-file", default=OBSERVATIONS_FILE)
    args = ap.parse_args()

    promote(dry_run=args.dry_run, data_file=args.data_file, observations_file=args.observations_file)
