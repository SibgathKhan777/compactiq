"""
fix_project_requirements.py - Scans a whole project directory for every
requirements*.txt file (requirements.txt, requirements-dev.txt,
services/*/requirements.txt, etc.), checks each one against a real deployment
target, and reports what would break -- with an explicit --apply flag to write
the corrected versions back to disk.

Built on the same validate_install_code() pipeline as the dashboard's Deploy
Check tab / chat / API -- ML + live PyPI + joint dependency conflicts +
platform-marker exclusion + (optionally) real Docker verification. Running it
across a whole project in one pass is what this script adds: point it at a
project root instead of pasting one file at a time.

SAFE BY DEFAULT: dry-run unless you pass --apply. Even with --apply, every file
it actually changes gets a .bak backup written alongside it first -- this
touches real project files, so it never silently overwrites anything.

Known limitation, carried over from validate.py's PIN_RE parser: only exact
`package==version` pins are recognized. Lines using >=, ~=, extras like
package[extra]==1.2.3, -r includes, or --index-url options are left untouched
(not validated, not flagged, not modified) -- this only fixes what it's certain
about the syntax of.

Also accepts a GitHub repo directly in place of a local path -- a full URL
(https://github.com/owner/repo, with or without .git), an SSH remote
(git@github.com:owner/repo.git), or the "owner/repo" shorthand. It's cloned
(shallow, --depth 1) into ./<repo-name> in the current directory so --apply's
changes land somewhere you can inspect, diff, and push from afterward -- not a
throwaway temp dir. Refuses to clone over an existing directory of the same
name rather than silently reusing or overwriting whatever's already there.

Usage:
    python fix_project_requirements.py /path/to/project
    python fix_project_requirements.py https://github.com/owner/repo
    python fix_project_requirements.py owner/repo
    python fix_project_requirements.py /path/to/project --deploy-target linux_aarch64
    python fix_project_requirements.py /path/to/project --docker-verify
    python fix_project_requirements.py /path/to/project --apply
    python fix_project_requirements.py /path/to/project --pattern "requirements*.txt"
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

from pycompat_model import PyCompatModel
from validate import validate_install_code, parse_pip_install

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "env", "__pycache__",
             "site-packages", "dist", "build", ".tox", ".mypy_cache"}

_GITHUB_URL_RE = re.compile(r"^(?:https?://github\.com/|git@github\.com:)([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
_SHORTHAND_RE = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def is_repo_reference(source):
    """True if `source` looks like a GitHub repo reference rather than a local path."""
    if os.path.exists(source):
        return False
    return bool(_GITHUB_URL_RE.match(source)) or bool(_SHORTHAND_RE.match(source))


def resolve_project_source(source, log=print):
    """
    If `source` is a local path, returns it unchanged. If it's a GitHub repo
    reference, clones it (shallow) into ./<repo-name> and returns that local
    path. Raises FileExistsError if that directory already exists, rather
    than guessing whether to reuse or overwrite someone's existing checkout.
    """
    if not is_repo_reference(source):
        return source

    m = _GITHUB_URL_RE.match(source) or _SHORTHAND_RE.match(source)
    owner, repo = m.group(1), m.group(2)
    clone_url = f"https://github.com/{owner}/{repo}.git"
    dest = os.path.join(os.getcwd(), repo)

    if os.path.exists(dest):
        raise FileExistsError(
            f"'{dest}' already exists -- refusing to clone over it. "
            f"Remove it, rename it, or point directly at that path instead."
        )

    log(f"Cloning {clone_url} -> {dest} ...")
    subprocess.run(["git", "clone", "--depth", "1", clone_url, dest], check=True)
    return dest


def find_requirements_files(project_dir, pattern="requirements*.txt", max_depth=6):
    """Recursively find requirements files, skipping common noise directories."""
    project_dir = os.path.abspath(project_dir)
    matches = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        depth = root[len(project_dir):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                matches.append(os.path.join(root, f))
    return sorted(matches)


def fix_project(project_dir, python_version="3.12", deploy_platform="linux_x86_64",
                 pattern="requirements*.txt", model=None, live=True, docker_verify=False,
                 use_llm=False, apply=False, log=None):
    """
    Returns a list of per-file result dicts:
        { "file", "skipped", "reason" }  -- if no recognizable pins found, or
        { "file", "skipped": False, "changed", "safe", "risk_score",
          "original", "corrected", "packages", "applied" }
    """
    model = model or PyCompatModel.load(os.path.join(os.path.dirname(__file__), "model"))
    files = find_requirements_files(project_dir, pattern)
    log = log or (lambda msg: None)

    results = []
    for path in files:
        log(f"Checking {path} ...")
        with open(path, "r") as f:
            original_content = f.read()

        pins = parse_pip_install(original_content)
        if not pins:
            results.append({"file": path, "skipped": True, "reason": "no package==version pins found"})
            continue

        result = validate_install_code(
            original_content, python_version=python_version, platform=deploy_platform,
            model=model, live=live, docker_verify=docker_verify, use_llm=use_llm,
        )

        changed = result["corrected_code"] != original_content
        applied = False
        if apply and changed:
            backup_path = path + ".bak"
            with open(backup_path, "w") as f:
                f.write(original_content)
            with open(path, "w") as f:
                f.write(result["corrected_code"])
            applied = True
            log(f"  -> applied, backup saved to {backup_path}")
        elif changed:
            log(f"  -> would change (dry run, no files written)")
        elif result["fully_resolved"]:
            log(f"  -> safe, no changes needed")
        else:
            log(f"  -> UNSAFE, no automatic fix available")

        results.append({
            "file": path,
            "skipped": False,
            "changed": changed,
            "safe": result["fully_resolved"],
            "risk_score": result["risk_score"],
            "original": original_content,
            "corrected": result["corrected_code"],
            "packages": result["packages"],
            "applied": applied,
        })

    return results


def _print_summary(results, apply):
    total = len(results)
    checked = [r for r in results if not r["skipped"]]
    changed = [r for r in checked if r["changed"]]
    unsafe = [r for r in checked if not r["safe"]]

    print(f"\n{'='*70}")
    print(f"Scanned {total} file(s): {len(checked)} checked, {total - len(checked)} skipped (no pins found)")
    print(f"{len(changed)} file(s) need changes for this deploy target")
    if unsafe:
        print(f"⚠️  {len(unsafe)} file(s) still have unresolved issues even after correction")
    print(f"{'='*70}\n")

    for r in results:
        if r["skipped"]:
            print(f"⏭️  {r['file']} -- {r['reason']}")
            continue
        if not r["changed"]:
            if not r["safe"]:
                # No correction was possible (e.g. a misspelled/hallucinated
                # package name doesn't exist under ANY version, so there's
                # nothing to change TO) -- that's the opposite of "safe as-is",
                # and reporting it that way would silently hide exactly the
                # hallucinated-package case this tool exists to catch.
                print(f"❌ {r['file']} -- UNSAFE, no automatic fix available")
                for p in r["packages"]:
                    if not p["is_clean"]:
                        print(f"     {p['package']}=={p['requested_version']}: {p['explanation']}")
                continue
            print(f"✅ {r['file']} -- safe as-is")
            continue
        status = "✅ fixed" if r["safe"] else "⚠️  best attempt, still not fully safe"
        action = "APPLIED" if r["applied"] else "dry run -- use --apply to write this"
        print(f"🔧 {r['file']} -- {status} ({action})")
        for p in r["packages"]:
            if p["changed"]:
                print(f"     {p['package']}: {p['requested_version']} -> {p['corrected_version']}")
            elif not p["is_clean"]:
                print(f"     {p['package']}=={p['requested_version']}: still unresolved -- {p['explanation']}")
    print()
    if changed and not apply:
        print("This was a dry run -- nothing was written. Re-run with --apply to write the fixes")
        print("(a .bak backup is created for every file actually changed).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scan a project for requirements*.txt files and check/fix them for a deploy target")
    ap.add_argument("project_dir", help="Path to the project root to scan, or a GitHub repo "
                                         "(https://github.com/owner/repo, git@github.com:owner/repo.git, "
                                         "or owner/repo) to clone and scan")
    ap.add_argument("--python-version", default="3.12")
    ap.add_argument("--deploy-target", default="linux_x86_64",
                     help="Target platform_key (default linux_x86_64 -- standard AWS EC2/most cloud targets)")
    ap.add_argument("--pattern", default="requirements*.txt", help="Filename glob to match (default: requirements*.txt)")
    ap.add_argument("--docker-verify", action="store_true", help="Also run real Docker install tests (slow, needs Docker running)")
    ap.add_argument("--use-llm", action="store_true",
                     help="Consult an LLM (via Groq, needs GROQ_API_KEY set) for any package outside the "
                          "trained catalog -- labeled, unverified second opinion, and automatically forces "
                          "real Docker verification for that package too")
    ap.add_argument("--no-live", action="store_true", help="Skip live PyPI checks (ML-only, faster, less accurate)")
    ap.add_argument("--apply", action="store_true", help="Actually write fixes to disk (default: dry run only)")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the summary")
    args = ap.parse_args()

    log = (lambda msg: None) if args.json else print
    try:
        project_dir = resolve_project_source(args.project_dir, log=log)
    except (FileExistsError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = fix_project(
        project_dir,
        python_version=args.python_version,
        deploy_platform=args.deploy_target,
        pattern=args.pattern,
        live=not args.no_live,
        docker_verify=args.docker_verify,
        use_llm=args.use_llm,
        apply=args.apply,
        log=log,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        _print_summary(results, args.apply)
