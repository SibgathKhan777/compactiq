"""
llm_fallback.py - Optional LLM opinion for packages the ML model has never
seen (is_known_package=False), via Groq's OpenAI-compatible chat API.

Scope, deliberately narrow: this ONLY fires for packages outside the trained
catalog, where pycompat_model.py's own prediction is explicitly documented as
"an extrapolation, not a real signal" (package-holdout eval shows worse than
the majority-class baseline there). An LLM has no more ground-truth authority
than that ML prediction does -- it can hallucinate a package's existence or
behavior just as easily as the classifier can guess wrong. So its opinion is
surfaced as a labeled, unverified hint alongside the existing explanation,
NEVER as a fact, and it does not replace or block live PyPI / Docker checks --
if anything, an unknown-to-ML package is exactly the case that most needs the
real ground truth those checks provide, LLM opinion or not.

Off by default (GROQ_API_KEY unset -> every call here is a no-op returning
"unavailable"), consistent with docker_verify's existing opt-in pattern:
nothing about compactiq's default behavior should depend on an external API
key or network call being present.

No extra dependency: uses urllib.request like live_verify.py, not `requests`
(never a declared dependency in this repo).
"""

import json
import os
import urllib.error
import urllib.request

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


def _load_dotenv(path=None):
    """Minimal .env loader -- no python-dotenv dependency. Only sets a var if
    it isn't already in the environment, so a real env var always wins."""
    path = path or os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def llm_available():
    return bool(os.environ.get("GROQ_API_KEY"))


def ask_llm_about_package(package, version, python_version, platform_key, timeout=12):
    """
    Returns a dict:
        { "available": bool,       -- was a real call made (API key present, no transport error)
          "verdict": "real" | "suspicious" | "unknown" | None,
          "opinion": str | None,   -- the LLM's raw explanation, always labeled as unverified by callers
          "error": str | None }

    "available": False means no signal here at all (missing key, network
    failure, malformed response) -- callers must treat this exactly like "the
    LLM wasn't consulted," never as "the LLM said it's fine."
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"available": False, "verdict": None, "opinion": None, "error": "GROQ_API_KEY not set"}

    prompt = (
        f"A Python dependency checker found the package \"{package}\" version \"{version}\" "
        f"is NOT in its training catalog (never seen before). It needs a quick second opinion "
        f"before deciding whether to trust an install of this on platform \"{platform_key}\" "
        f"with Python {python_version}.\n\n"
        f"Answer in exactly this format:\n"
        f"VERDICT: <REAL | SUSPICIOUS | UNKNOWN>\n"
        f"REASON: <one or two sentences>\n\n"
        f"REAL = you're confident this is a real, currently-published PyPI package at roughly "
        f"this version. SUSPICIOUS = the name looks like a likely typo of a well-known package, "
        f"or you don't believe this version/package exists. UNKNOWN = you genuinely don't have "
        f"reliable knowledge either way -- say so rather than guessing."
    )

    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        # GROQ_MODEL is a reasoning model -- it spends completion tokens on a
        # hidden chain-of-thought before the final answer, so a small budget
        # here truncates to an EMPTY content field, not a short one (verified
        # directly: max_tokens=20 produced content="" with 36 reasoning
        # tokens burned and finish_reason="length"). 400 leaves real headroom
        # for both the reasoning and the two-line VERDICT/REASON answer.
        "max_tokens": 400,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_CHAT_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Groq's endpoint is behind Cloudflare, which blocks Python's
            # default "Python-urllib/x.y" User-Agent outright (a bare HTTP 403
            # with Cloudflare error 1010, not a Groq auth error -- confirmed
            # by testing with a known-good key). Any browser-like UA clears it.
            "User-Agent": "Mozilla/5.0 (compactiq/1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return {"available": False, "verdict": None, "opinion": None, "error": f"http_error_{e.code}"}
    except Exception as e:
        return {"available": False, "verdict": None, "opinion": None, "error": f"{type(e).__name__}: {e}"}

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return {"available": False, "verdict": None, "opinion": None, "error": "malformed_response"}

    verdict = None
    for candidate in ("REAL", "SUSPICIOUS", "UNKNOWN"):
        if f"VERDICT: {candidate}" in content.upper().replace(" ", " "):
            verdict = candidate.lower()
            break

    return {"available": True, "verdict": verdict, "opinion": content, "error": None}


if __name__ == "__main__":
    import sys
    pkg = sys.argv[1] if len(sys.argv) > 1 else "reqests"
    ver = sys.argv[2] if len(sys.argv) > 2 else "2.31.0"
    result = ask_llm_about_package(pkg, ver, "3.12", "linux_x86_64")
    print(json.dumps(result, indent=2))
