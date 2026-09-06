"""
FILE PURPOSE:
Verify, against the live internet, that every search provider actually works
from this machine — and say exactly what is wrong when one doesn't.

    python check_providers.py
    python check_providers.py --claim "the prime minister of India resigned"

WHY THIS EXISTS:
The provider tests in tests/ verify PARSING against captured payload shapes.
They make no network calls, so they cannot tell you whether Google News is
reachable from where you are sitting, whether your GNEWS_API_KEY is valid, or
whether DuckDuckGo is rate-limiting you today. Those are the failures that
actually ruin a demo, and they are invisible to the test suite by design.

They are also the failures most easily misread. A provider that is blocked
returns nothing, the pipeline reports "insufficient evidence", and the natural
conclusion is that the fact-checker is broken — when the truth is that no
search ran. This script separates those two cases in about ten seconds.

Run it before a demo, and after changing anything about providers or keys.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import urllib.error
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from main import load_env_file  # noqa: E402
from providers.duckduckgo import search as ddg_search  # noqa: E402
from providers.google_news import search as google_news_search  # noqa: E402
from providers.news_apis import (  # noqa: E402
    search_gnews,
    search_guardian,
    search_newsapi,
)
from providers.wikipedia import search as wikipedia_search  # noqa: E402


# (label, callable, env key or None, why it matters)
PROVIDERS = [
    ("google_news", google_news_search, None,
     "primary source of recent coverage; no key needed"),
    ("wikipedia", wikipedia_search, None,
     "background knowledge for timeless claims; no key needed"),
    ("duckduckgo", ddg_search, None,
     "fallback web search; frequently rate-limited"),
    ("gnews", search_gnews, "GNEWS_API_KEY",
     "optional, improves recent-headline recall"),
    ("guardian", search_guardian, "GUARDIAN_API_KEY",
     "optional, returns full article bodies"),
    ("newsapi", search_newsapi, "NEWSAPI_KEY",
     "optional, broad publisher coverage"),
]

GREEN, RED, YELLOW, GREY, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"
if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
    GREEN = RED = YELLOW = GREY = RESET = ""


def diagnose(error: Exception) -> str:
    """Turn an exception into something worth acting on."""
    text = str(error)

    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            return (
                f"HTTP {error.code} — rejected. If this provider needs a key, it is "
                "missing, invalid, or out of quota. If it needs no key, the host is "
                "blocked from this network."
            )
        if error.code == 429:
            return "HTTP 429 — rate limited. Wait, or configure a keyed provider instead."
        return f"HTTP {error.code} — {error.reason}"

    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        if isinstance(reason, socket.timeout) or "timed out" in text:
            return "timed out — the host is unreachable or very slow from this network."
        if "CONNECT" in text or "Tunnel" in text or "403" in text:
            return (
                "connection refused by a proxy — this network blocks the host. "
                "Nothing to fix in the code."
            )
        if "Name or service not known" in text or "nodename nor servname" in text:
            return "DNS lookup failed — check your internet connection."
        if "certificate" in text.lower():
            return "TLS verification failed — a proxy is intercepting HTTPS."
        return f"network error — {reason}"

    if isinstance(error, RuntimeError):
        # The keyed providers raise this for an error payload returned with HTTP 200.
        return f"provider returned an error — {text}"

    return f"{type(error).__name__} — {text}"


def check(label, search, env_key, note, claim, timeout_note):
    """Run one provider and print a single line describing what happened."""
    if env_key and not os.getenv(env_key):
        print(f"  {GREY}skip {label:<12} {env_key} not set — {note}{RESET}")
        return "skipped"

    started = time.monotonic()
    try:
        results = search(claim, max_results=3)
    except Exception as error:  # noqa: BLE001 - every failure mode is reportable
        elapsed = time.monotonic() - started
        print(f"  {RED}FAIL {label:<12} ({elapsed:.1f}s) {diagnose(error)}{RESET}")
        return "failed"

    elapsed = time.monotonic() - started
    if not results:
        print(
            f"  {YELLOW}EMPTY{label:<12} ({elapsed:.1f}s) reachable, but returned no "
            f"results for this claim{RESET}"
        )
        return "empty"

    top = results[0]
    print(f"  {GREEN}ok   {label:<12} ({elapsed:.1f}s) {len(results)} results{RESET}")
    print(f"       {GREY}{(top.source or 'unknown'):<22} {top.title[:60]}{RESET}")

    # A result with no usable text is reachable but useless downstream.
    if not (top.snippet or top.text):
        print(f"       {YELLOW}note: no snippet or body text — parsing may have changed{RESET}")
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim", default="India prime minister resigned",
        help="what to search for (default: a recent-news style query)",
    )
    args = parser.parse_args()

    load_env_file()

    print(f"\nChecking search providers with: {args.claim!r}\n")
    outcomes = [
        check(label, search, env_key, note, args.claim, None)
        for label, search, env_key, note in PROVIDERS
    ]

    working = outcomes.count("ok")
    failed = outcomes.count("failed")
    skipped = outcomes.count("skipped")

    print(f"\n{working} working · {failed} failing · {skipped} not configured\n")

    if working == 0:
        print(
            f"{RED}No provider returned anything. Every check will report "
            f"insufficient evidence, and that will be a retrieval failure rather "
            f"than a finding about the claim.{RESET}\n"
        )
        return 1
    if working < 2:
        print(
            f"{YELLOW}Only one provider is working. Verdicts will be thin, and "
            f"'no credible source reports this' needs several providers before it "
            f"is trustworthy.{RESET}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
