#!/usr/bin/env python3
"""Check GitHub Actions/combined status for a commit SHA.

This is a small helper for maintainers: it queries the public GitHub API
(endpoints that do not require auth for public repos) and summarizes status.

Usage:
  python scripts/check_github_ci.py --repo OJWatson/motac --sha <sha>

Exit codes:
  0: success / all checks passed
  1: checks pending
  2: checks failed / error
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "motac-ci-check/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
        data = resp.read().decode("utf-8")
    return json.loads(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="owner/name, e.g. OJWatson/motac")
    p.add_argument("--sha", required=True, help="full commit SHA")
    args = p.parse_args(argv)

    owner, name = args.repo.split("/", 1)
    sha = args.sha

    base = f"https://api.github.com/repos/{owner}/{name}"

    # 1) Combined commit status API.
    status_url = f"{base}/commits/{sha}/status"
    try:
        status = _get_json(status_url)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to fetch/parse {status_url}: {e}")
        return 2

    state = status.get("state", "unknown")
    total = status.get("total_count", "?")
    print(f"combined_status: {state} (contexts={total})")

    # 2) Check-runs API (more GitHub Actions oriented).
    checks_url = f"{base}/commits/{sha}/check-runs"
    try:
        checks = _get_json(checks_url)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"WARN: failed to fetch/parse {checks_url}: {e}")
        checks = {}

    runs = checks.get("check_runs") or []
    if runs:
        # Print a small stable summary.
        for r in runs:
            name_ = r.get("name", "")
            status_ = r.get("status", "")
            conclusion = r.get("conclusion", "")
            print(f"check_run: {name_}: {status_} {conclusion}")

        any_in_progress = any(r.get("status") != "completed" for r in runs)
        any_failed = any(
            (r.get("status") == "completed")
            and (r.get("conclusion") not in ("success", "neutral", "skipped"))
            for r in runs
        )

        if any_failed:
            return 2
        if any_in_progress:
            return 1
        return 0

    # Fall back to combined status.
    if state == "success":
        return 0
    if state in {"pending", "expected"}:
        return 1
    if state == "failure":
        return 2

    print("WARN: unknown combined status state")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
