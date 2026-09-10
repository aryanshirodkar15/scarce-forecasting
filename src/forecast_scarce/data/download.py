"""Fetching raw data into the gitignored cache.

Kaggle competition data cannot be fetched anonymously: it needs API credentials
and, separately, acceptance of each competition's rules on the website. We try
the CLI and otherwise print exactly what to do by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"


def kaggle_competition(slug: str, target: Path) -> Path:
    """Download and unzip a Kaggle competition into `target`."""
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        return target

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            _manual_instructions(slug, target, reason="the kaggle CLI is not installed")
        )

    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _manual_instructions(
                slug, target, reason=result.stderr.strip() or "the kaggle CLI failed"
            )
        )

    for archive in target.glob("*.zip"):
        shutil.unpack_archive(archive, target)
        archive.unlink()
    return target


def _manual_instructions(slug: str, target: Path, reason: str) -> str:
    return (
        f"Could not download '{slug}' automatically: {reason}.\n\n"
        f"To do it by hand:\n"
        f"  1. Accept the competition rules at https://www.kaggle.com/c/{slug}/rules\n"
        f"  2. Download the data from https://www.kaggle.com/c/{slug}/data\n"
        f"  3. Unzip everything into {target}\n\n"
        f"For the CLI route instead: pip install kaggle, then put an API token at\n"
        f"~/.kaggle/kaggle.json (Kaggle account settings -> Create New Token), chmod 600."
    )


def socrata(domain: str, resource: str, target: Path, page: int = 50_000) -> Path:
    """Page a Socrata dataset to newline-delimited JSON. No auth needed."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://{domain}/resource/{resource}.json"
    offset = 0
    tmp = target.with_suffix(".partial")

    with tmp.open("w") as fh:
        while True:
            response = requests.get(
                url,
                params={"$limit": page, "$offset": offset, "$order": ":id"},
                timeout=120,
            )
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            offset += len(rows)
            print(f"  fetched {offset} rows")
            if len(rows) < page:
                break

    tmp.rename(target)
    return target
