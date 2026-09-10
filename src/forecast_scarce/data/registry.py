"""Content hashes for raw downloads and normalized panels.

Two hashes per dataset, deliberately. The raw hash catches an upstream file
changing under us (Kaggle re-releases, a Socrata backfill). The normalized hash
catches our own loader code changing the panel without anyone noticing. A result
is only comparable to an earlier result if both still match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "data_registry.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(directory: Path, suffix: str = ".parquet") -> str:
    """Hash of a directory's contents, stable under file ordering."""
    parts = []
    for path in sorted(directory.rglob(f"*{suffix}")):
        parts.append(f"{path.relative_to(directory)}:{sha256_file(path)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")


def check(dataset: str, kind: str, digest: str, *, update: bool = False) -> None:
    """Verify a digest against the registry, or record it the first time.

    Raises on mismatch unless `update` is set, which is the deliberate escape
    hatch for when a loader change is intended.
    """
    registry = _load()
    known = registry.get(dataset, {}).get(kind)

    if known is None or update:
        registry.setdefault(dataset, {})[kind] = digest
        _save(registry)
        verb = "updated" if known else "recorded"
        print(f"  {verb} {dataset}/{kind} hash {digest[:12]}")
        return

    if known != digest:
        raise ValueError(
            f"{dataset}/{kind} hash mismatch.\n"
            f"  expected {known}\n"
            f"  got      {digest}\n"
            f"Results computed before and after this change are not comparable. "
            f"Re-run with --update-hashes only if the change was intended."
        )
