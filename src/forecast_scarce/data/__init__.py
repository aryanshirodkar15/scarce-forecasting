"""Dataset loaders behind one interface.

Every loader returns a `Dataset` whose panel is long format with NaN meaning
absence and 0.0 meaning observed zero demand. Normalized panels are cached to
parquet so the expensive read happens once.
"""

from __future__ import annotations

from . import cta, favorita, m5
from .base import Dataset
from .download import CACHE_DIR
from .registry import check, sha256_dir

LOADERS = {"m5": m5.load, "favorita": favorita.load, "cta": cta.load}


def load(name: str, *, refresh: bool = False, update_hashes: bool = False, **kwargs) -> Dataset:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(LOADERS)}")

    suffix = "_".join(f"{k}-{v}" for k, v in sorted(kwargs.items()) if v is not None)
    cache = CACHE_DIR / (f"{name}_{suffix}" if suffix else name)

    if cache.exists() and not refresh:
        dataset = Dataset.from_parquet(cache, name=name)
    else:
        dataset = LOADERS[name](**kwargs)
        dataset.to_parquet(cache)

    check(name, "normalized", sha256_dir(cache), update=update_hashes)
    return dataset


def available() -> list[str]:
    return sorted(LOADERS)


__all__ = ["Dataset", "load", "available", "LOADERS"]
