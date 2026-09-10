"""The scarcity sampling protocol.

Draws a reproducible subset of a dataset at a given point on the scarcity grid:
how much history, how many series, how much missingness and of what shape. Every
cell carries a manifest recording exactly what was drawn and how, so any cell can
be rebuilt byte for byte.

Two seeds, not one. Series selection is keyed on the sampling frame only, so the
same series appear at every history length and every missingness level and those
comparisons are paired. Missingness injection is keyed on the full spec, so each
cell gets its own independent gaps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .data.base import Dataset
from .intermittency import classify

STRATA = ("smooth", "erratic", "intermittent", "lumpy")
Pattern = Literal["none", "mcar", "burst"]


@dataclass(frozen=True)
class ScarcitySpec:
    dataset: str
    history_days: int
    n_series: int
    missing_rate: float = 0.0
    missing_pattern: Pattern = "none"
    seed: int = 0
    # Eligibility is judged over this window, not over history_days, so that the
    # same series are available at every history length. Otherwise the short
    # cells would quietly draw from a different, longer-lived population.
    frame_days: int = 730
    min_coverage: float = 0.8
    burst_mean_days: int = 7
    end_date: str | None = None

    def __post_init__(self) -> None:
        if self.history_days > self.frame_days:
            raise ValueError(
                f"history_days {self.history_days} exceeds frame_days {self.frame_days}"
            )
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError(f"missing_rate must be in [0, 1), got {self.missing_rate}")
        if self.missing_pattern not in ("none", "mcar", "burst"):
            raise ValueError(f"unknown missing_pattern {self.missing_pattern!r}")
        if self.missing_pattern == "none" and self.missing_rate:
            raise ValueError("missing_rate is nonzero but missing_pattern is 'none'")
        if self.n_series < len(STRATA):
            raise ValueError(f"n_series must be at least {len(STRATA)} to fill every stratum")


@dataclass
class Manifest:
    spec: dict
    selection_seed: int
    injection_seed: int
    frame_start: str
    frame_end: str
    window_start: str
    window_end: str
    requested_per_stratum: dict
    actual_per_stratum: dict
    n_series_requested: int
    n_series_actual: int
    shortfall: dict
    native_absent_rate: float
    injected_rate: float
    final_absent_rate: float
    window_stratum_counts: dict
    quadrant_drift_rate: float
    series_ids: list = field(repr=False)
    panel_sha256: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


@dataclass
class ScarceSample:
    panel: pd.DataFrame
    static: pd.DataFrame
    stats: pd.DataFrame
    manifest: Manifest


def _digest(payload: dict) -> int:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _selection_key(spec: ScarcitySpec) -> dict:
    # Deliberately excludes history_days and everything about missingness.
    return {
        "dataset": spec.dataset,
        "n_series": spec.n_series,
        "seed": spec.seed,
        "frame_days": spec.frame_days,
        "min_coverage": spec.min_coverage,
        "end_date": spec.end_date,
    }


def sample(dataset: Dataset, spec: ScarcitySpec, stats: pd.DataFrame | None = None) -> ScarceSample:
    """Draw one cell of the scarcity grid."""
    if stats is None:
        stats = classify(dataset.panel)

    panel = dataset.panel
    frame_end = pd.Timestamp(spec.end_date) if spec.end_date else panel["ds"].max()
    frame_start = frame_end - pd.Timedelta(days=spec.frame_days - 1)
    window_start = frame_end - pd.Timedelta(days=spec.history_days - 1)

    frame = panel.loc[panel["ds"].between(frame_start, frame_end)]
    eligible = _eligible(frame, stats, spec)

    selection_rng = np.random.default_rng(_digest(_selection_key(spec)))
    chosen, requested, actual = _stratified_draw(eligible, spec, selection_rng)

    window = frame.loc[
        frame["ds"].between(window_start, frame_end) & frame["series_id"].isin(chosen)
    ].copy()
    window["series_id"] = pd.Categorical(window["series_id"], categories=chosen)
    window = window.sort_values(["series_id", "ds"]).reset_index(drop=True)

    native_absent = float(window["y"].isna().mean())

    injection_rng = np.random.default_rng(_digest(asdict(spec)))
    window, injected = _inject_missingness(window, spec, injection_rng)
    final_absent = float(window["y"].isna().mean())

    window_stats = classify(window)
    drift = _drift_rate(stats, window_stats, chosen)

    manifest = Manifest(
        spec=asdict(spec),
        selection_seed=_digest(_selection_key(spec)),
        injection_seed=_digest(asdict(spec)),
        frame_start=str(frame_start.date()),
        frame_end=str(frame_end.date()),
        window_start=str(window_start.date()),
        window_end=str(frame_end.date()),
        requested_per_stratum=requested,
        actual_per_stratum=actual,
        n_series_requested=spec.n_series,
        n_series_actual=len(chosen),
        shortfall={q: requested[q] - actual[q] for q in STRATA if requested[q] > actual[q]},
        native_absent_rate=round(native_absent, 6),
        injected_rate=round(injected, 6),
        final_absent_rate=round(final_absent, 6),
        window_stratum_counts={
            str(k): int(v) for k, v in window_stats["quadrant"].value_counts().items()
        },
        quadrant_drift_rate=round(drift, 6),
        series_ids=list(chosen),
    )
    manifest.panel_sha256 = hashlib.sha256(
        pd.util.hash_pandas_object(window, index=False).values.tobytes()
    ).hexdigest()

    static = dataset.static.loc[dataset.static.index.isin(chosen)]
    return ScarceSample(panel=window, static=static, stats=window_stats, manifest=manifest)


def _eligible(frame: pd.DataFrame, stats: pd.DataFrame, spec: ScarcitySpec) -> pd.DataFrame:
    """Series with enough observed history in the frame and a usable quadrant."""
    coverage = (
        frame.assign(seen=frame["y"].notna())
        .groupby("series_id", observed=True)["seen"]
        .agg(["sum", "size"])
    )
    covered = coverage["sum"] / spec.frame_days
    keep = covered[covered >= spec.min_coverage].index

    usable = stats.loc[stats.index.isin(keep) & stats["quadrant"].isin(STRATA)]
    return usable


def _stratified_draw(
    eligible: pd.DataFrame, spec: ScarcitySpec, rng: np.random.Generator
) -> tuple[list[str], dict, dict]:
    """Equal allocation across the four quadrants, remainder spread deterministically."""
    base, remainder = divmod(spec.n_series, len(STRATA))
    requested = {q: base for q in STRATA}
    for q in STRATA[:remainder]:
        requested[q] += 1

    chosen: list[str] = []
    actual: dict[str, int] = {}
    for quadrant in STRATA:
        pool = eligible.index[eligible["quadrant"] == quadrant].to_numpy()
        want = requested[quadrant]
        take = min(want, len(pool))
        if take:
            # Sort first so the draw does not inherit the panel's row order.
            picked = rng.choice(np.sort(pool), size=take, replace=False)
            chosen.extend(picked.tolist())
        actual[quadrant] = take

    return chosen, requested, actual


def _inject_missingness(
    window: pd.DataFrame, spec: ScarcitySpec, rng: np.random.Generator
) -> tuple[pd.DataFrame, float]:
    if spec.missing_pattern == "none" or spec.missing_rate == 0:
        return window, 0.0

    values = window.pivot(index="series_id", columns="ds", values="y")
    array = values.to_numpy(dtype="float64", copy=True)
    n_masked = 0
    n_eligible = 0

    for row in range(array.shape[0]):
        observed = ~np.isnan(array[row])
        n_eligible += int(observed.sum())
        target = int(round(spec.missing_rate * observed.sum()))
        if target == 0:
            continue

        if spec.missing_pattern == "mcar":
            positions = np.flatnonzero(observed)
            drop = rng.choice(positions, size=min(target, len(positions)), replace=False)
        else:
            drop = np.flatnonzero(
                _burst_mask(len(observed), observed, target, rng, spec.burst_mean_days)
            )
        array[row, drop] = np.nan
        n_masked += len(drop)

    masked = pd.DataFrame(array, index=values.index, columns=values.columns)
    out = (
        masked.stack(future_stack=True)
        .rename("y")
        .reset_index()
        .astype({"y": "float32"})
    )
    out["series_id"] = pd.Categorical(out["series_id"], categories=values.index)
    out = out.sort_values(["series_id", "ds"]).reset_index(drop=True)
    return out, (n_masked / n_eligible if n_eligible else 0.0)


def _burst_mask(
    n: int, eligible: np.ndarray, target: int, rng: np.random.Generator, mean_len: int
) -> np.ndarray:
    """Contiguous outage runs rather than independent dropouts."""
    mask = np.zeros(n, dtype=bool)
    guard = 0
    while int((mask & eligible).sum()) < target and guard < 10 * n:
        guard += 1
        length = max(1, int(rng.geometric(1.0 / mean_len)))
        start = int(rng.integers(0, n))
        mask[start : start + length] = True

    # Trim any overshoot so the realized rate matches the requested one. This
    # nicks the edges of a few runs but keeps the burst structure intact.
    over = int((mask & eligible).sum()) - target
    if over > 0:
        masked_positions = np.flatnonzero(mask & eligible)
        mask[rng.choice(masked_positions, size=over, replace=False)] = False
    return mask & eligible


def _drift_rate(full: pd.DataFrame, window: pd.DataFrame, chosen: list[str]) -> float:
    """Share of sampled series whose quadrant changes when seen through the window."""
    if not chosen:
        return 0.0
    before = full.loc[chosen, "quadrant"].astype(str)
    after = window["quadrant"].reindex(chosen).astype(str)
    return float((before.to_numpy() != after.to_numpy()).mean())
