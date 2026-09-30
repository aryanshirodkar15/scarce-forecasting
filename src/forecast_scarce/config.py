"""YAML experiment configuration.

A config names the datasets, the models and the grid axes. Expanding it gives the
list of cells to run, each of which is a fully determined `ScarcitySpec`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sampling import ScarcitySpec
from .splits import DEFAULT_HORIZON


@dataclass(frozen=True)
class Cell:
    spec: ScarcitySpec
    models: tuple[str, ...]
    horizon: int

    @property
    def cell_id(self) -> str:
        """Readable slug plus a digest, so filenames are both greppable and unique."""
        s = self.spec
        slug = (
            f"{s.dataset}_h{s.history_days}_n{s.n_series}"
            f"_m{int(s.missing_rate * 100):02d}{s.missing_pattern[:1]}_s{s.seed}"
        )
        digest = hashlib.sha256(repr(s).encode()).hexdigest()[:8]
        return f"{slug}_{digest}"


@dataclass
class RunConfig:
    name: str
    datasets: list[str]
    models: list[str]
    history_days: list[int]
    n_series: list[int]
    missingness: list[dict]
    seeds: list[int]
    allocation: dict[str, str] = field(default_factory=dict)
    horizon: int = DEFAULT_HORIZON
    test_days: int = 168
    frame_days: int = 898
    n_jobs: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        grid = raw.pop("grid", {})
        missingness = grid.pop("missingness", [{"rate": 0.0, "pattern": "none"}])
        return cls(
            history_days=grid.pop("history_days"),
            n_series=grid.pop("n_series"),
            seeds=grid.pop("seeds", [0]),
            missingness=missingness,
            **raw,
        )

    def cells(self) -> list[Cell]:
        out = []
        for dataset in self.datasets:
            allocation = self.allocation.get(dataset, "equal")
            for history in self.history_days:
                for count in self.n_series:
                    for gap in self.missingness:
                        for seed in self.seeds:
                            spec = ScarcitySpec(
                                dataset=dataset,
                                history_days=history,
                                n_series=count,
                                missing_rate=float(gap.get("rate", 0.0)),
                                missing_pattern=gap.get("pattern", "none"),
                                seed=seed,
                                test_days=self.test_days,
                                frame_days=self.frame_days,
                                allocation=allocation,
                            )
                            out.append(
                                Cell(spec=spec, models=tuple(self.models), horizon=self.horizon)
                            )
        return out
