"""Execute a configured grid and write tidy results.

Each cell writes its own parquet and manifest as soon as it finishes, and cells
whose parquet already exists are skipped. A full grid is long enough that it will
be interrupted at some point, and losing a day of compute to a closed laptop is
not acceptable.

A model that fails on one cell is recorded and stepped over rather than killing
the run. Auto-ARIMA in particular will occasionally fail to converge on a short
intermittent window, and that is a data point, not a reason to stop.
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

from .config import Cell, RunConfig
from .data import Dataset, load
from .intermittency import classify
from .metrics import evaluate
from .models import build
from .sampling import sample
from .splits import rolling_origin, split

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def run(config: RunConfig, results_dir: Path | None = None, refresh: bool = False) -> Path:
    out_dir = (results_dir or RESULTS_DIR) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = config.cells()
    cache: dict[str, tuple[Dataset, pd.DataFrame]] = {}
    started = time.perf_counter()
    done = skipped = 0

    for index, cell in enumerate(cells, start=1):
        target = out_dir / f"{cell.cell_id}.parquet"
        if target.exists() and not refresh:
            skipped += 1
            continue

        name = cell.spec.dataset
        if name not in cache:
            print(f"loading {name}")
            dataset = load(name)
            cache[name] = (dataset, classify(dataset.panel))
        dataset, stats = cache[name]

        print(f"[{index}/{len(cells)}] {cell.cell_id}")
        rows, manifest = _run_cell(cell, dataset, stats, config)

        if rows.empty:
            print("  no rows produced, skipping write")
            continue
        rows.to_parquet(target, index=False)
        (out_dir / f"{cell.cell_id}.manifest.json").write_text(manifest)
        done += 1

    elapsed = time.perf_counter() - started
    print(f"\n{done} cells run, {skipped} already present, {elapsed / 60:.1f} min")
    return out_dir


def _run_cell(
    cell: Cell, dataset: Dataset, stats: pd.DataFrame, config: RunConfig
) -> tuple[pd.DataFrame, str]:
    drawn = sample(dataset, cell.spec, stats=stats)
    folds = rolling_origin(
        pd.Timestamp(drawn.manifest.test_start),
        cell.spec.history_days,
        cell.spec.test_days,
        cell.horizon,
    )
    quadrant = stats["quadrant"].astype(str)

    rows = []
    for model_name in cell.models:
        for fold in folds:
            train, test = split(drawn.panel, fold)
            try:
                scored, timing = _score(model_name, train, test, drawn.static, config)
            except Exception as error:  # noqa: BLE001 - a failure is a data point
                print(f"  {model_name} fold {fold.index} failed: {error}")
                traceback.print_exc(limit=1)
                rows.append(_failure_row(cell, model_name, fold.index, error))
                continue

            scored["model"] = model_name
            scored["fold"] = fold.index
            scored["fit_seconds"] = timing
            scored["error"] = None
            rows.append(scored)

    if not rows:
        return pd.DataFrame(), drawn.manifest.to_json()

    table = pd.concat(rows, ignore_index=True)
    table["quadrant"] = table["series_id"].map(quadrant)
    for key, value in _spec_columns(cell).items():
        table[key] = value

    return table, drawn.manifest.to_json()


def _score(model_name, train, test, static, config) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build(model_name, n_jobs=config.n_jobs)
        model.fit(train, static)
        prediction = model.predict(test[["series_id", "ds"]])
    elapsed = time.perf_counter() - started
    return evaluate(test, prediction, train), elapsed


def _failure_row(cell: Cell, model_name: str, fold: int, error: Exception) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": None,
                "n_scored": 0,
                "mase": float("nan"),
                "rmsse": float("nan"),
                "wape": float("nan"),
                "model": model_name,
                "fold": fold,
                "fit_seconds": float("nan"),
                "error": f"{type(error).__name__}: {error}",
            }
        ]
    )


def _spec_columns(cell: Cell) -> dict:
    s = cell.spec
    return {
        "dataset": s.dataset,
        "history_days": s.history_days,
        "n_series": s.n_series,
        "missing_rate": s.missing_rate,
        "missing_pattern": s.missing_pattern,
        "seed": s.seed,
        "allocation": s.allocation,
        "horizon": cell.horizon,
        "cell_id": cell.cell_id,
    }


def collect(results_dir: Path) -> pd.DataFrame:
    """Every cell's rows in one tidy long frame."""
    files = sorted(Path(results_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no result parquet files under {results_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_manifests(results_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(Path(results_dir).glob("*.manifest.json"))
    ]
