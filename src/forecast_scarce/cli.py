from __future__ import annotations

import argparse
from pathlib import Path

from . import data
from .config import RunConfig
from .runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast-scarce")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("data", help="download and normalize a dataset")
    fetch.add_argument("dataset", choices=[*data.available(), "all"])
    fetch.add_argument("--level", default=None, help="aggregation level (m5 only)")
    fetch.add_argument("--refresh", action="store_true", help="ignore the parquet cache")
    fetch.add_argument("--update-hashes", action="store_true", help="accept a changed panel")

    experiment = sub.add_parser("run", help="execute a configured grid")
    experiment.add_argument("--config", required=True, type=Path)
    experiment.add_argument("--results", default=None, type=Path)
    experiment.add_argument("--refresh", action="store_true", help="rerun completed cells")
    experiment.add_argument("--dry-run", action="store_true", help="list cells and stop")

    args = parser.parse_args(argv)

    if args.command == "data":
        return _data(args)
    return _run(args)


def _data(args) -> int:
    names = data.available() if args.dataset == "all" else [args.dataset]
    for name in names:
        kwargs = {"level": args.level} if name == "m5" and args.level else {}
        print(f"{name}:")
        dataset = data.load(
            name, refresh=args.refresh, update_hashes=args.update_hashes, **kwargs
        )
        for key, value in dataset.summary().items():
            print(f"  {key}: {value}")
    return 0


def _run(args) -> int:
    config = RunConfig.from_yaml(args.config)
    cells = config.cells()

    if args.dry_run:
        print(f"{config.name}: {len(cells)} cells x {len(config.models)} models")
        for cell in cells:
            print(f"  {cell.cell_id}")
        return 0

    out = run(config, results_dir=args.results, refresh=args.refresh)
    print(f"results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
