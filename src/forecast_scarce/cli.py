from __future__ import annotations

import argparse

from . import data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast-scarce")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("data", help="download and normalize a dataset")
    fetch.add_argument("dataset", choices=data.available() + ["all"])
    fetch.add_argument("--level", default=None, help="aggregation level (m5 only)")
    fetch.add_argument("--refresh", action="store_true", help="ignore the parquet cache")
    fetch.add_argument("--update-hashes", action="store_true", help="accept a changed panel")

    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
