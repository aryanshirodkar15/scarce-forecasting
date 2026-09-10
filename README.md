# forecast-scarce

A benchmark for demand forecasting under data scarcity.

## The question

At what data volume do deep forecasting models stop outperforming classical
statistical baselines?

Published forecasting benchmarks train on years of clean history across thousands
of SKUs. Small retailers have 6 to 18 months of noisy, intermittent, gap-filled
data across dozens of SKUs. The regime that most papers evaluate is not the regime
most businesses operate in. This benchmark looks for the crossover point and tries
to characterize what drives it.

## Status

Under construction, built in stages: **loaders (done)**, **sampling protocol
(done)**, metrics, model wrappers, runner, analysis.

## Datasets

| name | what | series | access |
| --- | --- | --- | --- |
| `m5` | Walmart hierarchical retail | 30,490 at item-store level | Kaggle competition |
| `favorita` | Corporacion Favorita grocery, perishables flagged | ~200k item-store | Kaggle competition |
| `cta` | Chicago rail station entries, non-retail control | ~145 stations | open Socrata API |

Raw data is never committed. `data/` is gitignored, downloads are scripted, and
both the raw files and the normalized panels are hashed into `data_registry.json`
so a result can be traced to the exact bytes that produced it.

The two Kaggle sets need credentials and competition-rule acceptance. If the
`kaggle` CLI is not set up, the loader prints the manual steps instead of failing
silently.

## Reproducing

```
make setup    # uv venv on Python 3.11, locked dependencies
make data     # download and normalize all three datasets
make test
```

## Panel conventions

Every loader returns the same long-format panel: `series_id`, `ds`, `y`.

The one convention worth stating up front is that **`NaN` means the demand
process was not observable and `0.0` means it was observed and produced zero**.
Retail data conflates these constantly, and telling them apart is what makes the
intermittency statistics trustworthy. See `DESIGN_NOTES.md`.
