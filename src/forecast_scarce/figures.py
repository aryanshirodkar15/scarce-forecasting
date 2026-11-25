"""Every figure in the paper, regenerated from a results directory.

Each figure is written as a vector PDF for the paper, a PNG for quick review, and
a CSV holding the exact numbers plotted, so no value is only readable off a chart.

Colour follows the model, never its rank: a model keeps the same hue in every
figure and in every grid, including the deep models before they exist, so adding
them later repaints nothing. The order is a validated categorical palette (checked
for colour-vision deficiency against white paper), and each model also carries
its own marker shape so the figures survive greyscale printing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import (  # noqa: E402
    bootstrap_difference,
    compare_grid,
    regime_stability,
    seed_stability,
)
from .runner import collect, collect_daily, collect_shap, load_manifests  # noqa: E402

MODEL_ORDER = [
    "lightgbm",
    "seasonal_naive",
    "ets",
    "auto_arima",
    "croston",
    "tsb",
    "nbeats",
    "tft",
]
MODEL_COLORS = dict(
    zip(
        MODEL_ORDER,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        strict=True,
    )
)
MODEL_MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P", "X", "*"], strict=True))
MODEL_NAMES = {
    "lightgbm": "LightGBM",
    "seasonal_naive": "Seasonal naive",
    "ets": "ETS",
    "auto_arima": "Auto-ARIMA",
    "croston": "Croston",
    "tsb": "TSB",
    "nbeats": "N-BEATS",
    "tft": "TFT",
}
QUADRANT_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"

# Blue and red poles with a neutral grey midpoint, so "no difference" reads as
# nothing and the two directions read as opposite.
DIVERGING = LinearSegmentedColormap.from_list(
    "scarce_diverging", ["#184f95", "#6da7ec", WASH, "#ee9796", "#b83232"]
)

COLUMN_WIDTH = 6.5  # inches, the paper's text width


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 8.5,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,  # embed real text so the PDF stays searchable
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _save(fig, table: pd.DataFrame, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    table.to_csv(out / f"{name}.csv", index=False)
    plt.close(fig)
    return out / f"{name}.pdf"


def _history_axis(ax, histories) -> None:
    """History lengths span an order of magnitude; on a linear axis 60 and 120
    would sit on top of each other."""
    ax.set_xscale("log")
    ax.set_xticks(histories)
    ax.set_xticklabels([str(h) for h in histories])
    ax.minorticks_off()


def _usable(results: pd.DataFrame) -> pd.DataFrame:
    ok = results["mase"].notna()
    if "error" in results:
        ok &= results["error"].isna()
    return results.loc[ok]


# Figure 1: the crossover.


def crossover(
    results: pd.DataFrame,
    out: Path,
    n_series: int | None = None,
    missing_rate: float = 0.0,
) -> Path:
    """LightGBM minus each baseline, in MASE, against history length.

    The research question is where this difference crosses zero, so that is what
    is plotted, with series-clustered bootstrap intervals. One panel per baseline
    rather than one shared panel: a misspecified baseline (Croston on a smooth
    series) sits far from zero and would otherwise flatten every crossover that
    matters. Panels therefore have their own vertical scales, and each is read
    against its own zero line.
    """
    clean = _usable(results)
    clean = clean.loc[clean["missing_rate"] == missing_rate]
    n_series = n_series or int(clean["n_series"].max())
    clean = clean.loc[clean["n_series"] == n_series]

    baselines = [m for m in MODEL_ORDER if m in set(clean["model"]) and m != "lightgbm"]
    quadrants = [q for q in QUADRANT_ORDER if q in set(clean["quadrant"].dropna())]
    histories = sorted(clean["history_days"].unique())

    rows = []
    for quadrant in quadrants:
        for baseline in baselines:
            for history in histories:
                cell = clean.loc[
                    (clean["quadrant"] == quadrant) & (clean["history_days"] == history)
                ]
                point, lower, upper = bootstrap_difference(cell, "lightgbm", baseline)
                rows.append(
                    {
                        "quadrant": quadrant,
                        "baseline": baseline,
                        "history_days": history,
                        "difference": point,
                        "lower": lower,
                        "upper": upper,
                        "n_series": cell["series_id"].nunique(),
                    }
                )
    table = pd.DataFrame(rows)

    _style()
    fig, axes = plt.subplots(
        len(quadrants),
        len(baselines),
        figsize=(COLUMN_WIDTH, 0.5 + 1.8 * len(quadrants)),
        sharex=True,
        squeeze=False,
    )
    for i, quadrant in enumerate(quadrants):
        for j, baseline in enumerate(baselines):
            ax = axes[i][j]
            part = table.loc[
                (table["quadrant"] == quadrant) & (table["baseline"] == baseline)
            ].sort_values("history_days")
            color = MODEL_COLORS[baseline]

            ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
            ax.fill_between(
                part["history_days"], part["lower"], part["upper"],
                color=color, alpha=0.14, linewidth=0, zorder=2,
            )
            ax.plot(
                part["history_days"], part["difference"],
                color=color, linewidth=1.5, solid_capstyle="round",
                marker=MODEL_MARKERS[baseline], markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.9, zorder=3,
            )
            _history_axis(ax, histories)
            if i == 0:
                ax.set_title(f"vs {MODEL_NAMES[baseline]}", pad=4)
            if j == 0:
                ax.set_ylabel(quadrant.capitalize())

    fig.supxlabel("History length (days)", fontsize=8, color=INK_2)
    fig.supylabel(
        "LightGBM minus baseline (MASE)\nbelow zero: LightGBM better",
        fontsize=8, color=INK_2, linespacing=1.4,
    )
    fig.tight_layout()
    return _save(fig, table, out, "crossover")


# Figure 2: quadrant drift.


def drift(manifests: list[dict], out: Path, missing_rate: float = 0.0) -> Path:
    """Share of sampled series whose quadrant changes when classified from the
    window instead of from full history."""
    table = pd.DataFrame(
        [
            {
                "dataset": m["spec"]["dataset"],
                "history_days": m["spec"]["history_days"],
                "missing_rate": m["spec"]["missing_rate"],
                "drift": m["quadrant_drift_rate"],
            }
            for m in manifests
        ]
    )
    table = (
        table.loc[table["missing_rate"] == missing_rate]
        .groupby(["dataset", "history_days"], as_index=False)["drift"]
        .agg(drift="mean", lowest="min", highest="max", n_cells="size")
    )
    datasets = sorted(table["dataset"].unique())
    histories = sorted(table["history_days"].unique())
    top = max(0.1, float(table["highest"].max()) * 1.25)

    _style()
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(min(COLUMN_WIDTH, 2.3 * len(datasets) + 0.6), 1.9),
        sharey=True, squeeze=False,
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        part = table.loc[table["dataset"] == dataset].sort_values("history_days")
        ax.vlines(part["history_days"], part["lowest"], part["highest"], color=INK_2, linewidth=0.8)
        ax.plot(
            part["history_days"], part["drift"], color=INK_2, linewidth=1.5,
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.9,
        )
        _history_axis(ax, histories)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        # The axis tops out at 10% or more, so ticks land on whole percents.
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(dataset.upper() if len(dataset) <= 3 else dataset.capitalize(), pad=4)
    axes[0][0].set_ylabel("Series changing quadrant")
    fig.supxlabel("History length (days)", fontsize=8, color=INK_2)
    fig.tight_layout()
    return _save(fig, table, out, "drift")


# Figure 3: attribution stability.


def _regime_label(row) -> str:
    regime, reference = row["regime"], row["reference"]
    if isinstance(regime, tuple):
        rate, pattern = regime
        pattern = {"mcar": "MCAR"}.get(pattern, pattern)
        return f"{rate:.0%} {pattern} vs clean"
    return f"{regime} vs {reference} days"


def attribution(shap: pd.DataFrame, out: Path) -> Path:
    """Rank agreement of LightGBM attributions across regimes, read against the
    agreement between seeds of the same cell. A regime comparison only means
    something to the extent it falls outside that seed-noise band."""
    seeds = seed_stability(shap)
    regimes = pd.concat(
        [regime_stability(shap, "history_days"), regime_stability(shap, "missingness")],
        ignore_index=True,
    )
    regimes["comparison"] = regimes.apply(_regime_label, axis=1)
    multi = regimes["dataset"].nunique() > 1
    if multi:
        regimes["comparison"] = regimes["dataset"].str.upper() + ": " + regimes["comparison"]

    table = (
        regimes.groupby("comparison", sort=False)
        .agg(
            rho=("rho", "mean"), rho_low=("rho", "min"), rho_high=("rho", "max"),
            jaccard=("top_k_jaccard", "mean"),
            jaccard_low=("top_k_jaccard", "min"), jaccard_high=("top_k_jaccard", "max"),
            n_configs=("rho", "size"),
        )
        .reset_index()
    )
    floor = {
        "rho": (seeds["rho"].min(), seeds["rho"].max()),
        "jaccard": (seeds["top_k_jaccard"].min(), seeds["top_k_jaccard"].max()),
    }
    table["seed_floor_rho_low"], table["seed_floor_rho_high"] = floor["rho"]
    table["seed_floor_jaccard_low"], table["seed_floor_jaccard_high"] = floor["jaccard"]

    _style()
    y = np.arange(len(table))[::-1]
    fig, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 0.7 + 0.32 * len(table)), sharey=True
    )
    panels = [("rho", "Spearman rank correlation"), ("jaccard", "Top-10 overlap (Jaccard)")]
    for ax, (metric, label) in zip(axes, panels, strict=True):
        low, high = floor[metric]
        ax.axvspan(low, high, color=MUTED, alpha=0.18, linewidth=0, zorder=1)
        ax.hlines(
            y, table[f"{metric}_low"], table[f"{metric}_high"],
            color=MODEL_COLORS["lightgbm"], linewidth=1.0, zorder=2,
        )
        ax.plot(
            table[metric], y, linestyle="none", marker="o", markersize=5.5,
            color=MODEL_COLORS["lightgbm"], markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )
        ax.set_xlim(min(0, float(table[f"{metric}_low"].min()) - 0.05), 1.02)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.text(
            low - 0.015, y.max() + 0.62, "seed noise", ha="right", va="bottom",
            fontsize=7, color=INK_2,
        )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(table["comparison"])
    axes[0].set_ylim(-0.6, y.max() + 1.1)
    fig.tight_layout()
    return _save(fig, table, out, "attribution")


# Figure 4: pairwise significance.


def significance(daily: pd.DataFrame, out: Path, horizon: int = 28) -> Path:
    """Mean loss difference for every model pair at each history length, with a
    dot where the difference survived FDR correction in every cell pooled into
    that square. Unanimity is the most conservative pooling rule available; the
    exact share of significant cells is in the CSV."""
    tests = compare_grid(daily, horizon=horizon)
    models = [m for m in MODEL_ORDER if m in set(daily["model"])]
    histories = sorted(tests["history_days"].unique())

    rows = []
    for history in histories:
        part = tests.loc[tests["history_days"] == history]
        for (a, b), pair in part.groupby(["model_a", "model_b"]):
            difference = float(pair["mean_difference"].mean())
            share = float(pair["significant"].mean())
            rows.append({"history_days": history, "row": a, "column": b,
                         "difference": difference, "share_significant": share,
                         "n_cells": len(pair)})
            rows.append({"history_days": history, "row": b, "column": a,
                         "difference": -difference, "share_significant": share,
                         "n_cells": len(pair)})
    table = pd.DataFrame(rows)
    limit = float(table["difference"].abs().max()) or 1.0

    _style()
    size = len(models)
    fig, axes = plt.subplots(
        1, len(histories), figsize=(COLUMN_WIDTH, 0.9 + 0.36 * size), squeeze=False
    )
    mesh = None
    for ax, history in zip(axes[0], histories, strict=True):
        grid = np.full((size, size), np.nan)
        stars = []
        part = table.loc[table["history_days"] == history]
        for _, r in part.iterrows():
            i, j = models.index(r["row"]), models.index(r["column"])
            grid[i, j] = r["difference"]
            if r["share_significant"] == 1.0:
                stars.append((j + 0.5, i + 0.5))

        mesh = ax.pcolormesh(
            np.ma.masked_invalid(grid), cmap=DIVERGING, vmin=-limit, vmax=limit,
            edgecolors="white", linewidth=1.5,
        )
        for x, yy in stars:
            ax.plot(x, yy, marker="o", markersize=2.6, color=INK, markeredgewidth=0)
        ax.set_xticks(np.arange(size) + 0.5)
        ax.set_yticks(np.arange(size) + 0.5)
        ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha="right")
        ax.set_yticklabels([MODEL_NAMES[m] for m in models])
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{history} days of history", pad=4)
        if ax is not axes[0][0]:
            ax.set_yticklabels([])

    bar = fig.colorbar(mesh, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    bar.set_label("Row minus column, mean scaled error\n(blue: row model better)", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=AXIS, labelcolor=INK_2, length=2)
    return _save(fig, table, out, "significance")


def make_all(results_dir: Path, out: Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(out) if out else results_dir / "figures"
    made = [
        crossover(collect(results_dir), out),
        drift(load_manifests(results_dir), out),
        significance(collect_daily(results_dir), out),
    ]
    try:
        made.append(attribution(collect_shap(results_dir), out))
    except FileNotFoundError:
        print("no attribution files; skipping the attribution figure")
    return made
