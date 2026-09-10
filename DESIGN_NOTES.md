# Design notes

A running record of the decisions behind this benchmark, in plain language, so
that each one can be defended if someone asks about it directly.

## Stage 1: loaders

### Zero is not the same as missing

The decision with the longest reach is how each loader represents the difference
between "this product sold nothing today" and "this product was not on the shelf
today". Both appear as a zero in the raw files. They mean opposite things.

An item that launches partway through the M5 window has a long run of leading
zeros that are not demand observations at all. If those are treated as real
zeros, the series looks far more intermittent than it is: the average interval
between demands is inflated, the coefficient of variation is distorted, and the
series drifts into the wrong Syntetos-Boylan quadrant. Since the whole benchmark
stratifies on exactly those statistics, that error would propagate into every
result rather than staying local.

So `y` is `NaN` for unobservable and `0.0` for observed-zero, and each loader
trims each series back to its first strictly positive observation. Series that
are never positive are dropped, since they carry no demand signal.

This is a deliberate divergence from a lot of published M5 code, which zero-fills
and moves on. The tradeoff is that headline numbers here will not line up exactly
with M5 competition leaderboards. That is the right trade for this question: the
benchmark is about the intermittent small-data regime, and this distinction is
load-bearing in precisely that regime. Zero-filling is still reachable, since the
information is preserved rather than destroyed, but it has to be asked for.

It also keeps native gaps distinguishable from the gaps the scarcity protocol
injects in stage 2. Without that separation, a 15 percent injected-missingness
cell would be measuring an unknown mixture of injected and pre-existing absence.

### One panel shape for everything

All three loaders return the same long frame: `series_id`, `ds`, `y`, with static
attributes in a separate table indexed by `series_id`.

Long format rather than a series-by-time matrix, because series here have
genuinely different lengths and a matrix would have to pad them, which
reintroduces exactly the zero-versus-missing ambiguity described above. It is
also the shape `neuralforecast` and `darts` already expect, so the deep models
need no translation layer.

Static attributes live apart from the panel because they do not vary over time
and repeating them across every row of a 59-million-row frame is wasteful. When
a panel is aggregated above item level, attributes that are no longer constant
within a series are set to null rather than silently taking a first value.

### Dataset-specific handling

**M5** arrives as a complete rectangle with explicit zeros, so no reindexing is
needed. Bottom level is 3,049 items across 10 stores, so 30,490 series. (Earlier
I quoted 42,840, which is the count across all twelve hierarchy levels, not the
bottom level.) Bottom level is the default because that is where intermittency
actually lives and it is the honest analogue of a small retailer's SKUs. Higher
levels stay available since aggregation is one of the things that makes a series
easy to forecast, and showing that is useful.

**Favorita** omits rows entirely for no-sale days, so absence is ambiguous
between a closed store and a genuine zero. We reindex to a daily grid and leave
those `NaN` rather than guessing, which is the conservative choice: a wrong
zero is a fabricated observation, a `NaN` is an honest one. Negative `unit_sales`
values are returns and are kept rather than clipped, because clipping would
distort the same intermittency statistics we stratify on. The default loads from
2015 onward, since the full file is around 125 million rows and the protocol
never asks for more than 730 days.

**CTA** rail entries is the non-retail control. A single national ridership
aggregate would have been easier to fetch but useless here, since the protocol
samples 10 to 1,000 series and there would be nothing to sample. CTA gives about
145 stations of daily history since 2001 over an open endpoint with no auth, plus
a real structural break in 2020 that usefully punishes models assuming a stable
level. No leading-zero mask is applied, because a station that opens mid-panel
simply has no rows before it opened.

### Reproducibility

Two hashes per dataset, not one. The raw hash catches the upstream file changing
underneath us. The normalized hash catches our own loader code changing the panel
without anyone noticing. Results are only comparable when both still match, and
overriding a mismatch takes an explicit flag.

### What was deliberately not built

No loader base class, no registration decorator, no config-driven source plugin
system. There are three datasets and a dictionary of three functions covers it.
Also no imputation, no outlier handling and no calendar feature engineering in
the loaders: those are modeling choices, and burying them in the data layer would
apply them unequally across models and quietly compromise the comparison.
