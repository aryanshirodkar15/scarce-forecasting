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

## Stage 2: the scarcity sampling protocol

### Where the intermittency statistics are computed

ADI and CV2 depend on how long you look. A series with four sales in sixty days
gives a CV2 estimated from four numbers, which is noise wearing a quadrant label.
Since the protocol deliberately varies history from 60 to 730 days, quadrant
membership would move around as a side effect of the very axis being studied.

So the classification is done twice. The full-history label is fixed per series
and drives both the stratified sampling and the analysis grouping, which keeps
history length and quadrant membership as independent axes. The within-window
statistics are computed as well and recorded per cell, and the disagreement
between the two is reported as `quadrant_drift_rate`.

That second number turns a nuisance into a result. On synthetic series with
cleanly separated quadrants it already shows the expected shape: zero drift at
730 and 365 days, 2.5 percent at 120, and 7.5 to 10 percent at 60. Real catalogues
are far less separable, so expect substantially more. The practical reading is
that a practitioner classifying their own catalogue off six months of history
gets the quadrant wrong a measurable fraction of the time, and picks the wrong
forecasting method as a result.

The label is never fed to a model. It only groups results, so it leaks nothing
into the forecasts even though it uses information from outside the window.

### Unobserved periods are dropped, not spanned

ADI is periods divided by demand occurrences, counted over observed periods only.

The alternative, counting an unobserved period as part of the interval, has a
failure mode that would have quietly wrecked the grid. Under missingness at rate
p, the demand count falls to (1-p)n while the period count stays at N, so ADI
inflates by 1/(1-p). At the 30 percent cell that is a 43 percent inflation, more
than enough to push series across the 1.32 cutoff. The missingness axis would
have silently become an intermittency axis and the two effects would be
inseparable.

Dropping unobserved periods instead takes both counts down proportionally, so
ADI is unchanged in expectation. The measured drift rate confirms it: 0.075 at
zero missingness and 0.075 at 30 percent MCAR on the same series.

Only strictly positive values count as demand occurrences, because Favorita
encodes returns as negative sales and a return is not a demand.

### Two seeds, so that comparisons are paired

Series selection is keyed only on the sampling frame: dataset, series count,
seed, frame length, coverage threshold and anchor date. It deliberately ignores
history length and missingness. The result is that the same series appear in
every cell along those two axes, so comparing 60 days against 730 days is a
paired comparison on identical series rather than two independent draws. That is
a substantial gain in power for free, and it removes series composition as an
explanation for any difference found.

Missingness injection is keyed on the entire spec, so each cell still gets its
own independent gaps.

### Eligibility is judged over the frame, not the history

A series qualifies if it has at least 80 percent observed coverage over the
730-day frame, regardless of how much history the cell actually hands to the
model.

Judging eligibility against `history_days` instead would have meant the 60-day
cells drew from a much larger pool that included short-lived products, while the
730-day cells drew only from long-lived ones. History length would then be
confounded with series longevity, and any crossover found could be explained
away by the population changing underneath the comparison. Fixing the frame
costs some realism, since a real new business genuinely has only short-lived
series, but it buys a clean attribution, which is what the research question
needs.

### Missingness shapes

MCAR masks observed cells independently. Burst places contiguous runs with
geometric lengths averaging seven days, which is what a real outage looks like:
a till breaking, a store closing, an export job failing for a week. Both are
trimmed to hit the requested rate exactly, so the two patterns are compared at
identical volumes of absence and differ only in shape.

Injection only ever masks cells that were observed, so the requested rate means
the same thing regardless of how much native absence a dataset already carries.
Native, injected and final absence rates are all recorded separately.

Gaps are injected across the whole window including the part that later becomes
the evaluation region. That is deliberate, since real evaluation periods have
gaps too, but it puts an obligation on the metrics stage: NaN actuals must be
excluded from every metric rather than treated as zeros.

### Shortfall is recorded, not silently filled

When a quadrant has fewer eligible series than the equal allocation asks for, the
sampler takes what exists and records the gap. It does not top up from other
quadrants, because that would silently unbalance the strata that the whole design
depends on. Cells that cannot be filled are visible in the manifest instead of
being quietly smaller than they claim.

## Stage 2 addendum: what real data changed

### Two allocation modes, because the control dataset forced it

CTA classifies as 145 smooth, 1 erratic, 1 intermittent and 0 lumpy out of 148
stations. That is not a defect in the data or the classifier. Rail ridership is a
daily continuous process and simply has no intermittency in it.

The consequence is that equal allocation degenerates there: asking for 20 series
returns 6, because three of the four strata are empty. So allocation is now a
switch. M5 and Favorita use equal allocation, which maximizes power per quadrant
and is the point of stratifying. CTA uses proportional, which for it means
effectively all-smooth.

This changes CTA's role in the benchmark, for the better. It is not a fourth
source of stratified evidence, it is the contrast case: what the crossover point
looks like when there is no intermittency present at all. If deep models hold
their advantage further down the data-volume curve on CTA than on M5, that is
direct evidence that intermittency rather than sheer volume is what breaks them.

Proportional allocation uses largest remainder so the parts sum to exactly the
requested count instead of drifting through rounding.

### Quadrant drift is driven by intermittency, not by short windows alone

On CTA the drift rate is 0.000 at every history length including 60 days. Smooth
series are robustly classified from very little data. The drift seen on synthetic
intermittent and lumpy series is therefore not a generic small-sample artifact;
it is specifically the instability of CV2 estimated from a handful of demand
occurrences. That sharpens the claim: it is not that short histories mislabel
everything, it is that short histories mislabel exactly the series where the
choice of forecasting method matters most.

### A defect only real data could show

The CTA feed reports 1,236 rows as duplicate station-days, all inside a 41-day
window in summer 2011, each giving two different ride counts for the same day.
Around 0.09 percent of the panel, median disagreement 0.3 percent, maximum 20
percent. No basis exists for preferring either figure and averaging them would
invent an observation, so those days are marked unobserved. `to_daily_grid` now
rejects unresolved duplicates with an actionable message rather than surfacing a
cryptic pandas reindexing error.
