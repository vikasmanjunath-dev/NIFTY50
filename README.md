# Nifty 50 Stock Analyser (STM)

A research codebase for building a **point-in-time-correct, leakage-free** machine-learning
system that forecasts short-horizon direction for Nifty 50 constituents — and, on top of
that, a **retrieval-augmented, LLM-explained intelligence layer** over the same universe's
corporate announcements and filings.

This README is the technical map of the project: what it does, *why* it is built this way,
and how every major concept works, from first principles. It is a companion to, not a
replacement for, **[TRACKER.md](TRACKER.md)** (the line-item build log — every task, every
real bug found and fixed, every honestly-reported result) and **[TRACKER.xlsx](TRACKER.xlsx)**
(the same tracker as a structured, filterable spreadsheet).

> **Current state (2026-11-24):** the repository was found reduced to a partial snapshot and has been **rebuilt and
> re-verified on real data** (see [TRACKER.md](TRACKER.md), Session 7). Present and tested (**361 tests passing**): the base
> layer, all Phase 2 layers (macro, regime, cross-sectional, delivery %, fundamentals incl. insurance PDFs, FII/DII,
> calibration, selective prediction, ensemble, costs, drift), Phase 3 (RAG + LLM + guardrails) and the new point-in-time
> universe. Phase 4 is built (below). Bug/test counts and the *historical* Phase 1/2
> numbers quoted further down predate the rebuild; the current, re-run results are below and in `reports/`.
>
> **Current headline result (point-in-time universe, 101 scale-free features):** accuracy 54.2% (= the up-day base rate; balanced
> accuracy 0.507), ROC-AUC 0.520, IC +0.048 ± 0.082 — **no detectable directional edge** (a faint cross-sectional *ranking* signal exists but does not survive costs — see Results). Cost-stressed Sharpe 0.51 at 20 bps, *below* an
> equal-weight buy-and-hold of the same universe (0.89). All model inputs are scale-free (no raw price/volume/macro/flow/revenue
> levels); a paired old-vs-new comparison on identical folds found no measurable loss (AUC -0.004 to -0.008, IC -0.009 to -0.013,
> within noise and the pre-set margin). Adding fundamentals + FII/DII earlier: paired-ablation IC change -0.0086, sign-flipping.

## Survivorship-bias-free universe (new, 2026-11-24)

Backtesting only today's Nifty 50 projected backward is a documented bias (Blueprint §3.4). `nifty_analyser/data/index_membership.py`
reconstructs the *real* membership on any date from 2021-09-20 (sourced, dated replacement history; announcement dates kept as
`available_time`), `data/instrument_master.py` holds the 60-name pool (59 ingestible; HDFC's data no longer exists on Yahoo), and
`scripts/run_survivorship_comparison.py` runs identical features, labels, date folds and models on both universes.

| Measured on real data (2021-09 → 2026-11) | Result |
|---|---|
| Phantom rows (stock before joining the index) | 5,756 rows, 9 names, **+32.4%/yr** forward return |
| Omitted rows (members that later left) | 5,785 rows, 9 names, **+3.4%/yr** |
| Equal-weight universe, biased vs point-in-time | **+11.8% vs +9.2%/yr** → premium **+2.6%/yr, t = 3.13** |
| Model skill (LightGBM / Elastic-Net, 4 folds) | accuracy 50.3–51.1%, AUC 0.507–0.510 in **both** universes — no edge to inflate |

Run: `python3 scripts/backfill_bars.py && python3 scripts/run_survivorship_comparison.py` → `reports/survivorship_comparison.md`.

---

## Table of contents

1. [What this project is trying to do](#what-this-project-is-trying-to-do)
2. [Status at a glance](#status-at-a-glance)
3. [Architecture — how the whole system fits together](#architecture--how-the-whole-system-fits-together)
4. [The core discipline: point-in-time correctness](#the-core-discipline-point-in-time-correctness)
5. [Data layer — sources, ingestion, quality](#data-layer--sources-ingestion-quality)
6. [Feature engineering — every family, explained](#feature-engineering--every-family-explained)
7. [Labelling — turning prices into targets](#labelling--turning-prices-into-targets)
8. [Validation — purged & embargoed walk-forward CV](#validation--purged--embargoed-walk-forward-cv)
9. [Modelling — baseline, calibration, ensemble, selective prediction](#modelling--baseline-calibration-ensemble-selective-prediction)
10. [Regime detection](#regime-detection)
11. [Backtesting with realistic costs](#backtesting-with-realistic-costs)
12. [Monitoring — feature and data drift](#monitoring--feature-and-data-drift)
13. [The RAG + LLM intelligence layer (Phase 3)](#the-rag--llm-intelligence-layer-phase-3)
14. [Repository layout](#repository-layout)
15. [Setup](#setup)
16. [Running things](#running-things)
17. [Results, honestly reported](#results-honestly-reported)
18. [Design principles this codebase holds itself to](#design-principles-this-codebase-holds-itself-to)
19. [What's next — Phase 4](#whats-next--phase-4)
20. [Keeping these docs current](#keeping-these-docs-current)

---

## What this project is trying to do

Most retail/hobbyist "stock prediction" projects fail silently in one of two ways: they leak
information from the future into training data (so backtests look great and live performance
is worthless), or they report a backtest's *gross* return while ignoring the transaction costs
that would have eaten it alive. This project is built specifically to avoid both failure modes,
even at the cost of reporting unglamorous, near-zero-edge results — which is exactly what it
has found and reported honestly at every phase so far.

The build follows a phased roadmap from a private research blueprint
([Nifty50_Complete_Blueprint_From_Scratch_to_Advanced.pdf](Nifty50_Complete_Blueprint_From_Scratch_to_Advanced.pdf),
Part 6's phase plan). Four phases, each with a distinct goal:

| Phase | Goal |
|---|---|
| **0** | Project scaffolding, environment, tooling |
| **1** | A solid, honest baseline: ingest real data, engineer simple features, label outcomes, validate without leakage, train two simple models, and lock the resulting (unimpressive, expected) metrics on disk *before* any tuning is allowed to touch them |
| **2** | Enrichment: more technical indicators, fundamentals, macro data, FII/DII flows, cross-sectional ranking, probability calibration, a proper stacked ensemble, regime awareness, realistic transaction costs, and drift monitoring |
| **3** | A retrieval-augmented generation (RAG) layer over corporate announcements/filings, fused into the ML feature matrix, with mechanical hallucination guardrails around the LLM explanation layer, and an honest ablation test of whether any of it actually helps |
| **4** | Production operations: an API contract, a monitoring dashboard, an audit log, a kill switch (built, see below) |

---

## Status at a glance

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup & tooling | ✅ done |
| 1 | Baseline: ingestion, labels, technical features, walk-forward validation | ✅ complete |
| 2 | Enrichment: fundamentals (incl. insurance PDFs), macro, FII/DII flows, cross-sectional features, calibration, ensemble, regime, costs, drift | ✅ 10/10 items, all data-coverage gaps closed |
| 3 | RAG + LLM intelligence layer over NSE announcements | ✅ 9/9 items complete |
| 4 | Operations: `/v3/analyse` API, monitoring dashboard, hash-chained audit log + replay, kill switch | ✅ 4/4 built and verified (`nifty_analyser/ops/`) |

**361/361 tests passing**. See [TRACKER.md](TRACKER.md) for the full
itemised checklist, every known simplification, every real bug found and fixed (27 documented
so far), and the complete, dated work log.

---

## Architecture — how the whole system fits together

There are two pipelines in this repository that share a data layer: a **predictive ML
pipeline** (Phases 1–2) and a **RAG/LLM intelligence pipeline** (Phase 3) that feeds one
extra family of features into the same pipeline via late fusion.

```
                              ┌─────────────────────────────────────────┐
                              │              RAW DATA SOURCES             │
                              │  yfinance OHLCV · NSE bhavcopy (delivery) │
                              │  NSE XBRL filings · IRDAI PDF filings     │
                              │  India VIX/USD-INR/Brent · NSDL FII/DII   │
                              │  NSE corporate announcements              │
                              └───────────────┬───────────────────────────┘
                                              │  ingest + validate (PASS/WARN/FAIL)
                                              ▼
                              ┌─────────────────────────────────────────┐
                              │   POINT-IN-TIME DATA CONTRACT (schemas)   │
                              │  every record carries event_time AND      │
                              │  available_time — "what did we know,      │
                              │  and when did we know it?"                │
                              └───────────────┬───────────────────────────┘
                                              │  data/store.py: as_of() / assert_no_lookahead()
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                 FEATURE ENGINEERING                      │
                    │  price/volume · technical (20+ indicators) · delivery %  │
                    │  fundamentals (XBRL + insurance PDFs) · macro · FII/DII  │
                    │  cross-sectional ranking · regime · RAG event features   │
                    └───────────────┬─────────────────────┬────────────────────┘
                                    │                       │
                                    ▼                       │
                    ┌───────────────────────────┐          │
                    │   LABELS (labels/targets)  │          │
                    │ forward return · direction │          │
                    │ · triple-barrier           │          │
                    └───────────────┬────────────┘          │
                                    ▼                       │
                    ┌───────────────────────────────────────┘
                    │        PURGED & EMBARGOED WALK-FORWARD CV
                    │        (validation/walk_forward.py)
                    ▼
        ┌───────────────────────────────────────────────────────┐
        │        MODELLING: Logistic / LightGBM / CatBoost         │
        │        → out-of-fold meta-model → calibration             │
        │        (models/baseline, ensemble, calibration, selective)│
        └───────────────┬─────────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────────────────────────────┐
        │   EVALUATION: metrics (IC/AUC/log loss) + cost-stressed   │
        │   backtest (eval/) + regime-segmented results + drift    │
        │   monitoring (monitoring/drift.py)                        │
        └───────────────────────────────────────────────────────┘

  ── separate but connected pipeline ──────────────────────────────────────

┌───────────────────────────────────────────────────────────────────────┐
│                     RAG + LLM INTELLIGENCE LAYER (Phase 3)               │
│                                                                           │
│  NSE announcements ──ingest──▶ chunk ──▶ embed (local Ollama) ──▶ index  │
│                                                                     │     │
│  query + as_of ──▶ hybrid retrieve (BM25 + dense + RRF) ◀──────────┘     │
│         │            with a point-in-time FIREWALL                       │
│         ▼                                                                │
│   evidence pack (claim-level, hashed, provenance-tagged)                  │
│         │                                                                 │
│         ├──▶ structured extraction (materiality, event counts) ──┐        │
│         │                                                        │        │
│         └──▶ local LLM (Ollama) ──▶ mechanical guardrails         │        │
│              (evidence-only prompting)   (abstention, citation,   │        │
│                                            numeric grounding,      │        │
│                                            prompt-injection scan)  │        │
│                                                                    ▼        │
│                                          late-fused into the main feature  │
│                                          matrix (features/events.py) ──────┼──▶ (feeds
└───────────────────────────────────────────────────────────────────────┘      back into
                                                                                  the ML
                                                                                  pipeline)
```

The two pipelines are deliberately kept separable: the RAG layer's output is just another
feature source that merges in via the same point-in-time-safe pattern as delivery %, macro
data, or FII/DII flow — it does not replace or bypass the core ML pipeline's validation
discipline. And per the blueprint's own operating rule, a feature only earns a place in the
live signal if a **paired ablation test** shows it actually helps out-of-sample (see
[Results](#results-honestly-reported) — so far, it doesn't, and is honestly reported as such).

Every layer in the diagram is both *built* and *actually called* by `scripts/run_phase2_pipeline.py`
— fundamentals and FII/DII flow were the last two to be wired in (2026-09-22); see
[Feature engineering](#feature-engineering--every-family-explained) for what that changed and
[Results](#results-honestly-reported) for the re-run numbers.

---

## The core discipline: point-in-time correctness

This is the single idea the entire codebase is organized around, so it is worth explaining
properly rather than just naming it.

**The problem it solves.** A naive backtest asks "given this row of features, what happened
next?" using a dataset built *after the fact*, with hindsight. But hindsight leaks in subtle
ways: a company's Q2 earnings are dated to the quarter, but weren't *known* until the filing
was actually published weeks later; a stock's index membership at any past date wasn't
necessarily its membership today; even a same-day closing price wasn't knowable until the
market actually closed. A model trained on data that "knows the future" relative to its own
decision point will look spectacular in-sample and be worthless live — this is called
**look-ahead bias**, and it is the single most common way research-grade trading systems lie
to their own authors.

**The fix.** Every single record in this system — a raw bar, a fundamentals figure, a
delivery-percentage row, an FII/DII flow figure, a corporate announcement, a derived feature —
carries **two** timestamps, not one:

- **`event_time`** — when the thing actually happened in the world (e.g., the trading day a
  bar covers, the quarter a results filing reports on).
- **`available_time`** — the earliest moment a real system could legitimately have known that
  value. For an OHLCV bar this is NSE's 15:30 IST close; for an XBRL filing it's the filing's
  own exact `broadCastDate`; for an NSE announcement it's the exchange's own
  to-the-second `exchdisstime` (the most precise timestamp of any source in this project).

This is enforced as a **typed, self-validating data contract** (`nifty_analyser/data/schemas.py`,
Pydantic models `MarketBar` and `FeatureRecord`), which mechanically *rejects* a record on
construction if `available_time < event_time` — a value literally cannot be knowable before it
exists. Every ingestion module also tags each record `PASS` / `WARN` / `FAIL` for data quality,
and a `FAIL` record must carry a `missing_reason` — nothing gets silently dropped or silently
trusted.

On top of the contract sits a small but load-bearing **feature store** (`nifty_analyser/data/store.py`):

```python
as_of(df, decision_time)          # filter any frame down to rows knowable at decision_time
latest_as_of(df, decision_time)   # per instrument, the most recent knowable row (live-inference pattern)
assert_no_lookahead(df, decision_time)  # hard-fail (raises LookaheadError) if anything leaked
```

`assert_no_lookahead` is deliberately not a warning. It is called as the *last* step before
handing any frame to a model, a feature merge, or a retrieval query — across every single data
layer in the project, all the way through the RAG pipeline's own retrieval firewall (see
[§13](#the-rag--llm-intelligence-layer-phase-3)). Every feature and label module additionally has a
**mechanical proof**, not just an assertion, that it respects this: the "truncate-the-future"
test pattern used throughout `tests/` builds a feature matrix, then rebuilds it after deleting
the most recent 30 rows, and asserts every overlapping value is bit-for-bit identical — if a
feature secretly used a future value, truncating the future would change a past row, and the
test would catch it.

This same discipline is why several real, non-hypothetical bugs were caught during
development rather than shipped silently — e.g., NSE's delivery-report archive returning
HTTP 200 with a *stale* previous-day file instead of 404ing for a non-trading date (caught by
cross-checking the file's own embedded date column), or a rolling 20-day window silently
propagating `NaN` across its entire span from a single bad bar (caught by reading real output,
not just a green test suite). Every one of these is documented in full, with its exact
mechanism, in [TRACKER.md's Known Simplifications section](TRACKER.md#known-simplifications--deviations-kept-honest-on-purpose).

---

## Data layer — sources, ingestion, quality

Every data source below is real (no synthetic data anywhere in this project) and follows the
same shape: an ingestion module fetches and validates, tags each row `PASS`/`WARN`/`FAIL`,
attaches `event_time`/`available_time`, and caches to Parquet under `data/raw/`.

| Source | What it provides | Module | `available_time` field | Real coverage |
|---|---|---|---|---|
| yfinance | Daily OHLCV bars, 48/50 Nifty constituents | `data/ingest.py` | Bar date @ 15:30 IST (NSE close) | Full history to present |
| NSE `sec_bhavdata_full` bhavcopy | Delivery quantity/%, whole-exchange bulk report | `data/bhavcopy.py` | Bar date + conservative same-day publish estimate | 1234/1241 dates, 100% PASS |
| NSE `corporates-financial-results` + raw XBRL | Revenue, profit, EPS, margins, YoY growth | `data/fundamentals.py` | Exact filing `broadCastDate` | 46/48 instruments, 1128 filings, 99.6% PASS |
| IRDAI-format PDF filings (via `pdfplumber`) | Same fundamentals fields for insurers | `data/insurance_fundamentals.py` | Exact filing broadcast date | SBILIFE + HDFCLIFE, 24/24 filings, 100% PASS — closes the 48/48 gap |
| India VIX, USD-INR, Brent crude | Macro time series | `data/macro.py` | Series-native publish time | 1,229 / 1,300 / 1,258 rows, 100% PASS |
| NSDL FPI/DII Monitor archive | Daily FII & DII net equity flows | `data/fii_dii.py` | Depository-confirmed report date | FII: 1,379/1,379 days (2021–2026); DII: 76/76 days (2026-06 onward — genuine source-side gap) |
| NSE Nifty 50 index (`^NSEI`) | Index level, for cross-sectional/regime features | `data/ingest.py` (same adapter) | Bar date @ 15:30 IST | Full history |
| NSE `corporate-announcements` (whole-exchange) | Filings, board outcomes, disclosures, free text | `documents/ingest.py` | Exact `exchdisstime` (to the second) | 43,623 announcements, 48/48 instruments, 100% PASS, 2021–2026 |

**Why this matters conceptually:** a naive project would treat "data I have" as "data that
existed." This project treats them as two different questions, and every source above was
*investigated* (not assumed) to find its real availability semantics — several times finding
that the obvious source doesn't actually support what's needed (e.g., NSE's own live FII/DII
page exposes only the latest trading day; NSDL's archive was found instead, and even then
DII's real history turned out to start only in June 2026 — reported as an honest asymmetry,
not silently backfilled with an estimate).

---

## Feature engineering — every family, explained

All feature functions operate strictly on data up to and including row `t` (no look-ahead;
see above), and are computed per-instrument (a rolling window never mixes one stock's history
with another's). Every family below is both built, tested, **and actually called** by
`scripts/run_phase2_pipeline.py` — fundamentals and FII/DII flow were the two most recently
wired in (2026-09-22), which grew the combined feature matrix from 98 to 110 columns and
surfaced one more real bug on the very first live run: two instruments' YoY growth ratios
divided by a real zero value, producing `inf` and crashing the Elastic-Net model's imputer —
fixed with the same zero-guard `profit_margin` already used (see
[TRACKER.md](TRACKER.md#known-simplifications--deviations-kept-honest-on-purpose), Known
Simplification #27). The re-run result is in [Results](#results-honestly-reported).

**Price & volatility** (`features/price_volume.py`)
Multi-horizon log returns (1/2/3/5/10/20/60 day), overnight vs. intraday return split, gap %,
high-low range %, realised volatility over multiple windows, and ATR(14) (Average True Range —
a volatility measure that accounts for gaps between days, not just the high-low range).

**Technical indicators** (`features/technical.py`) — 20+ indicators, all hand-implemented
(not pulled from a library) specifically so every formula is auditable line-by-line:
- *Trend*: SMA/EMA across six windows, MACD(12,26,9), price/SMA ratios, a golden-cross flag.
- *Momentum*: RSI(14) + its slope, stochastic %K/%D, Rate of Change (10/12/20-day).
- *Volume*: On-Balance Volume + slope, Chaikin Money Flow, Accumulation/Distribution line.
- *Volatility/bands*: Bollinger Bands (20, 2σ), Keltner Channels, Donchian Channels.
- *Trend strength/exhaustion*: Aroon, CCI (Commodity Channel Index), TSI (True Strength
  Index), Williams %R.

**Delivery %** (`features/delivery.py`) — the fraction of traded volume that resulted in
actual share delivery (vs. intraday squaring-off) — an India-specific signal not available
from plain OHLCV, sourced from NSE's own bulk bhavcopy report.

**Fundamentals** (`features/fundamentals.py`) — Revenue, Profit, EPS, margins, YoY growth,
merged onto the daily bar frame via `pandas.merge_asof(direction="backward")`, which by
construction can only carry a value *forward* from its last known filing — never backward
from a future one. A filing that's 18 months stale correctly stays pinned at its last known
value rather than being silently refreshed. Loaded via `data/store.py: load_combined_fundamentals()`,
which concatenates the general XBRL backfill with the insurance-sector PDF backfill (disjoint
instruments by construction) before the ratios and merge run.

**Macro** (`features/macro_features.py`) — level, 1-day return, and 252-day rolling percentile
for India VIX / USD-INR / Brent crude, broadcast onto every instrument (macro conditions are
market-wide, not stock-specific) with a "max of both series' `available_time`s" merge rule.

**FII/DII flow** (`features/fii_dii_features.py`) — daily net equity flow from Foreign and
Domestic Institutional Investors, plus 5-day/20-day rolling sums as trend signals, on the same
broadcast-merge pattern as macro. Loaded via `data/store.py: load_all_fii_dii_flows()`, which
reads FII and DII from two separate cached files (their real coverage windows differ sharply —
see the data-layer table above). Because DII's real history only starts June 2026, its columns
are entirely `NaN` in most of the walk-forward folds' training windows — a direct, expected
consequence of that coverage gap, not a bug, and `sklearn`'s own imputer warns about it rather
than silently guessing a value.

**Cross-sectional** (`features/cross_sectional.py`) — relative strength vs. the real Nifty 50
index, an equal-weighted sector-relative return proxy, rolling beta/correlation to the index,
and same-day percentile ranks across the universe. These answer "how is this stock doing
*relative to its peers today*", a signal family a single-stock time series can never see on
its own.

**Regime** (`features/regime.py`) — see [Regime detection](#regime-detection) below.

**RAG-derived event features** (`features/events.py`) — daily announcement count, high/medium
materiality counts, and "days since last high-materiality event" (a novelty signal), fused
in from the Phase-3 document pipeline. See
[§13](#the-rag--llm-intelligence-layer-phase-3).

---

## Labelling — turning prices into targets

`nifty_analyser/labels/targets.py` implements three label families (Blueprint §3.3's
recommended multi-task design), and every one of them carries an **`outcome_available_time`**
column — the moment the label's outcome is actually resolved and knowable, distinct from the
decision timestamp at row `t`. This field is what makes leakage-free validation possible
(see next section) — without it, "purging" overlapping labels would be guesswork.

- **Forward return** (`add_forward_return`) — a regression target: the log return from `t` to
  `t + horizon`.
- **Direction label** (`add_direction_label`) — Up / Down / Flat classification on the same
  horizon, with a configurable flat-threshold band.
- **Triple-barrier labelling** (`add_triple_barrier_label`) — an event-based label (from
  López de Prado's *Advances in Financial Machine Learning*): from each row `t`, walk forward
  up to `max_holding` bars and label whichever of three barriers is touched first — an
  **upper** return barrier (+1), a **lower** return barrier (−1), or the **vertical** (time)
  barrier (0, neither touched in time). This produces a label that reflects how a real
  position would actually have been exited, not an arbitrary fixed-horizon snapshot, and
  records the realised holding period alongside it.

---

## Validation — purged & embargoed walk-forward CV

Plain k-fold cross-validation is wrong for time series with overlapping labels: if two
adjacent rows' triple-barrier labels both resolve using overlapping future windows, a random
train/test split will leak one label's resolution period into the other's training data. This
project implements the correct alternative end-to-end (`nifty_analyser/validation/walk_forward.py`):

```
 TRAIN ──────────────▶ | PURGE | EMBARGO | ─── VALIDATE ─── TEST      (then slide forward)
```

- **Expanding-window walk-forward**: candidate training data for a fold is *everything*
  strictly before that fold's validation block — never a random split.
- **Purge**: any candidate training row whose *label outcome* (`outcome_available_time`)
  falls on or after the validation block's start is dropped — its label was determined using
  information from inside or after the validation window.
- **Embargo**: an additional small buffer (default 5 days) immediately before the validation
  start is also dropped, guarding against serial correlation in engineered features (not just
  labels) bleeding across the boundary.

A companion function, `assert_folds_are_leakage_free`, hard-fails if any fold violates this —
proven in tests by deliberately smuggling a validation-window row into training and confirming
it's caught.

The same primitive is reused **recursively** for two more purposes that need "safely
out-of-fold" data: `carve_calibration_slice` splits a fold's training window again into a
chronologically-earlier `fit` slice and a later `calibration` slice (needed so a calibrator
never sees data the base model trained on — see next section), and the champion/challenger
ensemble applies it *twice* to get three mutually-separated slices (base model fit → meta-model
fit → calibration), each proven leakage-free against every later stage by its own dedicated test.

---

## Modelling — baseline, calibration, ensemble, selective prediction

**Baseline** (`models/baseline.py`) — Logistic Regression / Elastic Net (a simple, well
understood benchmark) and LightGBM (gradient-boosted trees), both run with **fixed
hyperparameters and no search** in Phase 1 by explicit design: tuning before an honest,
untuned baseline is locked on disk introduces selection bias that has to be tracked
separately (§3.1's own warning) — so Phase 1 deliberately doesn't tune anything.

**Champion/challenger ensemble** (`models/ensemble.py`) — three structurally different base
models (LightGBM, CatBoost, Elastic Net) feed a Logistic meta-model that learns how to blend
their **out-of-fold** predictions, which then get calibrated. "Structurally different" matters:
a tree ensemble and a linear model make different kinds of mistakes, so blending them
out-of-sample can genuinely add information a single model type can't — but only if the
stacking is done correctly, which is why the recursive purge/embargo slicing above exists.

**Calibration** (`models/calibration.py`) — a model can be *directionally* useful (its ranking
of "more likely up" vs. "less likely up" is informative) while being badly *calibrated* (its
raw probability of 0.9 doesn't actually mean 90% of such predictions come true). Two standard
methods are implemented: **Platt/sigmoid scaling** (fits a 1-D logistic curve from raw score to
calibrated probability — robust with small calibration sets) and **isotonic regression** (a
non-parametric monotonic step function — more flexible, needs more data). Both are fit *only*
on the purged/embargoed calibration slice, never on data the base model trained on. This layer
also produces Expected Calibration Error and a reliability table for diagnosing miscalibration
directly.

A real bug surfaced and was fixed here: isotonic calibration, when its calibration slice's
tail happened to be one-class-only, produced a monotonic "shelf" of *exactly* probability 1.0
for a whole score region — on real data this meant 703 predictions at literal 100% confidence
in one fold, 303 of them wrong, which blew up that fold's log loss to 1.87 (worse than a naive
coin flip's 0.693). Fixed by clipping every calibrated probability to `[0.01, 0.99]` — a
system forecasting single-stock direction should never claim literal certainty regardless of
what a finite calibration sample implies.

**Selective prediction / abstention** (`models/selective.py`) — turns a calibrated probability
into an explicit **trade / abstain** decision via a confidence signal
(`|proba − 0.5| × 2`, i.e. distance from maximal ambiguity), and measures quality as
*accuracy at a given coverage level* (the blueprint's own "precision at target coverage"
metric). The underlying principle: a model saying "I don't have enough edge to call this one"
is more useful in production than one that always guesses.

---

## Regime detection

`features/regime.py` implements a deliberately simple, transparent, **rule-based** (not
learned) regime classifier — this is one of the blueprint's two explicitly "acceptable
designs": prove that conditioning on regime helps at all before building a more complex
learned model (HMM / clustering) to do it.

Two independent axes, both market-wide (not per-instrument):
- **Trend regime** — Uptrend / Downtrend / Range, from the real Nifty 50 index's price
  relative to its own 50-day and 200-day moving averages.
- **Volatility regime** — Low / Medium / High, from India VIX's rolling 1-year percentile,
  split into terciles.

---

## Backtesting with realistic costs

`eval/costs.py` exists specifically to prevent "paper alpha" — a backtest that looks
profitable gross of costs but would have been destroyed by real trading frictions. It converts
a probability signal into a position size, applies a **turnover-proportional blended cost**
(brokerage + STT + slippage combined into one basis-point figure — a simplified but honest
stand-in for the blueprint's fuller execution-cost model), and reports Sharpe ratio, total
return, and max drawdown **both at baseline cost and stressed at 2× and 5×** — exactly the
kind of stress table the blueprint calls for, because a backtest is only as trustworthy as its
approximation of how orders would actually have been filled.

The current locked result makes this concrete: converting the Phase-2 ensemble's raw
probabilities into a simple linearly-sized position, annualised Sharpe is a modest **+0.149 at
baseline cost** but degrades sharply under 2×/5× stress (−0.685, −3.057) — a real, measured
demonstration that even a fragile, barely-positive result (which is what this system currently
has, not a genuine edge) can be wiped out by costs, one of the blueprint's central warnings.
See [Results](#results-honestly-reported) for the full before/after-wiring comparison.

---

## Monitoring — feature and data drift

`monitoring/drift.py` implements the **Population Stability Index (PSI)**, the standard
metric for detecting whether a feature's distribution has shifted between a reference window
(e.g., training data) and a current window (e.g., a recent slice) — using the industry-standard
thresholds `PSI < 0.10` (no shift), `0.10–0.25` (moderate, worth watching), `≥ 0.25`
(significant, investigate/retrain). A companion data-freshness/staleness check flags when a
source hasn't updated as recently as expected. Run for real on this project's own data, PSI
correctly flagged USD-INR/VIX/Brent (genuine multi-year macro regime shifts) and level-based
technicals (SMA/EMA/OBV, which drift with the price level itself) as drifted, while
return-based features stayed stable — exactly the expected pattern, which is itself a form of
validation that the metric is measuring something real.

---

## The RAG + LLM intelligence layer (Phase 3)

This is a genuine, **locally-run** retrieval-augmented-generation pipeline over NSE corporate
announcements — not a mocked stand-in. It runs against a local Ollama server (embeddings via
`mxbai-embed-large`, generation via `llama3.2:3b`), chosen deliberately over installing
`sentence-transformers`/`torch`/`faiss` to avoid a third heavy, versioned-dependency risk after
two real scipy/numpy ABI conflicts earlier in the project.

**Ingestion** (`documents/ingest.py`, `documents/schemas.py`) — NSE's whole-exchange
`corporate-announcements` endpoint, carrying the most precise point-in-time field in the whole
project: `exchdisstime`, the exchange's own dissemination timestamp *to the second*.

**Chunking** (`documents/chunking.py`) — word-bounded chunks with stable per-chunk IDs (most
NSE announcement text is already short enough to become a single chunk; the machinery exists
for the longer ones and for future full-PDF parsing).

**Embeddings** (`documents/embeddings.py`) — real 1024-dimensional dense vectors from the
local Ollama server, with an on-disk cache so repeated runs don't re-embed unchanged text.

**Hybrid retrieval with a point-in-time firewall** (`documents/retrieval.py`) — this is the
conceptual heart of the RAG layer, and worth explaining precisely:

1. A **hand-rolled Okapi BM25** index (auditable, no new dependency — the same
   philosophy as the technical indicators) finds documents matching exact terminology.
2. **Dense cosine-similarity search** over the Ollama embeddings bridges paraphrases BM25
   would miss.
3. **Reciprocal Rank Fusion** (`score = Σ 1/(k + rank)`) combines the two ranked lists without
   needing to calibrate BM25's unbounded scores against cosine similarity's [-1, 1] range —
   only each list's *rank order* matters, which is exactly why RRF needs no tuning to combine
   fundamentally different signal types.
4. Before any of that scoring happens, **`filter_as_of()`** drops every document whose
   `available_time` is after the query's `as_of` timestamp — structurally, not by convention.
   `hybrid_retrieve()` cannot accidentally surface a future document because it never even
   sees one; the filter runs first, not as a post-hoc check. This is proven by a dedicated
   test: a future document engineered to score highest on *both* BM25 and dense similarity
   must never appear in results.

**Evidence packs** (`documents/evidence.py`) — the contract between retrieval and generation:
each evidence item carries its own ID, a content hash (so a later correction can't silently
alter the basis of an old decision), and full provenance — deliberately a plain, serialisable
structure meant to be logged verbatim as an audit trail, not a class with hidden behaviour.

**Structured extraction** (`documents/extraction.py`) — converts retrieved evidence into
daily, time-stamped features rather than feeding raw text into a predictor: a materiality
classification (Low/Medium/High) grounded in NSE's own announcement-category taxonomy, daily
event counts, and a "days since last high-materiality event" novelty signal.

**RAG → ML fusion** (`features/events.py`) — a deliberate **late-fusion** design (per the
blueprint's own recommendation to start there rather than a more complex early-fusion
architecture): event counts and the novelty signal merge onto the same daily feature matrix
every other data layer uses. One subtlety handled explicitly: a day with *zero* announcements
is real, known information ("nothing happened"), so counts are zero-filled after the merge —
unlike fundamentals or delivery data, where an absent row genuinely means "not yet known" and
must stay `NaN`.

**LLM explanation + hallucination guardrails** (`documents/llm.py`, `documents/guardrails.py`)
— the local LLM is given *only* an evidence pack, never a raw question, and is instructed to
answer using only that evidence, cite every claim to a specific evidence item, and say
"Insufficient evidence to answer" rather than guess. Crucially, **the model is never trusted to
grade its own groundedness** — that would just relocate the hallucination risk, not remove it.
Instead, five *mechanical*, auditable checks run against its raw output:

- **Abstention check** — no evidence ⇒ no claim.
- **Citation-reference validity** — every cited evidence number must actually exist.
- **Numeric grounding** — every number in the response must trace back to the evidence shown.
- **Claim overlap** — a lexical (word-overlap) faithfulness check (a documented simplification:
  not a trained entailment model, so it can miss a paraphrase with no shared vocabulary, but it
  reliably catches the common case of wholesale invention).
- **Prompt-injection scanning** — pattern-matches for text trying to override the system prompt.

Two real bugs were found and fixed here via genuine end-to-end runs against the actual local
model (not mocks): the numeric-grounding check initially false-flagged a numbered-list answer's
"1.", "2.", "3." as invented figures (fixed by excluding bare 1–2 digit integers), and it
missed that a response correctly citing a publication year shown in the prompt's
`publication_time` field wasn't in the guardrail's own notion of "the evidence" (fixed by
making the guardrail check everything the model was actually shown, not just one field) — the
general lesson recorded in the code: a guardrail's definition of "the evidence" must exactly
match what the model was given, or it cannot correctly judge groundedness.

**Evaluation and the ablation test** (`documents/eval.py`, `scripts/run_phase3_pipeline.py`,
`scripts/run_phase3_ablation.py`) — retrieval quality is measured with Recall@k, Precision@k,
and Mean Reciprocal Rank (MRR 0.886 on a real evaluation pool — the correct company's document
is almost always ranked first). But the test that actually matters is the **ablation**: a
paired, identical-fold comparison of the Phase 2 model *with* vs. *without* the new RAG event
features. Reported exactly as found — **no meaningful incremental value** (mean IC difference
−0.011, negative in all 4 folds). Per the blueprint's own operating rule, these features stay
in the explanation/research layer and are *not* promoted into the live trading signal on this
evidence. This is the correct, disciplined outcome of doing an ablation test honestly, not a
failure of the Phase 3 build.

---

## Repository layout

```
nifty_analyser/
  config.py               project-wide paths/constants (IST tz, NSE close time)

  data/                   ingestion & storage layer
    schemas.py              point-in-time data contract (MarketBar, FeatureRecord)
    ingest.py                yfinance OHLCV adapter
    bhavcopy.py              NSE delivery %/bhavcopy ingestion
    fundamentals.py           NSE XBRL fundamentals extraction
    insurance_fundamentals.py IRDAI-format PDF fundamentals extraction (SBILIFE/HDFCLIFE)
    macro.py                  India VIX / USD-INR / Brent ingestion
    fii_dii.py                 NSDL FII/DII flow ingestion
    instrument_master.py       Nifty 50 constituent list + ticker mapping
    timestamps.py               IST timezone + as-of availability helpers
    store.py                    point-in-time feature store (as_of, assert_no_lookahead)

  features/                feature engineering, one family per module
    price_volume.py, technical.py, delivery.py, fundamentals.py, macro_features.py,
    fii_dii_features.py, cross_sectional.py, regime.py, events.py, build.py (assembler)

  labels/targets.py        forward return, direction, triple-barrier labelling
  validation/walk_forward.py  purged & embargoed walk-forward CV + calibration-slice carving
  models/                   baseline.py, calibration.py, ensemble.py, selective.py
  eval/                     metrics.py (IC/AUC/log loss/...), costs.py (cost-stressed backtest)
  monitoring/drift.py       Population Stability Index + data-freshness checks

  documents/                Phase 3: the RAG + LLM pipeline
    schemas.py, ingest.py, chunking.py, embeddings.py, retrieval.py, evidence.py,
    extraction.py, llm.py, guardrails.py, eval.py

scripts/                  end-to-end pipeline runners & backfill jobs (one per phase)
tests/                    pytest suite — one file per area, 361 tests, one file per module
data/{raw,processed}/     parquet cache (gitignored except .gitkeep)
reports/                  locked metrics output from each phase's pipeline run (.md + .json)
*.pdf                     blueprint & reference documents driving the roadmap
TRACKER.md / TRACKER.xlsx  the detailed, line-item build tracker (see below)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ (for `zoneinfo`). Key dependencies: pandas, numpy, scikit-learn,
lightgbm, xgboost, catboost, yfinance, pydantic, pyarrow, pdfplumber, pytest.

**Phase 3 (RAG/LLM) additionally requires a local [Ollama](https://ollama.com) server**
running with:
- `mxbai-embed-large` pulled (embeddings)
- `llama3.2:3b` pulled (generation — see `documents/llm.py:DEFAULT_MODEL`)

The Phase 1/2 pipelines have no such dependency and run with only the packages in
`requirements.txt`.

---

## Running things

```bash
# Run the full test suite (361 tests)
pytest

# End-to-end phase pipelines (produce reports/ output)
python scripts/run_phase1_baseline.py
python scripts/run_phase2_pipeline.py
python scripts/run_phase3_pipeline.py       # requires local Ollama running
python scripts/run_phase3_ablation.py       # the RAG-vs-no-RAG paired comparison
python scripts/run_phase2_fundamentals_fii_dii_ablation.py   # fundamentals+FII/DII-vs-without paired comparison

# Data backfills (each hits real external sources)
python scripts/backfill_delivery.py
python scripts/backfill_fundamentals.py
python scripts/backfill_announcements.py
```

FII/DII and insurance-fundamentals backfills were run inline during development (see
[TRACKER.md](TRACKER.md) Session 5's work log for the exact commands/output); dedicated
`scripts/backfill_*.py` wrappers for them are a natural, low-effort addition if you need to
re-run them regularly — currently their cached output lives at
`data/raw/fii_flow_nifty50.parquet` and `data/raw/dii_flow_nifty50.parquet` (FII and DII are
separate files, since their real coverage windows differ), and `data/raw/fundamentals_nifty50.parquet`
(general XBRL) plus `data/raw/insurance_fundamentals_nifty50.parquet` (SBILIFE/HDFCLIFE PDFs).
`data/store.py: load_combined_fundamentals()` concatenates the latter two at load time — it does
not read the separate `data/raw/fundamentals_all_nifty50.parquet` file some earlier one-off
verification scripts produced, which is not part of the actual pipeline.

---

## Results, honestly reported

Every number below is the actual locked result from a real pipeline run against real data —
not a cherry-picked or illustrative figure.

| Phase | Result |
|---|---|
| **1** (technicals-only baseline) | LightGBM: 51.4% ± 3.3% accuracy, IC +0.012 ± 0.053. Logistic: 51.5% ± 3.4% accuracy, IC −0.010 ± 0.049. **Statistically indistinguishable from no edge** — expected and correct for an untuned, technicals-only baseline. |
| **2** (98-feature: technical + delivery + cross-sectional + macro + regime + calibrated ensemble, 2026-09-19) | 50.4% ± 4.4% accuracy, ROC-AUC 0.519 ± 0.044, IC +0.045 ± 0.093. Still no statistically robust edge. Cost-stressed Sharpe: **−0.48 at baseline cost, −1.50 at 2×, −4.21 at 5×.** |
| **2, re-run** (110-feature: adds fundamentals + FII/DII flow, 2026-09-22) | 50.10% ± 5.23% accuracy, ROC-AUC 0.5069 ± 0.0212, IC +0.0065 ± 0.0540 — flat to slightly worse than the 98-feature version on classification metrics (both within noise of "no edge"). Cost-stressed Sharpe: **+0.149 at baseline cost, −0.685 at 2×, −3.057 at 5×** — improved at baseline, still not robust to cost stress. DII columns were entirely `NaN` in 3 of 4 folds' training windows (its real coverage starts June 2026). |
| **2, fundamentals + FII/DII ablation** (2026-09-23) | Paired, identical-fold comparison (`scripts/run_phase2_fundamentals_fii_dii_ablation.py`): mean IC difference **−0.0089, but not uniformly negative** (per-fold: +0.0418, −0.0783, +0.0020, −0.0012) — a high-variance, sign-flipping pattern that is itself the signature of no real incremental signal, not a consistent effect either way. Same disciplined non-conclusion as the Phase 3 ablation below: not evidence to promote, not grounds to remove. |
| **3** (RAG event-feature ablation) | Mean IC difference with vs. without RAG features: **−0.011, negative in all 4 folds** — no incremental trading value found; features kept in the research/explanation layer only. |
| **Robustness** (2026-11-24, `reports/robustness.md`) | **CPCV** (8 groups, 7 paths): ROC-AUC **0.494** (0.488–0.502), IC −0.009 — no path shows edge. **Impact-cost model**: strategy Sharpe 0.68 gross → −0.24 net at Rs 10 cr (negative from Rs 1 cr to Rs 1000 cr) vs equal-weight buy&hold 0.77. **Learned GMM regime** vs rule-based: AUC −0.004 ± 0.004 — no benefit, not adopted. |
| **Nested HPO + reranker** (2026-11-24) | Nested random search (5 outer folds): tuned AUC 0.513 vs default 0.509, **+0.003 ± 0.007** — no edge; inner scores overstate outer by +0.0074 (selection bias). LLM reranker (opt-in `documents/rerank.py`): nDCG@5 0.760 → 0.791 (+0.031 ± 0.015) on a 40-query metadata-labelled benchmark — modest, small sample. |
| **Cross-sectional refinement** (2026-11-24) | Pooled CPCV AUC ≈ 0.49–0.50 for LightGBM/logistic/ensemble (no directional edge) — but **within each date** LightGBM ranks stocks slightly better than chance: daily IC +0.015, market-neutral top/bottom-quintile **gross Sharpe ≈ 1.0, permutation p = 0.025**. **Net of impact costs it is negative** (Sharpe −0.28 at Rs 10 cr; break-even ≈ 14 bps/side); only a post-hoc 20-day hold nets +0.32 (+2.1%/yr). Faint ranking signal, not a tradable strategy; multiple-testing caveats in TRACKER. |
| **Label design** (2026-11-24) | Rebuilt vol-scaled triple-barrier labels, uniqueness weights, meta-labels (the Phase-1 triple-barrier code had been lost). Paired 6-fold × 3-seed study on the same target: **no variant beats plain direction labels** (within-date AUC 0.5137 vs 0.5140 / 0.5145; ±0.002); uniqueness weights are a no-op for fixed-horizon labels; meta-labelling +0.08 pp hit rate (noise). |
| **Scalable retrieval** (2026-11-24) | Persistent PIT vector index over all **58,445** announcements (embedded once in 500 s; 0.35 s/query). **Full-corpus** retrieval is much harder than the earlier 600–700-doc pools suggested: **RRF MRR 0.47** (vs 0.91 on the small pool — that figure was optimistic). LLM rerank: nDCG@5 0.327 → 0.360 (+0.034 ± 0.013), P@5 +0.035 ± 0.012, consistent but modest; opt-in. Ablation: restricting retrieval to the instrument named in the query lifts nDCG@5 **0.32 → 0.71**; unfiltered BM25 is the weakest component at scale (0.19) and drags RRF below dense-only; rerank on top of the filter adds +0.024 ± 0.013. A tune/holdout fusion study (200 queries) then chose the weights: whole-corpus search is best dense-only (held-out nDCG@5 0.44 vs 0.27 classic RRF), instrument-filtered search is best BM25-heavy (0.67 vs 0.64); these are now `retrieve()`'s defaults (BM25 weight 16 and ticker-stripped queries when instrument-filtered). Embedding comparison on 200 queries: mxbai-embed-large 0.451 ≈ bge-m3 0.429 > qwen3-embedding 0.310 (unfiltered); identical (~0.69) when instrument-filtered — mxbai kept. |

The consistent, honest theme across all three phases: this is a hard problem, an
untuned/lightly-tuned technical+fundamental+macro+news-event system does not currently show a
statistically robust directional edge on Nifty 50 daily direction, and every phase's job was to
measure that rigorously and say so plainly — not to keep adding complexity until a number
looked good by chance.

---

## Design principles this codebase holds itself to

- **No lookahead, ever.** Every feature/label/retrieval module carries (and tests prove)
  point-in-time correctness — see `assert_no_lookahead()`, the retrieval firewall, and the
  truncate-the-future proofs throughout `tests/`.
- **Honest reporting over flattering numbers.** Locked results are reported even when they
  show no edge; nothing is tuned before an honest baseline is recorded; an ablation that finds
  "no value" is reported as such and the feature is *not* promoted anyway.
- **Real data over mocked convenience**, wherever a genuine source exists — including running
  a real local LLM/embedding stack for Phase 3 rather than mocking model calls.
- **Investigate before building.** Nearly every data source in this project required directly
  testing assumptions (does this endpoint support historical ranges? does this heuristic
  generalise past one example?) before writing extraction code — several times finding the
  obvious approach doesn't work and a better one exists.
- **Bugs are documented with their mechanism, not glossed over.** 27 real, non-hypothetical
  bugs have been found and fixed across this project so far, every one recorded in
  [TRACKER.md](TRACKER.md) with exactly what broke, how it was caught, and how it was fixed.
- **Simplifications are documented, not hidden** — see "Known Simplifications / Deviations" in
  TRACKER.md for every deliberate shortcut and why it's safe to leave as-is for now.

---

## Phase 4 — operations (`nifty_analyser/ops/`)

`python3 scripts/build_serving_artifacts.py` builds a point-in-time serving feature store and a checksummed model registry entry;
`STM_ADMIN_TOKEN=... python3 scripts/serve.py` serves it on `127.0.0.1:8310`.

| Endpoint | Purpose |
|---|---|
| `POST /v3/analyse` | `{instrument_id, as_of?}` → status **OK / ABSTAIN / DEGRADED / REFUSED**, signal, regime, index-membership flag, health, warnings, disclaimer, audit seq/hash. Only data available at `as_of` is used; future `as_of` → 422. |
| `GET /v3/health`, `GET /v3/dashboard` | Serving state (GREEN/AMBER/RED from data freshness + feature drift), drift table, model, audit integrity. |
| `POST /v3/admin/kill` / `reset` | Manual kill switch (persistent; only an explicit reset clears it). **Disabled unless `STM_ADMIN_TOKEN` is set.** |
| `GET /v3/audit/verify`, `POST /v3/audit/replay/{seq}` | Verify the SHA-256 hash chain; recompute any logged signal and detect a changed model, edited features or altered output. |

Drift monitoring is deliberately **scale-free and self-calibrated**: unit-bearing levels (prices, ₹ flows, USD-INR, Brent, VIX level,
revenue) are excluded, market-wide features are monitored per distinct day, and the alert levels are learned from the model's own history
(textbook PSI cut-offs would have alarmed in 100% of historical months). Honest operating note: with the validated no-edge model, all 50
current constituents return **ABSTAIN** (confidence below 0.10); all 101 model features are scale-free and monitored. The API
has no authentication of its own — keep it on localhost or behind a gateway. `python3 scripts/calibrate_drift.py` re-learns the alert levels.

Remaining refinements (all optional): a trained cross-encoder reranker for retrieval, a learned regime model, combinatorial purged CV, a fuller
cost/market-impact model, scale-free drift features, and rebuilding `TRACKER.xlsx`.

---

## Robustness tooling (2026-11-24)
`validation/cpcv.py` (combinatorial purged CV), `eval/costs.py: ImpactCostModel` (spread + square-root impact vs ADV),
`features/regime_learned.py` (fold-wise GMM regime), `eval/dataset.py` (fast labelled PIT dataset); run
`python3 scripts/run_robustness.py` -> `reports/robustness.{md,json}`; `validation/nested.py` (nested HPO, `scripts/run_nested_hpo.py`); `documents/index.py` (`scripts/build_vector_index.py`, `scripts/run_phase3_fullscale_eval.py`); `labels/triple_barrier.py` (`scripts/run_label_study.py`); `eval/longshort.py` (market-neutral test, `scripts/run_longshort.py`, `scripts/run_cpcv_full.py`); `documents/rerank.py` (LLM reranker, `scripts/run_phase3_rerank_eval.py`). Result: no edge, costs and complexity do not help.

## Forward paper trading (`ops/paper.py`)
Historical results have been looked at many times, so the only untainted evidence is the future. A **pre-registered**
market-neutral spec (top/bottom quintile, 20-day hold, impact-cost model; hashed into every entry) is recorded as books in
a hash-chained, append-only ledger (`data/processed/paper_ledger.jsonl`) using only data published at decision time;
`python3 scripts/paper_trade.py` adds a book when a rebalance is due and scores past books on later prices. First book:
2026-11-20; no realised P&L yet — a meaningful sample needs 1–2+ years. Back up the ledger (gitignored, irreproducible).

## Content-judged retrieval evaluation (`documents/judge.py`)
Metadata-derived relevance (same instrument + category) was a weak proxy, so retrieval is now also scored on 60 natural-language
topical queries with graded 0–3 relevance from two local LLM judges (κ = 0.89): `python3 scripts/run_judged_eval.py` →
`reports/judged_eval.json`. Result: classic RRF (w=1) beats the earlier metadata-tuned BM25 weight (default changed to 1.0);
LLM reranking is a robust gain (qwen3:8b +0.075 ± 0.015 nDCG@5, now the default reranker model). A label sheet
(`reports/label_sheet.csv`) is exported for a **human** to fill `human_grade` (0–3); `load_human_labels` overrides model grades.
LLM judges are a proxy, not ground truth. `scripts/run_longshort_models.py` runs the market-neutral test for every model family:
LightGBM/CatBoost carry the faint gross cross-sectional signal; the production stacked ensemble carries none.

## Cost calibration & robustness add-ons
`eval/calibrate_impact.py` + `python3 scripts/calibrate_impact.py fills.csv` fit the impact-cost model from real executions (none exist yet:
the coefficient is still an assumption). `scripts/run_cost_sensitivity.py`: at Rs 10 cr the 20-day-hold long/short stays net-positive across all
tested fee/impact assumptions (Sharpe 0.24–0.59) while the 5-day hold does not. `scripts/run_blend_seeds.py`: the faint cross-sectional signal is
positive in 9/9 seed runs of LightGBM/CatBoost/blend; the blend is no better than its parts. Selection is in-sample — only the forward ledger is clean.
`python3 scripts/build_tracker_xlsx.py` regenerates `TRACKER.xlsx` from `TRACKER.md`.

## Cross-encoders, spread estimates, second paper arm (2026-11-24)
`documents/crossencoder.py` (MiniLM-L6 and bge-reranker-v2-m3 via sentence-transformers): +0.025 / +0.014 nDCG@5 on the content-judged benchmark, consistent
but weaker than the LLM reranker (qwen3:8b, +0.09), which stays the default. `eval/spread_estimators.py` (Corwin–Schultz / Abdi–Ranaldo on Yahoo OHLC) gives 27–69 bps
half-spreads — loose upper bounds, not usable for calibration; real fills are still required. Forward testing now has two pre-registered arms
(`production`, `blend`); `python3 scripts/daily_refresh.py` (Yahoo refresh → serving rebuild when due → both books) keeps them accruing. A third LLM rater
(240 graded pairs, κ 0.855 vs the local judges) supports the judged benchmark but is not a human labelling.

## Selection-aware statistics (`eval/stat_rigor.py`)
`python3 scripts/run_stat_rigor.py`: deflated Sharpe, effective sample size, bootstrap CIs and a sector-neutral long/short on a seed-averaged tree blend.
Honest bottom line: gross Sharpe ≈0.5 (earlier single-seed ≈1.0 was optimistic), hold-20 net +0.14 with a CI spanning zero, deflated Sharpe ≈0.03, and the signal
vanishes when sector bets are removed — the faint ranking signal is mostly sector rotation. No statistically established, tradable edge.

## `/v3/evidence` (2026-11-25)
`STM_ADMIN_TOKEN=... python3 scripts/serve.py --evidence [--rerank]` adds `POST /v3/evidence` (`{query, instrument_id?, as_of?, top_k, rerank?, window_days?}`):
point-in-time, audited retrieval of company announcements as a citable evidence pack, with graceful degradation (BM25-only if the embedder is down, first-stage
order if the reranker is down). Same guarantees as `/v3/analyse` (future `as_of` rejected, unknown instrument refused, every call in the hash-chained audit log).

## `/v3/explain` (2026-11-25)
`python3 scripts/serve.py --evidence --explain` adds `POST /v3/explain` (same request as `/v3/evidence`): the local LLM answers only from the retrieved evidence, then guardrails
(citations exist, numbers appear in the evidence, claim overlap, abstention) decide whether the text is released — otherwise it is withheld (BLOCKED) and only the evidence is returned.
No evidence → the model is not called; LLM down → evidence only. Guardrails check support, not full semantic faithfulness. A 50-question study (`scripts/run_explain_eval.py`) found and fixed an over-blocking bug (explicit abstentions were withheld) and moved the default generator to `qwen3:8b` (answers 50% of answerable questions vs 29% for `llama3.2:3b`, no unfaithful releases under a `qwen3:14b` judge; the judge is lenient, see TRACKER).

## Where high accuracy is real (2026-11-26)
Return/direction forecasting stays at chance (AUC ≈ 0.50, LLM 52%). Understanding the filings is a different story: `scripts/run_event_classifier.py` recovers the announcement type
(category string masked, time-split holdout) at **85% on all rows (v1, conservative)**, **92%** (v2, 14-way, content-bearing texts only) and **97%** (v2 with NSE's 2025 results/board-outcome taxonomy drift merged); `scripts/run_filing_qa.py` answers dividend-record-date / results-period questions through `/v3/explain`
at **96.7%** on fresh questions (the pipeline now lists evidence newest-first with an as-of time, which moved a hard-window dev set from 57.5% to 91.2%). Recency is now handled by default — `ops/temporal.py` restricts a question that names a month/year to that period, otherwise company-specific questions favour the freshest filing (`prefer_recent` overrides) — which takes the no-flag pipeline from **57.5% to 100% on a fresh 120-question draw and 99.1% on 108 historical-period questions** (with a guardrail overlap fix and a former-names hint) on 120 fresh questions; the benchmark asks for the *latest* filing, so it rewards recency by construction. Caveats are in TRACKER.md.

## `/v3/classify` (2026-11-26)
`python3 scripts/train_event_classifier.py [--final]` trains/evaluates/saves the event-type classifier; `python3 scripts/serve.py --classify` serves `POST /v3/classify` (`{text, category?}` → label +
confidence, ABSTAIN when unsure). Untouched test period: 97.3% raw, **99.5% on the 94% of texts it accepts** (threshold chosen on validation). Labels are NSE's own category field.

## Topical search & answer quality (2026-11-26)
`CategoryRouter` (question → likely NSE categories → fused category-restricted search) lifts judged nDCG@5 0.665 → 0.693, **0.768 with the `qwen3:8b` reranker**; `/v3/evidence` and `/v3/explain` use it by default. The answer guardrails were rebuilt and calibrated
(`scripts/run_guardrail_probe.py`: faithful answers pass 100%, planted fabrications blocked 95.7%; leaks are fabrications reusing evidence words). `/v3/explain` answers 54% of answerable packs with `qwen3:8b` (61.5% with `qwen3:14b`), nothing unfaithful released
under the judge; the rest abstain. These open-ended tasks are inherently below the templated date questions (99–100%).

## Pretrained models (2026-11-26)
Pretrained NLI (`documents/nli.py`, DeBERTa-v3) is available as an opt-in strict faithfulness veto (`serve.py --nli`, request `strict_faithfulness=true`): blocks 99.9% of planted fabrications
but withheld 3 of 21 real faithful answers in a replay, so it is not the default. FinBERT sentiment and the Chronos-Bolt time-series foundation model were tested zero-shot on direction: 50–53%, i.e. chance — pretrained
knowledge does not create a forecasting edge on these stocks.

## Digest fallback (2026-11-26)
When the model refuses (or is down) but the reranker rates the top filing ≥ 7/10, `/v3/explain` answers with a deterministic digest of the retrieved filings (`answer_source: "digest"`, faithful by construction):
answer rate on answerable questions 54% → **76.9%**, every released answer judged faithful. NLI strict mode (`strict_faithfulness`) withheld 3 of 21 real answers in a replay (earlier "half" figure was from a mismatched replay and is corrected in TRACKER.md).

## Overnight (US-close) direction signal (2026-11-26)
`python3 scripts/overnight_signal.py` / `ops/overnight.py`: following the S&P 500's sign predicts the Nifty's next **opening gap** 68% of the time on all days, **78% when |S&P| ≥ 0.5% (54% of days), 85% at ≥ 1% (26%)**; close-to-close 58–72%.
This is real and leak-free but **not a trading edge**: the gap is set in the pre-open auction after the US move is known, and the tradable part (open → close) is at chance (47–48%). Tradable direction accuracy remains ≈ 50%.

## Tradable direction — every horizon tested (2026-11-26)
Hourly-bar test (`scripts/run_intraday_direction.py`): predicting the remainder-of-day direction after the first or fourth hour is 51% (AUC 0.51), naive intraday momentum 49–50%. With the daily, LLM, FinBERT, Chronos
and overnight-US results, **no tradable direction signal exists at any horizon from intraday to 20 days**; only the non-tradable opening gap is predictable (78–86% on big US-move days).

## Pre-open direction model (2026-11-26)
`python3 scripts/run_preopen_direction.py`: US indices, VIX, dollar, Brent, gold, US yields and USD/INR (all known at 09:00 IST) give the Nifty **opening-gap direction AUC 0.75** (accuracy 68.6%; 88% on the most confident quarter), stocks AUC 0.69, close-to-close AUC 0.60/0.56 —
but the open-to-close part, the only part tradable after the open, stays at AUC ≈ 0.50.

## Needs a human (2026-11-25)
`python3 scripts/label_cli.py --n 120` (blind relevance labels, ~15 min) then `python3 scripts/human_agreement.py`. To keep the forward test accruing unattended:
`cp scripts/launchd/com.stm.daily-refresh.plist ~/Library/LaunchAgents/ && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.stm.daily-refresh.plist`
(not installed by me; the harness blocked it). Real execution fills for `scripts/calibrate_impact.py` must come from your broker.

## Volatility forecasting (2026-11-27)
Direction stays at chance; the **size** of moves is forecastable. `features/volatility.py` (range-based HAR terms, own-median ratios, market and India VIX state,
results-season timing; 21 scale-free features) + `models/volatility.py` (pre-registered blend of linear HAR and shallow LightGBM), purged walk-forward on the point-in-time
universe (`scripts/run_vol_forecast_v2.py`). Target: will the next h-day realised volatility exceed the stock's own trailing-year median?

| | 5-day | 20-day |
|---|---|---|
| Persistence rule AUC | 0.598 | 0.669 |
| **Blend AUC** | **0.671** | **0.702** |
| AUC gain, 95% date-block bootstrap | +0.056 to +0.089 | -0.003 to +0.065 (**not significant**) |
| Accuracy, all / most-confident 20% | 62.7% / 73.8% | 61.4% / 79.0% |
| Accuracy of persistence rule | 57.1% | 62.5% |
| Out-of-sample R² of log volatility | 0.198 | 0.143 (LightGBM alone 0.267, post-hoc) |

At 20 days the blend ranks better than persistence but its 0.5-threshold accuracy is slightly **worse**. Zero-shot Hugging Face `amazon/chronos-bolt-small`
(`scripts/run_vol_chronos.py`, non-overlapping dates): AUC 0.638 vs our 0.661 at 5 days, 0.676 vs 0.673 at 20 days (only 38 dates); averaging the two looks helpful at 20 days
and is untested forward. The same history has been examined many times, so `scripts/forecast_volatility.py` records live forecasts in a hash-chained ledger
(`data/processed/vol_ledger.jsonl`, first entry 2026-11-25) and grades them as they mature; `daily_refresh.py` runs it and `scripts/serve.py` serves
`GET /v3/volatility?instrument_id=NSE:INFY&horizon_days=5`. This is a risk forecast, not a direction or trading signal.

### Expected-move ranges, ablation and portfolio use
`scripts/run_vol_ranges.py` turns a forecast into a 90% price-move range (`models/ranges.py`, quantiles of realised outcomes only).

| Walk-forward | 5-day model / EWMA / trailing | 20-day model / EWMA / trailing |
|---|---|---|
| Coverage of the 90% range | 88.7% / 89.4% / 89.6% | 87.1% / 88.4% / 88.6% |
| Mean range width | 10.7% / 11.7% / 14.3% | 21.2% / 23.1% / 23.9% |
| Pinball loss gain | significant over both | **not significant** over either |

The model's ranges are narrower but cover slightly less than 90% (widths are log-return units). **Portfolio test:** inverse-forecast-volatility weighting
(net Sharpe 0.96 / 1.03) beats inverse-trailing-volatility (0.83 / 0.98) with far less turnover, but does **not** meaningfully beat equal weight (0.97 / 1.02).
**Ablation** (`scripts/run_vol_ablation.py`): delivery % and announcement counts add nothing; FII flows with FX/Brent volatility make it worse. None adopted.

**Calibration follow-up** (`scripts/run_vol_calibration.py`, rules set first): adaptive ranges restore coverage (89.8% / 89.0%) but cost 1.8% pinball loss at 20 days,
and probability recalibration does not improve the Brier score — **neither adopted**. `/v3/dashboard` now shows the ten names most likely to be more volatile than usual.

### AUC search (`scripts/run_vol_auc.py`)
Seven candidates (more frequent refits, 18 extra features, zero-shot Chronos as a feature, a four-model ensemble) were chosen on the first 60% of out-of-sample dates and
the winner scored once on the rest. Holdout AUC: **5-day 0.6973 → 0.7011** (95% interval of gain -0.0002 to +0.0084), **20-day 0.7288 → 0.7355** (-0.005 to +0.021).
Neither interval clears zero, so **nothing was adopted**; the extra features made 20-day development AUC clearly worse. The holdout period scores higher than the full
history for both models, so compare only within a row. The model is near what daily data supports; finer-grained data is the next lever.

### Longer training history — adopted (current live model)
`scripts/backfill_history.py` adds 2008–2020 Yahoo bars and India VIX (57 stocks, ~170k rows, `data/raw/history/`) as **training rows only**; scoring uses the same 2021+ rows.
One pre-set variant, adoption rule fixed first (`scripts/run_vol_history.py`, `reports/vol_history.json`):

| Same out-of-sample rows | 5-day before → after | 20-day before → after |
|---|---|---|
| **AUC** | 0.671 → **0.684** | 0.702 → **0.739** |
| 95% interval of gain | +0.003 to +0.023 | +0.006 to +0.062 |
| Accuracy | 62.7% → 63.6% | 61.4% → 67.9% (persistence rule 62.5%) |
| R² of log volatility | 0.198 → 0.204 | 0.143 → 0.325 |
| 90% range coverage / width | 89.5% / 11.0% | 88.3% / 21.4% |

The ranges now beat both EWMA and trailing-volatility ranges on pinball loss at both horizons. The tables above this one describe the earlier model without history.
Caveats: the older rows carry survivorship bias (today's pool, training only), have no announcement features, and this history has been examined repeatedly — the forward
ledger (entries tagged `v3-history` from the next data date) is the clean test. Run order: `run_vol_forecast_v2.py`, `run_vol_history.py`, `run_vol_ranges.py`.

### Second AUC search, with history — 20-day extra features adopted
`scripts/build_chronos_history.py` then `scripts/run_vol_auc2.py` (`reports/vol_auc_search2.json`); same develop-then-holdout protocol against the history model.

| | winner | holdout AUC live → winner | 95% interval of gain | decision |
|---|---|---|---|---|
| 5-day | larger trees | 0.7054 → 0.7077 | -0.002 to +0.008 | not adopted |
| 20-day | extra features | 0.7632 → 0.7761 | +0.007 to +0.029 | **adopted** |

Over all out-of-sample dates the 20-day AUC goes 0.739 → **0.752** (that figure includes the selection dates; candidate used 12 refits, and refitting alone added nothing).
The live 20-day classifier now uses `features/volatility_extra.py` (without index features); the 5-day model, the volatility-level regression and the ranges are unchanged.
**Direction** with the same data stays at chance: AUC 0.517 / 0.494 (up, 5/20-day) and about 0.51 for beating the median stock.

Round 4 (selection on 2008-2020 only, `scripts/run_vol_auc3.py`) adopted nothing; every trial and the plan are in [docs/AUC_PLAN.md](docs/AUC_PLAN.md). Round 5 added overnight US and Nifty-index features (`scripts/backfill_global.py`, `scripts/run_vol_auc4.py`): also no gain.

Round 6 tested true intraday realised volatility from public 1-minute data (`scripts/backfill_intraday.py`, `features/volatility_intraday.py`, `scripts/run_vol_intraday.py`): also no reliable gain.

### Announced board meetings — adopted at 5 days
`scripts/backfill_board_meetings.py` + `features/volatility_events.py`: days to the next announced meeting/results date, using only intimations already published.
Chosen on 2010-2020, confirmed on 2021+: **5-day AUC 0.684 → 0.691** (gain +0.004 to +0.011); 20-day did not confirm. Live for the 5-day classifier; `daily_refresh.py` refreshes the dates.

Round 8 tested option-implied volatility from NSE F&O files (`scripts/backfill_options.py`, `features/volatility_options.py`, `scripts/run_vol_options.py`): not adopted.

Ingestion drops the still-forming bar of an open session, and forecasts use the newest date for which at least 95% of members have data.

## Web app
Double-click `Nifty Analyser.command` in Finder to start and open it (adds the Ask page if Ollama is running). System has a refresh button that runs
`scripts/daily_refresh.py`; it accepts requests only from the app, and the server rejects foreign `Host` names.

`python3 scripts/serve.py` then open http://127.0.0.1:8310/. Apple-style, no build step, no external scripts; light and dark, desktop and phone, installable to the home screen.
Stock pages show past forecasts against outcomes; Model shows reliability. Today also lists announced meetings and exports CSV. Pages: Today (volatility board, sectors, watchlist), Portfolio (risk of your own mix; amounts stay in the browser), stock detail (price, forecasts, board meetings, filings), Filings search, Ask (guarded local-model answers with sources; start with `--evidence --explain`), Model (progress and what was rejected), System
(health, audit, forward scorecard). Code: `nifty_analyser/web/`, `ops/web.py`. The app shell works offline (data never cached). Portfolio swing figures are calibrated by `scripts/run_portfolio_backtest.py`
(`reports/portfolio_backtest.json`): the bell-curve 1-in-10 figure held only 83% (5-day) / 88% (20-day), so the measured multiplier is used. No login: keep it on localhost.

## Keeping these docs current

This README, [TRACKER.md](TRACKER.md), and [TRACKER.xlsx](TRACKER.xlsx) must be kept in sync
with the project as it evolves — every phase completion, every new module, every closed gap
gets reflected in all three, not just committed to code. TRACKER.md remains the detailed,
line-item source of truth (every task, every bug, every dated work-log entry); TRACKER.xlsx
mirrors it in spreadsheet form; this README stays the conceptual/architectural map — update its
status table, architecture diagram, and relevant technical section whenever the underlying
system changes, rather than letting it drift out of date.
