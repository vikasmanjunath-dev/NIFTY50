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

> **Current state (2026-09-22):** Phases 0–3 are complete — data ingestion, feature
> engineering, labelling, validation, modelling, calibration, cost-aware backtesting, drift
> monitoring, **and** a full local RAG + LLM layer over NSE announcements. Fundamentals and
> FII/DII flow are now wired into the live Phase 2 pipeline, not just built and verified in
> isolation. 321/321 tests passing. Phase 4 (production operations) has not started. See
> [Status at a glance](#status-at-a-glance).

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
| **4** | Production operations: an API contract, a monitoring dashboard, an audit log, a kill switch (not started) |

---

## Status at a glance

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup & tooling | ✅ done |
| 1 | Baseline: ingestion, labels, technical features, walk-forward validation | ✅ complete |
| 2 | Enrichment: fundamentals (incl. insurance PDFs), macro, FII/DII flows, cross-sectional features, calibration, ensemble, regime, costs, drift | ✅ 10/10 items, all data-coverage gaps closed |
| 3 | RAG + LLM intelligence layer over NSE announcements | ✅ 9/9 items complete |
| 4 | Operations: production API, monitoring, audit log, kill switch | ⬜ not started |

**321/321 tests passing** across 39 test files. See [TRACKER.md](TRACKER.md) for the full
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
tests/                    pytest suite — 39 files, 321 tests, one file per module
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
# Run the full test suite (321 tests)
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

## What's next — Phase 4

Phase 4 (production operations) has not started. Per the blueprint, it covers:
- A production API contract (`/v3/analyse`)
- A monitoring dashboard surfacing data/feature/prediction/calibration drift live
- An audit log with reproducible decision replay
- A kill switch for degraded data or model health

There are also several explicitly flagged, non-blocking refinements documented in
[TRACKER.md's Known Simplifications](TRACKER.md#known-simplifications--deviations-kept-honest-on-purpose)
— e.g., a true cross-encoder reranker beyond RRF fusion, a learned (rather than rule-based)
regime model, and making `hybrid_retrieve` safe to call against the full 43,000+ document
corpus in one pass (currently a real, flagged architecture limit for large-corpus retrieval).

---

## Keeping these docs current

This README, [TRACKER.md](TRACKER.md), and [TRACKER.xlsx](TRACKER.xlsx) must be kept in sync
with the project as it evolves — every phase completion, every new module, every closed gap
gets reflected in all three, not just committed to code. TRACKER.md remains the detailed,
line-item source of truth (every task, every bug, every dated work-log entry); TRACKER.xlsx
mirrors it in spreadsheet form; this README stays the conceptual/architectural map — update its
status table, architecture diagram, and relevant technical section whenever the underlying
system changes, rather than letting it drift out of date.
