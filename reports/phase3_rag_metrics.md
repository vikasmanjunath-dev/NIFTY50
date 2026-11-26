# Phase 3 RAG + LLM Layer — Locked Metrics Report

Generated: 2026-09-20T15:23:12.453984+00:00

## Corpus

```json
{
  "n_documents": 43623,
  "n_instruments": 48,
  "date_range": [
    "2021-09-20",
    "2026-09-17"
  ]
}
```

## RAG -> ML fusion check

```json
{
  "n_rows": 59213,
  "n_columns": 90,
  "no_lookahead_check": "PASSED",
  "mean_announcement_count_30d": 19.843226994072246
}
```

## Retrieval evaluation (same-instrument relevance proxy)

```json
{
  "n_queries": 30,
  "eval_pool_size": 600,
  "mean_recall_at_10": 0.18386205899363794,
  "mean_reciprocal_rank": 0.8861111111111111
}
```

## Real end-to-end RAG + LLM + guardrail examples

### NSE:AXISBANK

**Query:** What were AXISBANK's most recent financial results or board decisions?

**Evidence items:** 5 | **Guardrail passed:** True (all guardrails passed) | **Claim overlap:** 0.54

**Response:** Based on the evidence provided, here are AXISBANK's most recent financial results and board decisions:

Axis Bank Limited has submitted their unaudited financial results (Standalone & Consolidated) for the quarter and nine months ended December 31, 2022. Additionally, they have informed the Exchange about a Board Meeting held on January 23, 2023.

There is no more recent information available in the provided evidence, but AXISBANK has submitted their financial results for the period ended March 31, 2023, and their financial results for the quarter and nine months ended December 31, 2022, are the most recent before that.

### NSE:HCLTECH

**Query:** What were HCLTECH's most recent financial results or board decisions?

**Evidence items:** 5 | **Guardrail passed:** True (all guardrails passed) | **Claim overlap:** 0.41

**Response:** Based on the provided evidence, I can answer the following questions:

* The most recent board decision was regarding the consideration of un-audited financial results and payment of an interim dividend (Evidence [1] and Evidence [2]).
* The record date for payment of the interim dividend is not specified in any of the given evidence items.
* HCL Technologies Limited has submitted their financial results for the following periods:
  - December 31, 2021 (Evidence [3])
  - September 30, 2021 (Evidence [4])
  - March 31, 2023 (Evidence [5])

### NSE:UPL

**Query:** What were UPL's most recent financial results or board decisions?

**Evidence items:** 5 | **Guardrail passed:** True (all guardrails passed) | **Claim overlap:** 0.36

**Response:** Based on the provided evidence, UPL's most recent financial result was announced as part of the Board Meeting Outcome on August 3, 2026 (Evidence [2]). The specific financial results mentioned are Unaudited Consolidated and Standalone Financial Results for the Quarter Ended June 30, 2026.

Insufficient evidence to answer the question about board decisions other than financial result announcements.

### NSE:BHARTIARTL

**Query:** What were BHARTIARTL's most recent financial results or board decisions?

**Evidence items:** 5 | **Guardrail passed:** True (all guardrails passed) | **Claim overlap:** 0.41

**Response:** Based on the evidence provided, I can answer the following questions:

* BHARTIARTL's most recent board decision was approved by the Board of Directors at its meeting held on Monday, August 08, 2022 (Evidence item [4]).

Insufficient evidence to answer. There is no information in the evidence regarding BHARTIARTL's most recent financial results.

### NSE:SUNPHARMA

**Query:** What were SUNPHARMA's most recent financial results or board decisions?

**Evidence items:** 5 | **Guardrail passed:** True (all guardrails passed) | **Claim overlap:** 0.51

**Response:** Based on the provided evidence, here are SUNPHARMA's most recent financial results or board decisions:

* Most recent financial results: [5] (2024-05-22T15:05:20+05:30, Financial Result Updates, doc=NSE_ANN_162185) states that Sun Pharmaceutical Industries Limited has submitted the financial results for the period ended March 31, 2024.
* Most recent board decision: [3] (2022-07-29T13:39:20+05:30, Outcome of Board Meeting, doc=NSE_ANN_5066293) states that a Board Meeting was held on July 29, 2022, to consider financial statements for the period ended June 2022.

## Reality check

Relevance in the retrieval evaluation is defined operationally as "belongs to the same instrument" (Blueprint §4.10's own warning against a single black-box 'RAG accuracy' number) -- a real, falsifiable, but narrow slice of true relevance, not a hand-labelled relevance judgment set. The claim-overlap guardrail is lexical, not a trained entailment model (see guardrails.py docstring) -- it catches wholesale fabrication, not a subtly unsupported paraphrase.