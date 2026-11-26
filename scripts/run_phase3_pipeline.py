"""End-to-end Phase 3 pipeline (Blueprint Part 4, §6.2 weeks 12-16).

ingest (cached announcements) -> materiality classification -> daily event
features -> fusion into the Phase 1/2 feature matrix -> retrieval evaluation
(Recall@k / MRR, using a same-instrument relevance proxy across the whole
universe) -> a handful of real RAG+LLM+guardrail question/answer examples ->
a locked report.

Run with: python3 scripts/run_phase3_pipeline.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from nifty_analyser.config import DATA_RAW_DIR, REPORTS_DIR
from nifty_analyser.data.instrument_master import get_instrument
from nifty_analyser.documents.embeddings import EmbeddingCache
from nifty_analyser.documents.eval import evaluate_ranking, mean_recall_at_k, mean_reciprocal_rank
from nifty_analyser.documents.evidence import build_evidence_pack
from nifty_analyser.documents.extraction import add_materiality, classify_materiality, compute_daily_event_features
from nifty_analyser.documents.embeddings import embed_many
from nifty_analyser.documents.guardrails import verify_response
from nifty_analyser.documents.llm import generate_explanation
from nifty_analyser.documents.retrieval import hybrid_retrieve
from nifty_analyser.features.events import add_rolling_event_intensity, merge_events_into_features


def load_announcements() -> pd.DataFrame:
    path = DATA_RAW_DIR / "announcements_nifty50.parquet"
    df = pd.read_parquet(path)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df[df["quality_flag"] == "PASS"].reset_index(drop=True)


def run_retrieval_evaluation(
    documents: pd.DataFrame, n_queries: int = 30, pool_size: int = 600, seed: int = 42
) -> dict:
    """Evaluate hybrid retrieval using a same-instrument relevance proxy (see
    documents/eval.py docstring for why this definition is used): does
    querying for one company's high-materiality event actually surface that
    company's documents, not a different one's?

    KNOWN LIMITATION, found and fixed during development: ``hybrid_retrieve``
    embeds its entire point-in-time-eligible candidate pool on every call --
    fine for one instrument's few-hundred-document history (as used in
    ``run_rag_llm_examples`` below), but not for the full 43,000+ document,
    5-year, 48-instrument corpus, where a query near the end of the history
    would need to embed nearly the whole corpus (tens of thousands of Ollama
    calls) before it could even start scoring. A real production system
    would maintain a persistent, incrementally-updated embedding index
    rather than re-embed the eligible set per call -- out of scope here (see
    TRACKER.md). For this evaluation, a fixed, bounded, stratified sample of
    ``pool_size`` documents (embedded once, up front) stands in for that
    index, which is standard practice for retrieval evaluation generally
    (evaluating against a curated test collection, not inline against a live
    production-scale index).
    """
    materiality = documents["category"].apply(classify_materiality)
    high_materiality_docs = documents[materiality == "High"]
    if high_materiality_docs.empty:
        return {"error": "no high-materiality documents available for evaluation"}

    query_docs = high_materiality_docs.sample(min(n_queries, len(high_materiality_docs)), random_state=seed)

    # Build a fixed, bounded evaluation pool: every query document itself,
    # plus a random sample of the rest of the corpus up to pool_size -- so
    # each query's own relevant document is guaranteed to be retrievable.
    remaining_budget = max(pool_size - len(query_docs), 0)
    other_docs = documents.drop(query_docs.index)
    filler = other_docs.sample(min(remaining_budget, len(other_docs)), random_state=seed)
    eval_pool = pd.concat([query_docs, filler]).drop_duplicates(subset="doc_id").reset_index(drop=True)

    cache = EmbeddingCache()
    print(f"  pre-embedding a fixed {len(eval_pool)}-document evaluation pool...")
    embed_many(eval_pool["text"].tolist(), cache=cache)
    cache.save()

    per_query_metrics = []
    for _, row in query_docs.iterrows():
        instrument = get_instrument(row["instrument_id"])
        query = f"{instrument.symbol} {row['category']}"
        as_of = row["available_time"] + pd.Timedelta(minutes=1)  # just after this doc published

        retrieved = hybrid_retrieve(query, eval_pool, as_of, top_k=10, embedding_cache=cache)
        relevant_ids = set(eval_pool[eval_pool["instrument_id"] == row["instrument_id"]]["doc_id"])
        metrics = evaluate_ranking(retrieved["doc_id"].tolist(), relevant_ids, k=10)
        per_query_metrics.append(metrics)

    return {
        "n_queries": len(per_query_metrics),
        "eval_pool_size": len(eval_pool),
        "mean_recall_at_10": mean_recall_at_k(per_query_metrics),
        "mean_reciprocal_rank": mean_reciprocal_rank(per_query_metrics),
    }


def run_rag_llm_examples(documents: pd.DataFrame, instrument_ids: list[str]) -> list[dict]:
    """A handful of real, end-to-end RAG + LLM + guardrail examples, across
    different instruments -- not just the one used during initial development.
    """
    cache = EmbeddingCache()
    examples = []
    for instrument_id in instrument_ids:
        instrument = get_instrument(instrument_id)
        instrument_docs = documents[documents["instrument_id"] == instrument_id]
        if instrument_docs.empty:
            continue
        as_of = instrument_docs["available_time"].max() + pd.Timedelta(days=1)
        query = f"What were {instrument.symbol}'s most recent financial results or board decisions?"

        retrieved = hybrid_retrieve(query, instrument_docs, as_of, top_k=5, embedding_cache=cache)
        pack = build_evidence_pack(query, as_of, retrieved)
        response = generate_explanation(query, pack)
        verification = verify_response(response, pack)

        examples.append(
            {
                "instrument_id": instrument_id,
                "query": query,
                "n_evidence_items": len(pack.evidence_items),
                "response": response,
                "guardrail_passed": verification.passed,
                "guardrail_reason": verification.reason,
                "claim_overlap_score": verification.claim_overlap_score,
            }
        )
    cache.save()
    return examples


def run_fusion_check(documents: pd.DataFrame) -> dict:
    """Prove the RAG->ML fusion (features/events.py) works on the real, full
    document corpus -- not just synthetic test fixtures.
    """
    from nifty_analyser.data.store import assert_no_lookahead, load_universe_bars
    from nifty_analyser.features.build import build_feature_matrix

    bars = load_universe_bars()
    features = build_feature_matrix(bars)

    docs_with_materiality = add_materiality(documents)
    daily_events = compute_daily_event_features(docs_with_materiality)

    merged = merge_events_into_features(features, daily_events)
    merged = add_rolling_event_intensity(merged, windows=(7, 30))

    decision_time = pd.Timestamp("2024-06-15 16:00", tz="Asia/Kolkata")
    snapshot = merged[merged["available_time"] <= decision_time]
    assert_no_lookahead(snapshot, decision_time)

    return {
        "n_rows": len(merged),
        "n_columns": merged.shape[1],
        "no_lookahead_check": "PASSED",
        "mean_announcement_count_30d": float(merged["announcement_count_30d"].mean()),
    }


def run() -> dict:
    print("Loading cached announcements...")
    documents = load_announcements()
    print(f"Loaded {len(documents)} PASS-quality announcements across "
          f"{documents['instrument_id'].nunique()} instruments, "
          f"{documents['event_time'].min().date()} .. {documents['event_time'].max().date()}")

    print("\nRunning fusion check (RAG event features -> ML feature matrix)...")
    fusion_result = run_fusion_check(documents)
    print(fusion_result)

    print("\nRunning retrieval evaluation (same-instrument relevance proxy, 30 queries)...")
    retrieval_result = run_retrieval_evaluation(documents)
    print(retrieval_result)

    print("\nRunning real end-to-end RAG+LLM+guardrail examples across 5 instruments...")
    sample_instruments = documents["instrument_id"].drop_duplicates().sample(5, random_state=7).tolist()
    rag_examples = run_rag_llm_examples(documents, sample_instruments)
    for ex in rag_examples:
        print(f"\n--- {ex['instrument_id']} ---")
        print(f"Query: {ex['query']}")
        print(f"Evidence items: {ex['n_evidence_items']} | Guardrail passed: {ex['guardrail_passed']} "
              f"({ex['guardrail_reason']})")
        print(f"Response: {ex['response'][:300]}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "n_documents": len(documents),
            "n_instruments": int(documents["instrument_id"].nunique()),
            "date_range": [str(documents["event_time"].min().date()), str(documents["event_time"].max().date())],
        },
        "fusion_check": fusion_result,
        "retrieval_evaluation": retrieval_result,
        "rag_llm_examples": rag_examples,
    }
    write_report(report)
    return report


def write_report(report: dict) -> None:
    json_path = REPORTS_DIR / "phase3_rag_metrics.json"
    md_path = REPORTS_DIR / "phase3_rag_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Phase 3 RAG + LLM Layer — Locked Metrics Report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Corpus",
        "",
        f"```json\n{json.dumps(report['corpus'], indent=2)}\n```",
        "",
        "## RAG -> ML fusion check",
        "",
        f"```json\n{json.dumps(report['fusion_check'], indent=2)}\n```",
        "",
        "## Retrieval evaluation (same-instrument relevance proxy)",
        "",
        f"```json\n{json.dumps(report['retrieval_evaluation'], indent=2)}\n```",
        "",
        "## Real end-to-end RAG + LLM + guardrail examples",
        "",
    ]
    for ex in report["rag_llm_examples"]:
        lines += [
            f"### {ex['instrument_id']}",
            "",
            f"**Query:** {ex['query']}",
            "",
            f"**Evidence items:** {ex['n_evidence_items']} | **Guardrail passed:** {ex['guardrail_passed']} "
            f"({ex['guardrail_reason']}) | **Claim overlap:** {ex['claim_overlap_score']:.2f}",
            "",
            f"**Response:** {ex['response']}",
            "",
        ]
    lines += [
        "## Reality check",
        "",
        "Relevance in the retrieval evaluation is defined operationally as "
        "\"belongs to the same instrument\" (Blueprint §4.10's own warning against "
        "a single black-box 'RAG accuracy' number) -- a real, falsifiable, but "
        "narrow slice of true relevance, not a hand-labelled relevance judgment set. "
        "The claim-overlap guardrail is lexical, not a trained entailment model "
        "(see guardrails.py docstring) -- it catches wholesale fabrication, not a "
        "subtly unsupported paraphrase.",
    ]
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    run()
