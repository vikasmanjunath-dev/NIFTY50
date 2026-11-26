"""Second-stage reranker (Blueprint §4.6 "learned reranker"): a local LLM judges each fused candidate's relevance to the
query pointwise (0-10, temperature 0), and the final score blends that judgement with the first-stage RRF score. Strictly
a re-ordering of candidates ``hybrid_retrieve`` already returned (so point-in-time filtering is untouched; no document can
enter that the first stage did not admit). The scorer is injectable so tests need no LLM."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd

from nifty_analyser.documents.llm import OLLAMA_GENERATE_URL, OllamaUnavailableError

RERANK_MODEL = "qwen3:8b"          # content-judged eval (reports/judged_eval.json): +0.075 nDCG@5 vs +0.050 for llama3.2:3b; thinking disabled
LLM_WEIGHT = 1.0                 # pre-specified (not tuned on the evaluation): final = LLM/10 + 0.5 * normalised RRF tiebreak
RRF_WEIGHT = 0.5
_PROMPT = ("You judge search results. Query: {q}\nDocument: {d}\n"
           "How relevant is the document to the query? Answer with a single integer from 0 (irrelevant) to 10 (exactly what was asked). Integer only.")


def parse_score(text: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return min(max(float(m.group()), 0.0), 10.0) if m else 0.0     # unparseable = irrelevant, never an exception


def ollama_scorer(model: str = RERANK_MODEL, workers: int = 4, max_chars: int = 600, think: bool | None = None) -> Callable[[str, list[str]], list[float]]:
    if think is None and model.startswith("qwen3"):
        think = False                  # qwen3 would otherwise spend the 4-token budget on a <think> preamble and score everything 0

    def one(args) -> float:
        q, d = args
        payload = json.dumps({"model": model, "prompt": _PROMPT.format(q=q, d=d[:max_chars]), "stream": False,
                              "options": {"temperature": 0, "num_predict": 4}, **({"think": think} if think is not None else {})}).encode()
        for _ in range(3):
            try:
                req = urllib.request.Request(OLLAMA_GENERATE_URL, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return parse_score(json.loads(r.read()).get("response", ""))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        raise OllamaUnavailableError(f"rerank call failed: {last}")

    def score(query: str, texts: list[str]) -> list[float]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, [(query, t) for t in texts]))
    return score


def rerank(query: str, candidates: pd.DataFrame, scorer: Callable[[str, list[str]], list[float]], top_k: int | None = None) -> pd.DataFrame:
    """Reorder ``candidates`` (needs ``text`` and ``retrieval_score``); adds ``rerank_llm`` and ``rerank_score``."""
    if candidates.empty:
        return candidates.assign(rerank_llm=pd.Series(dtype=float), rerank_score=pd.Series(dtype=float))
    out = candidates.copy().reset_index(drop=True)
    llm = list(scorer(query, out["text"].tolist()))
    assert len(llm) == len(out), "scorer must return one score per candidate"
    out["rerank_llm"] = llm
    rrf = out["retrieval_score"].to_numpy(float)
    span = rrf.max() - rrf.min()
    rrf_n = (rrf - rrf.min()) / span if span > 0 else rrf * 0.0
    out["rerank_score"] = LLM_WEIGHT * out["rerank_llm"] / 10.0 + RRF_WEIGHT * rrf_n
    out = out.sort_values("rerank_score", ascending=False, kind="stable").reset_index(drop=True)
    return out.head(top_k) if top_k else out
