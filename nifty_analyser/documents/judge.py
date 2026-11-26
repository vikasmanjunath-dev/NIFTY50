"""Content-based relevance judgments for retrieval evaluation (replaces metadata-derived relevance).

Graded 0-3 relevance of a document to a natural-language query, produced by one or more local LLM judges and/or humans.
Human grades (``human_grade`` column of the exported CSV, see ``export_label_sheet`` / ``load_human_labels``) always
override model grades. Judges are injectable so tests need no LLM. Pooled evaluation: only judged documents carry gain
(unjudged = 0), the standard TREC pooling convention; the ideal DCG is taken over the judged pool.
"""
from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from nifty_analyser.documents.llm import OLLAMA_GENERATE_URL, OllamaUnavailableError

GRADES = "0 = unrelated or only the same company; 1 = loosely related; 2 = relevant (clearly about the topic); 3 = directly answers the query"
_PROMPT = ("You are a strict relevance judge for a financial-announcement search engine.\n"
           "Query: {q}\nAnnouncement ({when}, {company}): {d}\n"
           f"Grade relevance of the announcement to the query: {GRADES}.\n"
           "Judge only the announcement text. Answer with a single digit 0-3.")


def parse_grade(text: str) -> float:
    m = re.search(r"[0-3]", text or "")
    return float(m.group()) if m else 0.0          # unparseable = unrelated, never an exception


def ollama_judge(model: str, workers: int = 4, max_chars: int = 700) -> Callable[[str, list[dict]], list[float]]:
    """``judge(query, [{"text","when","company"}...]) -> grades`` using a local Ollama model (thinking disabled)."""
    def one(args) -> float:
        q, d = args
        payload = json.dumps({"model": model, "think": False, "stream": False,
                              "prompt": _PROMPT.format(q=q, d=d["text"][:max_chars], when=d.get("when", "?"), company=d.get("company", "?")),
                              "options": {"temperature": 0, "num_predict": 4}}).encode()
        last = None
        for _ in range(3):
            try:
                req = urllib.request.Request(OLLAMA_GENERATE_URL, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return parse_grade(json.loads(r.read()).get("response", ""))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        raise OllamaUnavailableError(f"judge call failed: {last}")

    def judge(query: str, docs: list[dict]) -> list[float]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, [(query, d) for d in docs]))
    return judge


def graded_ndcg_at_k(retrieved_ids: list[str], gains: dict[str, float], k: int) -> float:
    """nDCG@k with graded gains (2**g - 1); ideal ordering is over every judged doc of the query. NaN if nothing is relevant."""
    gain = lambda g: 2.0 ** g - 1.0
    ideal_gains = sorted((gain(g) for g in gains.values() if g > 0), reverse=True)[:k]
    if not ideal_gains:
        return float("nan")
    dcg = sum(gain(gains.get(d, 0.0)) / math.log2(i + 2) for i, d in enumerate(retrieved_ids[:k]))
    return dcg / sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))


def consensus(grades: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean grade over judges per doc: ``{judge: {doc_id: grade}} -> {doc_id: mean}``; human grades, if present under key
    ``"human"``, replace the model consensus for the docs they cover."""
    models = [g for j, g in grades.items() if j != "human"]
    docs = set().union(*[set(g) for g in models]) if models else set()
    out = {d: float(np.mean([g[d] for g in models if d in g])) for d in docs}
    out.update(grades.get("human", {}))
    return out


def weighted_kappa(a: Iterable[float], b: Iterable[float], n_grades: int = 4) -> float:
    """Quadratic-weighted Cohen's kappa between two judges' integer grades (1 = perfect, 0 = chance)."""
    a = np.asarray(list(a), int); b = np.asarray(list(b), int)
    if len(a) == 0:
        return float("nan")
    O = np.zeros((n_grades, n_grades))
    for x, y in zip(a, b):
        O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 for j in range(n_grades)] for i in range(n_grades)], float) / (n_grades - 1) ** 2
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else float("nan")


def export_label_sheet(pool: pd.DataFrame, path: Path, sample_per_query: int | None = None, seed: int = 0) -> Path:
    """CSV for a human to fill ``human_grade`` (0-3). ``pool`` columns: query_id, query, as_of, doc_id, company, text, llm_grade.
    Model grades are included only as a hint column and can be hidden for blind labelling by dropping ``llm_grade``."""
    sheet = pool.copy()
    if sample_per_query:
        sheet = sheet.groupby("query_id", group_keys=False).apply(lambda g: g.sample(min(len(g), sample_per_query), random_state=seed))
    sheet = sheet.sort_values(["query_id", "doc_id"]).assign(human_grade="")
    sheet.to_csv(path, index=False)
    return path


def load_human_labels(path: Path) -> dict[str, dict[str, float]]:
    """``{query_id: {doc_id: grade}}`` from a filled label sheet; blank/invalid cells are skipped, out-of-range rejected."""
    df = pd.read_csv(path, dtype={"query_id": str, "doc_id": str})
    out: dict[str, dict[str, float]] = {}
    for r in df.itertuples():
        v = pd.to_numeric(getattr(r, "human_grade"), errors="coerce")
        if pd.isna(v):
            continue
        if not 0 <= v <= 3:
            raise ValueError(f"human_grade out of range 0-3 for doc {r.doc_id}: {v}")
        out.setdefault(str(r.query_id), {})[str(r.doc_id)] = float(v)
    return out
