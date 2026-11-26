"""Persistent, incremental, point-in-time vector index (closes Known Simplification #23: ``hybrid_retrieve`` re-embedded its
whole eligible pool on every call, impossible at corpus scale). Documents are embedded ONCE (deduplicated by text hash --
boilerplate announcements repeat thousands of times), stored as a normalised float32 matrix + a metadata table, and only new
documents are ever embedded afterwards. ``search`` masks by ``available_time <= as_of`` BEFORE ranking, so the index cannot
leak the future; the PIT guarantee is the same as ``hybrid_retrieve``'s."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from nifty_analyser.documents.embeddings import EMBEDDING_MODEL
from nifty_analyser.documents.retrieval import BM25Index, reciprocal_rank_fusion

AUTO_BM25_WEIGHT_UNFILTERED = 0.0     # whole corpus: BM25 hurts at scale -> dense-only ranking
AUTO_BM25_WEIGHT_FILTERED = 1.0       # classic RRF: on content-judged topical queries w=1 beat w=16 by +0.056 +/- 0.028 nDCG@5 (reports/judged_eval.json); w=16 was tuned on literal ticker+category queries and does not transfer
OLLAMA_EMBED_BATCH_URL = "http://localhost:11434/api/embed"


def _h(text: str, model: str = EMBEDDING_MODEL) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode()).hexdigest()


def make_ollama_embedder(model: str = EMBEDDING_MODEL) -> Callable[[list[str]], np.ndarray]:
    def embed(texts: list[str]) -> np.ndarray:
        req = urllib.request.Request(OLLAMA_EMBED_BATCH_URL, data=json.dumps({"model": model, "input": texts}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32)
    return embed


ollama_embed_batch = make_ollama_embedder()


class VectorIndex:
    def __init__(self, directory, embed_batch: Callable[[list[str]], np.ndarray] = ollama_embed_batch, model: str = EMBEDDING_MODEL):
        self.dir, self.embed_batch, self.model = Path(directory), embed_batch, model
        self.hashes: list[str] = []
        self.E = np.zeros((0, 0), dtype=np.float32)
        self.docs = pd.DataFrame({"doc_id": pd.Series(dtype=str), "hash": pd.Series(dtype=str), "available_time": pd.Series(dtype="datetime64[ns, Asia/Kolkata]")})
        if (self.dir / "docs.parquet").exists():
            self.docs = pd.read_parquet(self.dir / "docs.parquet")
            self.hashes = json.loads((self.dir / "hashes.json").read_text())
            self.E = np.load(self.dir / "E.npy")
        self._rebuild()

    def _rebuild(self) -> None:
        self._row = {h: i for i, h in enumerate(self.hashes)}
        self._doc_rows = self.docs["hash"].map(self._row).to_numpy() if len(self.docs) else np.zeros(0, dtype=int)
        self._seen = set(self.docs["doc_id"])

    def __len__(self) -> int:
        return len(self.docs)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "E.npy", self.E)
        (self.dir / "hashes.json").write_text(json.dumps(self.hashes))
        self.docs.to_parquet(self.dir / "docs.parquet")

    def add(self, documents: pd.DataFrame, batch: int = 64, save_every: int = 4096) -> int:
        """Index documents (doc_id, text, available_time) not yet present; embeds only unseen *texts*. Returns #texts embedded."""
        new = documents[~documents["doc_id"].isin(self._seen)].drop_duplicates("doc_id")
        if new.empty:
            return 0
        hs = new["text"].map(lambda t: _h(t, self.model))
        todo = pd.Series(new["text"].to_numpy(), index=hs.to_numpy())
        todo = todo[~todo.index.duplicated() & ~todo.index.isin(self._row)]
        embedded, since_save = 0, 0
        for s in range(0, len(todo), batch):
            chunk = todo.iloc[s:s + batch]
            v = np.asarray(self.embed_batch(chunk.tolist()), dtype=np.float32)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
            self.E = v if self.E.size == 0 else np.vstack([self.E, v])
            self.hashes += chunk.index.tolist()
            embedded += len(chunk); since_save += len(chunk)
            if since_save >= save_every:                              # resumable: texts are saved before docs are registered
                self._rebuild(); self.save(); since_save = 0
        add = pd.DataFrame({"doc_id": new["doc_id"].to_numpy(), "hash": hs.to_numpy(), "available_time": pd.to_datetime(new["available_time"]).to_numpy()})
        self.docs = pd.concat([self.docs, add], ignore_index=True)
        self._rebuild()
        self.save()
        return embedded

    def search(self, query_vec: np.ndarray, as_of: pd.Timestamp, top_k: int = 50, allowed_doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not len(self.docs):
            return []
        mask = (self.docs["available_time"] <= as_of).to_numpy()                       # point-in-time BEFORE ranking
        if allowed_doc_ids is not None:
            mask &= self.docs["doc_id"].isin(allowed_doc_ids).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32); q = q / (np.linalg.norm(q) + 1e-10)
        scores = self.E[self._doc_rows[idx]] @ q
        top = np.argsort(-scores, kind="stable")[:top_k]
        ids = self.docs["doc_id"].to_numpy()[idx[top]]
        return list(zip(ids.tolist(), scores[top].tolist()))


class CategoryRouter:
    """Maps a question onto the most likely NSE announcement categories by embedding similarity between the question and the category NAMES
    (e.g. "share buyback" -> "Buyback", "dividend declaration" -> "Dividend"). Used to run a category-restricted search that is fused with the normal one."""

    def __init__(self, categories: list[str], embed_batch: Callable[[list[str]], np.ndarray]):
        self.categories = sorted({c for c in categories if isinstance(c, str) and c.strip()}, key=str.lower)
        self.embed_batch = embed_batch
        E = np.asarray(embed_batch(self.categories), dtype=np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

    def route(self, query: str, top_n: int = 2, min_sim: float = 0.0, query_vec: np.ndarray | None = None) -> list[str]:
        q = np.asarray(query_vec if query_vec is not None else self.embed_batch([query])[0], dtype=np.float32); q = q / (np.linalg.norm(q) + 1e-10)
        sims = self.E @ q
        return [self.categories[i] for i in np.argsort(-sims)[:top_n] if sims[i] >= min_sim]


def hybrid_retrieve_indexed(query: str, documents: pd.DataFrame, as_of: pd.Timestamp, index: VectorIndex, top_k: int = 8,
                            candidate_k: int = 50, instrument_id: str | None = None, query_vec: np.ndarray | None = None, mode: str = "rrf", bm25_weight: float = 1.0) -> pd.DataFrame:
    """BM25 (over the eligible subset) + indexed dense search -> RRF. No document is embedded at query time except the query."""
    elig = documents[documents["available_time"] <= as_of]
    if instrument_id is not None:
        elig = elig[elig["instrument_id"] == instrument_id]
    if elig.empty:
        return elig.assign(retrieval_score=pd.Series(dtype=float))
    bm = BM25Index.build(elig["doc_id"].tolist(), elig["text"].tolist()).score(query)
    bm_ranked = [d for d, _ in sorted(bm.items(), key=lambda x: x[1], reverse=True)[:candidate_k]]
    dense_ranked: list[str] = []
    if mode != "bm25":                      # BM25-only needs no embedding (and must work while the embedder is down)
        qv = query_vec if query_vec is not None else index.embed_batch([query])[0]
        dense_ranked = [d for d, _ in index.search(qv, as_of, candidate_k, allowed_doc_ids=set(elig["doc_id"]))]
    lists = {"rrf": [bm_ranked, dense_ranked], "bm25": [bm_ranked], "dense": [dense_ranked]}[mode]    # single-list modes are ablations
    fused = reciprocal_rank_fusion(lists, weights=[bm25_weight, 1.0] if mode == "rrf" else None)[:top_k]
    scores = dict(fused)
    res = elig[elig["doc_id"].isin(scores)].copy()
    res["retrieval_score"] = res["doc_id"].map(scores)
    return res.sort_values("retrieval_score", ascending=False).reset_index(drop=True)


def retrieve(query: str, documents: pd.DataFrame, as_of: pd.Timestamp, index: VectorIndex, top_k: int = 8, instrument_id: str | None = None,
             scorer: Callable[[str, list[str]], list[float]] | None = None, rerank_depth: int = 20, bm25_weight: float | None = None,
             recency_weight: float = 0.0, recency_pool: int = 20, router: "CategoryRouter | None" = None, route_top_n: int = 2, route_weight: float = 1.0,
             route_min_sim: float = 0.0) -> pd.DataFrame:
    """Production entry point: indexed hybrid retrieval, optionally followed by LLM reranking of the top ``rerank_depth``
    candidates (``scorer`` given). The result feeds ``evidence.build_evidence_pack`` unchanged."""
    if instrument_id:           # the instrument is already enforced by the filter; its ticker in the text only adds noise
        from nifty_analyser.data.instrument_master import get_instrument
        try:
            symbol = get_instrument(instrument_id).symbol
        except KeyError:                                   # id outside the instrument master: leave the query alone
            symbol = None
        if symbol:
            query = " ".join(re.sub(rf"\b{re.escape(symbol)}\b", " ", query, flags=re.IGNORECASE).split()) or query
    if bm25_weight is None:     # chosen on a tune split of 100 queries, confirmed on 100 held-out (reports/fusion_study.json)
        bm25_weight = AUTO_BM25_WEIGHT_FILTERED if instrument_id else AUTO_BM25_WEIGHT_UNFILTERED
    use_rec = recency_weight > 0
    cand = hybrid_retrieve_indexed(query, documents, as_of, index, top_k=max(top_k, rerank_depth, recency_pool if use_rec else 0) if (scorer or use_rec) else top_k,
                                   instrument_id=instrument_id, bm25_weight=bm25_weight)
    if router is not None and not cand.empty:   # category routing: fuse the normal candidate list with a search restricted to the routed categories
        routed = router.route(query, route_top_n, route_min_sim)
        sub = documents[documents["category"].isin(routed)] if routed and "category" in documents else documents.iloc[0:0]
        if not sub.empty:
            depth = max(top_k, rerank_depth, recency_pool if use_rec else 0, 20)
            c2 = hybrid_retrieve_indexed(query, sub, as_of, index, top_k=depth, instrument_id=instrument_id, bm25_weight=bm25_weight)
            if not c2.empty:
                fused = dict(reciprocal_rank_fusion([cand["doc_id"].tolist(), c2["doc_id"].tolist()], weights=[1.0, route_weight]))
                allc = pd.concat([cand, c2]).drop_duplicates("doc_id"); allc["retrieval_score"] = allc["doc_id"].map(fused)
                cand = allc.sort_values("retrieval_score", ascending=False, kind="stable").head(depth if (scorer or use_rec) else top_k).reset_index(drop=True)
    if use_rec and not cand.empty:     # blend relevance with recency (newest = 1) inside the candidate pool; PIT filtering already happened upstream
        sc = cand["retrieval_score"].to_numpy(float); span = sc.max() - sc.min()
        rel = (sc - sc.min()) / span if span > 0 else np.ones_like(sc)
        rec = cand["available_time"].rank(method="first").to_numpy(float); rec = (rec - 1) / max(len(rec) - 1, 1)
        cand = cand.assign(retrieval_score=rel + recency_weight * rec).sort_values("retrieval_score", ascending=False, kind="stable")
        if not scorer:
            cand = cand.head(top_k)
    if scorer is None or cand.empty:
        return cand
    from nifty_analyser.documents.rerank import rerank
    return rerank(query, cand, scorer, top_k=top_k)
