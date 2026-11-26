import numpy as np
import pandas as pd
import pytest
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed

TZ = "Asia/Kolkata"


class FakeEmbed:
    """Deterministic bag-of-letters embedder; counts how many texts it was asked to embed."""
    def __init__(self): self.calls = 0
    def __call__(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z": out[i, ord(ch) - 97] += 1
        return out


def _docs(n=6):
    t = pd.date_range("2026-01-01", periods=n, freq="D", tz=TZ)
    texts = ["dividend declared board", "loss of share certificate", "resignation of director", "dividend declared board", "merger approved", "results quarterly profit"]
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "text": texts[:n], "available_time": t, "instrument_id": ["A", "A", "B", "B", "A", "B"][:n]})


def test_dedup_incremental_and_persistence(tmp_path):
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe)
    assert ix.add(_docs()) == 5 and fe.calls == 5            # 'dividend declared board' repeats -> embedded once
    assert ix.add(_docs()) == 0 and fe.calls == 5            # nothing new -> nothing embedded
    extra = _docs().iloc[:1].assign(doc_id="new", text="brand new text", available_time=pd.Timestamp("2026-02-01", tz=TZ))
    assert ix.add(extra) == 1 and fe.calls == 6
    again = VectorIndex(tmp_path, fe)                         # reload from disk
    assert len(again) == 7 and again.add(_docs()) == 0 and fe.calls == 6


def test_search_is_point_in_time_and_matches_bruteforce(tmp_path):
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe); d = _docs(); ix.add(d)
    q = fe(["dividend declared"])[0]
    as_of = d["available_time"].iloc[2] + pd.Timedelta(hours=1)        # d3..d5 not yet published
    res = ix.search(q, as_of, 10)
    assert {r[0] for r in res} == {"d0", "d1", "d2"}
    E = fe(d["text"].tolist()[:3]); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    brute = np.argsort(-(E @ (q / np.linalg.norm(q))), kind="stable")
    assert [r[0] for r in res] == [f"d{i}" for i in brute]
    assert ix.search(q, d["available_time"].iloc[0] - pd.Timedelta(days=1)) == []
    assert {r[0] for r in ix.search(q, d["available_time"].max(), 10, allowed_doc_ids={"d4"})} == {"d4"}


def test_hybrid_indexed_respects_pit_and_instrument_and_embeds_only_query(tmp_path):
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe); d = _docs(); ix.add(d); before = fe.calls
    as_of = d["available_time"].iloc[3] + pd.Timedelta(hours=1)
    out = hybrid_retrieve_indexed("dividend declared", d, as_of, ix, top_k=5, instrument_id="A")
    assert fe.calls == before + 1                                       # only the query was embedded
    assert set(out["doc_id"]) <= {"d0", "d1"} and out["doc_id"].iloc[0] == "d0"
    assert hybrid_retrieve_indexed("x", d, d["available_time"].min() - pd.Timedelta(days=1), ix).empty


def test_retrieve_with_and_without_rerank_feeds_evidence_pack(tmp_path):
    from nifty_analyser.documents.evidence import build_evidence_pack
    from nifty_analyser.documents.index import retrieve
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe); d = _docs(); ix.add(d)
    as_of = d["available_time"].max() + pd.Timedelta(hours=1)
    plain = retrieve("dividend declared", d, as_of, ix, top_k=3)
    boosted = retrieve("dividend declared", d, as_of, ix, top_k=3, scorer=lambda q, t: [10 if "merger" in x else 0 for x in t])
    assert len(plain) == 3 and boosted["doc_id"].iloc[0] == "d4" and len(boosted) == 3     # LLM favourite promoted, nothing outside candidates
    pack = build_evidence_pack("dividend declared", as_of, boosted)
    assert pack.temporal_valid and len(pack.evidence_items) == 3


def test_retrieval_modes_are_ablations_of_the_same_pipeline(tmp_path):
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe); d = _docs(); ix.add(d)
    as_of = d["available_time"].max() + pd.Timedelta(hours=1)
    for mode in ("rrf", "bm25", "dense"):
        assert not hybrid_retrieve_indexed("dividend declared", d, as_of, ix, top_k=3, mode=mode).empty
    with pytest.raises(KeyError):
        hybrid_retrieve_indexed("x", d, as_of, ix, mode="nope")


def test_weighted_rrf_default_is_classic_and_weight_shifts_ranking():
    from nifty_analyser.documents.retrieval import reciprocal_rank_fusion as rrf
    lists = [["a", "b", "c"], ["c", "b", "a"]]
    assert rrf(lists) == rrf(lists, weights=[1.0, 1.0])
    assert rrf(lists, weights=[3.0, 1.0])[0][0] == "a" and rrf(lists, weights=[1.0, 3.0])[0][0] == "c"


def test_retrieve_auto_weights_unfiltered_is_dense_only_filtered_is_bm25_heavy(tmp_path):
    from nifty_analyser.documents.index import retrieve
    fe = FakeEmbed(); ix = VectorIndex(tmp_path, fe); d = _docs(); ix.add(d)
    as_of = d["available_time"].max() + pd.Timedelta(hours=1)
    auto = retrieve("dividend declared", d, as_of, ix, top_k=4)
    dense = hybrid_retrieve_indexed("dividend declared", d, as_of, ix, top_k=4, mode="dense")
    assert auto["doc_id"].tolist() == dense["doc_id"].tolist()                 # w=0 reproduces dense-only ordering
    f_auto = retrieve("dividend declared", d, as_of, ix, top_k=3, instrument_id="A")
    f_bm25 = hybrid_retrieve_indexed("dividend declared", d, as_of, ix, top_k=3, instrument_id="A", bm25_weight=16.0)
    assert f_auto["doc_id"].tolist() == f_bm25["doc_id"].tolist()


def test_retrieve_strips_the_instrument_ticker_from_the_query_when_filtering(tmp_path):
    from nifty_analyser.documents.index import retrieve
    seen = []
    class Spy(FakeEmbed):
        def __call__(self, texts): seen.extend(texts); return super().__call__(texts)
    ix = VectorIndex(tmp_path, Spy()); d = _docs().assign(instrument_id="NSE:INFY"); ix.add(d); seen.clear()
    retrieve("INFY dividend declared", d, d["available_time"].max() + pd.Timedelta(hours=1), ix, top_k=2, instrument_id="NSE:INFY")
    assert seen == ["dividend declared"]
