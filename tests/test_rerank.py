import math
import pandas as pd
import pytest
from nifty_analyser.documents.eval import ndcg_at_k
from nifty_analyser.documents.rerank import parse_score, rerank


def _cands():
    return pd.DataFrame({"doc_id": list("abcd"), "text": ["x", "y", "z", "w"], "retrieval_score": [0.04, 0.03, 0.02, 0.01]})


def test_parse_score_is_robust():
    assert parse_score("8") == 8 and parse_score("Score: 7.5/10") == 7.5 and parse_score("15") == 10.0
    assert parse_score("none") == 0.0 and parse_score("") == 0.0 and parse_score(None) == 0.0


def test_rerank_promotes_llm_favourite_and_never_adds_or_drops_docs():
    out = rerank("q", _cands(), lambda q, t: [0, 0, 10, 0])
    assert out["doc_id"].iloc[0] == "c" and sorted(out["doc_id"]) == list("abcd")


def test_rerank_ties_fall_back_to_first_stage_order():
    out = rerank("q", _cands(), lambda q, t: [5, 5, 5, 5])
    assert out["doc_id"].tolist() == list("abcd")


def test_rerank_handles_empty_single_and_top_k():
    assert rerank("q", _cands().iloc[0:0], lambda q, t: []).empty
    assert len(rerank("q", _cands().iloc[:1], lambda q, t: [3])) == 1
    assert len(rerank("q", _cands(), lambda q, t: [1, 2, 3, 4], top_k=2)) == 2
    with pytest.raises(AssertionError):
        rerank("q", _cands(), lambda q, t: [1])


def test_ndcg():
    assert ndcg_at_k(["a", "b"], {"a"}, 5) == 1.0
    assert ndcg_at_k(["b", "a"], {"a"}, 5) == pytest.approx(1 / math.log2(3))
    assert math.isnan(ndcg_at_k(["a"], set(), 5))


def test_qwen3_scorer_disables_thinking_by_default(monkeypatch):
    import io, json as _json
    from nifty_analyser.documents import rerank as rr
    sent = []

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=0):
        sent.append(_json.loads(req.data)); return _R(b'{"response": "7"}')
    monkeypatch.setattr(rr.urllib.request, "urlopen", fake)
    assert rr.ollama_scorer("qwen3:8b", workers=1)("q", ["d"]) == [7.0]
    assert sent[0]["think"] is False
    rr.ollama_scorer("llama3.2:3b", workers=1)("q", ["d"])
    assert "think" not in sent[1]


def test_cross_encoder_scores_map_into_zero_to_ten_and_stay_monotone():
    from nifty_analyser.documents.crossencoder import sigmoid10
    s = sigmoid10([-8.0, 0.0, 8.0])
    assert s[0] < s[1] == 5.0 < s[2] and 0 < s[0] and s[2] < 10
