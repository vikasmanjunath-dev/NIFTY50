import math
import pandas as pd
import pytest
from nifty_analyser.documents.judge import (consensus, export_label_sheet, graded_ndcg_at_k, load_human_labels, parse_grade, weighted_kappa)


def test_parse_grade_clamps_and_survives_garbage():
    assert parse_grade("3") == 3 and parse_grade("Grade: 2.") == 2 and parse_grade("") == 0 and parse_grade("nine") == 0


def test_graded_ndcg_perfect_and_worse_orders():
    g = {"a": 3, "b": 2, "c": 1, "z": 0}
    assert graded_ndcg_at_k(["a", "b", "c"], g, 3) == pytest.approx(1.0)
    assert graded_ndcg_at_k(["c", "b", "a"], g, 3) < 1.0
    assert graded_ndcg_at_k(["z", "q"], g, 3) == 0.0                       # unjudged / irrelevant = no gain
    assert math.isnan(graded_ndcg_at_k(["a"], {"a": 0}, 3))


def test_consensus_mean_and_human_override():
    c = consensus({"j1": {"a": 3, "b": 1}, "j2": {"a": 1, "b": 1}, "human": {"b": 0}})
    assert c == {"a": 2.0, "b": 0.0}


def test_weighted_kappa_perfect_and_disagreement():
    assert weighted_kappa([0, 1, 2, 3, 0, 3], [0, 1, 2, 3, 0, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0


def test_label_sheet_roundtrip(tmp_path):
    pool = pd.DataFrame({"query_id": ["1", "1"], "query": "q", "as_of": "x", "doc_id": ["d1", "d2"], "company": "c", "text": "t", "llm_grade": [2, 0]})
    p = export_label_sheet(pool, tmp_path / "s.csv")
    df = pd.read_csv(p, dtype=str); df.loc[0, "human_grade"] = "3"; df.to_csv(p, index=False)
    assert load_human_labels(p) == {"1": {"d1": 3.0}}
    df.loc[1, "human_grade"] = "7"; df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_human_labels(p)
