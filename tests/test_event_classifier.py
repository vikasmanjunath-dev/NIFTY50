import numpy as np, pytest
from fastapi.testclient import TestClient
from nifty_analyser.documents.event_classifier import EventClassifier, label_for, mask_text
from nifty_analyser.ops.api import create_app
import tests.test_ops as T


def test_label_grouping_and_masking():
    assert label_for("Outcome of Board Meeting") == label_for("Financial Result Updates") == "results_or_board_outcome" and label_for("Credit Rating- New") == "credit_rating"
    assert label_for("Something Unknown") is None
    m = mask_text("Tata Motors Limited has informed the Exchange about Credit Rating CRISIL AA+ assigned", "Credit Rating")
    assert "Credit Rating" not in m and "Tata Motors" not in m and "CRISIL AA+" in m
    assert len(mask_text("Wipro Limited", None)) < 25                                  # company name only: no event information left


def _fit():
    X = ["record date for dividend fixed on friday", "interim dividend declared per share", "credit rating assigned crisil aa stable", "icra reaffirmed rating long term facilities"] * 30
    y = ["dividend_record_date", "dividend_record_date", "credit_rating", "credit_rating"] * 30
    return EventClassifier(threshold=0.6).fit(X, y)


def test_predict_and_abstain_threshold():
    c = _fit(); assert c.predict(["dividend record date fixed"])[0][0] == "dividend_record_date"
    hi = EventClassifier(threshold=0.999999).fit(["alpha bravo charlie delta"] * 30 + ["foxtrot golf hotel india"] * 30, ["x"] * 30 + ["y"] * 30)
    assert hi.predict(["alpha foxtrot"])[0][0] is None                                           # ambiguous input below threshold -> abstains


def test_save_load_roundtrip_and_endpoint(tmp_path):
    c = _fit(); c.save(tmp_path / "m.joblib"); c2 = EventClassifier.load(tmp_path / "m.joblib")
    assert c2.predict(["credit rating assigned stable outlook"])[0][0] == "credit_rating"
    cl = TestClient(create_app(T.mk(tmp_path / "s"), classifier=c2))
    r = cl.post("/v3/classify", json={"text": "Acme Limited has informed the Exchange about Credit Rating: ICRA reaffirmed rating long term facilities", "category": "Credit Rating"}).json()
    assert r["status"] == "OK" and r["label"] == "credit_rating"
    assert cl.post("/v3/classify", json={"text": "Acme Limited"}).json()["status"] == "ABSTAIN"
    assert TestClient(create_app(T.mk(tmp_path / "s2"))).post("/v3/classify", json={"text": "abc def"}).status_code == 404
