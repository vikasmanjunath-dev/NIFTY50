"""Announcement event-type classifier (13 classes: NSE categories grouped; results + board-meeting outcomes merged because NSE moved results filings
between those two categories in 2025). Input = the announcement text (the exact category string and exchange boilerplate are masked, as in training, so it
also works for texts that arrive without a category). Word + char TF-IDF -> logistic regression + calibrated linear SVM, probabilities averaged.
``predict`` abstains (label None) when the top probability is below ``threshold`` (chosen on a validation slice for >= 99% accepted accuracy)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import hstack

CLASSES = {"investor_meet": r"investor meet|con\. call|analyst|investor presentation|link of recording|earnings call|transcript",
           "share_admin": r"share certificate|letter of confirmation|duplicate|depositories and participants|74\(5\)",
           "results_or_board_outcome": r"financial result|result updates|quarterly|audited|unaudited|outcome of board|board meeting",
           "press_release": r"press release", "newspaper": r"newspaper", "esop_allotment": r"esop|esos|esps|allotment|stock option|restricted stock|rsu",
           "trading_window": r"trading window", "takeover_disclosure": r"takeover|regulation 31|substantial acquisition|disclosure under sebi",
           "credit_rating": r"credit rating|rating", "dividend_record_date": r"dividend|record date|book closure",
           "acquisition_mna": r"acquisition|amalgamation|merger|scheme of arrangement|demerger",
           "personnel": r"appointment|change in director|cessation|resignation|senior management|change in management",
           "shareholder_meeting": r"shareholders meeting|postal ballot|agm|egm|general meeting|scrutinizer|srutinizer"}
VAGUE = {"Updates", "General Updates", "General updates"}
MIN_CONTENT_CHARS = 25
_BOILER = re.compile(r"\b(has|have) (informed|submitted)( to)? the exchange( about| regarding| that)?\b", re.I)
_COMPANY = re.compile(r"^.*?\b(Limited|Ltd\.?|Corporation|Bank)\b[ ,.\-:]*", re.I)


def label_for(category: str) -> str | None:
    c = category.lower()
    return next((k for k, p in CLASSES.items() if re.search(p, c)), None)


def mask_text(text: str, category: str | None = None) -> str:
    t = text or ""
    if category:
        t = re.sub(re.escape(category), " ", t, flags=re.I)
    t = _BOILER.sub(" ", t)
    t = " ".join(re.sub(r"[()'\"]", " ", t).split())
    if re.search(r"\b(Limited|Ltd\.?)\b", t[:80], re.I):          # a leading company name carries no event information
        t = " ".join(_COMPANY.sub("", t, count=1).split())
    return t


@dataclass
class EventClassifier:
    threshold: float = 0.0
    classes_: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def _feats(self, texts, fit=False):
        from sklearn.feature_extraction.text import TfidfVectorizer
        if fit:
            self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000)
            return hstack([self.w.fit_transform(texts), self.c.fit_transform(texts)]).tocsr()
        return hstack([self.w.transform(texts), self.c.transform(texts)]).tocsr()

    def fit(self, texts, labels) -> "EventClassifier":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        X = self._feats(list(texts), fit=True)
        self.lr = LogisticRegression(max_iter=3000, C=20).fit(X, labels)
        self.svm = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3).fit(X, labels)
        self.classes_ = list(self.lr.classes_)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self._feats(list(texts))
        return (self.lr.predict_proba(X) + self.svm.predict_proba(X)) / 2

    def predict(self, texts) -> list[tuple[str | None, float]]:
        P = self.predict_proba(texts); i = P.argmax(1); p = P.max(1)
        return [(self.classes_[k] if q >= self.threshold else None, float(q)) for k, q in zip(i, p)]

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "EventClassifier":
        import joblib
        return joblib.load(path)
