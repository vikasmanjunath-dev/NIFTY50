"""Pretrained natural-language-inference faithfulness check (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, safetensors, runs locally).
Premise = the evidence the model was shown (dated, categorised); hypothesis = each factual sentence of the answer. A lexical check cannot tell that
"announced a share buyback" is unsupported by evidence that happens to contain the word "share"; an NLI model can. Heavy imports are deferred."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, use_safetensors=True).eval()        # refuse pickle weights
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, model.to(dev), dev, {v.lower(): k for k, v in model.config.id2label.items()}


def entailment_probs(premise: str, hypotheses: list[str]) -> np.ndarray:
    """(n, 3) probabilities ordered [entailment, neutral, contradiction]."""
    import torch
    tok, model, dev, lab = _load()
    if not hypotheses:
        return np.zeros((0, 3))
    enc = tok([premise] * len(hypotheses), hypotheses, truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    return p[:, [lab["entailment"], lab["neutral"], lab["contradiction"]]]


def split_claims(answer: str) -> list[str]:
    """Factual sentences of an answer: citation markers removed; pure hedges / refusals ('details are not stated', 'insufficient evidence') dropped."""
    a = re.sub(r"\[[^\]]*\]", " ", answer or "")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", a) if len(s.strip()) > 12]
    hedge = re.compile(r"\b(not (specif|stated|provided|disclosed|mentioned|available)|no (specific )?(details|information)|insufficient evidence|cannot (answer|determine|confirm))", re.I)
    return [s for s in sents if not (hedge.search(s) and not re.search(r"\d", s))]


def premise_from_pack(pack) -> str:
    return " ".join(f"On {i.publication_time[:10]}: {i.claim_span} (category: {i.category})." for i in pack.evidence_items)


def nli_check(answer: str, pack, entail_min: float = 0.5, contra_max: float = 0.5) -> dict:
    """Pass when every factual sentence is entailed (P >= entail_min) and none is contradicted (P(contra) <= contra_max)."""
    claims = split_claims(answer)
    if not claims or pack.is_empty:
        return {"passed": True, "n_claims": 0, "min_entail": None, "max_contra": None, "worst_claim": None}
    p = entailment_probs(premise_from_pack(pack), claims); i = int(p[:, 0].argmin())
    return {"passed": bool(p[:, 0].min() >= entail_min and p[:, 2].max() <= contra_max), "n_claims": len(claims), "min_entail": float(p[:, 0].min()), "max_contra": float(p[:, 2].max()), "worst_claim": claims[i]}
