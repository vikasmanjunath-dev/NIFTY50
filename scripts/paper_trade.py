"""Record today's pre-registered paper book if a rebalance is due (never edits history), then score the ledger so far.
Two pre-registered arms accrue in parallel: "production" (registered model) and "blend" (LightGBM+CatBoost logit mean).
Run: python3 scripts/paper_trade.py"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.dataset import liquidity_panel
from nifty_analyser.ops import paper
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.registry import load_model

from nifty_analyser.eval.dataset import load_pit_dataset
from nifty_analyser.models.ensemble import fit_blend

feats = pd.read_parquet(DATA_PROCESSED_DIR / "serving_features.parquet")
for c in ("event_time", "available_time"):
    feats[c] = pd.to_datetime(feats[c], utc=True).dt.tz_convert("Asia/Kolkata")
as_of = feats["available_time"].max(); dates = sorted(str(d.date()) for d in feats["event_time"].unique())
bars = load_universe_bars(); liq = liquidity_panel(bars)
model, meta = load_model(DATA_PROCESSED_DIR / "models")
cols = meta["feature_cols"]


def blend_model():
    df, c = load_pit_dataset(5)
    df = df[df["fwd_ret_5d_outcome_available_time"] <= as_of]       # only outcomes already public at decision time
    return fit_blend(df[c].to_numpy(float), df["y"].to_numpy(), random_state=42), f"blend-{len(df)}rows-{df.event_time.max().date()}"


for arm, path, mk, sh in (("production", "paper_ledger.jsonl", lambda: (model, meta["version"]), paper.SPEC_HASH),
                          ("blend", "paper_ledger_blend.jsonl", blend_model, paper.SPEC_BLEND_HASH)):
    ledger = AuditLog(DATA_PROCESSED_DIR / path)
    m, ver = mk()
    book = paper.target_book(m, feats, cols, as_of)
    if paper.is_due(ledger, book["decision_date"], dates):
        e = paper.record_book(ledger, book, ver, sh); print(f"[{arm}] recorded book seq={e['seq']} decision={book['decision_date']} long={book['long']} short={book['short']}")
    else:
        print(f"[{arm}] no rebalance due; last book", paper.books(ledger)[-1]["decision_date"])
    print(f"[{arm}] ledger intact:", ledger.verify())
    r = paper.realized(ledger, bars, liq)
    print(f"[{arm}] realised days: {len(r)}" + (f", gross {r.gross.sum():+.4f}, net {r.net.sum():+.4f}" if len(r) else " (no prices after the decision date yet)"))
