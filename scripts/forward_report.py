"""Write a plain-text report of the forward record (data/processed/forward_report.md): what has been forecast, what has matured,
and how it scored. Run: python3 scripts/forward_report.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.vol_forecast import score


def render(ledger, scores: dict) -> str:
    entries = [e["payload"] for e in ledger.read() if e["kind"] == "vol_forecast"]
    lines = ["# Forward record", "", f"Hash chain intact: {'yes' if ledger.verify()[0] else 'NO'}", "",
             "| Data date | Horizon | Stocks | Model |", "|---|---|---|---|"]
    lines += [f"| {p['date']} | {p['horizon_days']}-day | {len(p['forecasts'])} | {p.get('model', '-')} |" for p in entries]
    lines += ["", "## Scores (matured forecasts only)", ""]
    if not scores:
        lines.append("Nothing has matured yet.")
    for horizon, s in sorted(scores.items(), key=lambda kv: int(kv[0])):
        auc = "-" if s["auc"] is None else f"{s['auc']:.3f}"
        cover = f", price range held {s['coverage_90']:.0%}" if "coverage_90" in s else ""
        lines.append(f"- {horizon}-day: {s['n']} graded, accuracy {s['accuracy']:.1%}, AUC {auc}{cover}")
    dates = len({p["date"] for p in entries})
    lines += ["", f"Only {dates} data date(s) recorded. Forecasts made on nearby dates overlap, so judge the model after months, not days."]
    return "\n".join(lines) + "\n"


def main():
    ledger = AuditLog(DATA_PROCESSED_DIR / "vol_ledger.jsonl")
    text = render(ledger, score(ledger, load_universe_bars()))
    (DATA_PROCESSED_DIR / "forward_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
