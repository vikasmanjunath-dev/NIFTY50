"""One command to keep the forward test accruing (safe to run daily; a no-op between rebalances):
Yahoo bars refresh -> if a rebalance is due (>= hold_days trading days since the last book) rebuild serving features -> paper_trade.
Run: python3 scripts/daily_refresh.py      Cron (weekdays 19:00 IST):  0 19 * * 1-5 cd /Users/vkm/Desktop/STM && python3 scripts/daily_refresh.py >> data/processed/refresh.log 2>&1"""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.macro import MACRO_SERIES, ingest_macro
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.ops import paper
from nifty_analyser.ops.audit import AuditLog
run = lambda s: subprocess.run([sys.executable, str(root / "scripts" / s)], cwd=root, check=True)
run("backfill_bars.py")
for name in MACRO_SERIES:                                  # keeps India VIX etc. in step with the bars
    try:
        ingest_macro(name)
    except RuntimeError as exc:
        print(f"macro refresh failed, keeping cache: {exc}")
dates = sorted(str(d.date()) for d in load_universe_bars().event_time.unique())
bk = paper.books(AuditLog(DATA_PROCESSED_DIR / "paper_ledger.jsonl"))
if not bk or paper.is_due(AuditLog(DATA_PROCESSED_DIR / "paper_ledger.jsonl"), dates[-1], dates):
    print("rebalance due -> rebuilding serving artifacts"); run("build_serving_artifacts.py")
run("paper_trade.py")
run("forecast_volatility.py")
