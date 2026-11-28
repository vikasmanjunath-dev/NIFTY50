"""One command to keep the forward test accruing (safe to run daily; a no-op between rebalances):
Yahoo bars refresh -> if a rebalance is due (>= hold_days trading days since the last book) rebuild serving features -> paper_trade.
Run: python3 scripts/daily_refresh.py      Cron (weekdays 19:00 IST):  0 19 * * 1-5 cd /Users/vkm/Desktop/STM && python3 scripts/daily_refresh.py >> data/processed/refresh.log 2>&1"""
import subprocess, sys
from datetime import datetime
from pathlib import Path
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.macro import MACRO_SERIES, ingest_macro
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.ops import paper
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.refresh import single_instance
from nifty_analyser.ops.runlog import append_run
failed: list[str] = []


def run(script, *args):
    try:
        subprocess.run([sys.executable, str(root / "scripts" / script), *args], cwd=root, check=True)
    except subprocess.CalledProcessError:
        failed.append(script)
        raise


def main():
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
    try:
        run("backfill_board_meetings.py")                      # newly announced meeting dates
    except subprocess.CalledProcessError as exc:
        print(f"board-meeting refresh failed, keeping cache: {exc}")
    try:
        run("refresh_announcements.py")                        # new company filings
    except subprocess.CalledProcessError as exc:
        print(f"filings refresh failed, keeping cache: {exc}")
    run("forecast_volatility.py")
    destination = DATA_PROCESSED_DIR / "backup_dir.txt"   # one line: a folder outside the project
    if destination.exists():
        try:
            run("backup_ledgers.py", destination.read_text().strip())
        except subprocess.CalledProcessError as exc:
            print(f"ledger backup failed: {exc}")


with single_instance(DATA_PROCESSED_DIR / "refresh.lock") as mine:
    if not mine:
        sys.exit("another refresh is already running")
    started, ok = datetime.now().isoformat(timespec="seconds"), False
    try:
        main()
        ok = not failed
    finally:
        append_run(DATA_PROCESSED_DIR / "refresh_runs.jsonl", started, datetime.now().isoformat(timespec="seconds"), ok, failed)
