import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from backup_ledgers import backup

from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.runlog import append_run, missed_days, recent_runs


def test_run_log_keeps_newest_first_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "r.jsonl"
    assert recent_runs(path) == []
    for i in range(9):
        append_run(path, f"s{i}", f"f{i}", i != 8, [] if i != 8 else ["backfill_bars.py"])
    path.open("a").write('{"tor')
    runs = recent_runs(path)
    assert len(runs) == 7 and runs[0]["failed"] == ["backfill_bars.py"] and runs[1]["ok"]


def test_missed_days_start_at_the_first_record_and_skip_the_latest():
    days = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    assert missed_days(days, {"2026-01-02", "2026-01-06"}) == ["2026-01-05"] and missed_days(days, set()) == []


def test_backup_copies_verifies_prunes_and_refuses_unsafe_targets(tmp_path):
    source, target, project = tmp_path / "proj" / "data", tmp_path / "safe", tmp_path / "proj"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="no ledgers"):
        backup(source, target, "0", project=project)
    AuditLog(source / "vol_ledger.jsonl").append("x", {})
    with pytest.raises(ValueError, match="outside the project"):
        backup(source, project / "copy", "0", project=project)
    for i in range(14):
        assert backup(source, target, f"{i:02d}", project=project) == ["vol_ledger.jsonl"]
    kept = sorted(p.name for p in target.iterdir())
    assert len(kept) == 12 and kept[-1] == "nifty-ledgers-13"
    assert AuditLog(target / kept[-1] / "vol_ledger.jsonl").verify() == (True, None)
    (source / "vol_ledger.jsonl").open("a").write("junk\n")
    with pytest.raises(ValueError, match="does not verify"):
        backup(source, target, "99", project=project)
    assert len(list(target.iterdir())) == 13  # nothing pruned when the chain is broken


def test_daily_refresh_logs_the_run_and_failed_steps_without_touching_real_data(tmp_path, monkeypatch):
    import runpy
    import subprocess

    import pandas as pd

    import nifty_analyser.config as config
    import nifty_analyser.data.macro as macro
    import nifty_analyser.data.store as store
    import nifty_analyser.ops.paper as paper

    ran = []

    def fake_run(cmd, **kw):
        ran.append(Path(cmd[1]).name)
        if cmd[1].endswith("refresh_announcements.py"):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(macro, "ingest_macro", lambda name: None)
    monkeypatch.setattr(store, "load_universe_bars", lambda: pd.DataFrame({"event_time": pd.to_datetime(["2026-01-02"])}))
    monkeypatch.setattr(paper, "books", lambda log: ["book"])
    monkeypatch.setattr(paper, "is_due", lambda *a: False)
    (tmp_path / "backup_dir.txt").write_text("/somewhere\n")
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "daily_refresh.py"), run_name="__main__")
    assert ran[0] == "backfill_bars.py" and ran[-2:] == ["forecast_volatility.py", "backup_ledgers.py"]
    (run,) = recent_runs(tmp_path / "refresh_runs.jsonl")
    assert run["ok"] is False and run["failed"] == ["refresh_announcements.py"]
