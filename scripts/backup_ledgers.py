"""Copy the forward ledgers (they cannot be recreated) to a folder outside the project, verify the copies, keep the last 12.
Run: python3 scripts/backup_ledgers.py <folder>. The daily refresh runs it when data/processed/backup_dir.txt names a folder."""
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.ops.audit import AuditLog


def backup(source: Path, target: Path, stamp: str, keep: int = 12, project: Path = ROOT) -> list[str]:
    target = Path(target).expanduser().resolve()
    if target == project or project in target.parents:
        raise ValueError("choose a folder outside the project: a copy beside the original is not a backup")
    ledgers = sorted(Path(source).glob("*ledger*.jsonl"))
    if not ledgers:
        raise ValueError(f"no ledgers in {source}")
    folder = target / f"nifty-ledgers-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    broken = []
    for ledger in ledgers:
        shutil.copy2(ledger, folder / ledger.name)
        if not AuditLog(folder / ledger.name).verify()[0]:
            broken.append(ledger.name)
    if broken:  # keep older copies: they may be the last good ones
        raise ValueError(f"copied, but the hash chain does not verify: {', '.join(broken)}")
    for old in sorted(target.glob("nifty-ledgers-*"))[:-keep]:
        shutil.rmtree(old)
    return [ledger.name for ledger in ledgers]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        names = backup(DATA_PROCESSED_DIR, sys.argv[1], datetime.now().strftime("%Y%m%d-%H%M%S"))
    except (ValueError, OSError) as exc:
        sys.exit(f"backup failed: {exc}")
    print(f"backed up and verified: {', '.join(names)}")


if __name__ == "__main__":
    main()
