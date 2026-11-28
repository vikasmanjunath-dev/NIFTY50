"""What each daily refresh did (data/processed/refresh_runs.jsonl) and which trading days the forward record is missing."""
import json
from pathlib import Path


def append_run(path, started: str, finished: str, ok: bool, failed: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"started": started, "finished": finished, "ok": ok, "failed": failed}) + "\n")


def recent_runs(path, n: int = 7) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return runs[-n:][::-1]


def missed_days(trading_days: list[str], recorded: set[str]) -> list[str]:
    """Trading days after the first record with no forecast saved. The latest day is not yet due."""
    if not recorded:
        return []
    return [d for d in sorted(trading_days)[:-1] if d >= min(recorded) and d not in recorded]
