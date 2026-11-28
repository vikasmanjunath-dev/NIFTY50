"""Runs scripts/daily_refresh.py for the web app's refresh button: one run at a time, in a separate process."""
import fcntl
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "daily_refresh.py"


@contextmanager
def single_instance(path):
    """Yields True if this process holds the lock, False if another run already does (button versus daily task)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


class Refresher:
    def __init__(self, command=None, timeout: float = 3600):
        self.command, self.timeout = command or [sys.executable, str(SCRIPT)], timeout
        self._lock = threading.Lock()
        self.state = {"running": False, "ok": None, "message": ""}

    def start(self) -> bool:
        """False if a run is already in progress."""
        with self._lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "ok": None, "message": ""}
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        try:
            done = subprocess.run(self.command, capture_output=True, text=True, timeout=self.timeout, cwd=SCRIPT.parents[1])
            lines = (done.stdout + done.stderr).strip().splitlines()
            ok, message = done.returncode == 0, lines[-1][:300] if lines else ""
        except Exception as exc:  # noqa: BLE001
            ok, message = False, str(exc)[:300]
        self.state = {"running": False, "ok": ok, "message": message}
