"""Append-only, hash-chained audit log (Blueprint §5.6). Each entry's hash covers the previous
entry's hash, so any edit, deletion or reordering of history is detectable by ``verify()``."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def _entry_hash(body: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in body.items() if k != "hash"}).encode()).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        entries = self.read()
        self._seq, self._tail = (entries[-1]["seq"], entries[-1]["hash"]) if entries else (0, GENESIS)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, kind: str, payload: dict) -> dict:
        with self._lock:
            body = {"seq": self._seq + 1, "ts": datetime.now(timezone.utc).isoformat(), "kind": kind,
                    "payload": json.loads(canonical(payload)), "prev_hash": self._tail}
            body["hash"] = _entry_hash(body)
            with self.path.open("a") as f:
                f.write(canonical(body) + "\n")
            self._seq, self._tail = body["seq"], body["hash"]
            return body

    def verify(self) -> tuple[bool, int | None]:
        """(ok, first_bad_seq). Checks contiguous seq, prev-hash linkage and every entry hash."""
        prev, expected = GENESIS, 1
        for e in self.read():
            if e["seq"] != expected or e["prev_hash"] != prev or e["hash"] != _entry_hash(e):
                return False, e["seq"]
            prev, expected = e["hash"], expected + 1
        return True, None
