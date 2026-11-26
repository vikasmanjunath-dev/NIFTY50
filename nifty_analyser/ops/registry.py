"""Model registry (Blueprint §5.4): versioned artifacts with SHA-256 integrity checks. A model whose
bytes don't match its recorded checksum is refused, never silently served."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib


class RegistryError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(model, meta: dict, directory) -> str:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    blob = d / "model.tmp"
    joblib.dump(model, blob)
    sha = _sha(blob)
    version = f"{meta.get('trained_through', 'na')}-{sha[:8]}"
    final = d / f"model_{version}.joblib"
    blob.rename(final)
    (d / f"model_{version}.json").write_text(json.dumps({**meta, "version": version, "sha256": sha}, indent=2, default=str))
    (d / "LATEST").write_text(version)
    return version


def load_model(directory, version: str | None = None):
    d = Path(directory)
    if version is None:
        if not (d / "LATEST").exists():
            raise RegistryError("no model registered")
        version = (d / "LATEST").read_text().strip()
    meta_path, blob = d / f"model_{version}.json", d / f"model_{version}.joblib"
    if not meta_path.exists() or not blob.exists():
        raise RegistryError(f"model {version} not found")
    meta = json.loads(meta_path.read_text())
    if _sha(blob) != meta["sha256"]:
        raise RegistryError(f"model {version} failed its integrity check (checksum mismatch); refusing to load")
    return joblib.load(blob), meta
