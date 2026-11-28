import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_static import public_system


def test_hosted_snapshot_omits_owner_activity():
    out = public_system({"recent": [{"kind": "refresh", "time": "t"}], "audit_entries": 9, "audit_ok": True, "runs": [{"started": "t"}]})
    assert out == {"recent": [], "audit_entries": 0, "audit_ok": True, "runs": []}
