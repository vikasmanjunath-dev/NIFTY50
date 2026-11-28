"""scripts/daily_publish.sh must not deploy a working folder with unfinished work or failing tests. Runs the real script
in a throw-away repository with stand-ins for python3, vercel and curl."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_publish.sh"
pytestmark = pytest.mark.skipif(not shutil.which("zsh"), reason="zsh not installed")


def run(tmp_path, dirty=False, tests_fail=False):
    repo, bin_dir, calls = tmp_path / "repo", tmp_path / "bin", tmp_path / "calls"
    (repo / "scripts").mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy(SCRIPT, repo / "scripts" / "daily_publish.sh")
    (repo / ".gitignore").write_text("dist/\n")
    stubs = {"python3": f'echo "python3 $*" >> {calls}\n' + ('[[ "$*" == *pytest* ]] && exit 1\n' if tests_fail else "")
             + '[[ "$*" == *export_static* ]] && mkdir -p dist/nifty-tremor/data && echo \'{"data_date":"d"}\' > dist/nifty-tremor/data/overview.json\nexit 0',
             "vercel": f'echo "vercel $*" >> {calls}', "curl": 'echo \'{"data_date":"d"}\''}
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/zsh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "x")
    if dirty:
        (repo / "scripts" / "half_done.py").write_text("raise SystemExit")
    done = subprocess.run(["zsh", "scripts/daily_publish.sh"], cwd=repo, capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "ZDOTDIR": str(tmp_path)})
    return done, calls.read_text() if calls.exists() else ""


def test_unfinished_work_stops_before_anything_runs(tmp_path):
    done, calls = run(tmp_path, dirty=True)
    assert done.returncode != 0 and calls == "" and "uncommitted" in done.stdout and "half_done.py" in done.stdout


def test_failing_tests_stop_before_refresh_and_deploy(tmp_path):
    done, calls = run(tmp_path, tests_fail=True)
    assert done.returncode != 0 and "pytest" in calls and "vercel" not in calls and "daily_refresh" not in calls


def test_clean_and_green_publishes_in_order(tmp_path):
    done, calls = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert calls.index("pytest") < calls.index("daily_refresh") < calls.index("export_static") < calls.index("vercel deploy")
