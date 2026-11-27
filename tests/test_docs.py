"""The generated documentation matches the code, and hand-written links point somewhere."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_docs import render


def test_generated_docs_are_current():
    stale = [name for name, body in render().items() if not (ROOT / name).exists() or (ROOT / name).read_text() != body]
    assert not stale, f"run python3 scripts/build_docs.py; stale: {stale}"


def test_relative_links_resolve():
    broken = []
    for doc in [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), *ROOT.glob("*/README.md"), *(ROOT / "nifty_analyser").glob("*/README.md")]:
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", doc.read_text()):
            if not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, broken
