"""Every third-party package the code imports is listed in a requirements file."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = {"sklearn": "scikit-learn", "PIL": "pillow", "chronos": "chronos-forecasting", "sentence_transformers": "sentence-transformers",
           "huggingface_hub": "huggingface-hub"}
VIA_OTHERS = {"starlette"}  # installed with fastapi


def test_requirements_cover_imports():
    listed = {re.split(r"[=<>\[]", line)[0].strip().lower() for f in ("requirements.txt", "requirements-optional.txt")
              for line in (ROOT / f).read_text().splitlines() if line.strip() and not line.startswith(("#", "-"))}
    files = [p for d in ("nifty_analyser", "scripts", "tests") for p in (ROOT / d).rglob("*.py")]
    local = {"nifty_analyser", "scripts", "tests"} | {p.stem for p in (ROOT / "scripts").glob("*.py")}
    missing = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and not node.level else []
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local | VIA_OTHERS and PACKAGE.get(top, top).lower() not in listed:
                    missing.setdefault(top, path.name)
    assert not missing, missing
