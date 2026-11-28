"""Cheap static checks without extra dependencies: every module compiles cleanly and has no bare except or mutable default."""
import ast
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modules_compile_without_warnings_or_common_traps():
    problems = []
    for path in (p for d in ("nifty_analyser", "scripts", "tests") for p in (ROOT / d).rglob("*.py")):
        source = path.read_text()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(source, str(path), "exec")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                problems.append(f"{path.name}:{node.lineno} bare except")
            if isinstance(node, ast.FunctionDef) and any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in node.args.defaults):
                problems.append(f"{path.name}:{node.lineno} mutable default")
    assert not problems, problems
