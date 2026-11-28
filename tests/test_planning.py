"""Planning files stay consistent: unique ids, every key result and task traced, no references to things that do not exist."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda name: (ROOT / name).read_text()


def test_ids_trace_across_goals_blueprint_plan_and_tracker():
    goals, blueprint, plan = read("goals.md"), read("blueprint.md"), read("PLAN.md")
    tracker = read("TRACKER.md").split("## Execution tracker")[1]
    krs = re.findall(r"^\| (KR\d\.\d) ", goals, re.M)
    risks = re.findall(r"^\| (R-\d\d) ", blueprint, re.M)
    rows = [r.split("|") for r in re.findall(r"^\| T-\d{3} .*$", tracker, re.M)]
    tasks = [r[1].strip() for r in rows]
    for ids in (krs, risks, tasks):
        assert ids and len(ids) == len(set(ids))
    traced = " ".join(r[-2] for r in rows)
    assert not [k for k in krs if k not in traced], "key result without a task"
    assert not [t for t, r in zip(tasks, rows) if not re.search(r"KR\d\.\d|R-\d\d", r[-2])], "task traces to nothing"
    assert not [r for r in risks if r not in traced], "risk without a task"
    everything = goals + blueprint + plan + tracker
    assert set(re.findall(r"KR\d\.\d", everything)) <= set(krs)
    assert set(re.findall(r"\bR-\d\d\b", everything)) <= set(risks)
    assert set(re.findall(r"T-\d{3}", everything)) <= set(tasks)
    assert set(re.findall(r"D-\d{3}", everything)) <= set(re.findall(r"^\| (D-\d{3}) ", blueprint, re.M))
    assert {r[2].strip() for r in rows} <= set(re.findall(r"^\| (M\d)[ ]", plan, re.M))
    assert "TBD" not in everything


def test_planning_links_resolve():
    for name in ("goals.md", "blueprint.md", "PLAN.md"):
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", read(name)):
            assert (ROOT / target).exists(), f"{name} -> {target}"
