"""Blind human relevance labelling (model grades are NOT shown). Resumable; saves after every answer.
Run: python3 scripts/label_cli.py [--n 120]      grades: 0 unrelated/same-company only, 1 loosely related, 2 relevant, 3 directly answers; s = skip, q = quit.
Labels go into reports/label_sheet.csv (`human_grade`); afterwards: python3 scripts/human_agreement.py"""
import argparse, sys
from pathlib import Path
import pandas as pd
import os
P = Path(os.environ.get("STM_LABEL_SHEET") or Path(__file__).resolve().parents[1] / "reports" / "label_sheet.csv")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); n = ap.parse_args().n
df = pd.read_csv(P, dtype={"query_id": str, "doc_id": str, "human_grade": str})
todo = df[df.human_grade.isna() | (df.human_grade.str.strip() == "")]
todo = todo.sample(frac=1, random_state=0)                      # shuffled across queries so labelling order carries no signal
print(f"{len(df) - len(todo)} labelled, {len(todo)} remaining; doing up to {n}. Grades: 0 none / 1 loose / 2 relevant / 3 direct | s skip | q quit\n")
done = 0
for i, r in todo.head(n).iterrows():
    print(f"[{done + 1}/{min(n, len(todo))}] QUERY: {r['query']}   (as of {r.as_of})\n  {r.company}: {str(r.text)[:500]}")
    while True:
        a = input("  grade> ").strip().lower()
        if a in {"0", "1", "2", "3"}:
            df.loc[i, "human_grade"] = a; df.to_csv(P, index=False); done += 1; break
        if a == "s": break
        if a == "q": print(f"saved; {done} new labels"); sys.exit()
print(f"done; {done} new labels")
