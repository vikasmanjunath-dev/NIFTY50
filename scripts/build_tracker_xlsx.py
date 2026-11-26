"""Regenerate TRACKER.xlsx from TRACKER.md (source of truth): checklist items + known simplifications, styling kept from the existing workbook."""
import re
from copy import copy
from pathlib import Path
import openpyxl
root = Path(__file__).resolve().parents[1]; md = (root / "TRACKER.md").read_text().split("\n")
wb = openpyxl.load_workbook(root / "TRACKER.xlsx"); ws1, ws2 = wb.worksheets
i0 = md.index("## Checklist"); i1 = next(i for i, l in enumerate(md) if l.startswith("## Known Simplifications")); i2 = next(i for i, l in enumerate(md) if l.startswith("## Work Log"))
items, sec = [], None
for l in md[i0:i1]:
    if l.startswith("### "): sec = l[4:].strip()
    m = re.match(r"- \[( |x)\] (.*)", l)
    if m:
        parts = re.split(r" — | ✅ [0-9-]+ — ", m.group(2), maxsplit=1)
        items.append((sec, "Done" if m.group(1) == "x" else "Open", parts[0].replace("**", "").strip(), parts[1].strip() if len(parts) > 1 else None))
simp = [(int(m.group(1)), m.group(2)) for l in md[i1:i2] if (m := re.match(r"(\d+)\. (.*)", l))]
def rewrite(ws, rows):
    sty = [copy(c._style) for c in ws[2]]; ws.delete_rows(2, ws.max_row)
    for r in rows:
        ws.append(list(r))
        for c, st in zip(ws[ws.max_row], sty): c._style = copy(st)
rewrite(ws1, items); rewrite(ws2, simp); wb.save(root / "TRACKER.xlsx"); print(len(items), "checklist items;", len(simp), "simplifications")
