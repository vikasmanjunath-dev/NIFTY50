"""Embed the whole announcement corpus once (resumable, deduplicated), then benchmark indexed retrieval latency."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.documents.index import VectorIndex, hybrid_retrieve_indexed

d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet")
d = d[d.quality_flag == "PASS"].copy()
for c in ("event_time", "available_time"):
    d[c] = pd.to_datetime(d[c], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000)
d = d[d.text.str.len() > 0].reset_index(drop=True)
print(f"{len(d)} docs, {d.text.nunique()} unique texts", flush=True)
ix = VectorIndex(DATA_PROCESSED_DIR / "vector_index")
t = time.time(); n = ix.add(d); print(f"embedded {n} texts in {time.time()-t:.0f}s; index docs={len(ix)}", flush=True)
as_of = d.available_time.max(); inst = d.instrument_id.iloc[0]
t = time.time(); r = hybrid_retrieve_indexed("dividend declared by board", d, as_of, ix, top_k=8); print(f"full-corpus query {time.time()-t:.2f}s -> {len(r)} docs", flush=True)
t = time.time(); r = hybrid_retrieve_indexed("dividend declared by board", d, as_of, ix, top_k=8, instrument_id=inst); print(f"single-instrument query {time.time()-t:.2f}s -> {len(r)} docs", flush=True)
print("done", flush=True)
