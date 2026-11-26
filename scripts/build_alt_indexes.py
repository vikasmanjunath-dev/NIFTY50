"""Embed the corpus with alternative Ollama embedding models into separate indexes (resumable)."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.documents.index import VectorIndex, make_ollama_embedder
d = pd.read_parquet(DATA_RAW_DIR / "announcements_nifty50.parquet"); d = d[d.quality_flag == "PASS"].copy()
d["available_time"] = pd.to_datetime(d["available_time"], utc=True).dt.tz_convert("Asia/Kolkata")
d["text"] = d["text"].str.slice(0, 2000); d = d[d.text.str.len() > 0].reset_index(drop=True)
for model in sys.argv[1:]:
    ix = VectorIndex(DATA_PROCESSED_DIR / f"vector_index_{model.split(':')[0]}", make_ollama_embedder(model), model=model)
    t = time.time(); n = ix.add(d, batch=32); print(f"{model}: embedded {n} texts in {time.time()-t:.0f}s, dim={ix.E.shape[1]}", flush=True)
print("done", flush=True)
