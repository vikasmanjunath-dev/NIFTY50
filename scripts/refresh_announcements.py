"""Add company filings published since the newest one on disk (NSE, whole-exchange feed filtered to the pool), and embed
them into the search index if it exists and the local embedder is running. Existing records are never rewritten."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.documents.ingest import ingest_announcements_universe, merge_new_documents

PATH = DATA_RAW_DIR / "announcements_nifty50.parquet"
OVERLAP = pd.Timedelta(days=3)  # late-arriving records around the last run


def main():
    if not PATH.exists():
        sys.exit("no filings yet: run scripts/backfill_announcements.py first")
    old = pd.read_parquet(PATH)
    start = pd.to_datetime(old.available_time, utc=True).max().tz_convert("Asia/Kolkata").tz_localize(None).normalize() - OVERLAP
    end = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    new = ingest_announcements_universe(start, end)
    if new.empty:
        sys.exit(f"nothing returned for {start.date()}..{end.date()} (NSE unavailable?); keeping the old file")
    merged, added = merge_new_documents(old, new)
    tmp = PATH.with_suffix(".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(PATH)
    print(f"filings: {added} new, {len(merged)} total, window {start.date()}..{end.date()}")
    index_dir = DATA_PROCESSED_DIR / "vector_index"
    if added and index_dir.exists():
        try:
            from nifty_analyser.documents.index import VectorIndex

            docs = merged[merged.quality_flag == "PASS"].copy()
            for c in ("event_time", "available_time"):
                docs[c] = pd.to_datetime(docs[c], utc=True).dt.tz_convert("Asia/Kolkata")
            docs["text"] = docs.text.str.slice(0, 2000)
            index = VectorIndex(index_dir)
            embedded = index.add(docs[docs.text.str.len() > 0].reset_index(drop=True))
            index.save()
            print(f"search index: {embedded} new texts embedded; restart the server for the Ask page to see them")
        except Exception as exc:  # noqa: BLE001
            print(f"search index not updated (is Ollama running?): {str(exc)[:150]}")


if __name__ == "__main__":
    main()
