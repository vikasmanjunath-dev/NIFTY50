"""Daily intraday-volatility summaries from the public 1-minute NSE dataset on Hugging Face (MIT, sourced from Upstox).

Only the row groups holding our symbols are read, into memory; nothing but the daily summary is written to disk. A byte
budget is checked from the file metadata before any data is fetched. Needs no account or token."""
import sys
from pathlib import Path


import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import ingestible_instrument_ids
from nifty_analyser.features.volatility_intraday import daily_summary

BASE = "datasets/xxparthparekhxx/indian-stock-market-minute-data/minute"
COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
BUDGET = 1.9e9
OUT = DATA_RAW_DIR / "intraday" / "daily_summary.parquet"


def main():
    fs = HfFileSystem(token=False)
    want = sorted({i.split(":")[1] for i in ingestible_instrument_ids()} | {"NIFTY"})
    plan, total = [], 0
    for path in sorted(fs.ls(BASE, detail=False)):
        with fs.open(path, "rb", block_size=2**20) as f:
            meta = pq.ParquetFile(f).metadata
        groups = []
        for i in range(meta.num_row_groups):
            st = meta.row_group(i).column(0).statistics
            if any(st.min <= w <= st.max for w in want):
                groups.append(i)
                total += sum(meta.row_group(i).column(c).total_compressed_size for c in range(len(COLUMNS)))
        plan.append((path, groups))
    print(f"planned transfer {total / 1e6:.0f} MB in {sum(len(g) for _, g in plan)} row groups", flush=True)
    if total > BUDGET:
        sys.exit("over the byte budget; nothing downloaded")
    parts, carry = [], None
    for path, groups in plan:
        with fs.open(path, "rb", block_size=4 * 2**20) as f:
            pf = pq.ParquetFile(f)
            for i in groups:
                rows = pf.read_row_group(i, columns=COLUMNS).to_pandas()
                rows = rows[rows.symbol.isin(want)]
                if carry is not None:
                    rows = pd.concat([carry, rows], ignore_index=True)
                if rows.empty:
                    carry = None
                    continue
                last = rows.symbol.iloc[-1]  # a symbol can continue in the next row group
                carry, done = rows[rows.symbol == last], rows[rows.symbol != last]
                if len(done):
                    parts.append(daily_summary(done))
        print(path.rsplit("/", 1)[1], "done", flush=True)
    if carry is not None and len(carry):
        parts.append(daily_summary(carry))
    out = pd.concat(parts, ignore_index=True)
    assert not out.duplicated(["symbol", "date"]).any(), "a symbol-day was split across summaries"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"symbols {out.symbol.nunique()} rows {len(out)} {out.date.min()}..{out.date.max()} missing {sorted(set(want) - set(out.symbol))}")


if __name__ == "__main__":
    main()
