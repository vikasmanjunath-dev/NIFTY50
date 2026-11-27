"""NSE board-meeting intimations (2010 onwards) for the instrument pool, with the exchange's publication timestamp."""
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument, ingestible_instrument_ids
from nifty_analyser.documents.ingest import _http_get_json
from nifty_analyser.features.volatility_events import normalise

OUT = DATA_RAW_DIR / "board_meetings.parquet"
URL = "https://www.nseindia.com/api/corporate-board-meetings?index=equities&symbol={}&from_date=01-01-2010&to_date={}"


def main():
    today = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d-%m-%Y")
    parts, failed = [], []
    for iid in ingestible_instrument_ids():
        inst = get_instrument(iid)
        records = []
        for symbol in (inst.symbol, *inst.aliases):
            try:
                records += _http_get_json(URL.format(quote(symbol), today))
            except Exception as exc:  # noqa: BLE001
                failed.append((symbol, str(exc)[:80]))
            time.sleep(0.5)
        parts.append(normalise(records, iid))
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    if out.instrument_id.nunique() < 50:
        sys.exit(f"only {out.instrument_id.nunique()} instruments returned data; keeping the old file. failures: {failed[:5]}")
    out.to_parquet(OUT, index=False)
    print(f"rows {len(out)} instruments {out.instrument_id.nunique()} {out.available_time.min().date()}..{out.available_time.max().date()} "
          f"results meetings {int(out.is_results.sum())} failed {failed[:5]}")


if __name__ == "__main__":
    main()
