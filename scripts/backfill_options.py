"""Stream NSE daily F&O bhavcopy zips (2021 onwards), keep only a per-symbol option summary. Nothing raw is written.
Resumable: finished days are skipped. Stops if the transfer reaches the byte budget."""
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_RAW_DIR
from nifty_analyser.data.instrument_master import get_instrument, ingestible_instrument_ids
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.documents.ingest import _HEADERS
from nifty_analyser.features.volatility_options import parse, summarise

OUT = DATA_RAW_DIR / "options" / "daily_summary.parquet"
BUDGET = 1.9e9
NEW_FORMAT_FROM = pd.Timestamp("2024-07-08")


def url(day: pd.Timestamp) -> str:
    if day >= NEW_FORMAT_FROM:
        return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    month = f"{day:%b}".upper()
    return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{day:%Y}/{month}/fo{day:%d}{month}{day:%Y}bhav.csv.zip"


def fetch(day):
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url(day), headers={"User-Agent": _HEADERS["User-Agent"]})
            return day, urllib.request.urlopen(req, timeout=40).read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return day, None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * attempt)
    return day, None


def main():
    names = {}  # NSE symbol (including former names) -> current symbol
    for iid in ingestible_instrument_ids():
        inst = get_instrument(iid)
        for symbol in (inst.symbol, *inst.aliases):
            names[symbol] = iid.split(":", 1)[1]
    names["NIFTY"] = "NIFTY"
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["symbol", "date"])
    done = set(old.date)
    days = sorted(d.tz_localize(None).normalize() for d in load_universe_bars().event_time.unique())
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in done]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts, transferred, missing = [old], 0, []

    def save():
        out = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
        out.to_parquet(OUT, index=False)
        return out

    with ThreadPoolExecutor(max_workers=3) as pool:
        for start in range(0, len(todo), 30):
            if transferred >= BUDGET:
                print("byte budget reached; stopping")
                break
            for day, blob in pool.map(fetch, todo[start: start + 30]):
                if blob is None:
                    missing.append(day.strftime("%Y-%m-%d"))
                    continue
                transferred += len(blob)
                z = zipfile.ZipFile(io.BytesIO(blob))
                contracts = parse(pd.read_csv(z.open(z.namelist()[0]), low_memory=False))
                contracts["symbol"] = contracts.symbol.map(names)
                parts.append(summarise(contracts.dropna(subset=["symbol"]), day, set(names.values())))
            out = save()
            print(f"{min(start + 30, len(todo))}/{len(todo)} days, {transferred / 1e6:.0f} MB, {len(missing)} missing", flush=True)
    out = save()
    print(f"DONE rows {len(out)} symbols {out.symbol.nunique()} {out.date.min()}..{out.date.max()} transferred {transferred / 1e6:.0f} MB "
          f"missing days {len(missing)} {missing[:5]} median near IV {out.iv_near.median():.3f}")


if __name__ == "__main__":
    main()
