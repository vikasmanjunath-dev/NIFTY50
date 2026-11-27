"""Option-implied volatility from NSE's end-of-day F&O files.

At-the-money volatility uses the straddle approximation IV ~ (call + put) / (0.798 * forward * sqrt(T)), on the nearest
expiry at least five days away. Prices are the day's closing/settlement values; NSE publishes the file that evening."""
import numpy as np
import pandas as pd

EPS = 1e-8
OLD = {"INSTRUMENT": "kind", "SYMBOL": "symbol", "EXPIRY_DT": "expiry", "STRIKE_PR": "strike", "OPTION_TYP": "type",
       "CLOSE": "close", "SETTLE_PR": "settle", "CONTRACTS": "volume", "OPEN_INT": "oi"}
NEW = {"FinInstrmTp": "kind", "TckrSymb": "symbol", "XpryDt": "expiry", "StrkPric": "strike", "OptnTp": "type",
       "ClsPric": "close", "SttlmPric": "settle", "TtlTradgVol": "volume", "OpnIntrst": "oi"}
KINDS = {"FUTSTK": "F", "FUTIDX": "F", "STF": "F", "IDF": "F", "OPTSTK": "O", "OPTIDX": "O", "STO": "O", "IDO": "O"}


def parse(csv: pd.DataFrame) -> pd.DataFrame:
    names = NEW if "TckrSymb" in csv.columns else OLD
    out = csv.rename(columns=names)[list(names.values())].copy()
    out["kind"] = out.kind.map(KINDS)
    out["expiry"] = pd.to_datetime(out.expiry, format="mixed", dayfirst=names is OLD)
    return out.dropna(subset=["kind"])


def summarise(contracts: pd.DataFrame, day: pd.Timestamp, symbols: set[str]) -> pd.DataFrame:
    """One row per symbol: near and next at-the-money IV, downside skew, put/call open interest and traded volume."""
    rows = []
    data = contracts[contracts.symbol.isin(symbols)]
    for symbol, g in data.groupby("symbol"):
        options, futures = g[(g.kind == "O") & (g.volume > 0) & (g.close > 0)], g[g.kind == "F"].sort_values("expiry")
        expiries = sorted(e for e in options.expiry.unique() if (e - day).days >= 5)
        if not expiries or futures.empty:
            continue
        row = {"symbol": symbol, "date": day.strftime("%Y-%m-%d")}
        for label, expiry in zip(("near", "next"), expiries[:1] + [e for e in expiries[1:] if (e - expiries[0]).days >= 20][:1]):
            chain = options[options.expiry == expiry].pivot_table(index="strike", columns="type", values="close")
            if not {"CE", "PE"} <= set(chain.columns):
                continue
            fut = futures[futures.expiry >= expiry]
            forward = float((fut if len(fut) else futures).iloc[0].settle)
            years = (expiry - day).days / 365
            both = chain.dropna(subset=["CE", "PE"])
            if both.empty or forward <= 0:
                continue
            strike = both.index[np.abs(both.index - forward).argmin()]
            if abs(strike / forward - 1) > 0.05:
                continue
            row[f"iv_{label}"] = float((both.CE[strike] + both.PE[strike]) / (0.7979 * forward * np.sqrt(years)))
            if label == "near":
                low, high = chain.PE.dropna(), chain.CE.dropna()
                if len(low) and len(high):
                    k_low = low.index[np.abs(low.index - 0.95 * forward).argmin()]
                    k_high = high.index[np.abs(high.index - 1.05 * forward).argmin()]
                    row["skew"] = float(np.log((low[k_low] + EPS) / (high[k_high] + EPS)))
        all_options = g[g.kind == "O"]
        oi = all_options.groupby("type").oi.sum()
        volume = all_options.groupby("type").volume.sum()
        row["pcr_oi"] = float(np.log((oi.get("PE", 0) + 1) / (oi.get("CE", 0) + 1)))
        row["opt_volume"] = float(volume.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def option_features(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Trailing-window features keyed by (instrument_id, date)."""
    s = summary.sort_values(["symbol", "date"]).reset_index(drop=True)
    s = s.reindex(columns=list(dict.fromkeys([*s.columns, "iv_near", "iv_next", "skew", "pcr_oi", "opt_volume"])))
    parts = []
    for symbol, g in s.groupby("symbol", sort=False):
        iv = np.log(g.iv_near.astype(float) + EPS)
        out = pd.DataFrame({"symbol": symbol, "date": g.date})
        out["iv_log"] = iv
        out["iv_rel"] = iv - iv.rolling(252, min_periods=60).median()
        out["iv_chg_1"], out["iv_chg_5"] = iv.diff(), iv.diff(5)
        out["iv_term"] = np.log(g.iv_next.astype(float) + EPS) - iv
        out["iv_skew"] = g["skew"] - g["skew"].rolling(66, min_periods=20).median()
        out["iv_pcr"] = g.pcr_oi - g.pcr_oi.rolling(66, min_periods=20).median()
        volume = np.log(g.opt_volume + 1)
        out["iv_volume_surprise"] = volume.rolling(3, min_periods=1).mean() - volume.rolling(66, min_periods=20).median()
        parts.append(out)
    f = pd.concat(parts, ignore_index=True)
    index = f[f.symbol == "NIFTY"][["date", "iv_log", "iv_rel", "iv_chg_5", "iv_term"]].rename(columns=lambda c: c if c == "date" else f"index_{c}")
    f = f[f.symbol != "NIFTY"].merge(index, on="date", how="left")
    f["iv_vs_index"] = f.iv_log - f.index_iv_log
    f["instrument_id"] = "NSE:" + f.pop("symbol")
    return f, [c for c in f.columns if c not in ("instrument_id", "date")]
