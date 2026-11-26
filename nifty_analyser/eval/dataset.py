"""Fast labelled point-in-time dataset: join labels onto the serving feature store (already membership-filtered,
unlabelled) instead of rebuilding ~100 features from bars -- seconds instead of ~10 minutes. Also liquidity inputs
(ADV, volatility) for the impact cost model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nifty_analyser.labels.targets import add_direction_label, add_forward_return


def attach_labels(serving: pd.DataFrame, bars: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Same rows/labels as ``eval.survivorship.label_dataset`` (flat and unlabellable rows dropped), RangeIndex sorted by date."""
    parts = [add_forward_return(add_direction_label(g.sort_values("event_time"), horizon), 1)
             [["instrument_id", "event_time", f"fwd_ret_{horizon}d", f"fwd_ret_{horizon}d_outcome_available_time", f"dir_label_{horizon}d", "fwd_ret_1d"]]
             for _, g in bars.groupby("instrument_id", sort=False)]
    df = serving.merge(pd.concat(parts, ignore_index=True), on=["instrument_id", "event_time"], how="inner")
    df = df.dropna(subset=[f"dir_label_{horizon}d", f"fwd_ret_{horizon}d_outcome_available_time", "fwd_ret_1d"])
    df = df[df[f"dir_label_{horizon}d"] != 0].copy()
    df["y"] = (df[f"dir_label_{horizon}d"] == 1).astype(int)
    return df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)


def liquidity_panel(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per (instrument, date): trailing average daily traded VALUE and daily return volatility, both lagged one day
    (nothing from the trade date itself is used to price its own impact)."""
    out = []
    for iid, g in bars.groupby("instrument_id", sort=False):
        g = g.sort_values("event_time")
        value = (g["close"] * g["volume"]).rolling(window, min_periods=window // 2).mean().shift(1)
        sigma = np.log(g["close"]).diff().rolling(window, min_periods=window // 2).std().shift(1)
        out.append(pd.DataFrame({"instrument_id": iid, "event_time": g["event_time"].to_numpy(), "adv_value": value.to_numpy(), "sigma": sigma.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def load_pit_dataset(horizon: int = 5) -> tuple[pd.DataFrame, list[str]]:
    from nifty_analyser.config import DATA_PROCESSED_DIR
    from nifty_analyser.data.store import load_universe_bars

    mdir = DATA_PROCESSED_DIR / "models"
    version = (mdir / "LATEST").read_text().strip()          # registry pointer -- NOT the lexicographically last file
    meta = json.loads((mdir / f"model_{version}.json").read_text())
    sv = pd.read_parquet(DATA_PROCESSED_DIR / "serving_features.parquet")
    for c in ("event_time", "available_time"):
        sv[c] = pd.to_datetime(sv[c], utc=True).dt.tz_convert("Asia/Kolkata")
    return attach_labels(sv, load_universe_bars(), horizon), meta["feature_cols"]
