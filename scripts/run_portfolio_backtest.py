"""Do the Portfolio page's swing figures hold? Random long-only mixes on past out-of-sample forecast dates (non-overlapping):
the page says moves stay within +/-1 "typical" about 68% of the time and within the "rough 1-in-10" figure 90% of the time.
Uses the walk-forward volatility forecasts (data/processed/vol_oos_h*.parquet) and trailing-250-day correlations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.data.store import load_universe_bars

bars = load_universe_bars()
close = bars.pivot_table(index="event_time", columns="instrument_id", values="close")
daily = np.log(close).diff()
rng = np.random.default_rng(0)
out = {}
for h in (5, 20):
    d = pd.read_parquet(DATA_PROCESSED_DIR / f"vol_oos_h{h}.parquet")
    vol = d.assign(v=np.exp(d.r)).pivot_table(index="event_time", columns="instrument_id", values="v")  # h-day volatility
    ahead = np.expm1(np.log(close).shift(-h) - np.log(close))
    z = []
    for day in vol.index[::h]:
        names = vol.columns[vol.loc[day].notna() & ahead.loc[day].reindex(vol.columns).notna()]
        corr = daily.loc[:day, names].tail(250).corr().fillna(0).to_numpy(copy=True)
        np.fill_diagonal(corr, 1.0)
        for _ in range(40):
            k = int(rng.integers(3, 11))
            pick = rng.choice(len(names), k, replace=False)
            w = rng.dirichlet(np.ones(k))
            s = vol.loc[day, names[pick]].to_numpy() * w
            sigma = np.sqrt(s @ corr[np.ix_(pick, pick)] @ s)
            z.append((day, float(w @ ahead.loc[day, names[pick]].to_numpy()) / sigma))
    z = pd.DataFrame(z, columns=["day", "z"])
    a = z.z.abs()
    by_date = z.assign(in90=a <= 1.645).groupby("day").in90.mean()
    # fit the multiplier on the first half of dates and check it on the second half
    days = np.sort(z.day.unique())
    first, second = a[z.day.isin(days[: len(days) // 2 - 1])], a[z.day.isin(days[len(days) // 2:])]
    fitted = float(first.quantile(0.9))
    out[str(h)] = {"calibrated_multiplier_from_first_half": round(fitted, 3), "second_half_within_calibrated": round(float((second <= fitted).mean()), 4),
                   "second_half_within_bell_curve": round(float((second <= 1.645).mean()), 4), "dates": int(z.day.nunique()), "mixes": len(z), "within_typical": round(float((a <= 1).mean()), 4),
                   "within_1_in_10": round(float((a <= 1.645).mean()), 4), "worst_date_within_1_in_10": round(float(by_date.min()), 3),
                   "multiplier_for_90pct": round(float(a.quantile(0.9)), 3)}
    print(h, json.dumps(out[str(h)]))
Path("reports/portfolio_backtest.json").write_text(json.dumps(out, indent=1))
