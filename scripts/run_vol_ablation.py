"""Does any other cached data layer improve the volatility blend? Paired, identical-row walk-forward.

Adoption rule, fixed before running: the 95% date-block bootstrap interval of the AUC gain must be above zero at BOTH horizons."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vol_forecast_v2 import block_bootstrap, load

from nifty_analyser.data.store import load_all_fii_dii_flows, load_all_macro_series, load_delivery_universe
from nifty_analyser.models.volatility import fit_predict_classifiers
from nifty_analyser.ops.vol_forecast import load_inputs
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward


def asof(df, right, by=None):
    right = right.sort_values("available_time").astype({"available_time": df.available_time.dtype})
    return pd.merge_asof(df.sort_values("available_time"), right, on="available_time", by=by, direction="backward",
                         tolerance=pd.Timedelta(days=7))


def delivery(df):
    d = load_delivery_universe().sort_values(["instrument_id", "event_time"])
    g = d.groupby("instrument_id")
    med = g.delivery_pct.transform(lambda s: s.rolling(60, min_periods=30).median())
    qty = g.traded_qty.transform(lambda s: s.rolling(60, min_periods=30).median())
    d = d.assign(dlv_rel=g.delivery_pct.transform(lambda s: s.rolling(5).mean()) - med, qty_rel=np.log((d.traded_qty + 1) / (qty + 1)))
    return asof(df, d[["instrument_id", "available_time", "dlv_rel", "qty_rel"]], by="instrument_id"), ["dlv_rel", "qty_rel"]


def announcements(df):
    a = load_inputs()[2]
    out = df.reset_index(drop=True).copy()
    n5, n66 = np.zeros(len(out)), np.zeros(len(out))
    for iid, rows in out.groupby("instrument_id"):
        times = np.sort(pd.DatetimeIndex(a.loc[a.instrument_id == iid, "available_time"]).asi8)
        now = pd.DatetimeIndex(rows.available_time).asi8
        upto = np.searchsorted(times, now, side="right")
        n5[rows.index] = upto - np.searchsorted(times, now - 7 * 86_400 * 10**9, side="right")
        n66[rows.index] = upto - np.searchsorted(times, now - 90 * 86_400 * 10**9, side="right")
    out["ann_surprise"] = np.log1p(n5) - np.log1p(n66 * 7 / 90)
    return out, ["ann_surprise"]


def flows_and_global(df):
    cols = []
    f = load_all_fii_dii_flows()["fii"].sort_values("available_time")
    scale = f.net_investment_cr.rolling(252, min_periods=60).std()
    f = f.assign(fii_5=f.net_investment_cr.rolling(5).sum() / scale, fii_22=f.net_investment_cr.rolling(22).sum() / scale)
    df = asof(df, f[["available_time", "fii_5", "fii_22"]])
    cols += ["fii_5", "fii_22"]
    for name in ("usd_inr", "brent_crude"):
        m = load_all_macro_series()[name].sort_values("available_time")
        v = np.log(m.close).diff().rolling(22).std()
        col = f"{name}_vol_rel"
        df = asof(df, m.assign(**{col: np.log(v / v.rolling(252, min_periods=120).median())})[["available_time", col]])
        cols.append(col)
    return df, cols


def main():
    out = {}
    for h in (5, 20):
        df, base_cols = load(h)
        df = df.dropna(subset=["fwd_rv", "med_rv", "cur_rv", "outcome_time", "har_66"])
        df = df[(df.fwd_rv > 0) & (df.cur_rv > 0)].reset_index(drop=True)
        groups = {}
        for name, fn in (("delivery", delivery), ("announcements", announcements), ("flows_global", flows_and_global)):
            df, groups[name] = fn(df)
        df = df.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
        y = (df.fwd_rv > df.med_rv).astype(int).to_numpy()
        folds = purged_embargoed_walk_forward(df, 6, "outcome_time", pd.Timedelta(days=int(h * 1.6) + 3))
        preds = {}
        for name, extra in [("base", [])] + list(groups.items()):
            X = df[base_cols + extra].to_numpy(float)
            p = np.full(len(df), np.nan)
            for f in folds:
                p[f.val_idx] = fit_predict_classifiers(X[f.train_idx], y[f.train_idx], X[f.val_idx])["blend"]
            preds[name] = p
        m = ~np.isnan(preds["base"])
        yy, dates, base = y[m], df.event_time.to_numpy()[m], preds["base"][m]
        res = {"base_auc": round(float(roc_auc_score(yy, base)), 4)}
        for name in groups:
            p = preds[name][m]
            res[name] = {"auc_gain": round(float(roc_auc_score(yy, p) - roc_auc_score(yy, base)), 4),
                         "ci95": block_bootstrap(dates, lambda i: roc_auc_score(yy[i], p[i]) - roc_auc_score(yy[i], base[i]), max(2 * h, 20), reps=300)}
        out[str(h)] = res
        print(h, json.dumps(res, separators=(",", ":")))
    out["adopted"] = [n for n in ("delivery", "announcements", "flows_global") if all(out[h][n]["ci95"][0] > 0 for h in ("5", "20"))]
    print("adopted", out["adopted"])
    Path("reports/vol_ablation.json").write_text(json.dumps(out, indent=1))


main()
