"""Label-design study (Blueprint §3.3): does vol-scaled triple-barrier labelling and/or uniqueness weighting beat the plain
5-day direction label -- judged on the SAME target (5d direction) with paired purged walk-forward folds x seeds -- and does a
meta-label filter raise primary-trade hit rate? Pre-set: k=1.0, h=5 (no tuning)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.data.store import load_universe_bars
from nifty_analyser.eval.dataset import load_pit_dataset
from nifty_analyser.labels.triple_barrier import add_triple_barrier_label, meta_labels, uniqueness_weights
from nifty_analyser.models.baseline import make_lightgbm_baseline
from nifty_analyser.validation.cpcv import path_metrics
from nifty_analyser.validation.walk_forward import purged_embargoed_walk_forward

H, OUT, EMB = 5, "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5)
df, cols = load_pit_dataset(H)
parts = []
for _, g in load_universe_bars().groupby("instrument_id", sort=False):
    t = add_triple_barrier_label(g.sort_values("event_time").reset_index(drop=True), H, 1.0)
    n = len(t)
    t["w_tb"] = uniqueness_weights(t["_tb_start"].to_numpy(), t["_tb_end"].to_numpy())
    t["w_dir"] = uniqueness_weights(np.arange(n), np.where(np.arange(n) + H < n, np.arange(n) + H, -1))
    parts.append(t[["instrument_id", "event_time", "tb_y", "tb_label", "w_tb", "w_dir"]])
df = df.merge(pd.concat(parts), on=["instrument_id", "event_time"], how="left").dropna(subset=["tb_y", "w_tb", "w_dir"]).reset_index(drop=True)
print(df.shape, "tb label share up/down/vertical:", df.tb_label.value_counts(normalize=True).round(3).to_dict(), "| tb_y vs y agree:", round(float((df.tb_y == df.y).mean()), 3), flush=True)
X = df[cols].to_numpy(float); w_norm = lambda w: w / w.mean()
variants = {"direction": ("y", None), "tb_vol_scaled": ("tb_y", None), "direction+uniq_w": ("y", "w_dir"), "tb+uniq_w": ("tb_y", "w_tb")}
folds = purged_embargoed_walk_forward(df, 6, OUT, EMB); rows = []
for f in folds:
    va = df.iloc[f.val_idx]
    for seed in (42, 7, 123):
        for name, (lab, wc) in variants.items():
            m = make_lightgbm_baseline(seed)
            m.fit(X[f.train_idx], df[lab].to_numpy()[f.train_idx], sample_weight=None if wc is None else w_norm(df[wc].to_numpy()[f.train_idx]))
            p = m.predict_proba(X[f.val_idx])[:, 1]
            pm = path_metrics(va.reset_index(drop=True), p, "fwd_ret_5d")
            rows.append({"fold": f.fold_id, "seed": seed, "variant": name, "auc": pm["roc_auc"], "within_date_auc": pm["within_date_auc"], "daily_ic": pm["daily_ic"]})
r = pd.DataFrame(rows); res = {"mean": r.groupby("variant")[["auc", "within_date_auc", "daily_ic"]].mean().round(4).to_dict("index")}
fm = r.groupby(["fold", "variant"])[["auc", "within_date_auc", "daily_ic"]].mean().unstack("variant")
for v in ("tb_vol_scaled", "direction+uniq_w", "tb+uniq_w"):
    for met in ("within_date_auc", "daily_ic"):
        d = fm[met][v] - fm[met]["direction"]
        res[f"{v}_minus_direction_{met}"] = {"mean": round(float(d.mean()), 5), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), "folds_better": int((d > 0).sum()), "n": len(d)}
# meta-labelling: primary on first 60% of each train window, secondary on the rest (primary OOS there)
mrows = []
for f in folds:
    tr = f.train_idx; cut = tr[int(len(tr) * 0.6)]; a, b = tr[tr < cut], tr[tr >= cut]
    a = a[df[OUT].iloc[a].to_numpy() < df["event_time"].iloc[b].min()]            # purge primary's labels before secondary slice
    prim = make_lightgbm_baseline(42).fit(X[a], df["y"].to_numpy()[a])
    pb = prim.predict_proba(X[b])[:, 1]; side_b = np.where(pb > 0.5, 1, -1)
    Xb = np.column_stack([X[b], pb]); mb = meta_labels(side_b, df["tb_y"].to_numpy()[b])
    sec = make_lightgbm_baseline(42).fit(Xb, mb)
    pv = prim.predict_proba(X[f.val_idx])[:, 1]; side_v = np.where(pv > 0.5, 1, -1)
    mv = meta_labels(side_v, df["tb_y"].to_numpy()[f.val_idx]); ps = sec.predict_proba(np.column_stack([X[f.val_idx], pv]))[:, 1]
    keep = ps >= np.median(ps)
    mrows.append({"fold": f.fold_id, "primary_hit": mv.mean(), "filtered_hit": mv[keep].mean(), "rejected_hit": mv[~keep].mean(), "meta_auc": roc_auc_score(mv, ps) if len(set(mv)) > 1 else 0.5})
mr = pd.DataFrame(mrows)
res["meta_label"] = {"mean": mr.mean(numeric_only=True).round(4).to_dict(), "filtered_minus_primary_hit": {"mean": round(float((mr.filtered_hit - mr.primary_hit).mean()), 4), "se": round(float((mr.filtered_hit - mr.primary_hit).std(ddof=1) / np.sqrt(len(mr))), 4), "folds_better": int((mr.filtered_hit > mr.primary_hit).sum()), "n": len(mr)}}
print(json.dumps(res, indent=1), flush=True); Path("reports/label_study.json").write_text(json.dumps(res, indent=1)); print("done", flush=True)
