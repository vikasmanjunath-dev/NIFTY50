"""CPCV with date-neutral metrics for LightGBM and the full calibrated ensemble -> reports/cpcv_full.json."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.eval.dataset import load_pit_dataset
from nifty_analyser.models.baseline import fit_lightgbm, fit_logistic
from nifty_analyser.models.ensemble import fit_ensemble
from nifty_analyser.validation.cpcv import cpcv_paths, path_metrics

OUT, EMB, SEED = "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5), 42
df, cols = load_pit_dataset(5)
X, y = df[cols].to_numpy(float), df["y"].to_numpy()
models = {
    "lightgbm": (8, lambda tr, te: fit_lightgbm(X[tr], y[tr], SEED).predict_proba_up(X[te])),
    "logistic": (8, lambda tr, te: fit_logistic(X[tr], y[tr], SEED).predict_proba_up(X[te])),
    "ensemble": (6, lambda tr, te: fit_ensemble(df, cols, "y", tr, OUT, EMB, random_state=SEED).predict_proba_up(X[te])),
}
res = {}
for name, (ng, fn) in models.items():
    t = time.time()
    rows = [path_metrics(df, p, "fwd_ret_5d", i) for i, p in enumerate(cpcv_paths(df, fn, ng, 2, OUT, EMB))]
    r = pd.DataFrame(rows)[["roc_auc", "within_date_auc", "information_coefficient", "daily_ic", "daily_ic_t", "log_loss"]]
    res[name] = {"n_groups": ng, "n_paths": len(r), "mean": r.mean().round(4).to_dict(), "min": r.min().round(4).to_dict(), "max": r.max().round(4).to_dict()}
    print(name, f"{time.time()-t:.0f}s", json.dumps(res[name]), flush=True)
Path("reports/cpcv_full.json").write_text(json.dumps(res, indent=1)); print("done", flush=True)
