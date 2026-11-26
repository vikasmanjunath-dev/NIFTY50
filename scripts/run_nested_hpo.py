"""Nested random search for LightGBM on the real PIT dataset -> reports/nested_hpo.json. Does tuning find edge?"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.eval.dataset import load_pit_dataset
from nifty_analyser.validation.nested import nested_random_search

df, cols = load_pit_dataset(5)
print(df.shape, len(cols), flush=True)
res = nested_random_search(df, cols, "fwd_ret_5d_outcome_available_time", pd.Timedelta(days=5), outer_splits=5, inner_splits=3, n_trials=12)
print(res.rows.drop(columns="best_params").round(4).to_string(), flush=True)
s = res.summary(); print(json.dumps(s, indent=1), flush=True)
Path("reports/nested_hpo.json").write_text(json.dumps({"summary": s, "folds": res.rows.to_dict("records")}, indent=1, default=str))
print("done", flush=True)
