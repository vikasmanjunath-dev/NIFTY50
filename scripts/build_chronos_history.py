"""Chronos zero-shot volatility features for the pre-2021 rows. Standalone on purpose: it must not import LightGBM,
because PyTorch and LightGBM in one macOS process segfault or deadlock."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from nifty_analyser.features.volatility import EPS, daily_variance

bars = pd.read_parquet(DATA_RAW_DIR / "history" / "bars.parquet")
bars = bars[bars.quality_flag != "FAIL"]
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="mps" if torch.backends.mps.is_available() else "cpu", torch_dtype=torch.float32)
keys, contexts, medians = [], [], []
for iid, g in bars.groupby("instrument_id"):
    g = g.sort_values("event_time")
    lv = np.log(daily_variance(g) + EPS)
    med, values, times = lv.rolling(252, min_periods=120).median().to_numpy(), lv.to_numpy(), g.event_time.to_numpy()
    for i in range(130, len(g)):
        w = values[max(0, i - 255): i + 1]
        keys.append((iid, times[i]))
        contexts.append(torch.tensor(w[np.isfinite(w)], dtype=torch.float32))
        medians.append(med[i])
q = np.concatenate([pipe.predict_quantiles(contexts[s: s + 256], prediction_length=20, quantile_levels=[0.5])[0][:, :, 0].cpu().numpy()
                    for s in range(0, len(contexts), 256)])
out = pd.DataFrame(keys, columns=["instrument_id", "event_time"])
out["event_time"] = pd.to_datetime(out.event_time, utc=True).dt.tz_convert("Asia/Kolkata")
for h in (5, 20):
    out[f"chronos_{h}"] = np.log(np.exp(q[:, :h]).mean(1)) - np.array(medians)
out.to_parquet(DATA_PROCESSED_DIR / "chronos_vol_features_history.parquet", index=False)
print("rows", len(out))
