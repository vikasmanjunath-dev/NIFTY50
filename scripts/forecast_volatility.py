"""Forecast 5- and 20-day volatility for current index members, record them in the forward ledger, score matured ones."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR, REPORTS_DIR
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.vol_forecast import EXTRA_FEATURE_HORIZONS, latest_forecasts, load_history_training, load_inputs, record, score

bars, vix, announcements = load_inputs()
ledger = AuditLog(DATA_PROCESSED_DIR / "vol_ledger.jsonl")
latest = {}
for horizon in (5, 20):
    rp = REPORTS_DIR / f"vol_ranges_h{horizon}.json"
    quantiles = json.loads(rp.read_text())["live_quantiles"] if rp.exists() else None
    f = latest_forecasts(bars, horizon, vix, announcements, quantiles=quantiles, history=(history := load_history_training(horizon)))
    new = record(ledger, f, "v2" if history is None else "v4-history-extra" if horizon in EXTRA_FEATURE_HORIZONS else "v3-history")
    date = str(f.event_time.iloc[0].date())
    latest[str(horizon)] = {"data_date": date, "forecasts": {r.instrument_id: {
        "prob_above_usual": round(float(r.prob_above_usual), 4), "forecast_vol_annualised": round(float(r.forecast_vol_annualised), 4),
        "usual_vol_annualised": round(float(r.usual_vol_annualised), 4),
        **({"move_low_90": round(float(r.move_low_90), 4), "move_high_90": round(float(r.move_high_90), 4)} if quantiles else {})} for r in f.itertuples()}}
    print(f"h={horizon} data={date} names={len(f)} recorded={new}")
    print(f.head(5).drop(columns=["event_time", "med_rv", "horizon_days"]).to_string(index=False))
(DATA_PROCESSED_DIR / "volatility_latest.json").write_text(json.dumps(latest))
print("forward scorecard:", json.dumps(score(ledger, bars)) or "{}", "| chain ok:", ledger.verify()[0])
