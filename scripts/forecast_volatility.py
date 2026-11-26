"""Forecast 5- and 20-day volatility for current index members, record them in the forward ledger, score matured ones."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.config import DATA_PROCESSED_DIR
from nifty_analyser.ops.audit import AuditLog
from nifty_analyser.ops.vol_forecast import latest_forecasts, load_inputs, record, score

bars, vix, announcements = load_inputs()
ledger = AuditLog(DATA_PROCESSED_DIR / "vol_ledger.jsonl")
latest = {}
for horizon in (5, 20):
    f = latest_forecasts(bars, horizon, vix, announcements)
    new = record(ledger, f)
    date = str(f.event_time.iloc[0].date())
    latest[str(horizon)] = {"data_date": date, "forecasts": {r.instrument_id: {
        "prob_above_usual": round(float(r.prob_above_usual), 4), "forecast_vol_annualised": round(float(r.forecast_vol_annualised), 4),
        "usual_vol_annualised": round(float(r.usual_vol_annualised), 4)} for r in f.itertuples()}}
    print(f"h={horizon} data={date} names={len(f)} recorded={new}")
    print(f.head(5)[["instrument_id", "prob_above_usual", "forecast_vol_annualised", "usual_vol_annualised"]].to_string(index=False))
(DATA_PROCESSED_DIR / "volatility_latest.json").write_text(json.dumps(latest))
print("forward scorecard:", json.dumps(score(ledger, bars)) or "{}", "| chain ok:", ledger.verify()[0])
