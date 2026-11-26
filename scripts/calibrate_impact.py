"""Fit ImpactCostModel from real fills:  python3 scripts/calibrate_impact.py fills.csv [--fee-bps 10] [--capital 1e8]
Writes reports/impact_calibration.json. `--dry-run` uses SYNTHETIC fills (pipeline check only; results are meaningless)."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nifty_analyser.eval.calibrate_impact import FILL_COLUMNS, fit_impact, simulate_fills
a = argparse.ArgumentParser(); a.add_argument("fills", nargs="?"); a.add_argument("--fee-bps", type=float, default=10.0); a.add_argument("--capital", type=float, default=1e8); a.add_argument("--dry-run", action="store_true")
a = a.parse_args()
if a.dry_run: fills = simulate_fills()
elif a.fills: fills = pd.read_csv(a.fills)
else: sys.exit(f"give a fills CSV with columns {FILL_COLUMNS} (or --dry-run)")
fit = fit_impact(fills); m = fit.to_model(a.fee_bps, a.capital)
out = {"synthetic": a.dry_run, "n": fit.n, "r2": round(fit.r2, 4), "half_spread_bps": round(fit.half_spread_bps, 2), "ci_half_spread": [round(x, 2) for x in fit.ci_half_spread],
       "impact_coef": round(fit.impact_coef, 3), "ci_impact_coef": [round(x, 3) for x in fit.ci_impact_coef], "model": {"fixed_bps": m.fixed_bps, "half_spread_bps": round(m.half_spread_bps, 2), "impact_coef": round(m.impact_coef, 3), "capital": m.capital}}
print(json.dumps(out, indent=1))
if not a.dry_run: Path("reports/impact_calibration.json").write_text(json.dumps(out, indent=1))
