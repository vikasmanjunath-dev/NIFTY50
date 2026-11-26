"""Today's pre-open Nifty direction signal from the latest S&P 500 close (needs network). Run: python3 scripts/overnight_signal.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, yfinance as yf
from nifty_analyser.ops.overnight import signal
c = yf.download("^GSPC", period="10d", progress=False, auto_adjust=True)["Close"].dropna().squeeze()
r = float(np.log(c.iloc[-1] / c.iloc[-2])); s = signal(r)
print(f"S&P 500 session {c.index[-1].date()}: {r:+.2%}  ->  {s.direction or 'ABSTAIN'}"); print(s.note)
if s.expected_accuracy: print("historical accuracy for this move size:", s.expected_accuracy)
