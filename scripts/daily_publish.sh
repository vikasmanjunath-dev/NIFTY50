#!/bin/zsh
# Refresh data, export the static snapshot, deploy it. Run daily by the Claude app scheduled task "nifty-tremor-daily-publish".
set -eo pipefail
cd "${0:A:h}/.."
echo "== $(date)"
python3 scripts/daily_refresh.py
python3 scripts/export_static.py
vercel deploy dist/nifty-tremor --prod --yes >/dev/null
echo "published $(date)"
# Fail loudly if the live site is not serving the data just exported.
want=$(grep -o '"data_date":"[^"]*"' dist/nifty-tremor/data/overview.json | head -1)
got=$(curl -fsS --retry 5 --retry-delay 5 https://nifty-tremor.vercel.app/data/overview.json | grep -o '"data_date":"[^"]*"' | head -1)
[[ -n $want && $want == $got ]] || { echo "live check failed: want $want, got $got"; exit 1; }
echo "live check ok: $got"
