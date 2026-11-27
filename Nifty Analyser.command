#!/bin/zsh
# Double-click in Finder: starts the local server and opens the app. Close the window to stop it.
cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:8310/"
if curl -s -o /dev/null -m 2 "$URL"; then open "$URL"; exit 0; fi
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python          # the environment from the setup steps, if there is one
"$PY" -c "import fastapi, uvicorn, pandas" 2>/dev/null || { echo "Missing packages. Run: $PY -m pip install -r requirements.txt"; exit 1; }
flags=()
curl -s -o /dev/null -m 2 http://127.0.0.1:11434/api/tags && flags=(--evidence --explain)   # Ask page, if Ollama is running
(for i in {1..90}; do curl -s -o /dev/null -m 2 "$URL" && { open "$URL"; break; }; sleep 1; done) &
exec "$PY" scripts/serve.py $flags
