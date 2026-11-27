#!/bin/zsh
# Double-click in Finder: starts the local server and opens the app. Close the window to stop it.
cd "$(dirname "$0")" || exit 1
URL="http://127.0.0.1:8310/"
if curl -s -o /dev/null -m 2 "$URL"; then open "$URL"; exit 0; fi
flags=()
curl -s -o /dev/null -m 2 http://127.0.0.1:11434/api/tags && flags=(--evidence --explain)   # Ask page, if Ollama is running
(for i in {1..90}; do curl -s -o /dev/null -m 2 "$URL" && { open "$URL"; break; }; sleep 1; done) &
exec python3 scripts/serve.py $flags
