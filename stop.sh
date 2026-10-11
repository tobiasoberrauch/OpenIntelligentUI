#!/bin/bash
# Stop OpenIntelligentUI (agent :8123 + app :3000), matched by port.
for port in 8123 3000; do
  pids=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null)
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null && echo "$port: gestoppt ($pids)"
  else
    echo "$port: war schon frei"
  fi
done
