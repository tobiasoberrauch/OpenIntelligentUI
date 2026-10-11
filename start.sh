#!/bin/bash
# Start OpenIntelligentUI locally: agent (:8123) + app (:3000).
# Requires: apps/agent/.env (see .env.example), built app (pnpm build).
# Note: port 3000 must be free (Hermes WhatsApp bridge was moved to 3900).
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$ROOT/apps/agent/.env" ]; then
  echo "FEHLT: apps/agent/.env — siehe .env.example" >&2
  exit 1
fi
if [ ! -f "$ROOT/apps/app/.next/standalone/apps/app/server.js" ]; then
  echo "FEHLT: Production-Build — erst 'pnpm build' laufen lassen" >&2
  exit 1
fi
if lsof -nP -iTCP:3000 -sTCP:LISTEN > /dev/null 2>&1; then
  echo "Port 3000 ist belegt:" >&2
  lsof -nP -iTCP:3000 -sTCP:LISTEN | head -3 >&2
  exit 1
fi

cd "$ROOT/apps/agent" && nohup uv run uvicorn main:app --host 0.0.0.0 --port 8123 > /tmp/oigu-agent.log 2>&1 &
cd "$ROOT/apps/app/.next/standalone/apps/app" && nohup env PORT=3000 HOSTNAME=0.0.0.0 node server.js > /tmp/oigu-app.log 2>&1 &
sleep 6
echo "Agent : $(curl -s -m 3 http://localhost:8123/health || echo DEFEKT)"
echo "App   : $(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:3000) (http://localhost:3000)"
echo "Logs  : /tmp/oigu-agent.log /tmp/oigu-app.log"
