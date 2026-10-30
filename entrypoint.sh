#!/bin/sh
set -eu
cd /app
exec /app/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}"