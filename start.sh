#!/usr/bin/env bash
# Start all ResearchAtlas servers in one terminal.
# Press Ctrl+C once to stop everything.

trap 'echo ""; echo "Stopping all servers..."; kill 0' EXIT

echo "Starting ResearchAtlas..."

uvicorn app.main:app --reload --port 8000 &
uvicorn app.mcp_server.server:app --reload --port 8001 &
python -m phoenix.server.main serve &

echo "  App    → http://localhost:8000"
echo "  Tools  → http://localhost:8001"
echo "  Phoenix→ http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop all servers."

wait
