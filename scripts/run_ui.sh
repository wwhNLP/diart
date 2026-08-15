#!/usr/bin/env bash
# 启动 Web 控制台（中继服务 + 页面服务）。
# 需先启动 diart serve（见 scripts/run_server.sh），然后浏览器访问 http://localhost:7008
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_DIR/src/diart/console/web_ui.py" \
    --host localhost \
    --port 7008 \
    --server-port 7007 \
    --server-host localhost \
    --log-dir "$PROJECT_DIR/logs" \
    --temp-voiceprint-dir "$PROJECT_DIR/temp_voiceprints" \
    --verbose
