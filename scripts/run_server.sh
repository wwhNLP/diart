#!/usr/bin/env bash
# 使用本地已下载的模型（ModelScope: models/segmentation-3.0、models/wespeaker-voxceleb-resnet34-LM）
# 离线启动 WebSocket 流式说话人确认服务（主声纹库 + 临时声纹库合并注册）。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_DIR/src/diart/console/serve.py" \
    --host localhost \
    --port 7007 \
    --cpu \
    --offline \
    --segmentation "$PROJECT_DIR/models/segmentation-3.0" \
    --embedding "$PROJECT_DIR/models/wespeaker-voxceleb-resnet34-LM" \
    --voiceprint-dir "$PROJECT_DIR/test_spk/说话人识别测试/声纹库" \
    --temp-voiceprint-dir "$PROJECT_DIR/temp_voiceprints" \
    --verify-mode identify
