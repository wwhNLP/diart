#!/usr/bin/env bash
# =============================================================================
# 流式说话人确认 — 一键复现评估脚本
#
# 功能：跑指定阈值（默认 0.5 EER 校准值）下全部 5 个会议，
#       导出逐段明细 CSV + 多阈值对比表，供其他人复现《技术报告》性能数据。
#
# 用法：
#   bash scripts/reproduce_verification.sh            # 只跑 0.5（默认）
#   bash scripts/reproduce_verification.sh 0.5        # 同上
#   bash scripts/reproduce_verification.sh 0.4,0.47,0.5   # 复现完整三档对比表
#
# 前置条件：
#   1. diart conda 环境（Python 3.10, torch, pyannote.audio 4.x）
#   2. 模型已下载：models/segmentation-3.0、models/wespeaker-voxceleb-resnet34-LM
#      （ModelScope: modelscope download --model pyannote/segmentation-3.0 --local_dir models/segmentation-3.0）
#   3. 数据齐全：test_spk/说话人识别测试/{声纹库,拼接会议}
#   4. 可从任意目录调用（脚本自动定位项目根目录）
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

THRESHOLDS="${1:-0.5}"
OUT="docs/results/reproduce"
# 便携解释器：优先 $PYTHON 环境变量，其次 PATH 中的 python/python3
# （set -e 下 command -v 失败会中止脚本，需 || true 让友好报错生效）
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"

# 阈值校验：非空、逗号分隔的 [0,1] 小数列表（允许 1 / 1.0）
if ! [[ "${THRESHOLDS}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)(,(0(\.[0-9]+)?|1(\.0+)?))*$ ]]; then
    echo "✗ 非法阈值列表: ${THRESHOLDS}（示例: 0.5 或 0.4,0.47,0.5）"
    exit 1
fi

echo "=============================================="
echo " 流式说话人确认 — 一键复现"
echo " 项目目录: ${PROJECT_DIR}"
echo " 阈值: ${THRESHOLDS}"
echo " 输出: ${OUT}"
echo "=============================================="

# ── 环境检查 ────────────────────────────────────────────────────────────────
echo "[1/3] 检查环境..."
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "✗ 找不到 Python: ${PYTHON}（可用 PYTHON=/path/to/python 指定）"; exit 1; }
# 模型目录（pyannote Model.from_pretrained 需要完整目录，而非单个 pytorch_model.bin）
for f in models/segmentation-3.0 \
         models/wespeaker-voxceleb-resnet34-LM; do
    [ -d "$f" ] || { echo "✗ 缺少模型目录: $f（先用 ModelScope 下载）"; exit 1; }
done
[ -d "test_spk/说话人识别测试/声纹库" ] || { echo "✗ 缺少声纹库目录"; exit 1; }
# 会议音频与真值文本：eval_verification.py 按 .wav 开会、按 .txt 匹配真值，
# 缺任一文件都会"成功"但产出空表，白耗算力
[ -n "$(find "test_spk/说话人识别测试/拼接会议/音频" -maxdepth 1 -name '*.wav' -print -quit 2>/dev/null)" ] \
    || { echo "✗ 未找到会议音频 .wav（test_spk/说话人识别测试/拼接会议/音频）"; exit 1; }
[ -n "$(find "test_spk/说话人识别测试/拼接会议/文本" -maxdepth 1 -name '*.txt' -print -quit 2>/dev/null)" ] \
    || { echo "✗ 未找到真值 .txt（test_spk/说话人识别测试/拼接会议/文本）"; exit 1; }
echo "  ✓ 环境就绪"

# ── 运行评估 ────────────────────────────────────────────────────────────────
echo "[2/3] 开始评估（每个会议约 2 分钟，阈值越多耗时越长）..."
PYTHONPATH=src "${PYTHON}" -u scripts/eval_verification.py \
    --thresholds "${THRESHOLDS}" --out "${OUT}"

# ── 汇总 ────────────────────────────────────────────────────────────────────
echo "[3/3] 完成。输出文件："
echo "  - 对比表:   ${OUT}/summary.md"
echo "  - JSON:     ${OUT}/summary.json"
echo "  - 逐段明细: ${OUT}/<阈值>/<会议>.csv"
echo ""
echo "================ 对比表 ================"
cat "${OUT}/summary.md"
