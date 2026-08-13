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
#   4. 工作目录为项目根目录（含 src/）
# =============================================================================
set -euo pipefail

THRESHOLDS="${1:-0.5}"
OUT="docs/results/reproduce"
PYTHON="${PYTHON:-/home/wwh/miniforge3/envs/diart/bin/python}"

echo "=============================================="
echo " 流式说话人确认 — 一键复现"
echo " 阈值: ${THRESHOLDS}"
echo " 输出: ${OUT}"
echo "=============================================="

# ── 环境检查 ────────────────────────────────────────────────────────────────
echo "[1/3] 检查环境..."
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "✗ 找不到 Python: ${PYTHON}（可用 PYTHON=/path/to/python 指定）"; exit 1; }
for f in models/segmentation-3.0/pytorch_model.bin \
         models/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin; do
    [ -f "$f" ] || { echo "✗ 缺少模型: $f（先用 ModelScope 下载）"; exit 1; }
done
[ -d "test_spk/说话人识别测试/声纹库" ] || { echo "✗ 缺少声纹库目录"; exit 1; }
[ -d "test_spk/说话人识别测试/拼接会议/音频" ] || { echo "✗ 缺少拼接会议音频目录"; exit 1; }
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
