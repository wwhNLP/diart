"""流式说话人确认端到端评估：diart 流式跑拼接会议音频，与时间戳真值比对。

支持：
- 单/多阈值一键评估（--thresholds "0.4,0.47,0.5"，默认 0.5 EER 校准值）
- 逐段明细导出（CSV：每段的预测标签/相似度/是否确认/真值/正确性/错误类型）
- 会议汇总 + 多阈值对比表（Markdown/JSON），格式与《技术报告》表 4.3(0) 一致

用法:
    diart_env python scripts/eval_verification.py \
        --thresholds 0.5 --out docs/results/reproduce

    # 复现完整三档对比表：
    diart_env python scripts/eval_verification.py \
        --thresholds 0.4,0.47,0.5 --out docs/results/reproduce
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import torch

from diart import models as m
from diart.blocks import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import FileAudioSource

BASE = "test_spk/说话人识别测试"
VP_DIR = f"{BASE}/声纹库"
AUDIO_DIR = f"{BASE}/拼接会议/音频"
TXT_DIR = f"{BASE}/拼接会议/文本"


def time_str_to_ms(time_str: str) -> int:
    m_, s = time_str.split(":")
    return int(int(m_) * 60000 + float(s) * 1000)


def parse_ground_truth(txt_path: str) -> list[dict]:
    pattern = re.compile(
        r"\[(\d{2,}:\d{2}\.\d{3})\]\s*->\s*\[(\d{2,}:\d{2}\.\d{3})\]\s*发言人:\s*(.+)"
    )
    segments = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                segments.append(
                    {
                        "start": time_str_to_ms(match.group(1)),
                        "end": time_str_to_ms(match.group(2)),
                        "speaker": match.group(3).strip(),
                    }
                )
    return segments


def true_speaker(pred_start: float, pred_end: float, gt: list[dict]) -> str:
    """重叠时间最长的真实说话人（与 test_spk 的 完整流程测试.py 同规则）。"""
    best, max_overlap = "未找到", 0
    for seg in gt:
        overlap = min(pred_end, seg["end"]) - max(pred_start, seg["start"])
        if overlap > max_overlap:
            max_overlap, best = overlap, seg["speaker"]
    return best


def classify_error(is_known_truth: bool, pred_name: str, true: str) -> str:
    """错误类型：错名(注册人->他人) / 误报(陌生人->注册人) / 漏报(注册人->空) / 正确。"""
    if pred_name == true:
        return "正确"
    if is_known_truth:
        return "漏报(注册人->空)" if pred_name == "" else "错名(注册人->他人)"
    return "误报(陌生人->注册人)" if pred_name != "" else "正确"


def evaluate_meeting(
    audio_path: str,
    config: SpeakerDiarizationConfig,
    meeting: str,
    out_dir: str | None = None,
) -> dict:
    """跑一个会议，返回汇总统计；out_dir 非 None 时导出逐段明细 CSV。"""
    txt_path = next(
        (t for t in glob.glob(os.path.join(TXT_DIR, "*.txt")) if meeting in t), None
    )
    if txt_path is None:
        return None
    gt = parse_ground_truth(txt_path)

    pipeline = SpeakerDiarization(config)
    padding = config.get_file_padding(audio_path)
    source = FileAudioSource(audio_path, config.sample_rate, padding, config.step)
    pipeline.set_timestamp_shift(-padding[0])
    inference = StreamingInference(
        pipeline, source, batch_size=1, do_profile=False, do_plot=False,
        show_progress=False,
    )

    segments = []

    def hook(ann_wav):
        ann, _ = ann_wav
        verifier = pipeline.verifier
        pending = verifier.pending if verifier is not None else {}
        confirmed = verifier.confirmed if verifier is not None else {}
        for turn, _, label in ann.itertracks(yield_label=True):
            if label.startswith("speaker"):
                g = int(label.replace("speaker", ""))
                sim = pending.get(g, (None, -1.0))[1]
                is_confirmed = g in confirmed
            else:
                g = next(
                    (k for k, vs in confirmed.items() if vs.name == label), -1
                )
                sim = pending.get(g, (None, -1.0))[1] if g >= 0 else -1.0
                is_confirmed = g >= 0
            segments.append(
                {
                    "start": turn.start * 1000,
                    "end": turn.end * 1000,
                    "label": label,
                    "similarity": round(float(sim), 4),
                    "confirmed": is_confirmed,
                }
            )

    inference.attach_hooks(hook)
    prediction = inference()

    # 导出原始 RTTM（流式预测本体：时间戳 + 标签，已确认段为人名）
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        prediction.uri = meeting
        rttm_path = os.path.join(out_dir, f"{meeting}.rttm")
        with open(rttm_path, "w", encoding="utf-8") as f:
            prediction.write_rttm(f)
        print(f"  原始输出已导出: {rttm_path} ({len(prediction)} 段)", flush=True)

    # 逐段与真值比对
    rows = []
    total = correct = total_known = correct_known = 0
    confusions = collections.Counter()
    err_types = collections.Counter()
    for seg in segments:
        true = true_speaker(seg["start"], seg["end"], gt)
        is_known_truth = "陌生人" not in true
        pred_name = "" if seg["label"].startswith("speaker") else seg["label"]
        err = classify_error(is_known_truth, pred_name, true)
        rows.append(
            {
                "会议": meeting,
                "开始ms": int(seg["start"]),
                "结束ms": int(seg["end"]),
                "预测标签": seg["label"],
                "相似度EMA": seg["similarity"],
                "是否确认": "是" if seg["confirmed"] else "否",
                "真值说话人": true,
                "是否正确": "是" if err == "正确" else "否",
                "错误类型": err,
            }
        )
        total += 1
        if err == "正确":
            correct += 1
            if is_known_truth:
                correct_known += 1
        else:
            err_types[err] += 1
            confusions[(true, pred_name or "(空)")] += 1
        if is_known_truth:
            total_known += 1

    # 导出明细 CSV（utf-8-sig 便于 Excel 打开）
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{meeting}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"  明细已导出: {csv_path} ({len(rows)} 段)", flush=True)

    summary = {
        "会议": meeting,
        "总段数": total,
        "正确段": correct,
        "准确率": round(correct / total * 100, 1) if total else 0.0,
        "已知说话人段": f"{correct_known}/{total_known}",
        "已知段准确率": round(correct_known / total_known * 100, 1) if total_known else 0.0,
        "错误分类": dict(err_types),
        "主要混淆": [
            {"真值": t, "预测": p or "(空)", "段数": n}
            for (t, p), n in confusions.most_common(5)
        ],
    }
    print(
        f"[{meeting}] 段数={total} 准确率={summary['准确率']}% "
        f"(已知说话人段 {correct_known}/{total_known})",
        flush=True,
    )
    print(f"    错误分类: {dict(err_types)}", flush=True)
    for (true, pred), n in confusions.most_common(5):
        print(f"    混淆: 真值[{true}] -> 预测[{pred}] x{n}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-new", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=None,
                        help="单阈值（旧参数，与 --thresholds 二选一）")
    parser.add_argument("--thresholds", type=str, default="0.5",
                        help="逗号分隔的阈值列表，默认 0.5（EER 校准值）")
    parser.add_argument("--min-chunks", type=int, default=3)
    parser.add_argument("--meeting", type=str, default=None, help="只跑包含该关键字的会议")
    parser.add_argument("--out", type=str, default="docs/results/reproduce",
                        help="输出目录（逐段明细 CSV + 汇总表）")
    args = parser.parse_args()

    thresholds = (
        [str(args.threshold)]
        if args.threshold is not None
        else [t.strip() for t in args.thresholds.split(",") if t.strip()]
    )
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    print(f"输出目录: {out_root}", flush=True)
    print(f"阈值列表: {thresholds}", flush=True)

    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    meetings = [
        os.path.splitext(os.path.basename(a))[0].replace("_完整版", "")
        for a in audio_files
        if args.meeting is None or args.meeting in os.path.basename(a)
    ]

    # {阈值: {会议: summary}}
    all_summaries: dict[str, dict[str, dict]] = {}

    for threshold in thresholds:
        t_out = os.path.join(out_root, threshold)
        os.makedirs(t_out, exist_ok=True)
        print(f"\n{'='*60}\n阈值 {threshold}\n{'='*60}", flush=True)
        config = SpeakerDiarizationConfig(
            segmentation=m.SegmentationModel.from_pretrained("models/segmentation-3.0"),
            embedding=m.EmbeddingModel.from_pretrained("models/wespeaker-voxceleb-resnet34-LM"),
            duration=5, step=0.5, latency=0.5, device=torch.device("cpu"),
            delta_new=args.delta_new,
            voiceprint_dir=VP_DIR,
            verify_threshold=float(threshold),
            verify_min_chunks=args.min_chunks,
        )
        for audio_path in audio_files:
            meeting = os.path.splitext(os.path.basename(audio_path))[0].replace("_完整版", "")
            if args.meeting is not None and args.meeting not in meeting:
                continue
            summary = evaluate_meeting(audio_path, config, meeting, t_out)
            if summary is not None:
                all_summaries.setdefault(threshold, {})[meeting] = summary

        # 阈值级汇总
        with open(os.path.join(t_out, "meeting_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_summaries[threshold], f, ensure_ascii=False, indent=2)

    # 多阈值对比表（与《技术报告》4.3(0) 同格式）
    th_labels = {t: t for t in thresholds}
    header = "| 会议 | " + " | ".join(f"{t}（{'EER' if t == '0.5' else '经验值' if t == '0.4' else 'FAR≤1%'}）" for t in thresholds) + " |"
    sep = "|" + "---|" * (len(thresholds) + 1)
    lines = ["# 流式说话人确认 — 多阈值对比（可复现输出）", "",
             f"- 参数：delta_new=0.8, min_chunks=3, 5 会议 2,663 段，全部可复现",
             f"- 逐段明细 CSV：`{out_root}/<阈值>/<会议>.csv`（预测标签/相似度EMA/是否确认/真值/错误类型）", "",
             header, sep]
    totals = {}
    for m_name in meetings:
        row = [m_name]
        for t in thresholds:
            s = all_summaries.get(t, {}).get(m_name)
            row.append(f"{s['准确率']:.1f}%" if s else "[待补充]")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**合计**"]
    for t in thresholds:
        ss = all_summaries.get(t, {}).values()
        if ss:
            tot = sum(s["正确段"] for s in ss)
            total_seg = sum(s["总段数"] for s in ss)
            row.append(f"**{tot / total_seg * 100:.1f}%**" if len(ss) == len(meetings) else "[待补充]")
        else:
            row.append("[待补充]")
    lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines) + "\n"

    with open(os.path.join(out_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\n对比表已生成: {os.path.join(out_root, 'summary.md')}\n{'='*60}")
    print(table)


if __name__ == "__main__":
    main()
