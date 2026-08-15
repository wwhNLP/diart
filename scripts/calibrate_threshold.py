"""EER 阈值校准工具（pyannote 官方策略迁移：方式 A）。

用拼接会议的真值标注段构建"同人/异人"验证对，提取声纹 embedding，
计算相似度分布与等错误率（EER）阈值，替代固定阈值 0.4。

用法:
    diart_env python scripts/calibrate_threshold.py [--out docs/results/calibration]

输出:
    docs/results/calibration/threshold.json   阈值与分布统计
    docs/results/calibration/eer_curve.png    FAR/FRR 曲线与 EER 点
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

from diart import models as m

BASE = Path("test_spk/说话人识别测试")
AUDIO_DIR = BASE / "拼接会议/音频"
TXT_DIR = BASE / "拼接会议/文本"
VP_DIR = BASE / "声纹库"
MODEL_DIR = Path("models/wespeaker-voxceleb-resnet34-LM")

SAMPLE_RATE = 16000
WINDOW = 3.0
STEP = 1.5
MAX_AUDIO_SECONDS = 30.0
MIN_SEGMENT_SECONDS = 0.5

_TIME_PATTERN = re.compile(
    r"\[(\d{2,}:\d{2}\.\d{3})\]\s*->\s*\[(\d{2,}:\d{2}\.\d{3})\]\s*发言人:\s*(.+)"
)


def time_to_seconds(time_str: str) -> float:
    m_, s = time_str.split(":")
    return int(m_) * 60 + float(s)


def parse_ground_truth(txt_path: Path) -> list[dict]:
    segments = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            match = _TIME_PATTERN.search(line)
            if match:
                segments.append(
                    {
                        "start": time_to_seconds(match.group(1)),
                        "end": time_to_seconds(match.group(2)),
                        "speaker": match.group(3).strip(),
                    }
                )
    return segments


def load_segment_embedding(
    waveform: torch.Tensor, start: float, end: float, embedding_model
) -> np.ndarray | None:
    """切出 [start, end) 语音段并滑窗提取 embedding（均值，同注册策略）。"""
    start_s = int(start * SAMPLE_RATE)
    end_s = min(int(end * SAMPLE_RATE), waveform.shape[1])
    if end_s - start_s < int(MIN_SEGMENT_SECONDS * SAMPLE_RATE):
        return None
    audio = waveform[0, start_s:end_s].float()
    audio = audio[: int(MAX_AUDIO_SECONDS * SAMPLE_RATE)]

    window_samples = int(WINDOW * SAMPLE_RATE)
    step_samples = int(STEP * SAMPLE_RATE)
    embeddings = []
    pos = 0
    while pos + window_samples <= audio.numel():
        chunk = audio[pos : pos + window_samples].reshape(1, 1, -1)
        with torch.no_grad():
            emb = embedding_model(chunk)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        pos += step_samples
    if not embeddings and audio.numel() >= int(0.1 * SAMPLE_RATE):
        with torch.no_grad():
            emb = embedding_model(audio.reshape(1, 1, -1))
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))

    valid = [e for e in embeddings if np.all(np.isfinite(e))]
    if not valid:
        return None
    mean = np.mean(np.vstack(valid), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-10 else None


def compute_eer(same_sims: np.ndarray, diff_sims: np.ndarray) -> dict:
    """网格搜索 EER 阈值，输出 FAR/FRR 曲线关键点。"""
    grid = np.arange(0.05, 0.95, 0.005)
    best = None
    curve = []
    for t in grid:
        far = float(np.mean(diff_sims >= t))  # 异人对误接受率
        frr = float(np.mean(same_sims < t))   # 同人对误拒绝率
        curve.append({"threshold": round(float(t), 3), "far": far, "frr": frr})
        if best is None or abs(far - frr) < abs(best["far"] - best["frr"]):
            best = {"threshold": round(float(t), 3), "far": far, "frr": frr}
    # 额外候选：FAR<=1% 与 FAR<=5% 的最严阈值（生产更保守）
    # 网格内可能没有阈值满足目标：此时不静默回退到最后一个网格点
    # （该点同样不满足约束，会误导用户部署），而是显式置 None 并提示
    far1 = next((c for c in curve if c["far"] <= 0.01), None)
    far5 = next((c for c in curve if c["far"] <= 0.05), None)
    if far1 is None:
        print(
            f"⚠ 提示: 网格内无阈值满足 FAR≤1%（最低 FAR="
            f"{min(c['far'] for c in curve):.3f}），threshold_far1 置为 null",
            flush=True,
        )
    if far5 is None:
        print(
            f"⚠ 提示: 网格内无阈值满足 FAR≤5%（最低 FAR="
            f"{min(c['far'] for c in curve):.3f}），threshold_far5 置为 null",
            flush=True,
        )
    far1 = far1 or {"threshold": None, "far": None, "frr": None}
    far5 = far5 or {"threshold": None, "far": None, "frr": None}
    return {
        "eer": (best["far"] + best["frr"]) / 2,
        "eer_threshold": best["threshold"],
        "far_at_eer": best["far"],
        "frr_at_eer": best["frr"],
        "threshold_far1": far1["threshold"],
        "far_at_threshold_far1": far1["far"],
        "frr_at_threshold_far1": far1["frr"],
        "threshold_far5": far5["threshold"],
        "far_at_threshold_far5": far5["far"],
        "frr_at_threshold_far5": far5["frr"],
        "num_same_pairs": int(len(same_sims)),
        "num_diff_pairs": int(len(diff_sims)),
        "same_mean": round(float(np.mean(same_sims)), 4),
        "same_std": round(float(np.std(same_sims)), 4),
        "diff_mean": round(float(np.mean(diff_sims)), 4),
        "diff_std": round(float(np.std(diff_sims)), 4),
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/results/calibration", type=str)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 注册库名单（用于同人对判定：同名且属于注册库才视为同人）
    registered = {d.name for d in VP_DIR.iterdir() if d.is_dir()}

    print("加载声纹模型...", flush=True)
    embedding_model = m.EmbeddingModel.from_pretrained(str(MODEL_DIR))
    embedding_model.eval()  # 关键：与管道一致（LazyModel 不自动 eval）

    # 1. 收集每个真值段的 embedding
    speaker_embeddings: dict[str, list[np.ndarray]] = {}
    for audio_path in sorted(AUDIO_DIR.glob("*.wav")):
        meeting = audio_path.name.replace("_完整版.wav", "")
        txt_path = next((p for p in TXT_DIR.glob("*.txt") if meeting in p.name), None)
        if txt_path is None:
            continue
        gt = parse_ground_truth(txt_path)
        print(f"加载 {audio_path.name} ...", flush=True)
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        for seg in gt:
            emb = load_segment_embedding(waveform, seg["start"], seg["end"], embedding_model)
            if emb is not None:
                speaker_embeddings.setdefault(seg["speaker"], []).append(emb)
        print(f"  已提取 {sum(len(v) for v in speaker_embeddings.values())} 段", flush=True)

    # 2. 构建同人/异人对
    same_sims, diff_sims = [], []
    names = list(speaker_embeddings)
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j < i:
                continue
            embs_a = speaker_embeddings[name_a]
            embs_b = speaker_embeddings[name_b]
            if name_a == name_b:
                # 同人对：注册库内同名（陌生人同名不代表同一人，跳过）
                if name_a not in registered:
                    continue
                for x in range(len(embs_a)):
                    for y in range(x + 1, len(embs_a)):
                        same_sims.append(float(embs_a[x] @ embs_a[y]))
            else:
                for x in embs_a:
                    for y in embs_b:
                        diff_sims.append(float(x @ y))

    same_sims, diff_sims = np.asarray(same_sims), np.asarray(diff_sims)
    if len(same_sims) == 0 or len(diff_sims) == 0:
        raise SystemExit(
            "✗ 无法构建同人/异人对（same=%d, diff=%d）：请检查声纹库目录与真值数据，"
            "至少需要 2 个已注册说话人且每人 >= 2 个可用段。"
            % (len(same_sims), len(diff_sims))
        )
    print(
        f"同人对 {len(same_sims)} 个 (均值 {same_sims.mean():.3f}) | "
        f"异人对 {len(diff_sims)} 个 (均值 {diff_sims.mean():.3f})",
        flush=True,
    )

    # 3. EER 计算
    result = compute_eer(same_sims, diff_sims)
    print(
        f"EER = {result['eer']*100:.1f}% @ 阈值 {result['eer_threshold']} "
        f"(FAR={result['far_at_eer']*100:.1f}%, FRR={result['frr_at_eer']*100:.1f}%)",
        flush=True,
    )
    print(
        f"FAR≤1% 阈值 {result['threshold_far1']} "
        f"(FRR={result['frr_at_threshold_far1']*100:.1f}%) | "
        f"FAR≤5% 阈值 {result['threshold_far5']} "
        f"(FRR={result['frr_at_threshold_far5']*100:.1f}%)",
        flush=True,
    )

    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 4. 曲线图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    curve = result["curve"]
    ts = [c["threshold"] for c in curve]
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(ts, [c["far"] for c in curve], label="FAR（异人误接受率）", color="#C44E52")
    ax1.plot(ts, [c["frr"] for c in curve], label="FRR（同人误拒绝率）", color="#4C72B0")
    ax1.axvline(result["eer_threshold"], color="gray", ls="--", lw=1)
    ax1.text(result["eer_threshold"] + 0.01, 0.85, f"EER 阈值 {result['eer_threshold']}",
             fontsize=9, color="gray")
    ax1.set_xlabel("相似度阈值"); ax1.set_ylabel("错误率")
    ax1.set_ylim(0, 1); ax1.legend(fontsize=9)
    ax1.set_title("EER 校准：FAR/FRR vs 相似度阈值", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "eer_curve.png", dpi=150)
    print(f"已保存: {out_dir / 'threshold.json'} / {out_dir / 'eer_curve.png'}", flush=True)


if __name__ == "__main__":
    main()
