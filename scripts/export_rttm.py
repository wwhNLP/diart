"""流式说话人确认 — 仅导出 RTTM（不评估）。

用最优参数（delta_new=0.8, EER 阈值 0.5, min_chunks=3）流式推理指定音频，
输出说话人确认后的 RTTM（已确认段为人名，未确认段保持 speakerX）。

用法:
    diart_env python scripts/export_rttm.py [--audio 拼接会议/音频] [--out 输出目录]

输出: <out>/<会议名>.rttm
"""
import argparse
import glob
import os

import torch

from diart import models as m
from diart.blocks import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import FileAudioSource

BASE = "test_spk/说话人识别测试"
VP_DIR = f"{BASE}/声纹库"
AUDIO_DIR = f"{BASE}/拼接会议/音频"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=AUDIO_DIR, type=str, help="音频目录（*.wav）")
    parser.add_argument("--out", default="docs/results/rttm_eer05", type=str,
                        help="RTTM 输出目录")
    parser.add_argument("--voiceprint-dir", default=VP_DIR, type=str, help="声纹库目录")
    parser.add_argument("--threshold", default=0.5, type=float, help="确认阈值（EER 校准值）")
    parser.add_argument("--delta-new", default=0.8, type=float, help="聚类新说话人距离阈值")
    parser.add_argument("--min-chunks", default=3, type=int)
    parser.add_argument("--meeting", default=None, type=str, help="只处理包含该关键字的音频")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    config = SpeakerDiarizationConfig(
        segmentation=m.SegmentationModel.from_pretrained("models/segmentation-3.0"),
        embedding=m.EmbeddingModel.from_pretrained("models/wespeaker-voxceleb-resnet34-LM"),
        duration=5, step=0.5, latency=0.5, device=torch.device("cpu"),
        delta_new=args.delta_new,
        voiceprint_dir=args.voiceprint_dir,
        verify_threshold=args.threshold,
        verify_min_chunks=args.min_chunks,
    )

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if args.meeting is not None:
        audio_files = [a for a in audio_files if args.meeting in os.path.basename(a)]
    print(f"待处理 {len(audio_files)} 个音频 -> {args.out}", flush=True)

    for audio_path in audio_files:
        meeting = os.path.splitext(os.path.basename(audio_path))[0].replace("_完整版", "")
        print(f"\n>>> {meeting}", flush=True)

        pipeline = SpeakerDiarization(config)
        padding = config.get_file_padding(audio_path)
        source = FileAudioSource(audio_path, config.sample_rate, padding, config.step)
        pipeline.set_timestamp_shift(-padding[0])
        inference = StreamingInference(
            pipeline, source, batch_size=1, do_profile=False, do_plot=False,
            show_progress=False,
        )
        prediction = inference()
        prediction.uri = meeting

        out_path = os.path.join(args.out, f"{meeting}.rttm")
        with open(out_path, "w", encoding="utf-8") as f:
            prediction.write_rttm(f)
        labels = sorted({s for _, _, s in prediction.itertracks(yield_label=True)})
        print(f"  已导出: {out_path}（{len(prediction)} 段, 标签: {labels}）", flush=True)

    print(f"\n完成，共 {len(audio_files)} 个 RTTM 文件在 {args.out}", flush=True)


if __name__ == "__main__":
    main()
