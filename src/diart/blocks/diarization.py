from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from pathlib import Path
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m
from .. import verification as verification


class SpeakerDiarizationConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        embedding: m.EmbeddingModel | None = None,
        duration: float = 5,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        normalize_embedding_weights: bool = False,
        device: torch.device | None = None,
        sample_rate: int = 16000,
        voiceprint_dir: str | None = None,
        temp_voiceprint_dir: str | None = None,
        verify_threshold: float = 0.5,
        verify_min_chunks: int = 3,
        verify_ema_alpha: float = 0.3,
        verify_mode: str = "verify",
        identify_min_similarity: float | None = None,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        # Default embedding model is pyannote/embedding
        self.embedding = embedding or m.EmbeddingModel.from_pyannote(
            "pyannote/embedding"
        )

        self._duration = duration
        self._sample_rate = sample_rate

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.normalize_embedding_weights = normalize_embedding_weights
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 流式说话人确认参数（voiceprint_dir 为 None 时关闭）
        self.voiceprint_dir = voiceprint_dir
        self.temp_voiceprint_dir = temp_voiceprint_dir
        self.verify_threshold = verify_threshold
        self.verify_min_chunks = verify_min_chunks
        self.verify_ema_alpha = verify_ema_alpha
        # 双模式：verify（确认，默认）/ identify（识别，open-set）
        if verify_mode not in ("verify", "identify"):
            raise ValueError(f"verify_mode 只能是 verify / identify，收到: {verify_mode!r}")
        self.verify_mode = verify_mode
        self.identify_min_similarity = identify_min_similarity

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class SpeakerDiarization(base.Pipeline):
    def __init__(self, config: SpeakerDiarizationConfig | None = None):
        self._config = SpeakerDiarizationConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device
        )
        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding,
            self._config.gamma,
            self._config.beta,
            norm=1,
            normalize_weights=self._config.normalize_embedding_weights,
            device=self._config.device,
        )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        # 流式说话人确认器（说话人确认功能）
        self.verifier = None
        self.verification: dict = {}
        self._voiceprint_provider = None
        if self._config.voiceprint_dir is not None:
            directories = [self._config.voiceprint_dir]
            if self._config.temp_voiceprint_dir is not None:
                temp_dir = Path(self._config.temp_voiceprint_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                directories.append(str(temp_dir))
                print(f"[说话人确认] 临时声纹库: {temp_dir}（与主库合并注册）")
            provider = verification.DirectoryVoiceprints(
                directories,
                self.embedding.embedding.model,
                device=self._config.device,
            )
            self._voiceprint_provider = provider
            voiceprints = provider.load()
            if voiceprints:
                self.verifier = verification.StreamingSpeakerVerifier(
                    voiceprints,
                    threshold=self._config.verify_threshold,
                    min_chunks=self._config.verify_min_chunks,
                    ema_alpha=self._config.verify_ema_alpha,
                    max_speakers=self._config.max_speakers,
                    mode=self._config.verify_mode,
                    identify_min_similarity=self._config.identify_min_similarity,
                )
                print(
                    f"[说话人确认] 已加载 {len(voiceprints)} 个注册说话人, "
                    f"阈值 {self._config.verify_threshold}, "
                    f"确认所需连续chunk数 {self._config.verify_min_chunks}, "
                    f"模式 {self._config.verify_mode}"
                )
            else:
                print("[说话人确认] 警告: 声纹库为空，说话人确认已禁用")

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive, base.RhoUpdate, base.DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reload_voiceprints(self):
        """热重载声纹库（主库 + 临时库重新扫描），保留已确认状态。

        支持惰性启用：启动时声纹库为空不创建 verifier，首次重载拿到声纹时
        再构造；库被清空时禁用匹配（清空矩阵）。
        """
        if self._voiceprint_provider is None:
            print("[说话人确认] 未启用声纹确认，忽略重载请求")
            return
        self._voiceprint_provider.refresh()
        voiceprints = self._voiceprint_provider.load()
        if self.verifier is None:
            if not voiceprints:
                print("[说话人确认] 警告: 声纹库为空，确认功能保持禁用")
                return
            # 惰性构造：库在启动后才有内容（如 web 上传首条声纹）
            self.verifier = verification.StreamingSpeakerVerifier(
                voiceprints,
                threshold=self._config.verify_threshold,
                min_chunks=self._config.verify_min_chunks,
                ema_alpha=self._config.verify_ema_alpha,
                max_speakers=self._config.max_speakers,
                mode=self._config.verify_mode,
                identify_min_similarity=self._config.identify_min_similarity,
            )
            print(
                f"[说话人确认] 已加载 {len(voiceprints)} 个注册说话人, "
                f"阈值 {self._config.verify_threshold}, "
                f"确认所需连续chunk数 {self._config.verify_min_chunks}, "
                f"模式 {self._config.verify_mode}"
            )
            return
        self.verifier.update_voiceprints(voiceprints)

    def reset(self):
        self.set_timestamp_shift(0)
        self.clustering = OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        self.chunk_buffer, self.pred_buffer = [], []
        if self.verifier is not None:
            self.verifier.reset()
        self.verification = {}

    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        """Diarize the next audio chunks of an audio stream.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            A sequence of consecutive audio chunks from an audio stream.

        Returns
        -------
        Sequence[tuple[Annotation, SlidingWindowFeature]]
            Speaker diarization of each chunk alongside their corresponding audio.
        """
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation and embeddings
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
        # embeddings has shape (batch, speakers, emb_dim)
        embeddings = self.embedding(batch, segmentations)

        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

        outputs = []
        for wav, seg, emb in zip(waveforms, segmentations, embeddings):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

            # Update clustering state and permute segmentation
            permuted_seg = self.clustering(seg, emb)

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction)

            # 流式说话人确认：匹配聚类质心 -> 重命名标签 -> 附确认信息
            if self.verifier is not None and self.clustering.centers is not None:
                self.verifier.update(
                    self.clustering.centers, self.clustering.active_centers
                )
                agg_prediction = self.verifier.rename_annotation(agg_prediction)
            # 输出属性：verify 模式 -> 确认映射；identify 模式 -> Top-1 识别映射（含 top3）
            if self.verifier is not None:
                source = (
                    self.verifier.confirmed
                    if self.verifier.mode == "verify"
                    else self.verifier.identifications
                )
                self.verification = {
                    g_spk: {
                        "name": v.name,
                        "id": v.id,
                        "similarity": v.similarity,
                        "top3": [(n, s) for n, s in v.top3],
                    }
                    for g_spk, v in source.items()
                }
            else:
                self.verification = {}
            agg_prediction.speaker_verification = dict(self.verification)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(
                    yield_label=True
                ):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs
