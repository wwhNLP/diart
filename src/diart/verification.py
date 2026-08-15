"""流式说话人确认（说话人验证 / 声纹匹配）。

参考 ASR-stream-SPK-PY 项目的说话人确认机制：
- 注册声纹与管道声纹在同一个向量空间（默认 256 维 wespeaker，与参考库兼容）
- 匹配方式：L2 归一化后的余弦相似度（numpy 批量矩阵乘法），阈值默认 0.5（EER 校准）
- 流式化：每个 chunk 用聚类累积质心匹配一次，EMA 平滑 + 连续命中确认状态机，
  确认后把 diart 的 "speaker0/speaker1..." 标签重命名为真实人名。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════════════
# 注册声纹数据结构
# ════════════════════════════════════════════════════════════════════


@dataclass
class RegisteredSpeaker:
    """一个注册说话人的声纹集合。

    embeddings 为 (n, dim) float32 矩阵，每行一条已 L2 归一化的声纹向量。
    """

    name: str
    embeddings: np.ndarray
    id: str = ""

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        assert self.embeddings.ndim == 2, "embeddings must be (n, dim) or (dim,)"

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]


class VoiceprintProvider(ABC):
    """声纹注册源统一接口。"""

    @abstractmethod
    def load(self) -> list[RegisteredSpeaker]:
        """返回注册说话人列表（向量已 L2 归一化）。"""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化，零向量保持不变（避免除零）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return matrix / norms


# ════════════════════════════════════════════════════════════════════
# 目录声纹库注册：<dir>/<说话人姓名>/*.wav
# ════════════════════════════════════════════════════════════════════


class DirectoryVoiceprints(VoiceprintProvider):
    """从本地声纹库目录注册声纹。

    目录结构：``<directory>/<说话人姓名>/<任意>.wav``。
    使用与管道相同的 embedding 模型提取声纹（与 ASR-stream-SPK-PY 的
    ``extract_spk_embedding_pyannote`` 同方案：滑窗取均值），保证注册向量
    与流式管道逐 chunk 的说话人质心在同一个向量空间。

    Parameters
    ----------
    directory: str | Path
        声纹库根目录。
    embedding_model: callable
        与管道一致的声纹提取模型，接受 ``(batch, channels, samples)``
        的 float32 张量，返回 ``(batch, dim)`` 嵌入。
    device: torch.device, optional
        模型推理设备。
    window: float
        滑窗时长（秒），默认 3.0（与参考项目一致）。
    step: float
        滑窗步长（秒），默认 1.5（与参考项目一致）。
    max_audio_seconds: float
        单个文件最多参与注册的时长（秒），默认 30（与参考项目一致）。
    sample_rate: int
        声纹模型要求的采样率，默认 16000。
    """

    SUPPORTED_EXTENSIONS = (".wav", ".m4a", ".mp3", ".flac", ".amr")

    def __init__(
        self,
        directory: str | Path | list[str | Path],
        embedding_model,
        device=None,
        window: float = 3.0,
        step: float = 1.5,
        max_audio_seconds: float = 30.0,
        sample_rate: int = 16000,
    ):
        # 支持单目录（兼容旧用法）或多目录（主库 + 临时库）
        if isinstance(directory, (str, Path)):
            self.directories = [Path(directory)]
        else:
            self.directories = [Path(d) for d in directory]
        for d in self.directories:
            assert d.is_dir(), f"声纹库目录不存在: {d}"
        self.embedding_model = embedding_model
        self.device = device
        self.window = window
        self.step = step
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate
        self._cache: list[RegisteredSpeaker] | None = None

    def refresh(self):
        """清空缓存，下次 load() 重新扫描（热重载用）。"""
        self._cache = None

    def load(self) -> list[RegisteredSpeaker]:
        """扫描目录并提取声纹（结果缓存，refresh() 后重新计算）。

        多目录时按说话人姓名合并：同名说话人的声纹行拼接（主库 + 临时库
        的增量声纹合为一个注册说话人）。
        """
        if self._cache is not None:
            return self._cache

        import torch
        import torchaudio

        by_name: dict[str, list[np.ndarray]] = {}
        for directory in self.directories:
            folders = sorted([f for f in directory.iterdir() if f.is_dir()])
            for folder in folders:
                name = folder.name
                wav_files = sorted(
                    [
                        f
                        for f in folder.iterdir()
                        if f.is_file()
                        and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
                    ]
                )
                if not wav_files:
                    continue
                all_embeddings: list[np.ndarray] = []
                for wav_path in wav_files:
                    waveform, sr = torchaudio.load(str(wav_path))
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(
                            waveform, sr, self.sample_rate
                        )
                    audio = waveform[0].float()  # (samples,)
                    if audio.numel() < int(0.1 * self.sample_rate):
                        continue
                    audio = audio[: int(self.max_audio_seconds * self.sample_rate)]
                    all_embeddings.extend(self._extract_sliding_windows(audio))
                valid = [e for e in all_embeddings if np.all(np.isfinite(e))]
                if valid:
                    by_name.setdefault(name, []).extend(valid)
                else:
                    print(
                        f"[声纹注册] 警告: 跳过 {name} ({wav_path.name}): "
                        "未能提取到有效声纹（音频过短或格式异常）",
                        flush=True,
                    )

        speakers: list[RegisteredSpeaker] = []
        for name in sorted(by_name):
            matrix = _l2_normalize(np.vstack(by_name[name]))
            speakers.append(RegisteredSpeaker(name=name, embeddings=matrix))
            print(
                f"[声纹注册] {name}: {matrix.shape[0]} 条声纹, "
                f"{matrix.shape[1]} 维"
            )

        self._cache = speakers
        return speakers

    def _extract_sliding_windows(self, audio) -> list[np.ndarray]:
        """对一段音频滑窗提取声纹，返回窗口均值。"""
        import torch

        embeddings: list[np.ndarray] = []
        window_samples = int(self.window * self.sample_rate)
        step_samples = int(self.step * self.sample_rate)
        start = 0
        while start + window_samples <= audio.numel():
            chunk = audio[start : start + window_samples]
            embeddings.append(self._embed_one(chunk))
            start += step_samples
        if not embeddings and audio.numel() >= int(0.1 * self.sample_rate):
            # 不足一个窗口：直接用整段
            embeddings.append(self._embed_one(audio))
        return embeddings

    def _embed_one(self, audio) -> np.ndarray:
        import torch

        batch = audio.reshape(1, 1, -1)
        with torch.no_grad():
            embedding = self.embedding_model(batch)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32).reshape(-1)


# ════════════════════════════════════════════════════════════════════
# PostgreSQL 声纹库注册（兼容 ASR-stream-SPK-PY 参考项目的 256 维库）
# ════════════════════════════════════════════════════════════════════


class DBVoiceprints(VoiceprintProvider):
    """从 PostgreSQL 读取注册声纹（参考项目 t_speaker / t_speaker_voiceprint /
    t_voiceprint 三表结构，256 维 pyannote 向量）。

    需安装 psycopg。连接参数通过 dsn 或环境变量 DB_HOST / DB_PORT / DB_NAME /
    DB_USER / DB_PASSWORD 提供。
    """

    VECTOR_DIMENSION = 256

    _SQL = (
        "SELECT spk.c_id AS id, spk.c_name AS name, voi.c_embedding AS embedding "
        "FROM t_speaker spk "
        "INNER JOIN t_speaker_voiceprint sv ON spk.c_id = sv.c_speaker_id "
        "INNER JOIN t_voiceprint voi ON sv.c_voiceprint_id = voi.c_id "
        "WHERE voi.c_embedding IS NOT NULL"
    )

    def __init__(self, dsn: str | None = None, threshold_dim_check: bool = True):
        self.dsn = dsn
        self.threshold_dim_check = threshold_dim_check

    def load(self) -> list[RegisteredSpeaker]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "DBVoiceprints 需要 psycopg 库: pip install psycopg[binary]"
            ) from exc

        conn = psycopg.connect(self.dsn) if self.dsn else psycopg.connect()
        try:
            conn.row_factory = dict_row
            rows = conn.execute(self._SQL).fetchall()
        finally:
            conn.close()

        by_speaker: dict[str, dict] = {}
        for row in rows:
            values = row["embedding"]
            if isinstance(values, str):
                import json

                try:
                    values = json.loads(values)
                except json.JSONDecodeError:
                    continue
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.VECTOR_DIMENSION:
                continue
            if not np.all(np.isfinite(arr)):
                continue
            by_speaker.setdefault(
                row["name"], {"id": row["id"], "embeddings": []}
            )["embeddings"].append(arr)

        speakers = []
        for name, info in by_speaker.items():
            matrix = _l2_normalize(np.vstack(info["embeddings"]))
            speakers.append(
                RegisteredSpeaker(name=name, id=info["id"], embeddings=matrix)
            )
        return speakers


# ════════════════════════════════════════════════════════════════════
# 流式说话人确认器
# ════════════════════════════════════════════════════════════════════


@dataclass
class VerifiedSpeaker:
    """一个已确认/已识别的说话人。"""

    name: str
    similarity: float
    id: str = ""
    # 识别模式下的 Top-N 候选：[(名字, 相似度), ...]（不含 Top-1）
    top3: list = field(default_factory=list)


class _PendingState:
    """单个全局说话人的候选状态。"""

    __slots__ = ("name", "id", "ema", "hits")

    def __init__(self, name: str, id: str, ema: float, hits: int = 0):
        self.name = name
        self.id = id
        self.ema = ema
        self.hits = hits


class StreamingSpeakerVerifier:
    """流式说话人确认/识别状态机（双模式）。

    每个 chunk 调用一次 :meth:`update`，传入说话人聚类累积质心
    （``OnlineSpeakerClustering.centers``，未归一化的 embedding 和）与
    当前活跃的全局说话人集合；内部完成：

    1. 质心 L2 归一化，与注册声纹矩阵批量余弦匹配（取最相似的一条注册声纹）
    2. 相似度 EMA 平滑，防止单 chunk 抖动
    3. 按模式决定标签替换规则：
       - verify（确认，默认）：连续 ``min_chunks`` 次平滑相似度 >= threshold
         才确认替换；确认后持续不达标自动撤销、恢复 speakerX 标签
       - identify（识别，open-set）：无 min_chunks 门禁，每 chunk 即时按
         EMA 平滑后的 Top-1 匹配替换；相似度 < 门禁（默认同 threshold）
         保持 speakerX，Top-1 变化时名字立即跟随

    两种模式均可通过 :meth:`set_mode` 运行中切换，EMA 状态保留。

    Parameters
    ----------
    voiceprints: list[RegisteredSpeaker]
        注册声纹库（向量已 L2 归一化）。
    threshold: float
        余弦相似度阈值，默认 0.5（EER 校准值，见
        scripts/calibrate_threshold.py；参考项目经验值为 0.4）。
    min_chunks: int
        确认模式：确认所需连续命中 chunk 数（=撤销所需连续不达标数），默认 3。
    ema_alpha: float
        相似度 EMA 平滑系数，默认 0.3。
    max_speakers: int
        与管道 max_speakers 一致（质心矩阵行数），默认 20。
    unconfirm_min_chunks: int | None
        确认模式：撤销确认所需连续不达标 chunk 数，默认与 min_chunks 相同。
    mode: str
        "verify"（确认）或 "identify"（识别，open-set），默认 "verify"。
    identify_min_similarity: float | None
        识别模式的替换门禁；None 时与 threshold 相同（open-set：低于门禁
        保留 speakerX）。
    """

    def __init__(
        self,
        voiceprints: list[RegisteredSpeaker],
        threshold: float = 0.5,
        min_chunks: int = 3,
        ema_alpha: float = 0.3,
        max_speakers: int = 20,
        unconfirm_min_chunks: int | None = None,
        mode: str = "verify",
        identify_min_similarity: float | None = None,
    ):
        assert voiceprints, "声纹库为空，无法进行说话人确认"
        self.voiceprints = voiceprints
        self.threshold = threshold
        self.min_chunks = max(1, int(min_chunks))
        self.ema_alpha = ema_alpha
        self.max_speakers = max_speakers
        # 撤销确认所需连续不达标 chunk 数（默认与确认所需一致）
        self.unconfirm_min_chunks = (
            self.min_chunks
            if unconfirm_min_chunks is None
            else max(1, int(unconfirm_min_chunks))
        )
        self.mode = "verify"
        self.identify_min_similarity = identify_min_similarity

        # 注册矩阵：(N, dim)，每行一条声纹；owners[i] 为行 i 所属说话人下标
        rows, owners = [], []
        for idx, spk in enumerate(voiceprints):
            for embedding in spk.embeddings:
                rows.append(embedding)
                owners.append(idx)
        self._matrix = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        self._owners = np.asarray(owners, dtype=np.int64)

        self._state: dict[int, _PendingState] = {}
        self._confirmed: dict[int, VerifiedSpeaker] = {}
        self._identifications: dict[int, VerifiedSpeaker] = {}
        # 标签跟踪：g -> 当前输出标签（"speakerX" 或 人名），用于撤销/改名回切
        self._label_state: dict[int, str] = {}

        self.set_mode(mode)

    # -- 模式 ------------------------------------------------------------ #
    @property
    def identify_gate(self) -> float:
        """识别模式的替换门禁（open-set 下限）。"""
        if self.identify_min_similarity is not None:
            return self.identify_min_similarity
        return self.threshold

    def set_mode(self, mode: str) -> None:
        """运行中切换模式（verify | identify），EMA 状态保留。"""
        if mode not in ("verify", "identify"):
            raise ValueError(f"未知模式: {mode}，可选 verify / identify")
        if mode != self.mode:
            self._label_state.clear()  # 标签语义变化，重置跟踪
            self.mode = mode

    # -- 状态视图 -------------------------------------------------------- #
    @property
    def confirmed(self) -> dict[int, VerifiedSpeaker]:
        """确认模式：已确认映射 {全局说话人下标 -> VerifiedSpeaker}。"""
        return self._confirmed

    @property
    def identifications(self) -> dict[int, VerifiedSpeaker]:
        """识别模式：每 chunk 的 Top-1 识别映射（含 top3 候选）。"""
        return self._identifications

    @property
    def pending(self) -> dict[int, tuple[str, float, int]]:
        """候选状态视图：{全局说话人下标 -> (名字, 平滑相似度, 连续命中数)}。"""
        return {
            g: (st.name, st.ema, st.hits) for g, st in self._state.items()
        }

    # -- 每 chunk 更新 ---------------------------------------------------- #
    def update(
        self, centers: np.ndarray, active_speakers: set[int]
    ) -> dict[int, VerifiedSpeaker]:
        """用当前质心更新状态，返回当前模式的识别/确认映射。

        verify 模式：未确认（未注册）不替换标签，误确认自动撤销恢复。
        identify 模式：open-set 识别，每 chunk 即时输出 Top-1（EMA 平滑），
        低于门禁保持 speakerX。
        """
        if self._matrix.shape[0] == 0:
            return self._confirmed if self.mode == "verify" else self._identifications

        topk = 3
        for g_spk in active_speakers:
            if g_spk >= centers.shape[0]:
                continue
            vec = centers[g_spk]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-10:
                continue
            query = vec / norm
            similarities = query @ self._matrix.T
            num = similarities.shape[0]

            if self.mode == "identify":
                # Top-3 候选（识别模式）
                k = min(topk, num)
                top_idx = np.argpartition(similarities, -k)[-k:]
                top_idx = top_idx[np.argsort(-similarities[top_idx])]
                best_index = int(top_idx[0])
                best_sim = float(similarities[best_index])
                owner = self.voiceprints[self._owners[best_index]]

                state = self._state.get(g_spk)
                if state is None or state.name != owner.name:
                    state = _PendingState(owner.name, owner.id, best_sim)
                else:
                    state.ema = self.ema_alpha * best_sim + (1 - self.ema_alpha) * state.ema
                self._state[g_spk] = state
                self._identifications[g_spk] = VerifiedSpeaker(
                    name=state.name,
                    id=state.id,
                    similarity=state.ema,
                    top3=[
                        (self.voiceprints[self._owners[int(i)]].name, float(similarities[i]))
                        for i in top_idx[1:]
                    ],
                )
                continue

            # ---- verify 模式（原逻辑） ----
            best_index = int(np.argmax(similarities))
            best_sim = float(similarities[best_index])
            owner = self.voiceprints[self._owners[best_index]]

            state = self._state.get(g_spk)
            if state is None or state.name != owner.name:
                state = _PendingState(owner.name, owner.id, best_sim)
            else:
                state.ema = self.ema_alpha * best_sim + (1 - self.ema_alpha) * state.ema
            self._state[g_spk] = state

            if g_spk in self._confirmed:
                # 已确认：持续监控，相似度持续低于阈值则撤销确认，恢复原标签
                if state.ema < self.threshold:
                    state.hits += 1
                    if state.hits >= self.unconfirm_min_chunks:
                        del self._confirmed[g_spk]
                        state.hits = 0
                        print(
                            f"[声纹确认] speaker{g_spk} 撤销确认 "
                            f"(相似度 {state.ema:.3f} < 阈值 {self.threshold})，"
                            "恢复原标签"
                        )
                else:
                    state.hits = 0
            else:
                # 未确认：连续达标才确认
                state.hits = state.hits + 1 if state.ema >= self.threshold else 0
                if state.hits >= self.min_chunks:
                    self._confirmed[g_spk] = VerifiedSpeaker(
                        name=state.name, id=state.id, similarity=state.ema
                    )
                    print(
                        f"[声纹确认] speaker{g_spk} = {state.name} "
                        f"(相似度 {state.ema:.3f})"
                    )

        # 已确认说话人的相似度随质心更新
        for g_spk, verified in self._confirmed.items():
            state = self._state.get(g_spk)
            if state is not None:
                verified.similarity = state.ema

        return self._confirmed

    # -- 标签改名 --------------------------------------------------------- #
    def _build_rename_mapping(self) -> dict:
        """按当前模式构建 {旧标签 -> 新标签} 映射。

        注意：每 chunk 的聚合输出中，说话人段标签总是新的 "speakerX"
        （聚类 -> binarize 生成），因此映射必须始终包含
        {speakerX -> 目标名}；同时对滑动窗口中尚未滑出的、已被改名的
        老段（标签为人名），提供 {旧人名 -> 新标签} 的改名/回切映射。
        """
        mapping: dict = {}
        if self.mode == "verify":
            # 已确认 -> 人名（新段 speakerX + 老段人名同步）
            for g_spk, verified in self._confirmed.items():
                mapping[f"speaker{g_spk}"] = verified.name
                old = self._label_state.get(g_spk)
                if old is not None and old != verified.name:
                    mapping[old] = verified.name
                self._label_state[g_spk] = verified.name
            # 撤销恢复：曾改名但现在未确认的 -> 恢复 speakerX
            for g_spk, old in list(self._label_state.items()):
                if g_spk not in self._confirmed and old != f"speaker{g_spk}":
                    mapping[old] = f"speaker{g_spk}"
                    self._label_state[g_spk] = f"speaker{g_spk}"
        else:
            # identify：每 chunk 即时跟随 Top-1（open-set：低于门禁保持 speakerX）
            gate = self.identify_gate
            for g_spk, verified in self._identifications.items():
                new = verified.name if verified.similarity >= gate else f"speaker{g_spk}"
                mapping[f"speaker{g_spk}"] = new
                old = self._label_state.get(g_spk)
                if old is not None and old != new:
                    mapping[old] = new
                self._label_state[g_spk] = new
        return mapping

    def rename_annotation(self, annotation) -> "Annotation":
        """按当前模式重命名标签（原地修改，含撤销恢复/改名回切）。"""
        mapping = self._build_rename_mapping()
        if mapping:
            annotation.rename_labels(mapping=mapping, copy=False)
        return annotation

    def update_voiceprints(self, voiceprints: list[RegisteredSpeaker]) -> None:
        """热重载声纹库：重建注册矩阵，保留仍有效的确认/候选状态。

        已确认/候选说话人若名字仍在新库中则保留（声纹行可能已扩充），
        否则失效：确认的自动恢复 speakerX，候选的直接清除。
        """
        assert voiceprints, "声纹库为空，无法进行说话人确认"
        self.voiceprints = voiceprints
        rows, owners = [], []
        for idx, spk in enumerate(voiceprints):
            for embedding in spk.embeddings:
                rows.append(embedding)
                owners.append(idx)
        self._matrix = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        self._owners = np.asarray(owners, dtype=np.int64)

        valid_names = {spk.name for spk in voiceprints}
        # 失效的已确认说话人：标签恢复 speakerX
        for g_spk in list(self._confirmed):
            if self._confirmed[g_spk].name not in valid_names:
                old = self._label_state.get(g_spk)
                if old is not None and old != f"speaker{g_spk}":
                    print(
                        f"[声纹重载] speaker{g_spk} ({old}) 已从声纹库移除，恢复原标签",
                        flush=True,
                    )
                self._label_state[g_spk] = f"speaker{g_spk}"
                del self._confirmed[g_spk]
        # 失效的候选状态清除
        for g_spk in list(self._state):
            if self._state[g_spk].name not in valid_names:
                del self._state[g_spk]
        # 失效的识别结果清除
        for g_spk in list(self._identifications):
            if self._identifications[g_spk].name not in valid_names:
                del self._identifications[g_spk]
        print(
            f"[声纹重载] 声纹库已更新: {len(voiceprints)} 个说话人 "
            f"(确认状态保留 {len(self._confirmed)} 个)",
            flush=True,
        )

    def reset(self):
        """重置状态（随管道 reset 调用）。"""
        self._state.clear()
        self._confirmed.clear()
        self._identifications.clear()
        self._label_state.clear()
