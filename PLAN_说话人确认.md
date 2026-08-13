# diart 流式说话人确认功能 — 实现方案（已实现 ✅）

## 0. 实施结果（2026-08-11）

已完成全部实现并通过端到端验证：

- **新增 `src/diart/verification.py`**：`RegisteredSpeaker` / `VoiceprintProvider` / `DirectoryVoiceprints`（目录注册）/ `DBVoiceprints`（PostgreSQL 兼容参考项目）/ `StreamingSpeakerVerifier`（EMA 平滑 + 连续命中确认状态机）
- **`blocks/diarization.py`**：config 新增 `voiceprint_dir / verify_threshold / verify_min_chunks / verify_ema_alpha`；`__call__` 中每 chunk 匹配聚类质心 → 重命名标签 → 输出 Annotation 附带 `speaker_verification` 属性
- **`models.py`**：torchaudio `list_audio_backends` 兼容 shim（speechbrain 导入保护）
- **`console/serve.py` + `stream.py`**：新增 `--voiceprint-dir / --verify-threshold / --verify-min-chunks / --verify-ema-alpha / --no-verify`
- **模型**：从 ModelScope 下载 `pyannote/segmentation-3.0` + `pyannote/wespeaker-voxceleb-resnet34-LM`（256 维，与参考库同空间）到 `models/`
- **验证脚本**：`scripts/eval_verification.py`

### 端到端验证结果

| 验证项 | 结果 |
|---|---|
| CPU 流式性能 | 0.11s/chunk，约 5-9 倍实时（无 GPU） |
| 流式确认 | speaker1=黄婷(0.589)@4s、speaker0=贺文泰(0.422)@20s（拼接会议音频） |
| WebSocket 全链路 | serve.py + client.py 流式返回 423 条 RTTM，418 条为真实人名（李笑康 224、刘磊 194），5.0s 处完成首次确认 |
| 陌生人处理 | 未注册说话人保持 speakerX 不误名 |
| 5 会议真值比对（delta_new=0.8, thr=0.4） | 段级准确率 45-67%，均值 ~60% |

> 准确率上限主要受限于在线聚类的合并/碎片化（wespeaker 向量下 delta_new 需调小，但过小会碎片化），
> 以及拼接会议数据本身 500ms 间隔打断连续性；验证器对纯净聚类的命名是正确的。
> 调参方向：`--delta-new 0.6~0.8`、`--verify-threshold 0.4~0.5`。

## 1. 背景与现状

### 参考项目 (ASR-stream-SPK-PY) 的说话人确认机制
- **注册**：声纹库存于 PostgreSQL（`t_speaker` / `t_speaker_voiceprint` / `t_voiceprint`），每条声纹是 **256 维 pyannote 声纹向量**（来自 `speaker-diarization-community-1` 管道的 embedding 模型）；另有本地 wav 注册方式（`extract_spk_embedding_pyannote`：3s 窗 / 1.5s 步滑窗取均值）。
- **匹配**：对 diarization 输出的每个说话人质心（`output.speaker_embeddings`，与注册向量同空间）做 **L2 归一化余弦相似度**，**阈值 0.4**，超过则命名为真实人名，否则留空由出现顺序编号 `speaker_XX`。
- **性质**：纯离线（文件级）流程。

### diart 项目现状
- 流式管道：`SpeakerDiarization` 以 5s 窗口 / 0.5s 步进逐 chunk 输出 `(Annotation, audio)`，`OnlineSpeakerClustering` 内部**已累积每个全局说话人的 embedding 质心**（`centers`，未归一化）。
- 输出说话人只有全局编号（`speaker0`/`speaker1`...），**无真实人名**。
- `console/serve.py` 已做过本机适配（hf-mirror 端点、offline/online 参数），WebSocket 流式输出 RTTM。
- 环境问题：
  - speechbrain 1.0.3 导入时调用 `torchaudio.list_audio_backends()`，而本机 torchaudio 缺少该属性 → 需 shim。
  - 模型未缓存：`pyannote/segmentation-3.0`、`pyannote/wespeaker-voxceleb-resnet34-LM`（256 维）均可在 **ModelScope** 下载（HF 直连 403/gated，hf-mirror 对 segmentation-3.0 也 403）。
  - 无 GPU，纯 CPU 推理。

## 2. 设计目标

在 diart 流式管道中内嵌**流式说话人确认器**：每个 chunk 到达时，将当前活跃全局说话人与注册声纹库比对，确认后把 `SPEAKER_XX` 重命名为真实人名，并附带相似度信息供下游消费；确认过程有状态（候选 → 确认），避免单帧抖动误报。

## 3. 总体架构

```
音频流 ──► SpeakerDiarization ──► (Annotation, audio)
              │   │
              │   └─► OnlineSpeakerClustering.centers（每全局说话人累积质心）
              │            │
              │            ▼
              │   StreamingSpeakerVerifier（新增）
              │      ├─ 质心 L2 归一化
              │      ├─ × 注册声纹矩阵（余弦）→ 相似度
              │      ├─ EMA 平滑 + 连续命中确认状态机
              │      └─ 确认表 {global_spk -> (name, similarity)}
              │            │
              ▼            ▼
         Annotation 标签改名          annotation.speaker_verification 属性
         （SPEAKER_00 → 王佳琪）      （hooks/sinks 可读，serve.py 流式下发）
```

## 4. 模块设计（文件改动清单）

### 4.1 新增 `src/diart/verification.py`
- `RegisteredSpeaker`（dataclass）：`id / name / embeddings: np.ndarray (n, dim)`
- `VoiceprintProvider`（抽象）：`load() -> list[RegisteredSpeaker]`，向量 L2 归一化
  - `DirectoryVoiceprints(dir, embedding_model, device)`：扫描 `声纹库/<人名>/*.wav`，用**与管道相同的 embedding 模型**滑窗提取（3s/1.5s，同参考项目），均值后注册；结果缓存
  - `DBVoiceprints(dsn, ...)`：读 PostgreSQL 256 维向量（参考项目兼容，可选实现）
- `StreamingSpeakerVerifier`：
  - 构造参数：`threshold=0.4`、`min_chunks=3`（确认所需连续命中数）、`ema_alpha=0.3`
  - 状态：`confirmed: dict[int, VerifiedSpeaker]`、`pending: dict[int, PendingState]`
  - `update(centers, active_global_speakers) -> dict[int, VerifiedSpeaker]`：
    1. 取活跃全局说话人的质心（`centers / ||centers||`）
    2. 与注册矩阵批量余弦（numpy `normalized @ registered.T`，与参考项目同数学）
    3. EMA 平滑相似度；连续 `min_chunks` 次 ≥ threshold → 提升为 confirmed
    4. 返回本轮确认结果
  - `rename_annotation(annotation) -> Annotation`：把确认过的 `SPEAKER_XX` 标签改名为真实人名
  - `reset()`：随管道 reset

### 4.2 修改 `src/diart/blocks/diarization.py`
- `SpeakerDiarizationConfig` 新增参数：
  - `voiceprint_dir: str | None = None`（声纹库目录，None = 关闭确认）
  - `verify_threshold: float = 0.4`（与参考项目一致）
  - `verify_min_chunks: int = 3`、`verify_ema_alpha: float = 0.3`
  - `verify_db_dsn: str | None = None`（可选 PostgreSQL 源）
- `SpeakerDiarization.__call__`：`clustering(seg, emb)` 之后：
  ```python
  if self.verifier is not None:
      centers, active = self.clustering.get_centers(), self.clustering.active_centers
      self.verification = self.verifier.update(centers, active)
      agg_prediction = self.verifier.rename_annotation(agg_prediction)
  ```
  并把 `speaker_verification` 附加到输出 Annotation 上（`setattr`，pyannote Annotation 无 `__slots__`，实测确认）。
- `reset()`：同步重置 verifier。

### 4.3 修改 `src/diart/blocks/clustering.py`
- 暴露 `get_centers()` / `active_centers` 只读访问（当前已有 `active_centers` 属性，补 `get_centers` 即可，最小改动）。

### 4.4 修改 `src/diart/models.py`
- 顶部加 torchaudio 兼容 shim：
  ```python
  try:
      import torchaudio
      if not hasattr(torchaudio, "list_audio_backends"):
          torchaudio.list_audio_backends = lambda: []
  except ImportError:
      pass
  ```
  修复 speechbrain 导入崩溃。

### 4.5 修改 `src/diart/console/serve.py` 与 `stream.py`
- 新增 CLI 参数：`--voiceprint-dir`、`--verify-threshold`、`--verify-min-chunks`、`--no-verify`
- serve.py 的 WebSocket 回复中，将确认结果并入（RTTM 标签即人名，天然流式）；stream.py 控制台打印 `[确认] SPEAKER_00 = 王佳琪 (相似度 0.62)`

### 4.6 修改 `src/diart/__init__.py`
- 导出 `verification` 模块（`StreamingSpeakerVerifier`、`DirectoryVoiceprints` 等）。

## 5. 模型与环境准备（实现第一步）

| 项 | 动作 |
|---|---|
| 模型下载 | ModelScope CLI：`pyannote/segmentation-3.0`（5.9MB）→ `models/segmentation-3.0`；`pyannote/wespeaker-voxceleb-resnet34-LM`（26.6MB）→ `models/wespeaker-voxceleb-resnet34-LM`（256 维，与参考 DB 同空间） |
| serve.py 默认参数 | `--segmentation models/segmentation-3.0`、`--embedding models/wespeaker-...`（本地路径，pyannote `Model.from_pretrained` 支持目录加载，需实测） |
| speechbrain shim | 见 4.4 |
| 兼容性 | 实测 pyannote.audio 4.0.7 + diart `PowersetAdapter`（`model.specifications`）是否可用 |

## 6. 端到端验证方案

1. 下载模型 → 用 `test_spk/说话人识别测试/声纹库/`（10 人 × 3 段 wav）注册
2. 流式回放 `拼接会议/音频/部门考勤与日常办公管理沟通会议_完整版.wav`
3. 对照 `拼接会议/文本/*_时间戳标注.txt` 真值：
   - 检查确认出的真实人名与真值一致（准确率统计可复用 test_spk 的 `完整流程测试.py` 思路）
4. 验证流式性：确认事件逐 chunk 产生（非文件级后处理），观察从说话人开口到确认的延迟（~min_chunks × step）

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| pyannote.audio 4.0.7 与 diart 0.7 API 不兼容 | 先跑最小加载测试（segmentation + embedding + 一次 forward），失败则按 4.x API 微调 diart 适配层 |
| CPU 推理速度（无 GPU） | wespeaker 模型小（26MB），5s chunk 单 speaker 推理约数百 ms，可跑通；若超实时再调大 step |
| 质心累积未归一化导致余弦失真 | 确认器内自行归一化质心（`centers` 是 embedding 的和，除以范数即可，与注册向量同空间） |
| 说话人开口初期质心不稳定 | EMA + 连续 min_chunks 确认机制；相似度打印供调阈值 |
