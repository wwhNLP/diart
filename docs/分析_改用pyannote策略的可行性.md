# 分析：改用 pyannote 官方策略做说话人确认的可行性

> 问题：当前实现的说话人确认基于"wespeaker 256 维 embedding + L2 归一化余弦 + 固定阈值 0.4"。若改用 **pyannote 官方策略**（`PretrainedSpeakerEmbedding` 包装器 + `cdist(cosine)` + EER 校准阈值），是否依然能做到流式？
>
> 结论：**完全可以流式，且与当前实现在数学上完全等价**。pyannote 策略的验证单元（embedding 对 + 余弦）天然流式友好；真正可迁移的增益是 EER 阈值校准与官方统一 API，但切换存在模型分发路径的坑（见 §3）。

## 1. pyannote 官方策略的构成

pyannote.audio 4.x 的说话人验证标准做法（`pyannote.audio.pipelines.speaker_verification`）：

| 环节 | 官方做法 |
|---|---|
| 声纹提取 | `PretrainedSpeakerEmbedding` 工厂：按模型名路由到 4 类包装器（PyannoteAudio / SpeechBrain / NeMo / **ONNXWeSpeaker**），构造时 `model_.eval()` |
| 归一化 | 包装器不内置归一化，由调用方 L2 归一化（官方文档示例） |
| 匹配 | `scipy.spatial.distance.cdist(emb1, emb2, metric="cosine")`，相似度 = 1 − 余弦距离 |
| 阈值 | `SpeakerVerification` pipeline 支持用开发集 **EER（等错误率）校准**（`set_threshold`），非拍脑袋定值 |
| 多声纹 | 对多 enrollment 逐条比较取最优（max），与"注册矩阵 argmax"一致 |

## 2. 实证：与当前实现数学等价

在本地环境（wespeaker 本地模型、贺文泰/黄婷音频）实测：

| 对比项 | 结果 |
|---|---|
| 官方包装器加载本地 torch 权重 | ✅ `PyannoteAudioPretrainedSpeakerEmbedding` 直接加载成功 |
| 两实现 embedding 一致性（eval 模式） | **1.000000**（完全一致） |
| 1v1 相似度：官方口径 `1−cdist` | 0.0119 |
| 1v1 相似度：当前矩阵乘 | **0.0119**（同值） |
| 单 chunk 匹配 80 路注册声纹 | **0.003 ms**（相对模型推理 ~100ms 可忽略） |

**结论**：当前实现 = pyannote 官方策略的等价实现（同模型、同归一化、同余弦），只是矩阵乘代替 cdist、固定阈值代替校准阈值。

## 3. 切换的现实成本（实证踩坑）

1. **模型分发路径的坑**：`PretrainedSpeakerEmbedding("models/wespeaker-voxceleb-resnet34-LM")` 会因路径含 `"wespeaker"` 子串被路由到 `ONNXWeSpeakerPretrainedSpeakerEmbedding`，要求 **onnxruntime + ONNX 权重**（pyannote 官方发布的 wespeaker 为 ONNX 格式）。ModelScope 下载的是 torch 权重，二者不通用。绕过方式：直接实例化 `PyannoteAudioPretrainedSpeakerEmbedding`。
2. **eval 模式陷阱（本次实证发现）**：diart 的 `EmbeddingModel`（LazyModel）**不自动 eval**，裸调用处于 train 模式会更新 BN 统计量（实测 `num_batches_tracked` +1、`running_var` 偏差 2.25），导致 embedding 偏差（实测余弦一致性仅 0.82）。当前管道已安全（`OverlapAwareSpeakerEmbedding` 构造时 `self.model.eval()`），但直接裸用 `EmbeddingModel.from_pretrained(...)` 注册声纹时需手动 `.eval()`。
3. **EER 校准需要开发集**：需要一个带标注的注册-验证对开发集做阈值校准，当前只有 5 个会议、10 人声纹库，可自建（同人/异人对），但样本量小。

## 4. 流式可行性分析

**结论：可以流式，且存在两种接入方式。**

### 方式 A（推荐，零结构改动）：保持当前架构，匹配层换成 pyannote 口径

```
diart 聚类质心（每 chunk 更新，已是 embedding）
   → 可选：PyannoteAudioPretrainedSpeakerEmbedding 包装（与现在等价的 eval 模式提取）
   → L2 归一化 → 与注册矩阵批量余弦（或 cdist）
   → 阈值判定（可用 EER 校准值替代固定 0.4）
```

- 流式依据：验证的输入单元是 **embedding 向量对**，而 diart 的在线聚类在每个 chunk 都产出稳定的说话人质心（embedding 的累积和）——即"测试声纹"每 chunk 就绪，匹配计算仅 0.003ms；
- 与现在的差异仅：API 换皮（可选）+ 阈值来源（校准 vs 固定），流式能力、确认状态机、撤销机制全部保留。

### 方式 B（不推荐）：把官方 SpeakerVerification pipeline 当黑盒逐 chunk 调用

- 每 chunk 把该说话人的**累积音频段**作为 test waveform 传入 pipeline → 每次重算 embedding（浪费，质心已提供）；
- 需自行维护每说话人音频缓冲、裁剪、重采样；
- 延迟更大、计算更重；pipeline 面向离线 1v1 验证场景设计（输入 AudioFile/waveform）。
- 唯一收益是获得 EER 校准的现成实现，但该逻辑可单独抽取（方式 A）。

## 5. 结论与建议

| 项 | 结论 |
|---|---|
| 流式可行性 | ✅ 完全可行；验证单元（embedding+余弦）天然流式友好，聚类质心每 chunk 即"测试声纹" |
| 与当前实现关系 | 数学等价（实证 1.000000 / 0.0119==0.0119）；当前实现即 pyannote 策略的等价复刻 |
| 建议采用方式 | 方式 A：保留现有流式架构，可选择性迁移两处——① `PretrainedSpeakerEmbedding` 统一 API；② **EER 校准阈值**（最有价值的改进，替代固定 0.4，需开发集） |
| 不建议 | 方式 B（黑盒 pipeline 逐 chunk 调用）：重算 embedding、维护音频缓冲、延迟更大 |
| 切换注意 | wespeaker 路径含关键字会被路由到 ONNX 分支（需 onnxruntime+ONNX 权重）；LazyModel 需手动 eval() |

**后续落地（已完成 2026-08-11）**：用 5 会议真值段构建验证对（同人 316 对 / 异人 4,909 对），EER 校准完成：**EER=0.6% @ 阈值 0.50**（FAR≤1% 阈值 0.47）。三档阈值端到端对比（5 会议 2,663 段）：0.4 → 59.6%、0.47 → 69.2%、**0.5 → 71.4%**。默认阈值已改为 0.5（`StreamingSpeakerVerifier` / config / CLI 同步），校准工具为 `scripts/calibrate_threshold.py`，输出见 `docs/results/calibration/`。误报（陌生人→注册人）由 255 段降至 105 段（-58.8%）。
