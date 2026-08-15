# 临时声纹库 + UI 添加声纹 — 实现计划

## 1. 需求

1. **临时声纹库**：serve 增加 `--temp-voiceprint-dir` 启动参数；声纹注册 = **主库 ∪ 临时库**（同名说话人声纹合并）
2. **UI 添加声纹**：web 控制台支持上传音频 + 指定说话人姓名 → 存入临时声纹库 → **热重载**，新声纹立即参与识别（会话不中断）

## 2. 架构链路

```
浏览器（新增"添加声纹"面板：姓名 + 音频文件）
   │  POST /api/voiceprints (multipart)
   ▼
web_ui.py（新增 REST API）
   ├─ 校验姓名/音频 → 保存 <temp_dir>/<姓名>/<时间戳>.<ext>
   ├─ 通知 serve 重载：WS 控制消息 {"type":"reload_voiceprints"}
   └─ 返回结果给浏览器
   │
   ▼
serve.py（新增控制消息分支）
   └─ verifier.update_voiceprints(新注册集)   ← 热重载：重建注册矩阵，保留确认状态
```

## 3. 分模块改动

### 3.1 `verification.py`
- **`DirectoryVoiceprints` 支持多目录**：`directories: str | Path | list` 参数（兼容现有单目录用法）；同名说话人（主库+临时库）**声纹合并**（embeddings 行拼接）
- 新增 `refresh()`：清空 `_cache`，下次 `load()` 重新扫描（热重载基础）
- **`StreamingSpeakerVerifier.update_voiceprints(voiceprints)`**：重建注册矩阵（`_matrix`/`_owners`），**保留状态**：
  - `_confirmed` / `_label_state`：仅保留"新库中仍存在"的说话人（按名字匹配），已移除者自动恢复 speakerX
  - `_state`（EMA pending）：名字仍在新库的保留，否则删除（重新积累）
  - 打印重载日志（新增/移除的说话人）

### 3.2 `blocks/diarization.py`
- config 新增 `temp_voiceprint_dir: str | None = None`
- verifier 构造处：主库 + 临时库合并注册（`directories=[主, 临时]`，临时目录不存在则自动创建）
- 暴露管道级热重载方法：`pipeline.reload_voiceprints()`：
  ```python
  provider.refresh() -> voiceprints = provider.load()
  self.verifier.update_voiceprints(voiceprints)
  ```

### 3.3 `serve.py`
- 新参数 `--temp-voiceprint-dir`（默认 None；提供时自动 `mkdir -p`）
- 控制消息新增 `{"type":"reload_voiceprints"}` 分支：调用 `pipeline.reload_voiceprints()`（`_on_control` 中 `ctrl["type"]` 分流，现有 mode 逻辑不变）
- 启动日志打印注册来源（主库 + 临时库）

### 3.4 `web_ui.py`（中继层）
- CLI 参数 `--temp-voiceprint-dir`（与 serve 相同路径；页面显示当前值）
- 新增 REST API：
  - `POST /api/voiceprints`：multipart（`name` + `audio` 文件）→ 校验（姓名非空、音频后缀 ∈ wav/m4a/mp3/flac/amr、大小 ≤ 50MB）→ 保存 `<temp_dir>/<name>/<yyyyMMdd_HHmmss>.<ext>` → 若会话在线发 `{"type":"reload_voiceprints"}` 控制消息 → 返回 `{ok, path, speaker}`
  - `GET /api/voiceprints`：列出临时库说话人及音频数（供 UI 展示）
- 无会话时添加声纹：仅保存（重载在下次 serve 收到控制消息时……需要 serve 无会话也能收控制消息——serve 的 WS 只有浏览器连接时才处理。无会话场景：web_ui 暂存"待重载"标记，会话建立后（start 时）自动发送 reload。简化：保存成功后总是尝试发送，失败则记录日志提示）

### 3.5 `web_ui/index.html`
- 新增"添加声纹"卡片（放在「推理服务连接」下方）：
  - 输入：说话人姓名（text）+ 音频文件选择（file input，accept 音频）
  - 按钮：添加声纹（POST → 显示结果 toast）
  - 展示：临时库当前列表（GET /api/voiceprints，含每人声纹数），可刷新
- i18n 中英补充

### 3.6 脚本
- `scripts/run_server.sh`：可加 `--temp-voiceprint-dir`（默认加一个项目内 `temp_voiceprints/` 目录？——预留，默认不传）
- `scripts/run_ui.sh`：加 `--temp-voiceprint-dir` 参数（与 serve 同路径）

## 4. 关键设计决策

| 决策点 | 方案 |
|---|---|
| 同名冲突（主库与临时库同人名） | **声纹合并**（拼接 embeddings），不覆盖——临时库用于增量补充，合并更合理 |
| 热重载是否打断会话 | 不打断：`update_voiceprints` 保留 confirmed/label_state，仅失效被移除的说话人 |
| 无会话时添加声纹 | 保存成功 + 记录待重载标记；会话开始（start）时若有待重载标记自动发送 reload |
| 音频格式 | 复用 DirectoryVoiceprints 支持列表（wav/m4a/mp3/flac/amr），上传后原样保存 |
| 临时库持久性 | 文件落盘（不清理），重启 serve 后自动重新注册 |

## 5. 验证方案

| 验证项 | 方法 |
|---|---|
| 合并注册 | 组件：主库+临时库同名说话人 embeddings 行数 = 两者之和；唯一说话人正常 |
| 热重载保留状态 | 组件：确认张三后 update_voiceprints（库不变）→ confirmed 保留；移除张三 → 标签恢复 speakerX |
| serve 端到端 | 启动（带临时库）→ WS 发 reload 控制消息 → 日志打印新增声纹；新说话人音频可被确认 |
| web_ui API | curl POST 上传 wav → 文件落盘正确 → 会话中发 reload → serve 生效 |
| 前端 | 浏览器添加声纹 → toast 成功 → 列表刷新；说话人确认出新名字 |
| 回归 | 不传 temp-voiceprint-dir 时行为与现状完全一致（单目录） |

## 6. 实施顺序

1. verification.py（多目录 + refresh + update_voiceprints，含组件测试）
2. diarization.py（config + reload_voiceprints）
3. serve.py（参数 + 控制消息分支）
4. web_ui.py（REST API + 参数）
5. index.html（添加声纹 UI）
6. 端到端验证 + 文档同步
