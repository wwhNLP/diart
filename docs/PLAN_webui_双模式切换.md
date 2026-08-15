# Web 控制台：说话人确认 / 说话人识别 双模式切换 — 实现计划

## 1. 现状与目标

### 现状
- `web_ui.py` 是**中继服务**：浏览器采集音频 → 转发 diart serve（WS）→ RTTM 原样透传回浏览器 + 落盘日志
- serve 管道内 `StreamingSpeakerVerifier` 只有一种行为：**确认模式**——相似度 ≥ 阈值（0.5）才把 `speakerX` 替换为人名，否则保持匿名（未注册不替换）
- serve 的 WS 协议：上行全是音频 base64，下行全是 RTTM 文本，**无控制消息通道**
- 前端 `index.html`（1211 行，中英双语）：RTTM 解析 + 说话人时间线渲染，无模式概念

### 目标
1. 在 web 控制台增加**模式切换**（说话人确认 ↔ 说话人识别），运行中即时生效、无需重启服务
2. 完善两种模式的行为定义与输出差异，前端可视化呈现

## 2. 两种模式的行为定义（关键设计决策）

| 维度 | 确认模式（verify，现状，默认） | 识别模式（identify，新增） |
|---|---|---|
| 判定哲学 | 宁缺毋滥：未达标不替换，符合"未注册说话人不替换标签" | 宁滥毋缺：总是给出最可能人名（closed-set 识别） |
| 标签替换 | 连续 min_chunks 次 EMA 相似度 ≥ 阈值 → 替换；持续不达标 → 撤销恢复 | 每 chunk 用 EMA 平滑后的 **Top-1 匹配**即时替换（无 min_chunks 门禁） |
| 阈值作用 | 替换与否的硬门禁 | 仅用于标注**置信度**（高/低置信，低置信前端高亮） |
| 输出附加 | `speaker_verification`：{确认者: {name, similarity}} | 新增 `speaker_identification`：{全部活跃者: {name, similarity, top3:[...]}} |
| 撤销机制 | 生效（误确认自动回退） | 不适用（始终替换；Top-1 变化即改名） |
| 可选参数 | — | `identify_min_similarity`（默认 -1 关闭）：低于此值保留 speakerX（可退化为 open-set） |
| 适用场景 | 会议纪要：只标确定的人 | 需始终有名字的下游（如坐席名显示） |

> 平滑共用：两种模式都使用 EMA（α=0.3）抗单帧抖动；状态机（pending 的 ema/hits）共用，仅"替换规则"分支不同。

## 3. 架构链路（切换如何贯通）

```
浏览器 UI（新增模式切换控件）
   │  WS 消息 {type:"mode", mode:"verify"|"identify"}
   ▼
web_ui.py（中继层）
   ├─ 新增 mode 消息类型：转发给上游 serve（Session.send_control，不走音频队列，立即生效）
   ├─ 启动/连接时向浏览器下发当前模式（init 消息扩展）
   └─ 会话日志记录模式切换事件
   │  WS（与音频同一连接）
   ▼
sources.py: WebSocketAudioSource._on_message_received（serve 侧）
   ├─ 先判定是否控制消息：JSON dict 且 {"type":"mode","mode":合法} → 触发回调
   └─ 否则按原逻辑 decode_audio 当音频
   ▼
serve.py：控制消息 → 运行时切换 pipeline.verifier.mode（无需重启）
   ▼
verification.py: StreamingSpeakerVerifier 新增 identify 模式分支
   └─ 输出 RTTM（识别模式下标签即 Top-1 人名）→ 浏览器渲染
```

## 4. 分模块改动清单

### 4.1 `src/diart/verification.py`（核心）
- `StreamingSpeakerVerifier.__init__` 新增 `mode: str = "verify"`（`"verify" | "identify"`）与 `identify_min_similarity: float = -1.0`
- 新增 `mode` property（可运行时切换，`set_mode()` 校验合法性）
- `update()` 增加 identify 分支：
  - 对每个活跃全局说话人计算 Top-1/Top-3（矩阵乘后 `argpartition` 取前 3）
  - EMA 平滑逻辑复用（best_name 变化时重置 ema，同现状）
  - identify 模式：**忽略 min_chunks 门禁**，`identifications[g] = {name, similarity(ema), top3}`；若 `ema >= identify_min_similarity`（默认恒真）则参与标签替换
  - verify 模式：完全保持现状（不回归）
- `rename_annotation()` 增加 identify 分支：识别模式下把**所有活跃且达标**的 `speakerX` 替换为 Top-1 人名（不再依赖 confirmed 集合）
- `speaker_verification` 属性保持（verify 口径），新增 `speaker_identification` 输出

### 4.2 `src/diart/sources.py`（serve 侧 WS 协议）
- `WebSocketAudioSource` 新增 `on_control: Callable[[dict], None] | None` 回调
- `_on_message_received`：先 `try json.loads`，若为 dict 且 `type=="mode"` 且 mode 合法 → 调 `on_control`，否则原逻辑音频解码
- 注意：base64 音频串一般不是合法 JSON dict，判定安全；`decode_audio` 失败路径不变

### 4.3 `src/diart/console/serve.py`
- 新增 CLI 参数 `--verify-mode verify|identify`（默认 verify）
- 管道构造后：`if verifier: verifier.set_mode(args.verify_mode)`
- `audio_source.on_control = lambda ctrl: verifier.set_mode(ctrl["mode"])`——运行中切换
- 日志打印模式切换事件

### 4.4 `src/diart/console/web_ui.py`（中继）
- `Session` 新增 `send_control(msg: dict)`：直接经当前 ws 发送（`enable_multithread=True` 已支持），不经过音频队列
- `_handle_browser_msg` 新增 `"mode"` 类型：校验合法 → `session.send_control(...)`；无会话时拒绝并提示
- `init` 消息扩展 `verify_mode` 字段（从启动参数传入）；`--verify-mode` CLI 参数
- 会话日志记录模式切换（`_write_log_line`）
- 透传模式切换后的 RTTM 不变（serve 侧输出即已按模式改名）

### 4.5 `src/diart/console/web_ui/index.html`（前端）
- 顶栏/设置区新增**模式切换控件**（两个按钮：确认 / 识别，或下拉），i18n 字典（中英）补充
- 状态区显示当前模式（`sess-mode` 旁或独立指示）
- 识别模式下解析 RTTM 时：人名段若伴随低置信标记显示高亮（置信度标注——RTTM 无相似度字段，采用"标签为人名即已识别"的呈现；相似度细节走 console 日志/后续扩展）
- 切换按钮点击 → `WS.send({type:"mode", mode})` → 服务端确认后 UI 更新模式状态

### 4.6 文档同步
- 技术报告/使用手册/README 补充双模式说明（行为对照表 + 切换方法）

## 5. 验证方案

| 验证项 | 方法 |
|---|---|
| 确认模式不回归 | 组件测试：verify 下确认/撤销/陌生人拒绝行为与现状一致（现有用例重跑） |
| 识别模式行为 | 组件测试：identify 下活跃说话人即时输出 Top-1 人名 + top3，低相似也替换（closed-set） |
| 运行时切换 | 单元：serve WS 发控制消息 → verifier.mode 变化；端到端：web_ui 切按钮 → 下游 RTTM 标签行为变化 |
| 端到端复现 | 用 `docs/results/rttm_eer05` 同一批 5 会议音频，识别模式输出 RTTM，对比确认模式（识别模式应有更高人名覆盖率、可能含低置信错名） |
| 前端 | 启动 serve+web_ui，浏览器手工切换验证 UI 状态与 RTTM 渲染 |

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 控制消息与音频消息区分误判（极端 base64 恰好是合法 JSON dict） | 双条件判定：`type=="mode"` 且 `mode ∈ {verify, identify}` 才走控制；否则一律音频 |
| 识别模式"总是替换"在低置信时引入错名 | 默认 closed-set 由用户显式选择；提供 `identify_min_similarity` 可退化为 open-set |
| 运行时切换状态不一致（pending/confirmed 残留） | 切换仅改替换规则，EMA 状态保留（平滑连续）；文档说明两种模式的状态语义 |
| 多线程发送控制消息与音频竞争 | websocket-client `enable_multithread=True` 已开启；控制消息量极小（切换事件），风险可控 |

## 7. 实施顺序

1. verification.py：identify 模式（核心，含组件测试）
2. sources.py + serve.py：WS 控制消息 + CLI 参数（运行时切换）
3. web_ui.py：中继转发 + init 扩展
4. index.html：切换控件 + i18n + 状态显示
5. 端到端验证（5 会议双模式 RTTM 对比 + 浏览器手工验证）
6. 文档同步
