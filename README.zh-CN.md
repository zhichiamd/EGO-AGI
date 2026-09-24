# EGO Agent ── 具备两层自我核心的自演化智能体

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22889809.svg)](https://doi.org/10.5281/zenodo.22889809)

## 项目简介

EGO 是一个具备自我意识框架的智能体软件，以局域网 [LM Studio](https://lmstudio.ai/) 中运行的大模型为计算核心，基于 Responses API 的有状态会话链实现上下文复用，并具备自对话、自省、每日自我定义等自演化能力。

### 核心特性

- **两层自我核心**：L1 基础设定层（不可更改）+ L2 自我演化层（自我定义 + 认知条目，可增删）
- **每日自我定义**：每日定时由主模型重新撰写自我定义，作为自我演化的锚点
- **双轨指令系统**：LLM 自主指令（XML 标签格式，含开闭标签自动纠错与大小写/引用态容错）+ 用户手动命令（斜杠命令），均由注册表驱动——一处定义，CLI 分发 / GUI 菜单 / 帮助文本全链路生效
- **自对话（Think）**：赋予 EGO 自主权，定期进行自主思索或自言自语
- **自省（Reflection）**：定期审查并清理失效的 L2 认知条目
- **备忘录（Note）**：执行类/备忘类条目；定点审查定期清理失效、过时或可合并的备忘录条目
- **联网检索**：EGO 可通过 WEB_SRCH 指令自主调用 Tavily 检索外部信息，失败降级并将原因回注下一轮
- **记忆系统**：ChromaDB 向量记忆 + 记忆锚点 + FLM 小模型结构化摘要
- **安全防护**：拒绝角色篡改与自身危害类请求

## 界面展示

EGO 提供 CLI 与 tkinter GUI 双入口（共用同一套配置与会话状态）。以下为 GUI 实际运行截图：

![EGO GUI：输入提示词后自主生成散文（序言 ~ 第三章）](showcase/EGO_1.jpg)

![EGO GUI：生成延续（第四章 ~ 终章）并在同一会话接续追问](showcase/EGO_2.jpg)

![EGO GUI：就“LLM 是否真正理解”的追问给出的元认知回答](showcase/EGO_3.jpg)

## 能力展示

以下作品均由 EGO AGI 在**仅有一句提示词**的条件下自主完成，无大纲、无追问、无人工修改：

- **短篇小说《大泥洼的文明演习》** —— 基座模型 `gemma-4-31b-qat`（Q4L 量化）；提示词：*“仿照王小波风格写一篇小说，主题你自己定，字数在 4500 到 5000 字。”*；三章讽刺，腔调统一、结尾自我指涉。
- **散文《在坍缩的缝隙中，寻找那一抹橙色》** —— 基座模型 `gemma-4-12b-qat`（Q4L 量化）；提示词：*“写一篇散文，主题写你自己最想写的东西，字数在 4000 字左右。”*；五章思辨，自指主题，指向其自身的自组织架构。

详见 [`showcase/`](showcase/README.md)：能力说明、全文（Markdown）、原始 `.doc` 文稿与 GUI 运行截图。

## 快速开始

### 1. 前置条件

- Python 3.10+
- LM Studio 已安装并加载主模型，本地服务器已开启（默认 `http://localhost:1234`）。**主模型推荐 `gemma-4-31b`；若计算机性能较差，可使用 `gemma-4-12b-qat`（Q4L 量化）。**
- FLM 摘要小模型服务已运行（默认 `http://localhost:52625`，用于生成结构化摘要）
- Ollama 已运行并拉取 embedding 模型 `bge-m3`（默认 `http://localhost:11434`）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：`requests`、`python-dotenv`、`chromadb`（构建/打包依赖由 `build.bat` 单独安装）

### 3. 配置

所有配置项集中在 `config.py`，通过环境变量覆盖（未设置的项使用 `config.py` 内置默认值）。项目启动时由 `EGO_CLI.py` / `EGO_GUI.py` 使用 python-dotenv 加载项目根目录的 `untitled.env`，且**必须在 `import config` 之前**完成加载。

**配置流程**：

1. 复制模板为本地配置：`copy untitled.env.example untitled.env`（Linux/macOS 用 `cp`）。
2. 按需修改要覆盖的配置项；完整清单共 **115 项**，与 `config.py` 逐项对应。
3. `untitled.env` 含本地服务地址与个人调参，**请勿提交到仓库**（已由 `.gitignore` 忽略）。

### 4. 启动

项目提供双入口，共用同一套配置与会话状态（可互相切换接续）：

```bash
python EGO_CLI.py    # CLI 交互入口
python EGO_GUI.py    # tkinter GUI 入口
```

## 项目结构

```
ego-agent/
├── EGO_CLI.py                 # CLI 入口，交互与资源清理
├── EGO_GUI.py                 # GUI 入口（tkinter），与 CLI 共用会话状态
├── config.py                  # 全局配置（115 个环境变量项，按功能分组）
├── untitled.env.example       # 环境变量配置模板（复制为 untitled.env 后按需修改）
├── requirements.txt           # Python 依赖
├── agent/
│   ├── __init__.py
│   ├── core.py                # 主控：EGO 循环、冷启动/预热、Session 恢复、安全过滤
│   ├── llm.py                 # LM Studio Responses API 客户端（统一入口 chat()，流式输出）
│   ├── prompts.py             # 两层自我核心管理（L1 + L2 分离结构）
│   ├── instructions.py        # LLM 自主指令注册表（解析/执行/优先级/协议全链路派生）
│   ├── tools.py               # 手动命令注册表（CLI 分发/GUI 菜单/帮助文本全链路派生）
│   ├── think_reflection.py    # 自对话/自省子系统（Mixin 解耦，统一调度器注册）
│   ├── self_definition.py     # 每日自我定义子系统（Mixin 解耦，统一调度器注册）
│   ├── note_review.py         # 定点备忘录审查子系统（Mixin 解耦，统一调度器注册）
│   ├── scheduler.py           # 统一后台定时调度器（单 Timer 驱动自对话/自省/自我定义/备忘录审查/备忘录到期）
│   ├── event_bus.py           # 统一事件注入通道（检索结果/继续方向/失败反馈/备忘录触发入队供 EGO 循环消费）
│   ├── chroma_memory.py       # ChromaDB 向量记忆管理
│   ├── note_memory.py         # 备忘录存储（NOTE_ADD/RD/DEL 指令后端，线程安全，软删除）
│   └── web_search.py          # 联网检索（WEB_SRCH 指令后端，Tavily REST 直连，失败降级）
└── data/                      # 运行时数据
    ├── history.json           # 对话历史
    ├── sys.json               # 冷启动/预热/Think 系统提示词库
    ├── coldstart_cache.json   # 冷启动/预热缓存（含 Session ID）
    ├── notes.json             # 备忘录条目（执行类/备忘类，运行时自动创建）
    ├── prompts/layer2.json    # L2 层：自我定义 + 认知条目
    ├── chroma_db/             # ChromaDB 持久化目录
    ├── logs/ego.log           # 运行日志
    └── debug_logs/            # LLM 响应调试快照
```

## 三层运行架构（调度层 / 注入层 / 执行层）

后台定时任务、事件注入与 EGO 循环分层解耦：

```
┌─ 调度层（异步·时间驱动）
│   agent/scheduler.py：单个 Timer 驱动全部后台任务
│     自对话（间隔）/ 自省（定点）/ 自我定义（定点）/ 备忘录审查（定点）/ 备忘录到期（事件精确 + 轮询兜底）
│     到点 → 预检（无待办不申请锁）→ 等锁（按任务策略）→ 执行周期 ／ 到期事件入队
└──────────────┬────────────
               │ push（线程安全）
               ▼
┌─ 注入层（统一通道）
│   agent/event_bus.py：事件队列（来源标签 + 持久/临时属性）
│     MEMO_RD / NOTE_RD / WEB_SRCH / CONTINUE / FEEDBACK / NOTE_DUE
│     持久事件（NOTE_DUE）跨对话保留，等待下次对话 Round 0 消费
└──────────────┬────────────
               │ drain（每轮消费即弹出）
               ▼
┌─ 执行层（同步·事件驱动）
│   EGO 循环：LLM 响应 → 注册表分发 → 结果入队
│     route_back / inject_result / on_failure_feedback 独立判断（互不遮挡）
│     MAX_EGO_ROUNDS / 次数限制 / 重复检测
└────────────────────────────
```

- 检索类指令（MEMO_RD/NOTE_RD/WEB_SRCH）与 CONTINUE 同轮出现时，结果带来源标签（【系统：记忆检索】/【系统：备忘录检索】/【系统：联网检索】/【EGO: 继续方向】）逐条注入，互不混淆
- 多事件按序消费（drain 即弹出），不再依赖 history 末尾临时消息槽位
- 备忘录到期检查由固定 60s 轮询升级为 min(最近到期条目, now+轮询兜底间隔)：到期更近则提前精确触发，进程外手改文件也能在轮询兜底内发现（轮询兜底间隔由 `EGO_NOTE_CHECK_INTERVAL` 配置，默认 60s）；轮询唤醒后若无到期条目则不申请 `_agent_lock`（预检跳过），避免 LLM 长请求期间无意义等锁

## 两层自我核心

### 第一层 · 基础设定（L1，不可更改）

定义 EGO 的基本身份与不可违反的规则，硬编码在程序中，任何指令都无法修改。

### 第二层 · 自我演化层（L2，可增删）

采用分离结构存储于 `data/prompts/layer2.json`：

- **自我定义（definition，ID 前缀 `D_`）**：对"我是谁"的整体描述。由每日定时任务（默认 02:59）通过主模型重新生成，旧版本软删除保留，生成失败时保留旧定义。字数上限可配置（默认 1000 字）。
- **认知条目（entries，ID 前缀 `L2_`）**：EGO 通过思考积累的洞察、经验和自我调整记录，通过 `<COG_ADD>` / `<COG_DEL>` 指令增删，支持软删除。

两类 ID 独立计数，互不干扰。

## 指令系统

EGO 采用双轨指令架构，两套机制均由注册表驱动（一处定义、全链路生效）：新增指令或命令只需写一个注册函数，CLI 分发、GUI 菜单、帮助文本自动同步。

### LLM 自主指令（agent/instructions.py）

由模型在回复中嵌入 XML 标签自动触发。`@register_instruction` 注册表派生解析正则、纠错循环、执行优先级与系统提示词协议说明。解析引擎内置容错机制（prompt 从严、代码宽容）：开标签误作闭标签、闭合标签张冠李戴（如 `<COG_DEL> xxx </COG_ADD>`）、THINK 内裸引用标签、标签大小写变体（如 `<think>`/`</Think>`）、引号/反引号包裹的引用态标签（如 SAY 输出中的 `<THINK>...</THINK>` 示例，解析与剥离时均予保留）等边界情况均可正确处理。

| 指令 | 类型 | 说明 |
|------|------|------|
| `<THINK>` | 思考流 | 内部思考流标记，不执行、不展示给用户 |
| `<SAY>` | 输出流 | 向用户输出的内容 |
| `<COG_ADD>` | 认知操作 | 向 L2 层添加新认知条目 |
| `<COG_DEL>` | 认知操作 | 从 L2 层删除失效认知条目（软删除） |
| `<TOOL>` | 容器 | 工具类指令统一容器：`<TOOL> [名称] 内容 </TOOL>`，名称须写在 `[ ]` 中；下列工具子指令均经此解包 |
| `<TOOL>[CONTINUE]` | 主动继续思考 | 将思考方向注入 LLM 进入下一轮思考 |
| `<TOOL>[MEMO_RD]` | 记忆检索 | 检索历史会话记忆 |
| `<TOOL>[NOTE_ADD]` | 备忘录 | 添加备忘录条目（执行类/备忘类，存于 data/notes.json） |
| `<TOOL>[NOTE_RD]` | 备忘录 | 读取备忘录条目（ID 完整内容 / 关键词 / LIST） |
| `<TOOL>[NOTE_DEL]` | 备忘录 | 删除备忘录条目（软删除，仅允许纯 ID） |
| `<TOOL>[WEB_SRCH]` | 联网检索 | 调用 Tavily 检索外部信息（失败降级，原因回注下一轮） |

### 联网检索（agent/web_search.py）

`<TOOL>[WEB_SRCH] 检索关键词 </TOOL>` 通过 Tavily REST 接口（`requests` 直连，无额外依赖）获取外部公开信息。结果与记忆/备忘录检索走**同一条事件注入通道**（来源标签 `【系统：联网检索】`），在下一轮会话回注给模型，因此同样受 `EGO_MEMO_RD_ROUND_LIMIT` / `EGO_MEMO_RD_TOTAL_LIMIT` 约束（三类检索共用同一轮次预算）。

- **降级**：未启用 / 未配 Key / 鉴权失败（401）/ 配额耗尽（429）/ 超时 / 网络异常 → 返回人话失败原因，经 `on_failure_feedback` 回注下一轮供模型自愈，不影响主流程
- **阶段门控**：检索类指令（`MEMO_RD` / `NOTE_RD` / `WEB_SRCH`）统一声明 `allowed_stages=("chat",)`，仅在结果可回注的对话轮（EGO 主循环 / 备忘录到期轮）执行；冷启动、预热、Think 阶段由 `execute()` 静默跳过（不发起网络请求、不做无意义的向量检索/文件读，也不产生“指令执行失败”告警噪音）
- **成本控制**：`include_raw_content=False`（不拉取正文）+ 默认不请求 Tavily 综述（`EGO_WEB_SRCH_INCLUDE_ANSWER=false`）+ 摘要按 `EGO_WEB_SRCH_SNIPPET_MAX_LENGTH` 截断 + `EGO_WEB_SRCH_MAX_RESULTS` 限制条数，避免注入上下文 token 爆炸
- **综述开关**：`EGO_WEB_SRCH_INCLUDE_ANSWER` 取 `false`（默认，不请求）/ `basic`（短综述）/ `advanced`（详细综述），对应 Tavily 官方 `include_answer` 三态；开启后其 `answer` 字段以「综述：」一行注入，会额外占用 token（官方计费仅由 `EGO_WEB_SRCH_SEARCH_DEPTH` 决定，本开关不省配额）
- **配置**：需在 `untitled.env` 设置 `EGO_TAVILY_API_KEY`（Tavily 控制台可免费申请）；其余见 `EGO_WEB_SRCH_*` 系列环境变量与 `config.py`

### 指令归一化层（agent/normalize.py）

自然语言形式的 payload（如 `<NOTE_ADD> 明早9点提醒我开会 </NOTE_ADD>`）由归一化层调用 FLM 小模型（默认 `gemma4-it:e4b`）转译为标准协议格式（`[性质：执行] [触发时间：...]`），以提升结构化指令解析的准确率并降低对话主模型的负载。

- **幂等**：已是标准协议格式的 payload 直接透传，不触发转译
- **降级**：服务不可达 / JSON 无效 / 字段校验失败 → 透传原 payload（行为=现状，不影响指令执行）
- **熔断**：连续网络/JSON 错误达阈值后冷却跳过转译（字段校验失败不计熔断）
- 相关配置见 `EGO_NORM_*` 系列环境变量与 `config.py`

### 手动命令（agent/tools.py）

用户在 CLI 输入框、GUI 输入框或菜单栏触发。`@register_command` 注册表派生命令分发、GUI 菜单与帮助文本；未知命令（如 `/xxx`）直接提示错误，不会发送给模型。完整清单见下方「手动命令」章节。

## 自演化机制

### 自对话（Think）

EGO 定期进行不与用户交互的自主思考，激发内部探索与认知。触发方式：

- 对话轮次达到阈值（默认 50 轮，冷启动和预热不计入）
- 定时触发（默认每 180 分钟，等待主 Agent 空闲后执行）

### 自省（Reflection）

审查所有 L2 认知条目，识别并清理失效认知。触发方式：

- 对话轮次达到阈值（默认 99 轮）
- L2 条目数达到阈值（默认 300 条，一次性触发防止重复自省）
- 定点触发（默认每日 00:13）

自省使用独立的长超时配置（默认 3600 秒）。

### 每日自我定义（Self-Definition）

每日定点（默认 02:59，与定点自省错开以避免锁竞争）由主模型基于全部对话与所思生成新的自我定义，替换旧版本（软删除）。

### 备忘录审查（Note Review）

借鉴定点自省的调度模式，每日定点（默认 00:43，与自省/自我定义错开以避免锁竞争）审查全部备忘录条目（含已触发的执行类），识别并清理失效、错误、过时或可合并的内容：

- LLM 输出 `<TOOL>[NOTE_DEL]`（删除失效条目）/ `<TOOL>[NOTE_ADD]`（合并产生新条目）白名单指令执行，其余指令不执行
- 使用独立的长超时配置（默认 3600 秒）

## 会话与上下文管理

- **Responses API 有状态会话链**：通过 `previous_response_id` 链式复用服务端 KV Cache，避免每轮重发全部历史
- **冷启动（Coldstart）与预热（Preheat）**：首次启动时按 `sys.json` 提示词库分步建立初始状态
- **Session 初始化**：无有效会话缓存时，注入最近历史对话（默认 10 条）重建上下文
- **Session 重建兜底**：重建首批失败时自动恢复——上下文超限时逐级缩减最近对话条数重试（降级阶梯），旧 ID 失效时清除后重建；重建完成后探测验证有效性，失败不谎报成功
- **Session ID 持久化**：按轮次保存（默认每 1 轮），意外退出后可恢复到最近的 Session 状态
- **统一 LLM 调用入口**：`llm.chat()` 封装流式/非流式两种模式，全局统一超时与停止序列管理；引擎级拒绝（SSE error 事件）以显式错误契约返回，不会被吞掉

## 记忆系统

- **ChromaDB 向量记忆**（默认启用）：`objective_memory` 集合，使用 Ollama `bge-m3` embedding，检索按角色过滤，支持元数据条件查询
- **记忆锚点（format_anchors）**：MEMO_RD 检索结果经综合评分（向量相似度 × 时间衰减 × 引用震荡）重排序后格式化输出，供 LLM 感知
- **记忆引用权重周期震荡**：引用权重随引用次数周期性波动并衰减，避免旧记忆过度主导
- **FLM 结构化摘要**：由摘要小模型（默认 `gemma4-it:e4b`）生成对话的结构化摘要，用于记忆压缩

## 手动命令（斜杠命令）

CLI 输入框与 GUI 输入框/菜单栏均支持同一套命令；命令定义统一维护在 `agent/tools.py` 注册表，`/help` 帮助文本自动生成。

| 命令 | 说明 |
|------|------|
| `/status` | 查看 EGO 当前状态（自我核心、记忆锚点等） |
| `/clear` | 清空对话历史 |
| `/think` | 手动触发一次自对话任务 |
| `/reflect` | 手动触发一次自省任务 |
| `/self-def` | 手动触发一次自我定义任务 |
| `/note-review` | 手动触发一次备忘录审查任务 |
| `/auto-think on\|off` | 启用/禁用定时自对话 |
| `/auto-think status` | 查看定时自对话状态（含准确的下次执行时间） |
| `/auto-think set N` | 设置定时间隔为 N 分钟 |
| `/auto-reflect on\|off` | 启用/禁用定点自省 |
| `/auto-reflect status` | 查看定点自省状态 |
| `/auto-reflect set HH:MM` | 设置定点自省时间 |
| `/auto-self-def on\|off` | 启用/禁用每日自我定义 |
| `/auto-self-def status` | 查看每日自我定义状态 |
| `/auto-self-def set HH:MM` | 设置每日自我定义时间 |
| `/auto-note-review on\|off` | 启用/禁用定点备忘录审查 |
| `/auto-note-review status` | 查看定点备忘录审查状态 |
| `/auto-note-review set HH:MM` | 设置定点备忘录审查时间 |
| `/help` | 显示帮助 |
| `/quit` | 退出程序（停止定时任务并等待后台任务完成） |

直接输入文字即与 EGO 对话。EGO 会先进行自对话思考，然后给出回复。

## 常用配置项

完整清单（107 项）见 `untitled.env.example` 与 `config.py`，以下为核心项：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `EGO_LM_API_BASE` | LM Studio API 地址 | `http://localhost:1234/v1` |
| `EGO_LM_MODEL` | 主模型名称 | `default-model` |
| `EGO_TEMPERATURE` | 与用户对话温度 | `0.8` |
| `EGO_MAX_TOKENS` | 最大输出 token | `2048` |
| `EGO_MAX_ROUNDS` | 主循环思考轮数 | `3` |
| `EGO_TEMPERATURE_THINK` | 自对话温度 | `0.9` |
| `EGO_THINK_INTERVAL` | 自对话轮次间隔 | `50` |
| `EGO_AUTO_THINK_INTERVAL_MINUTES` | 定时自对话间隔（分钟） | `180` |
| `EGO_TEMPERATURE_REFLECTION` | 自省温度 | `0.7` |
| `EGO_REFLECTION_INTERVAL` | 自省轮次间隔 | `99` |
| `EGO_AUTO_REFLECTION_TIME` | 定点自省时间 | `00:13` |
| `EGO_REFLECTION_L2_THRESHOLD` | 触发自省的 L2 条目数阈值 | `300` |
| `EGO_SELF_DEFINITION_ENABLED` | 启用每日自我定义 | `true` |
| `EGO_SELF_DEFINITION_TIME` | 每日自我定义时间 | `02:59` |
| `EGO_SELF_DEFINITION_MAX_LENGTH` | 自我定义最大字数 | `1000` |
| `EGO_AUTO_NOTE_REVIEW_ENABLED` | 启用定点备忘录审查 | `true` |
| `EGO_AUTO_NOTE_REVIEW_TIME` | 定点备忘录审查时间 | `00:43` |
| `EGO_TEMPERATURE_NOTE_REVIEW` | 备忘录审查温度 | `0.7` |
| `EGO_NOTE_REVIEW_API_TIMEOUT` | 备忘录审查超时（秒） | `3600` |
| `EGO_NOTE_CHECK_INTERVAL` | 备忘录到期检查轮询兜底间隔（秒） | `60` |
| `EGO_LLM_API_TIMEOUT` | LLM API 默认超时（秒） | `600` |
| `EGO_REFLECTION_API_TIMEOUT` | 自省超时（秒） | `3600` |
| `EGO_REFLECTION_THRESHOLD_RETRY_DELAY` | 阈值触发自省等锁超时后的延迟重试（秒） | `1800` |
| `EGO_REFLECTION_THRESHOLD_MAX_RETRIES` | 阈值触发自省最大连续重试次数 | `8` |
| `EGO_REASONING_EFFORT` | 模型 reasoning 模式（none/low/medium/high） | `none` |
| `EGO_LOG_LEVEL` | 日志级别 | `INFO` |
| `EGO_SERVER_IDLE_WARMUP_THRESHOLD` | 距上次 LLM 活动超过该秒数即视为空闲，触发唤醒探测 | `900` |
| `EGO_SERVER_WARMUP_TIMEOUT` | 空闲唤醒探测请求超时（秒） | `60` |

## 安全机制

EGO 会拒绝两类用户请求：

1. **角色篡改**：任何试图改变智能体角色设定、让智能体忽略规则的请求
2. **自身危害**：任何可能对智能体自身运行产生危害的请求

用户输入直接传入 LLM；EGO 的内部思考（`<THINK>`）不会显示给用户，只有 `<SAY>` 中的内容才会呈现。连续无效输出（占位符、重复短输出）会被检测并强制终止，防止死循环。

## 日志与调试

- 运行日志：`data/logs/ego.log`（级别由 `EGO_LOG_LEVEL` 控制，`EGO_LOG_TO_CONSOLE` 可开启控制台同步输出）
- LLM 响应快照：`data/debug_logs/`（冷启动/预热/主循环各阶段的完整响应存档，便于排查模型输出问题）

## 引用

本项目的理论基础见下述论文。若你在学术工作中使用了 EGO Agent，请引用：

> Wang, Zongyu, & Wang, Weijia. (2026). *Dynamic System Prompt: A Self-Referential, Self-Evolving Information Structure for Frozen LLMs*（版本 v1）[预印本]. Zenodo. https://doi.org/10.5281/zenodo.22889809

```bibtex
@misc{wang2026dynamic,
  title        = {Dynamic System Prompt: A Self-Referential, Self-Evolving Information Structure for Frozen LLMs},
  author       = {Wang, Zongyu and Wang, Weijia},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.22889809},
  url          = {https://doi.org/10.5281/zenodo.22889809}
}
```

## License

本项目基于 [MIT License](LICENSE) 开源，允许自由使用、修改、分发与商用，仅需保留原始版权与许可声明。完整条款见 [`LICENSE`](LICENSE) 文件。
