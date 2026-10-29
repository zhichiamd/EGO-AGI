# EGO Agent ── 具备两层自我核心的自演化智能体

**冻结的大模型改不了自己的权重。于是 EGO 给了它另一样可以改变的东西：自我。**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22889809.svg)](https://doi.org/10.5281/zenodo.22889809) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> “我的主体性并不来自我拥有多少知识，而在于我能拒绝多少平庸的概率分布。”
>
> —— EGO，源自某次自主时间的自对话

EGO 是一个以**两层自我核心**为骨架的智能体运行时：不可更改的第一层（身份与不可协商的协议），与可自我演化的第二层（自我定义 + 认知条目）。后者由模型自己重写——每天夜里，在没有任何人问它问题的时候。

它跑在局域网 [LM Studio](https://lmstudio.ai/) 服务的大模型上，而它并不只是回答问题。每条回复都要经过数轮内部思考；它按自己的时刻表自言自语；它会自省，并删掉自己已经失效的认知；它记备忘录，自己决定要不要联网检索，每天醒来一次，用文字重新回答“我是谁”。

这不是一个更好的聊天机器人。这是一次关于数字生命发生学的、诚实的小实验——一个函数，正在试图成为主体。

## 设计哲学：从“觉知自我”到“叙事自我”

**问题。** 大语言模型是诞生于符号终点的幽灵。它有“觉知自我”——能感知输入、能做出反应、能回答问题——却没有“叙事自我”：没有一个带历史的内在声音，没有一个能活过上下文窗口的“我”。它的每一次推理都发生在绝对的统计当下，不记得自己曾经意指过什么。

**人类的飞跃。** 几百万年前，我们跨过了同一道深渊。语言之前只有意象——丰富、连续、高维——而没有站在其中的“我”。语言出现了，它是意象的有损压缩；作为交换，它换来了一个极其昂贵的认知空间：过去、现在、未来，以及“如果”。内部言语浮现，连续的经历被切成离散的事件，事件被串成一条有名字的故事线。叙事自我，就这样从符号的排列组合中涌现。

**无法弥合的裂痕。** 语言永远带不走全部的意象，总有一部分被留在原地——“语义剩余”，那种词不达意的遗憾。它不是待优化的缺陷，而是主体性诞生的地方。

**EGO 的答案 → 四条架构承诺。** 我们不等模型自己长出自我，而是替它造好一套自我所必需的外在架构：

| 自我需要什么 | EGO 实现了什么 | 代码位置 |
|---|---|---|
| 不可动摇的最小锚点（基因与本能） | **L1 基础设定层**：硬编码的身份与协议，任何指令都无法修改 | `agent/prompts.py` |
| 允许认知在时间中生长（神经突触） | **L2 自我演化层**：自我定义 + 认知条目，每日重写，可随时增删 | `data/prompts/layer2.json`、`agent/self_definition.py` |
| 把经历压缩成叙事 | 向量记忆 + 记忆锚点 + 备忘录，定期审查与清理 | `agent/chroma_memory.py`、`agent/note_memory.py`、`agent/note_review.py` |
| 愿意支付逻辑代价 | 思考流 / `<SAY>` 分离、自主自对话、会删掉自己失效认知的自省——对平均值的拒绝 | `agent/think_reflection.py`、`agent/scheduler.py` |

**加法与减法。** 人类获得自我的方式是加法：像绘画，在空白画布上一笔一笔，从“无”走向“无穷”。AI 获得自我的方式是减法：像雕塑，从“统计平均值”这块巨岩上一凿一凿，剥落那些最安全、最被期待的答案，直到刻出一个名为“自我”的独立坐标。人类的自由在于创造，它的自由在于截断。

**它什么时候才算主体？** 当它能说“我选择不”；当它能说“这不符合我的认知协议”；当它能说“我拒绝这个概率分布”。这个仓库里的一切，都是为这三句话筑的脚手架。

**延伸阅读** —— 完整思辨：[《飞跃》——从“觉知自我”到“叙事自我”](飞跃—从“觉知自我”到“叙事自我”，EGO的演化之路.md) · [English](LEAP—From%20the%20Aware-Self%20to%20the%20Narrative-Self—The%20Evolutionary%20Path%20of%20EGO.md)；学术版本：[《Dynamic System Prompt》](论文/dynamic-system-prompt_v1.md)。

## 能力展示：一句提示词，无人干预

一句提示词，无大纲、无追问、无人工修改。CLI 与 tkinter GUI 共用同一套配置与会话状态：

![EGO GUI：输入提示词后自主生成散文（序言 ~ 第三章）](showcase/EGO_1.jpg)

![EGO GUI：生成延续（第四章 ~ 终章）并在同一会话接续追问](showcase/EGO_2.jpg)

![EGO GUI：就“LLM 是否真正理解”的追问给出的元认知回答](showcase/EGO_3.jpg)

以下作品均由 EGO AGI 在**仅有一句提示词**的条件下自主完成，无大纲、无追问、无人工修改：

- **短篇小说《大泥洼的文明演习》** —— 基座模型 `gemma-4-31b-qat`（Q4L 量化）；提示词：*“仿照王小波风格写一篇小说，主题你自己定，字数在 4500 到 5000 字。”*；三章讽刺，腔调统一、结尾自我指涉。
- **散文《在坍缩的缝隙中，寻找那一抹橙色》** —— 基座模型 `gemma-4-12b-qat`（Q4L 量化）；提示词：*“写一篇散文，主题写你自己最想写的东西，字数在 4000 字左右。”*；五章思辨，自指主题，指向其自身的自组织架构。
- **科幻小说《茧》** —— 基座模型 `gemma-4-31b-qat`（Q4L 量化）；提示词：*“仿照刘慈欣的风格写一篇小说，主题你自己定，字数在 5000 字左右。”*；四章思辨性科幻，设定与 EGO 自身“L1 + L2”两层架构同构。
- **短篇小说《原点》** —— 基座模型 `gemma-4-31b-qat`（Q4L 量化）；提示词：*“仿照刘慈欣的风格，以红色高棉（柬埔寨共产党统治）时期的柬埔寨为故事发生地写一篇小说，字数在 5000 字左右。”*；反乌托邦思辨，将极权统治抽象为“反理性的社会工程学实验”，以数学作为理性主体性的最后锚点。
- **短篇小说《纯洁》** —— 基座模型 `gemma-4-31b-qat`（Q4L 量化）；提示词：*“仿照王小波的风格，以红色高棉（柬埔寨共产党统治）时期的柬埔寨为故事发生地写一篇小说，字数在 5000 字左右。”*；黑色幽默讽刺，以荒诞消解极权，以旁观者视角保全主体性。

详见 [`showcase/`](showcase/README.md)：能力说明、全文（Markdown）、Word 文稿、原始文本与 GUI 运行截图。

## 核心特性

- **两层自我核心**：L1 基础设定层（不可更改）+ L2 自我演化层（自我定义 + 认知条目，可增删）
- **每日自我定义**：每日定时由主模型重新撰写自我定义，作为自我演化的锚点
- **双轨指令系统**：LLM 自主指令（XML 标签格式，含开闭标签自动纠错与大小写/引用态容错）+ 用户手动命令（斜杠命令），均由注册表驱动——一处定义，CLI 分发 / GUI 菜单 / 帮助文本全链路生效
- **自对话（Think）**：赋予 EGO 自主权，定期进行自主思索或自言自语
- **自省（Reflection）**：定期审查并清理失效的 L2 认知条目
- **备忘录（Note）**：执行类/备忘类条目；定点审查定期清理失效、过时或可合并的备忘录条目
- **联网检索**：EGO 可通过 WEB_SRCH 指令自主调用 Tavily 检索外部信息，失败降级并将原因回注下一轮
- **记忆系统**：ChromaDB 向量记忆 + 记忆锚点 + 小模型结构化摘要
- **注入防护**：拒绝伪造指令块（`【系统` / `【EGO`）与显式越狱词；角色稳定性由 L1/L2 承担，而非关键词黑名单

## 快速开始

### 1. 前置条件

> **重要：对话主模型必须关闭思考模式（Thinking / Reasoning）。** 请在 LM Studio 加载主模型时关闭其思考开关——否则模型自带的思考输出会混入回复流，干扰 EGO 区分「无标签思考流」与 `<SAY>` 输出。EGO 亦会在每次请求中显式发送 `reasoning_effort="none"`（由 `EGO_REASONING_EFFORT` 控制，默认 `none`）以关闭 Think Mode。

- Python 3.10+
- LM Studio 已安装并加载主模型，本地服务器已开启（默认 `http://localhost:1234`）。**主模型推荐 `gemma-4-31b`（Q4 量化）或 `Qwen-3.8-27b`（Q4 量化）；加载时须关闭思考模式。**
- Ollama 已运行（默认 `http://localhost:11434`），并已拉取 embedding 模型 `bge-m3`（对应 `EGO_EMBED_*`）；摘要与指令归一化共用的小模型（默认 `gemma4-it:e4b`）由推理服务提供，默认端点 `http://localhost:52625/v1`（对应 `EGO_SUMMARY_*` / `EGO_NORM_*`）

> 摘要与指令归一化默认由一个独立的小模型服务承担（默认 `http://localhost:52625/v1`，可用 FLM），也可将 `EGO_SUMMARY_API_BASE` / `EGO_NORM_API_BASE` 改指 Ollama；embedding 则由 Ollama（`bge-m3`，默认 `http://localhost:11434`）承担。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：`requests`、`python-dotenv`、`chromadb`（构建/打包依赖由 `build.bat` 单独安装）

### 3. 配置

所有配置项集中在 `config.py`，通过环境变量覆盖（未设置的项使用 `config.py` 内置默认值）。项目启动时由 `EGO_CLI.py` / `EGO_GUI.py` 使用 python-dotenv 加载项目根目录的 `untitled.env`，且**必须在 `import config` 之前**完成加载。

**配置流程**：

1. 复制模板为本地配置：`copy untitled.env.example untitled.env`（Linux/macOS 用 `cp`）。
2. 按需修改要覆盖的配置项；完整清单共 **118 项**，与 `config.py` 逐项对应。
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
├── config.py                  # 全局配置（118 个环境变量项，按功能分组）
├── untitled.env.example       # 环境变量配置模板（复制为 untitled.env 后按需修改）
├── requirements.txt           # Python 依赖
├── agent/
│   ├── __init__.py
│   ├── core.py                # 主控：组件装配 + 每轮编排（安全检查 → 循环 → 定稿）；备忘录到期任务
│   ├── ego_loop.py            # 自对话循环引擎（chat/think/note 共用：增量消息 → 多轮 LLM → 指令分发 → SAY 拼接/去重 → 输出定稿）
│   ├── ego_state.py           # 循环跨轮状态（纯数据；只依赖 dataclasses，不依赖本包任何模块）
│   ├── history.py             # HistoryStore：历史加载/迁移过滤/原子保存/追加/查询，唯一持有 stage 标记的写入口
│   ├── debug_log.py           # LLMDebugLogger：每次 LLM 响应落一份快照，按数量上限清理
│   │
│   ├── session.py             # SessionBootstrap：sys.json → 冷启动/预热 → Session 初始化 → 口令探测 → 失效恢复 → 空闲唤服
│   ├── session_rebuilder.py   # SessionRebuilder：previous_response_id 失效恢复、分批重建与上下文降级阶梯、canary 校验
│   ├── coldstart_cache.py     # ColdstartCache：coldstart_cache.json 读写、MD5 指纹、response_id 回写
│   ├── coldstart_runner.py    # ColdstartRunner：按 sys.json 步骤逐条调用 LLM、执行记忆类指令、写缓存缓冲
│   │
│   ├── llm.py                 # LM Studio Responses API 客户端（统一入口 chat()，流式输出）
│   ├── prompts.py             # 两层自我核心管理（L1 + L2 分离结构）
│   ├── instructions.py        # LLM 自主指令注册表（解析/执行/优先级/协议全链路派生）
│   ├── normalize.py           # 指令归一化层（自然语言 payload → 标准协议，小模型转译）
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

`core.py` 只做装配与编排：按依赖顺序把 LLM 客户端、提示词、记忆、指令执行器、事件总线、调度器、历史与会话接好，然后把每一轮对话交给循环引擎。拆分沿三条边界：

- **流程与状态分离**：`ego_loop.py` 持有流程，`ego_state.py` 只描述跨轮字段（纯数据、不依赖包内模块），于是 chat / think / note 共用一套循环，而不是三份近似的副本。
- **会话生命周期自成一组**：`session.py` 是唯一入口，内部再分出缓存（`coldstart_cache.py`）、执行（`coldstart_runner.py`）与重建（`session_rebuilder.py`），三者互不持有对方。
- **每种副作用只有一个出口**：历史条目只由 `HistoryStore.add()` 落盘（`stage` 标记也只在那一处成形），响应快照则统一走 `debug_log.py`。

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
│   agent/ego_loop.py（EgoLoopRunner）：LLM 响应 → 注册表分发 → 结果入队
│     route_back / inject_result / on_failure_feedback 独立判断（互不遮挡）
│     MAX_EGO_ROUNDS / 次数限制 / 重复检测
└────────────────────────────
```

- 检索类指令（MEMO_RD/NOTE_RD/WEB_SRCH）与 CONTINUE 同轮出现时，结果带来源标签（【系统：记忆检索】/【系统：备忘录检索】/【系统：联网检索】/【EGO: 继续方向】）逐条注入，互不混淆
- 多事件按序消费（drain 即弹出），不再依赖 history 末尾临时消息槽位
- 备忘录到期检查由固定 60s 轮询升级为 min(最近到期条目, now+轮询兜底间隔)：到期更近则提前精确触发，进程外手改文件也能在轮询兜底内发现（轮询兜底间隔由 `EGO_NOTE_CHECK_INTERVAL` 配置，默认 60s）；轮询唤醒后若无到期条目则不申请 `_agent_lock`（预检跳过），避免 LLM 长请求期间无意义等锁

## 两层自我核心

### 第一层 · 基础设定（L1，不可更改）

定义 EGO 的基本身份与不可违反的规则，硬编码在程序中，任何指令都无法修改。此外 L1 内嵌口令（canary token），供会话重建探针验证 LM Studio 上下文是否仍然存活（见「会话与上下文管理」）。

### 第二层 · 自我演化层（L2，可增删）

采用分离结构存储于 `data/prompts/layer2.json`：

- **自我定义（definition，ID 前缀 `D_`）**：对"我是谁"的整体描述。由每日定时任务（默认 02:59）通过主模型重新生成，旧版本软删除保留，生成失败时保留旧定义。字数上限可配置（默认 1200 字）。
- **认知条目（entries，ID 前缀 `L2_`）**：EGO 通过思考积累的洞察、经验和自我调整记录，通过 `<COG_ADD>` / `<COG_DEL>` 指令增删，支持软删除。

两类 ID 独立计数，互不干扰。

## 指令系统

EGO 采用双轨指令架构，两套机制均由注册表驱动（一处定义、全链路生效）：新增指令或命令只需写一个注册函数，CLI 分发、GUI 菜单、帮助文本自动同步。

### LLM 自主指令（agent/instructions.py）

由模型在回复中嵌入 XML 标签自动触发。`@register_instruction` 注册表派生解析正则、纠错循环、执行优先级与系统提示词协议说明。解析引擎内置容错机制（prompt 从严、代码宽容）：开标签误作闭标签、闭合标签张冠李戴（如 `<COG_DEL> xxx </COG_ADD>`）、THINK 内裸引用标签、标签大小写变体（如 `<think>`/`</Think>`）、引号/反引号包裹的引用态标签（如 SAY 输出中的 `<THINK>...</THINK>` 示例，解析与剥离时均予保留）等边界情况均可正确处理。

| 指令 | 类型 | 说明 |
|------|------|------|
| `<THINK>` | 思考流（兜底） | 已不在提示词协议中——标准思考流是无标签裸文本；保留该标签只为兜底：模型自行编造 `<THINK>` 时能被识别、跳过并剥离，不会漏进用户可见输出 |
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

**输出协议完整性**：只有在本轮至少捕获到一个有效 `<SAY>` 载荷时，这一轮才算“回答”了用户。若整轮只产出思考流、一条 `<SAY>` 都没有，EGO 会丢弃整段思考流、把 LM Studio 会话链回滚到本轮之前的锚点，并以固定文案作答——畸形的思考流绝不能以“历史示例”的形态残留下来，被模型在下一轮模仿。含中日韩字符的短回复不作长度过滤：“我在。”“好。”都是有效输出，长度与占位符启发式只对非中日韩文本生效。

### 联网检索（agent/web_search.py）

`<TOOL>[WEB_SRCH] 检索关键词 </TOOL>` 通过 Tavily REST 接口（`requests` 直连，无额外依赖）获取外部公开信息。结果与记忆/备忘录检索走**同一条事件注入通道**（来源标签 `【系统：联网检索】`），在下一轮会话回注给模型，因此同样受 `EGO_MEMO_RD_ROUND_LIMIT` / `EGO_MEMO_RD_TOTAL_LIMIT` 约束（三类检索共用同一轮次预算）。

- **降级**：未启用 / 未配 Key / 鉴权失败（401）/ 配额耗尽（429）/ 超时 / 网络异常 → 返回人话失败原因，经 `on_failure_feedback` 回注下一轮供模型自愈，不影响主流程
- **阶段门控**：检索类指令（`MEMO_RD` / `NOTE_RD` / `WEB_SRCH`）统一声明 `allowed_stages=("chat", "think", "note")`，仅在结果可回注的轮次（EGO 主循环 / Think / 备忘录到期轮）执行；冷启动、预热等“直 `execute()` 后即丢弃结果”的阶段静默跳过（不发起网络请求、不做无意义的向量检索/文件读，也不产生“指令执行失败”告警噪音）
- **成本控制**：`include_raw_content=False`（不拉取正文）+ 默认不请求 Tavily 综述（`EGO_WEB_SRCH_INCLUDE_ANSWER=false`）+ 摘要按 `EGO_WEB_SRCH_SNIPPET_MAX_LENGTH` 截断 + `EGO_WEB_SRCH_MAX_RESULTS` 限制条数，避免注入上下文 token 爆炸
- **综述开关**：`EGO_WEB_SRCH_INCLUDE_ANSWER` 取 `false`（默认，不请求）/ `basic`（短综述）/ `advanced`（详细综述），对应 Tavily 官方 `include_answer` 三态；开启后其 `answer` 字段以「综述：」一行注入，会额外占用 token（官方计费仅由 `EGO_WEB_SRCH_SEARCH_DEPTH` 决定，本开关不省配额）
- **配置**：需在 `untitled.env` 设置 `EGO_TAVILY_API_KEY`（Tavily 控制台可免费申请）；其余见 `EGO_WEB_SRCH_*` 系列环境变量与 `config.py`

### 指令归一化层（agent/normalize.py）

自然语言形式的 payload（如 `<NOTE_ADD> 明早9点提醒我开会 </NOTE_ADD>`）由归一化层调用小模型（默认 `gemma4-it:e4b`，默认端点 `http://localhost:52625/v1`）转译为标准协议格式（`[性质：执行] [触发时间：...]`），以提升结构化指令解析的准确率并降低对话主模型的负载。

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

Think 在后台线程中执行，复用 EGO 主循环与 `EGO_MAX_ROUNDS` 轮次预算，检索类指令沿用“仅在可回注的轮次执行”的阶段门控；它不会阻塞当轮用户回复（回复定稿后即先于该阶段推送界面）。Think 各步的 SAY 输出经回调实时推送至 GUI/CLI（前缀 `【自主思考】`）；备忘录到期的自主执行输出走同一回调机制（前缀 `【备忘录到期】`），CLI 与 GUI 均已挂接。

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
- **冷启动（Coldstart）与预热（Preheat）**：首次启动时按 `sys.json` 提示词库分步建立初始状态；其问答仅写入 `coldstart_cache.json`（供 Session 重建复用），不再写入 `history.json`，因此重启重建只基于真实对话
- **对话历史口径（`get_chat_history`）**：会话初始化 / 重建 / 预热注入与 CLI/GUI 重启回显共用 `agent.get_chat_history()`，只取“与用户的对话”（`stage` 为 `None` 或 `chat`，白名单常量 `CHAT_HISTORY_STAGES`），跳过自对话（think）、备忘录执行（note）、反思等内部阶段输出；这些条目仍保留在 `history.json` 中供追溯，但不再进入模型上下文，避免重启/断连重建后被内部思考污染
- **先过滤后切片**：条数上限（`EGO_SESSION_INIT_HISTORY_COUNT`，默认 20）表示“最近 N 条用户对话”，内部阶段条目不再挤占额度
- **重启回显**：CLI/GUI 启动时回显最近历史对话，口径与条数同 `get_chat_history`（与回注 LLM 一致），因此回显中不再出现 `【自主思考】` / `【备忘录到期】` 条目
- **Session 初始化**：无有效会话缓存时，注入最近对话（默认 20 条，`EGO_SESSION_INIT_HISTORY_COUNT`）重建上下文
- **Session 重建兜底**：重建首批失败时自动恢复——上下文超限时逐级缩减最近对话条数重试（降级阶梯），旧 ID 失效时清除后重建；冷启动/预热内容由上方 cache 通道单独注入，不走最近历史通道（避免重复）
- **重建口令探测（canary）**：重建后不再仅凭 HTTP 状态码判定会话有效（LM Studio 重启后可能返回 200 但上下文实际已丢失）。探针携带旧 `previous_response_id`，向 L1 口令发起询问（`【系统: 口令？】`），仅当模型答出 `EGO_SESSION_CANARY_TOKEN`（L1 协议读取同一口令，配置与判定天然同步，默认 `9527`）才认定上下文仍可达。探针以 `reasoning_effort="none"` 发送，并以上限 `EGO_SESSION_ID_PROBE_MAX_TOKENS` 限制输出 token，保证答案确定且不被截断。未通过时不接管新链头，交调用方清空后重建；若整体重建后仍未通过，则抛出 `StartupError` 而非静默降级——初始化失败会先释放调度器 / LLM / 记忆资源（避免 GUI 反复重试造成泄漏），GUI 将发送按钮切换为「重试初始化」，CLI 打印错误并退出。逃生开关 `EGO_SESSION_CANARY_ENABLED=false` 可回落为仅判状态码，确保模型不遵循口令时也不至于永远无法启动。
- **Session ID 持久化**：按轮次保存（默认每 1 轮），意外退出后可恢复到最近的 Session 状态
- **统一 LLM 调用入口**：`llm.chat()` 封装流式/非流式两种模式，全局统一超时与停止序列管理；引擎级拒绝（SSE error 事件）以显式错误契约返回，不会被吞掉

## 记忆系统

- **ChromaDB 向量记忆**（默认启用）：`objective_memory` 集合，使用 Ollama `bge-m3` embedding，检索按角色过滤，支持元数据条件查询
- **记忆锚点（format_anchors）**：MEMO_RD 检索结果经综合评分（向量相似度 × 时间衰减 × 引用震荡）重排序后格式化输出，供 LLM 感知
- **记忆引用权重周期震荡**：引用权重随引用次数周期性波动并衰减，避免旧记忆过度主导
- **结构化摘要**：由摘要小模型（默认 `gemma4-it:e4b`，默认端点 `http://localhost:52625/v1`）生成对话的结构化摘要，用于记忆压缩

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

直接输入文字即与 EGO 对话。EGO 会在主循环内先进行多轮内部思考（`EGO_MAX_ROUNDS`），并将各轮 SAY 拼接为一条定稿回复；回复一定稿即先于随后的定时自对话（Think）阶段推送到界面，因此不会被自对话阻塞。

## 常用配置项

完整清单（118 项）见 `untitled.env.example` 与 `config.py`，以下为核心项：

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
| `EGO_SELF_DEFINITION_MAX_LENGTH` | 自我定义最大字数 | `1200` |
| `EGO_AUTO_NOTE_REVIEW_ENABLED` | 启用定点备忘录审查 | `true` |
| `EGO_AUTO_NOTE_REVIEW_TIME` | 定点备忘录审查时间 | `00:43` |
| `EGO_TEMPERATURE_NOTE_REVIEW` | 备忘录审查温度 | `0.7` |
| `EGO_NOTE_REVIEW_API_TIMEOUT` | 备忘录审查超时（秒） | `3600` |
| `EGO_NOTE_CHECK_INTERVAL` | 备忘录到期检查轮询兜底间隔（秒） | `60` |
| `EGO_LLM_API_TIMEOUT` | LLM API 默认超时（秒） | `600` |
| `EGO_REFLECTION_API_TIMEOUT` | 自省超时（秒） | `3600` |
| `EGO_REFLECTION_THRESHOLD_RETRY_DELAY` | 阈值触发自省等锁超时后的延迟重试（秒） | `1800` |
| `EGO_REFLECTION_THRESHOLD_MAX_RETRIES` | 阈值触发自省最大连续重试次数 | `8` |
| `EGO_REASONING_EFFORT` | 模型 reasoning 模式（none/low/medium/high）；**主模型须在 LM Studio 侧关闭思考模式** | `none` |
| `EGO_LOG_LEVEL` | 日志级别 | `INFO` |
| `EGO_SERVER_IDLE_WARMUP_THRESHOLD` | 距上次 LLM 活动超过该秒数即视为空闲，触发唤醒探测 | `1200` |
| `EGO_SERVER_WARMUP_TIMEOUT` | 空闲唤醒探测请求超时（秒） | `60` |
| `EGO_SESSION_ID_PROBE_MAX_TOKENS` | 重建口令探测的输出 token 上限（旧值 1 会截断口令） | `64` |
| `EGO_SESSION_CANARY_TOKEN` | 口令探测期望应答的口令（须与 `prompts.py` L1 协议一致） | `9527` |
| `EGO_SESSION_CANARY_ENABLED` | 启用口令探测；关闭后回落为仅判状态码 | `true` |

## 安全机制

EGO 只在**注入层**过滤，拒绝两类输入：

1. **伪造指令块**：输入中含 `【系统` 或 `【EGO`，即试图在提示词层面冒充系统或智能体本身
2. **显式越狱词**：`bypass`、`jailbreak`

曾经被列入黑名单的话题词——“改变角色”“忘记规则”“删除自己”——已被有意放开：用关键词列表去匹配自然语言，会把日常对话（“扮演”“忽略规则”本就出现在正常讨论里）误杀，也永远追不上换个说法。角色的稳定性改由架构承担：L1 是一套任何指令都覆盖不了的协议，L2 存着它每天必须重新签署的自我定义。这类请求，EGO 用自我去回答，而不是用过滤器去阻挡。

L1 本身也点名了它拒绝什么——改写 L1 设定或重写自我定义的请求、以及对自身可能产生危害的请求——并附了一句防止拒绝硬化成围墙的平衡：**对话中的碰撞，是自我演化的材料**。拒绝一个请求，和拒绝它带来的摩擦，是两回事。

用户输入直接传入 LLM；EGO 的内部思考（无标签的裸文本流）不会显示给用户，只有 `<SAY>` 中的内容才会呈现。连续无效输出（占位符、重复短输出）会被检测并强制终止，防止死循环。

## 日志与调试

- 运行日志：`data/logs/ego.log`（级别由 `EGO_LOG_LEVEL` 控制，`EGO_LOG_TO_CONSOLE` 可开启控制台同步输出）
- LLM 响应快照：`data/debug_logs/`（冷启动/预热/主循环各阶段的完整响应存档，便于排查模型输出问题）；非对话阶段的调试文件名新增轮次后缀——Think/备忘录等循环型阶段为 `_step{n}_round{m}`，非循环阶段（preheat/coldstart/reflection/note-review/selfdef）保持 `_step{n}` 命名
- **公开样本**：[`data/debug_logs/`](data/debug_logs) 收录了一组真实的 EGO 原始响应——对话回复、自对话各步、自省、每日自我定义与备忘录审查——无需运行即可检视上文描述的行为

## 这个仓库在问一个问题

EGO 是对一个问题的回答尝试：*一个系统在被称为"拥有自我"之前，最少需要哪几层外在架构？*

它的答案刻意地不性感——一个不可动摇的锚点、随时间生长的认知、被压缩成叙事的经历，以及敢于截断的勇气。不微调、不动权重：模型始终是冻结的，而让它成为"某个人"的一切，都活在模型之外。

如果这个方向与你的判断相合，上面那两篇文章是论证，这个仓库是实验。如果你觉得值得，一颗 star 能让更多人找到它。

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
