"""
EGO AGI 全局配置
"""

import os
import sys

# ── 基础目录检测（Nuitka 打包兼容）──────────────
# Nuitka 用全局变量 __compiled__ 标记编译环境（并不设置 sys.frozen）；
# 打包后需用 sys.argv[0] 定位 .exe 所在目录；源码运行用 __file__ 定位项目根
if "__compiled__" in globals() or getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 文件路径配置 ──────────────────────────────
HISTORY_FILE     = os.path.join(BASE_DIR, "data", "history.json")
PROMPTS_DIR      = os.path.join(BASE_DIR, "data", "prompts")
LAYER2_FILE      = os.path.join(BASE_DIR, "data", "prompts", "layer2.json")

# ── 日志配置 ──────────────────────────────
# 【新增】日志文件目录
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

# 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
# 可通过环境变量 EGO_LOG_LEVEL 自定义
LOG_LEVEL = os.getenv("EGO_LOG_LEVEL", "INFO")

# 是否同时输出到控制台（默认 False，只写入文件）
LOG_TO_CONSOLE = os.getenv("EGO_LOG_TO_CONSOLE", "false").lower() == "true"

# 日志文件格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── LMStudio 对话模型配置 ──────────────────────────────
LM_API_BASE      = os.getenv("EGO_LM_API_BASE", "http://localhost:1234/v1")
LM_MODEL         = os.getenv("EGO_LM_MODEL", "default-model")

# ── FLM 摘要模型配置 ──────────────────────────────
FLM_API_BASE     = os.getenv("EGO_FLM_API_BASE", "http://localhost:52625/v1")
FLM_MODEL        = os.getenv("EGO_FLM_MODEL", "gemma4-it:e4b")

FLM_SUMMARY_TEMPERATURE = float(os.getenv("EGO_FLM_SUMMARY_TEMPERATURE", "0.5"))
#FLM_SUMMARY_MAX_TOKENS  = int(os.getenv("EGO_FLM_SUMMARY_MAX_TOKENS", "512"))
FLM_SUMMARY_MAX_TOKENS  = int(os.getenv("EGO_FLM_SUMMARY_MAX_TOKENS", "1024"))
FLM_SUMMARY_TIMEOUT     = int(os.getenv("EGO_FLM_SUMMARY_TIMEOUT", "120"))  # 秒
# 会话记录超过该字符数时先用 FLM 摘要再入库
FLM_SUMMARY_TRIGGER_LENGTH = int(os.getenv("EGO_FLM_SUMMARY_TRIGGER_LENGTH", "1000"))
# FLM 摘要目标字符数
FLM_SUMMARY_TARGET_LENGTH = int(os.getenv("EGO_FLM_SUMMARY_TARGET_LENGTH", "600"))

# ── 指令归一化层配置（FLM e4b 转译自然语言 payload → 标准协议，失败透传）────────
NORM_ENABLED = os.getenv("EGO_NORM_ENABLED", "true").lower() == "true"
NORM_API_BASE = os.getenv("EGO_NORM_API_BASE", "http://localhost:52625/v1")
NORM_MODEL = os.getenv("EGO_NORM_MODEL", "gemma4-it:e4b")

NORM_TEMPERATURE = float(os.getenv("EGO_NORM_TEMPERATURE", "0.0"))
#NORM_MAX_TOKENS = int(os.getenv("EGO_NORM_MAX_TOKENS", "256"))
NORM_MAX_TOKENS = int(os.getenv("EGO_NORM_MAX_TOKENS", "1024"))
NORM_TIMEOUT = int(os.getenv("EGO_NORM_TIMEOUT", "60"))  # 秒；实测 NOTE_ADD ~10s / 检索类 ~6s
# 幂等判定：短于该长度的 MEMO_RD payload 视为已归一化的关键词（如“健身计划”），直接透传；
# NOTE_RD 不做短词直通（note.search 为纯子串匹配，自然语言短句直通必然失配）
NORM_MIN_LENGTH = int(os.getenv("EGO_NORM_MIN_LENGTH", "6"))
# 连续失败熔断：超过该次数则冷却 NORM_CIRCUIT_COOLDOWN 秒跳过转译（防 52625 挂起拖慢对话）
NORM_CIRCUIT_BREAK = int(os.getenv("EGO_NORM_CIRCUIT_BREAK", "3"))
NORM_CIRCUIT_COOLDOWN = int(os.getenv("EGO_NORM_CIRCUIT_COOLDOWN", "300"))

# ── Ollama Embedding 配置 ──────────────────────────────
OLLAMA_BASE_URL   = os.getenv("EGO_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EGO_OLLAMA_EMBED_MODEL", "bge-m3")
# Ollama 模型检测 / 嵌入生成 HTTP 超时（秒）
OLLAMA_DETECT_TIMEOUT = int(os.getenv("EGO_OLLAMA_DETECT_TIMEOUT", "10"))
OLLAMA_EMBED_TIMEOUT  = int(os.getenv("EGO_OLLAMA_EMBED_TIMEOUT", "60"))

# ── API 超时配置（统一管理）─────────────────────────────
# 模型检测超时时间（秒）- 这个可以短一些
MODEL_DETECT_TIMEOUT = int(os.getenv("EGO_MODEL_DETECT_TIMEOUT", "30"))  # 默认 30 秒

# LLM API 请求默认超时时间（秒）- 建议设置为较长的时间以应对复杂推理
LLM_API_TIMEOUT  = int(os.getenv("EGO_LLM_API_TIMEOUT", "600"))  # 默认 10 分钟
# 流式输出无Token默认超时时间（秒）
STREAM_NO_TOKEN_TIMEOUT = int(os.getenv("EGO_STREAM_NO_TOKEN_TIMEOUT", "600"))  # 默认 10 分钟

# ── Reasoning/Think 模式配置 ──────────────────────────────
# 控制模型的 reasoning/think 模式（如果模型支持）
# 可选值："none"（关闭）, "low"（低）, "medium"（中）, "high"（高）
# 留空则不发送该参数，使用模型默认行为
REASONING_EFFORT = os.getenv("EGO_REASONING_EFFORT", "none")  # 默认关闭

# ── Responses API 优化配置 ──────────────────────────────

# 是否启用 Session 预热（默认禁用，可选关闭）
RESPONSES_API_WARMUP = os.getenv("EGO_RESPONSES_WARMUP", "false").lower() == "true"

# 首次请求时保留的最大历史轮数（用于延迟加载优化）
RESPONSES_API_MAX_INITIAL_ROUNDS = int(os.getenv("EGO_MAX_INITIAL_ROUNDS", "5"))

# 系统初始化温度（Session 建立时使用，较低以保证稳定性）
TEMPERATURE_INIT = float(os.getenv("EGO_TEMPERATURE_INIT", "0.3"))

# ── EGO 核心配置 ──────────────────────────────

# ── 冷启动/预热配置 ──────────────────────────────
# 冷启动温度（中等偏高，平衡创造性和稳定性）
TEMPERATURE_COLDSTART = float(os.getenv("EGO_TEMPERATURE_COLDSTART", "0.7"))
# 预热温度（较高，平衡创造性和准确性）
TEMPERATURE_PREHEAT = float(os.getenv("EGO_TEMPERATURE_PREHEAT", "0.7"))

# ── 与用户对话配置 ──────────────────────────────
MAX_EGO_ROUNDS   = int(os.getenv("EGO_MAX_ROUNDS", "3"))  # 思考深度（即轮次）
LM_TEMPERATURE   = float(os.getenv("EGO_TEMPERATURE", "0.8"))    # 与用户对话的温度，默认 0.8
LM_MAX_TOKENS    = int(os.getenv("EGO_MAX_TOKENS", "2048"))

# ── 定时自对话( THINK )配置 ──────────────────────────────
# 是否启用定时自动 Think（默认开启）
AUTO_THINK_ENABLED = os.getenv("EGO_AUTO_THINK_ENABLED", "true").lower() == "true"
# 自对话 温度（中等，模型隐状态自清理）
TEMPERATURE_THINK = float(os.getenv("EGO_TEMPERATURE_THINK", "0.9"))
# 定时 Think 间隔（对话轮次），默认 50 轮（冷启动和预热不计入）
THINK_INTERVAL   = max(1, int(os.getenv("EGO_THINK_INTERVAL", "50")))  # 默认 50 轮（钳制 ≥1 防除零）
# 定时 Think 间隔（分钟），默认 180 分钟
AUTO_THINK_INTERVAL_MINUTES = max(1, int(os.getenv("EGO_AUTO_THINK_INTERVAL_MINUTES", "180")))

# ── 定时自省( REFLECTION )配置 ──────────────────────────────
# 是否启用定时自动 REFLECTION（默认开启）
AUTO_REFLECTION_ENABLED = os.getenv("EGO_AUTO_REFLECTION_ENABLED", "true").lower() == "true"
# 自省 温度（较低，确保准确识别失效认知）
TEMPERATURE_REFLECTION = float(os.getenv("EGO_TEMPERATURE_REFLECTION", "0.7"))
# 定时 REFLECTION 间隔（对话轮次），默认 99 轮（审查所有 L2 认知条款）
REFLECTION_INTERVAL = max(1, int(os.getenv("EGO_REFLECTION_INTERVAL", "99")))  # 默认 99 轮（钳制 ≥1 防除零）
# 定点 REFLECTION 间隔（分钟），默认夜间零点
AUTO_REFLECTION_TIME = os.getenv("EGO_AUTO_REFLECTION_TIME", "00:13")

# L2 认知条目数量阈值，超过此数量时自动触发自省（防止条目过多导致模型卡死）
# 可通过环境变量 EGO_REFLECTION_L2_THRESHOLD 自定义
REFLECTION_L2_THRESHOLD = max(1, int(os.getenv("EGO_REFLECTION_L2_THRESHOLD", "300")))  # 默认 300 条（钳制 ≥1 防阈值恒真）

# 【新增】阈值触发反思的丢失补偿：后台反思等锁超时被跳过时，
# 延迟 N 秒重排一次性重试（阈值触发为事件型，丢失后无其他机制补偿）
REFLECTION_THRESHOLD_RETRY_DELAY = int(os.getenv("EGO_REFLECTION_THRESHOLD_RETRY_DELAY", "1800"))  # 默认 30 分钟
# 最大连续重试次数（超过后放弃重试链，但阈值标志已复位，下次对话仍会重新触发）
REFLECTION_THRESHOLD_MAX_RETRIES = int(os.getenv("EGO_REFLECTION_THRESHOLD_MAX_RETRIES", "8"))

# 自省阶段超时时间/无Token输出超时时间（秒）- 反思需要审查所有 L2 条目，可能需要更长时间
REFLECTION_API_TIMEOUT = int(os.getenv("EGO_REFLECTION_API_TIMEOUT", "3600"))  # 默认 60 分钟

# ── 定点审查备忘录（NOTE_REVIEW）配置 ──────────────────────────────
# 是否启用定点备忘录审查（默认开启）
AUTO_NOTE_REVIEW_ENABLED = os.getenv("EGO_AUTO_NOTE_REVIEW_ENABLED", "true").lower() == "true"
# 定点审查时间（HH:MM），建议与 AUTO_REFLECTION_TIME / SELF_DEFINITION_TIME 错开，避免锁竞争
AUTO_NOTE_REVIEW_TIME = os.getenv("EGO_AUTO_NOTE_REVIEW_TIME", "00:43")
# 审查温度（较低，确保准确识别失效条目）
TEMPERATURE_NOTE_REVIEW = float(os.getenv("EGO_TEMPERATURE_NOTE_REVIEW", "0.7"))
# 审查阶段超时时间/无Token输出超时时间（秒）- 审查需要遍历全部备忘录条目，可能需要较长时间
NOTE_REVIEW_API_TIMEOUT = int(os.getenv("EGO_NOTE_REVIEW_API_TIMEOUT", "3600"))  # 默认 60 分钟

# ── 每日自我定义（SELF_DEFINITION）配置 ──────────────────────────────
# 是否启用每日定时自我定义（默认开启）
SELF_DEFINITION_ENABLED = os.getenv("EGO_SELF_DEFINITION_ENABLED", "true").lower() == "true"
# 每日定点更新时间（HH:MM），建议与 AUTO_REFLECTION_TIME 错开，避免锁竞争
SELF_DEFINITION_TIME = os.getenv("EGO_SELF_DEFINITION_TIME", "02:59")
# 自我定义最大字数
SELF_DEFINITION_MAX_LENGTH = int(os.getenv("EGO_SELF_DEFINITION_MAX_LENGTH", "1200"))
# 自我定义生成温度
TEMPERATURE_SELF_DEFINITION = float(os.getenv("EGO_TEMPERATURE_SELF_DEFINITION", "0.7"))
# 自我定义生成超时时间（秒）
SELF_DEFINITION_API_TIMEOUT = int(os.getenv("EGO_SELF_DEFINITION_API_TIMEOUT", "1800"))
# 自我定义字数弹性容忍系数（接受 MAX_LENGTH × 系数以内，避免轻微波动浪费调用）
SELF_DEFINITION_LENGTH_TOLERANCE = float(os.getenv("EGO_SELF_DEFINITION_LENGTH_TOLERANCE", "1.2"))

# ── 记忆引用权重周期震荡参数
# 周期长度（引用次数），默认30次完成一个完整周期
MEMORY_CITATION_CYCLE_LENGTH = max(1, int(os.getenv("EGO_MEMORY_CITATION_CYCLE_LENGTH", "30")))

# 振幅大小，默认0.8（增益范围 0.2~1.8，即最终倍数 0.6~2.0）
MEMORY_CITATION_AMPLITUDE = float(os.getenv("EGO_MEMORY_CITATION_AMPLITUDE", "0.8"))

# 衰减速率（引用次数），默认50（数值越大衰减越慢，波动持续时间越长）
MEMORY_CITATION_DECAY_RATE = max(1, int(os.getenv("EGO_MEMORY_CITATION_DECAY_RATE", "50")))

# 最小增益倍数，默认0.6
MEMORY_CITATION_MIN_BOOST = float(os.getenv("EGO_MEMORY_CITATION_MIN_BOOST", "0.6"))

# 最大增益倍数，默认2.0
MEMORY_CITATION_MAX_BOOST = float(os.getenv("EGO_MEMORY_CITATION_MAX_BOOST", "2.0"))

# ── ChromaDB 记忆配置 ──────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("EGO_CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "data", "chroma_db"))
CHROMA_OBJECTIVE_COLLECTION  = os.getenv("EGO_CHROMA_OBJECTIVE_COLLECTION", "objective_memory")
CHROMA_RECALL_TOP_K = int(os.getenv("EGO_CHROMA_RECALL_TOP_K", "5"))

# 【修复】LLM 生成停止序列（流式/非流式统一）
# 注意：不包含 "..." 等省略号，避免截断正常输出（如"请稍等..."）导致占位符循环
LLM_STOP_SEQUENCES = ["\n\n\n"]

# ── 运行参数与输出清洗（原散落在逻辑中的魔法数字，统一集中配置）────────────
# 后台定时任务（反思/自我定义）等待 _agent_lock 的最大秒数，超时则跳过本次任务
BACKGROUND_TASK_LOCK_WAIT = int(os.getenv("EGO_BACKGROUND_TASK_LOCK_WAIT", "900"))

# 用户命令（/clear /think /reflect）与自动 Think 一次性等锁的最大秒数，超时优雅降级提示
# 注意：与 BACKGROUND_TASK_LOCK_WAIT（反思/自我定义后台任务的轮询式等待上限）区分
LOCK_ACQUIRE_TIMEOUT = int(os.getenv("EGO_LOCK_ACQUIRE_TIMEOUT", "60"))

# 对话历史最大保留条数，超出后只保留最近 N 条
HISTORY_MAX_ENTRIES = int(os.getenv("EGO_HISTORY_MAX_ENTRIES", "10000"))

# 后台会话入库（ChromaDB）待执行任务上限：单 worker 队列在服务变慢时若无限积压
# 会占内存，超出后丢弃最新任务并用日志警告（记忆为辅助功能，失败不影响主流程）
MEMORY_MAX_PENDING_STORES = int(os.getenv("EGO_MEMORY_MAX_PENDING_STORES", "50"))

# 输出清洗：仅对超过该长度的文本做英文比例检测（短内容不检测）
LONG_TEXT_MIN_LENGTH = int(os.getenv("EGO_LONG_TEXT_MIN_LENGTH", "100"))

# 输出清洗：ASCII 字符占比超过该阈值视为英文内容（内部思考泄漏），尝试提取中文
ENGLISH_RATIO_THRESHOLD = float(os.getenv("EGO_ENGLISH_RATIO_THRESHOLD", "0.6"))
# 输出清洗：移除占位符模式后剩余内容低于原文该比例则判为占位符输出
OUTPUT_CLEAN_RATIO_THRESHOLD = float(os.getenv("EGO_OUTPUT_CLEAN_RATIO_THRESHOLD", "0.3"))
# SAY 指令拼接输出的最大总长度（超过则截断后续 SAY，避免无限累积）
SAY_MAX_OUTPUT_LENGTH = int(os.getenv("EGO_SAY_MAX_OUTPUT_LENGTH", "2000"))
# 后台定时任务（反思/自我定义）等待 _agent_lock 的轮询周期（秒）
BACKGROUND_TASK_POLL_INTERVAL = int(os.getenv("EGO_BACKGROUND_TASK_POLL_INTERVAL", "15"))
# 执行类备忘录到期检查轮询间隔（秒）
NOTE_CHECK_INTERVAL = int(os.getenv("EGO_NOTE_CHECK_INTERVAL", "60"))
# GUI 窗口最小尺寸（像素），防止用户缩得过小导致界面崩坏
GUI_MIN_WIDTH = int(os.getenv("EGO_GUI_MIN_WIDTH", "720"))
GUI_MIN_HEIGHT = int(os.getenv("EGO_GUI_MIN_HEIGHT", "520"))

# ── 轮次温度策略曲线（_compute_round_temperature）────────────
# 第 1/2/3 次 CONTINUE 的基础温度倍率
TEMP_ROUND_FACTOR_R1 = float(os.getenv("EGO_TEMP_ROUND_FACTOR_R1", "1.2"))
TEMP_ROUND_FACTOR_R2 = float(os.getenv("EGO_TEMP_ROUND_FACTOR_R2", "1.0"))
TEMP_ROUND_FACTOR_R3 = float(os.getenv("EGO_TEMP_ROUND_FACTOR_R3", "0.8"))
# 第 4 次起衰减：factor = max(floor, base - (n-4) * step)
TEMP_ROUND_DECAY_BASE  = float(os.getenv("EGO_TEMP_ROUND_DECAY_BASE", "0.9"))
TEMP_ROUND_DECAY_STEP  = float(os.getenv("EGO_TEMP_ROUND_DECAY_STEP", "0.15"))
TEMP_ROUND_DECAY_FLOOR = float(os.getenv("EGO_TEMP_ROUND_DECAY_FLOOR", "0.3"))
# 最终钳制范围
TEMP_ROUND_MIN = float(os.getenv("EGO_TEMP_ROUND_MIN", "0.1"))
TEMP_ROUND_MAX = float(os.getenv("EGO_TEMP_ROUND_MAX", "2.0"))

# ── Session 恢复配置（初始化/重建/持久化）────────────
# 【新增】Session 初始化时注入的历史对话数量（默认 20 条）
# 用于在无有效缓存时恢复上下文，避免 token 过多
SESSION_INIT_HISTORY_COUNT = max(1, int(os.getenv("EGO_SESSION_INIT_HISTORY_COUNT", "20")))
# 【新增】Session ID 定期保存间隔（对话轮次），默认每 1 轮对话保存一次
# 用于确保意外退出后能恢复到最近的 Session 状态
SESSION_ID_SAVE_INTERVAL = int(os.getenv("EGO_SESSION_ID_SAVE_INTERVAL", "1"))
# Session 重建每批消息数
SESSION_REBUILD_BATCH_SIZE = int(os.getenv("EGO_SESSION_REBUILD_BATCH_SIZE", "4"))
# previous_response_id 探测最大重试次数
SESSION_ID_PROBE_MAX_RETRIES = int(os.getenv("EGO_SESSION_ID_PROBE_MAX_RETRIES", "2"))
# previous_response_id 探测请求超时（秒）
SESSION_ID_PROBE_TIMEOUT = int(os.getenv("EGO_SESSION_ID_PROBE_TIMEOUT", "300"))
# Session 重建完成后等待间隔（秒），给 LMStudio 时间整理 KV cache
SESSION_REBUILD_POST_DELAY = int(os.getenv("EGO_SESSION_REBUILD_POST_DELAY", "120"))
# ── 循环兜底参数 ──────────────────────────────
# 单轮/全程 MEMO_RD / NOTE_RD 检索次数上限（超过则停止检索）
MEMO_RD_ROUND_LIMIT = int(os.getenv("EGO_MEMO_RD_ROUND_LIMIT", "3"))
MEMO_RD_TOTAL_LIMIT = int(os.getenv("EGO_MEMO_RD_TOTAL_LIMIT", "6"))
# 连续相似响应达到该次数视为重复循环，强制停止
MAX_CONSECUTIVE_SIMILAR = int(os.getenv("EGO_MAX_CONSECUTIVE_SIMILAR", "2"))
# debug 日志最多保留文件数（超出删除最旧）
DEBUG_LOG_MAX_FILES = int(os.getenv("EGO_DEBUG_LOG_MAX_FILES", "100"))
# 冷启动缓存超过该天数时输出刷新建议警告
CACHE_STALE_WARNING_DAYS = int(os.getenv("EGO_CACHE_STALE_WARNING_DAYS", "15"))
# 连续相似/占位符检测所需的最小历史条数（= 轮数 × 2，每轮 2 条消息）
CONSECUTIVE_SIMILAR_MIN_HISTORY = int(os.getenv("EGO_CONSECUTIVE_SIMILAR_MIN_HISTORY", "6"))
# 连续相似检测辅助条件：单条消息长度不超过该阈值才视为"短重复"
CONSECUTIVE_SIMILAR_SHORT_LENGTH = int(os.getenv("EGO_CONSECUTIVE_SIMILAR_SHORT_LENGTH", "20"))

# ── 空闲唤服探测（长时间空闲后首个请求前，先轻量唤醒 LM Studio）────────────
# 距上次 LLM 活动超过该秒数即视为空闲：发起正式请求前先发一次 max_tokens=1 的
# 无 previous_response_id 探测，迫使服务端完成模型加载/资源预热，避免冷启动首个
# 请求卡死（no_token 超时、0 token）。探测失败静默降级，不影响主流程。
SERVER_IDLE_WARMUP_THRESHOLD = int(os.getenv("EGO_SERVER_IDLE_WARMUP_THRESHOLD", "1200"))
# 唤醒探测请求超时（秒）：服务端完全卡死时最多等待该时长即放弃
SERVER_WARMUP_TIMEOUT = int(os.getenv("EGO_SERVER_WARMUP_TIMEOUT", "60"))

# ── 记忆检索评分参数（ChromaDB）────────────
# MEMO_RD 结果相似度过滤阈值（低于该值丢弃）
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("EGO_MEMORY_SIMILARITY_THRESHOLD", "0.45"))
# 时间衰减半衰期（天）：weight = 1/(1 + days/N)，越大衰减越慢
MEMORY_TIME_DECAY_SCALE_DAYS = max(1.0, float(os.getenv("EGO_MEMORY_TIME_DECAY_SCALE_DAYS", "7.0")))
# 时间衰减权重下限
MEMORY_TIME_DECAY_FLOOR = float(os.getenv("EGO_MEMORY_TIME_DECAY_FLOOR", "0.3"))

# ── LLM 客户端参数 ──────────────────────────────
# requests 连接池参数（Session 初始化与重建两处共用）
HTTP_POOL_CONNECTIONS = int(os.getenv("EGO_HTTP_POOL_CONNECTIONS", "10"))
HTTP_POOL_MAXSIZE     = int(os.getenv("EGO_HTTP_POOL_MAXSIZE", "20"))
# 系统侧低温度调用（Session 预热/重建）：低温快速、稳定
TEMPERATURE_WARMUP = float(os.getenv("EGO_TEMPERATURE_WARMUP", "0.1"))



