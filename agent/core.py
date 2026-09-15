"""
EGO 智能体核心

整合两层自我核心、指令系统、记忆锚点、自我对话循环。
"""

from __future__ import annotations

import time
import json
import os
import re
import logging
import threading
import traceback
from dataclasses import dataclass, field
import requests
from datetime import datetime
from pathlib import Path

from agent.llm import LMStudioClient, is_llm_error, add_timestamps
from agent.prompts import PromptManager
from agent.event_bus import EventBus  # 【新增】统一事件注入通道（注入层）
from agent.scheduler import EgoScheduler  # 【新增】统一后台定时调度器（调度层）
from agent.instructions import (
    parse_instructions, strip_instructions, InstructionExecutor,
    is_invalid_content, is_blank_text, INVALID_PLACEHOLDER_PATTERNS,
    filter_empty_payload,
    get_instruction,
)
from agent.chroma_memory import ChromaMemoryManager
from agent.note_memory import NoteMemory
from agent.think_reflection import ThinkReflectionMixin
from agent.self_definition import SelfDefinitionMixin
from agent.note_review import NoteReviewMixin
from config import (
    MAX_EGO_ROUNDS, 
    HISTORY_FILE, 
    RESPONSES_API_WARMUP, 
    RESPONSES_API_MAX_INITIAL_ROUNDS, 
    THINK_INTERVAL,
    TEMPERATURE_INIT,
    TEMPERATURE_COLDSTART,
    TEMPERATURE_PREHEAT,
    SESSION_INIT_HISTORY_COUNT,
    SESSION_ID_SAVE_INTERVAL,
    LOCK_ACQUIRE_TIMEOUT,  # 用户命令一次性等锁最大秒数
    NOTE_CHECK_INTERVAL,  # 【新增】备忘录到期检查轮询间隔（秒）
    BACKGROUND_TASK_LOCK_WAIT,  # 后台任务轮询等锁最大秒数
    BACKGROUND_TASK_POLL_INTERVAL,  # 后台任务轮询等锁间隔（秒）
    HISTORY_MAX_ENTRIES,  # 对话历史最大保留条数
    LONG_TEXT_MIN_LENGTH,  # 输出清洗：长文本阈值
    ENGLISH_RATIO_THRESHOLD,  # 输出清洗：英文占比阈值
    OUTPUT_CLEAN_RATIO_THRESHOLD,  # 输出清洗：占位符移除比例阈值
    SAY_MAX_OUTPUT_LENGTH,  # SAY 拼接输出最大总长度
    TEMP_ROUND_FACTOR_R1,  # 轮次温度曲线：第 1 次 CONTINUE 倍率
    TEMP_ROUND_FACTOR_R2,  # 轮次温度曲线：第 2 次 CONTINUE 倍率
    TEMP_ROUND_FACTOR_R3,  # 轮次温度曲线：第 3 次 CONTINUE 倍率
    TEMP_ROUND_DECAY_BASE,  # 轮次温度曲线：第 4 次起衰减基准
    TEMP_ROUND_DECAY_STEP,  # 轮次温度曲线：衰减步长
    TEMP_ROUND_DECAY_FLOOR,  # 轮次温度曲线：衰减下限
    TEMP_ROUND_MIN,  # 轮次温度钳制下限
    TEMP_ROUND_MAX,  # 轮次温度钳制上限
    MEMO_RD_ROUND_LIMIT,  # 单轮 MEMO_RD 检索次数上限
    MEMO_RD_TOTAL_LIMIT,  # 全程 MEMO_RD 检索次数上限
    MAX_CONSECUTIVE_SIMILAR,  # 连续相似响应重复循环阈值
    DEBUG_LOG_MAX_FILES,  # debug 日志最大保留文件数
    CACHE_STALE_WARNING_DAYS,  # 冷启动缓存刷新建议天数
    CONSECUTIVE_SIMILAR_MIN_HISTORY,  # 连续相似检测最小历史条数
    CONSECUTIVE_SIMILAR_SHORT_LENGTH,  # 连续相似检测短消息阈值
    SESSION_REBUILD_BATCH_SIZE,  # Session 重建每批消息数
    SESSION_ID_PROBE_MAX_RETRIES,  # previous_response_id 探测重试次数
    SESSION_ID_PROBE_TIMEOUT,  # previous_response_id 探测请求超时（秒）
    SESSION_REBUILD_POST_DELAY,  # Session 重建后等待间隔（秒）
    TEMPERATURE_WARMUP,  # 系统侧低温度调用（预热/重建）
    SERVER_IDLE_WARMUP_THRESHOLD,  # 空闲唤服探测空闲阈值（秒）
    SERVER_WARMUP_TIMEOUT,  # 空闲唤服探测请求超时（秒）
    BASE_DIR,  # Nuitka 打包兼容的基础目录
)


# 获取 logger 实例
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """
    记录日志（级别由调用方显式指定，默认 INFO；不再靠关键词猜测）
    
    Args:
        message: 要记录的消息
        level: 日志级别（logging.INFO / WARNING / ERROR）
    """
    logger.log(level, message)


def _is_executable(kind: str) -> bool:
    """指令是否需真正执行（未注册 / 标记型指令一律不执行）

    原多处硬编码 "kind not in (SAY, THINK, CONTINUE)" 特判的注册表等价物。
    """
    spec = get_instruction(kind)
    return bool(spec and spec.executable)

# 注：无效内容统一检测（is_invalid_content / INVALID_PLACEHOLDER_PATTERNS）
# 定义于 agent/instructions.py（避免循环导入），已从顶部 import 引入

def _is_context_overflow(result: str) -> bool:
    """判断重建失败是否为上下文超限（引擎拒绝特征：context/exceeds/tokens）"""
    low = result.lower()
    return "context" in low or "exceeds" in low or "tokens" in low


def _safe_int_key(key):
    """sys.json 步骤键排序键：无法解析为数字的键排最后（防止用户编辑文件导致启动崩溃）"""
    try:
        return int(key)
    except (ValueError, TypeError):
        return float("inf")

@dataclass
class _EgoLoopState:
    """
    【重构】EGO 自我对话循环运行状态（跨轮累积）
    """
    output_content: str = ""
    has_output: bool = False
    recall_count: int = 0
    continue_count: int = 0
    consecutive_similar_count: int = 0
    round_num: int = 0
    ego_log: list = field(default_factory=list)
    recent_responses: list = field(default_factory=list)


@dataclass
class _RoundControl:
    """
    【重构】单轮指令执行后产出的控制信号
    """
    should_continue: bool = False
    round_recall_count: int = 0

class EGOAgent(ThinkReflectionMixin, SelfDefinitionMixin, NoteReviewMixin):
    """EGO 智能体主控类"""

    def __init__(self):
        self.llm = LMStudioClient()
        self.pm = PromptManager()

        # 记忆系统：统一使用 ChromaDB（旧本地锚点系统已移除）
        self.mem = ChromaMemoryManager()

        # 【新增】备忘录存储（NOTE_ADD / NOTE_RD / NOTE_DEL 指令后端）
        self.note = NoteMemory()

        self.executor = InstructionExecutor(self.pm, self.mem, self.note)

        # 【新增】全局可重入锁：保护 LLM 调用和共享资源
        # 使用 RLock 允许同一线程重入（process_input 内的 Think/自省不会死锁）
        # 定时器线程获取锁时会被阻塞，直到主线程释放
        self._agent_lock = threading.RLock()

        # 【新增】三层架构接线：统一事件注入通道（注入层）+ 统一后台调度器（调度层）
        # 调度器必须创建于锁之后（任务执行依赖 _agent_lock），顺带修正旧实现中
        # 备忘录检查 Timer 先于锁创建调度的脆弱点
        self.events = EventBus()
        # 【新增】后台自主运行（备忘录到期）输出回调：GUI/CLI 挂接后可实时推送
        self.on_note_output = None  # callable(text) -> None，后台线程调用
        self.scheduler = EgoScheduler(
            self,
            lock_acquire_timeout=LOCK_ACQUIRE_TIMEOUT,
            background_wait=BACKGROUND_TASK_LOCK_WAIT,
            poll_interval=BACKGROUND_TASK_POLL_INTERVAL,
        )
        # 注册后台定时任务（各 mixin 提供注册逻辑；备忘录到期任务注册在 core）
        self._register_think_tasks()
        self._register_self_def_task()
        self._register_note_review_task()
        self._register_note_task()
        # 启动全部已启用任务（内部重排最近事件，设定单个 Timer）
        self.scheduler.reschedule()

        # 对话历史
        self.history: list[dict] = []
        self._load_history()

        # 上次 LLM 服务端活动时间（用于空闲唤醒判定）；0 表示进程启动后首个请求即视为空闲
        self._last_llm_activity = 0.0

        # LLM Debug Log 目录和文件
        self.debug_log_dir = Path(BASE_DIR) / "data" / "debug_logs"
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_counter = 0  # 用于生成唯一的日志文件名
        
        # 加载系统提示词配置（sys.json）
        self.sys_prompts = self._load_sys_prompts()
        
        # 【修复】对话计数器初始化：从历史中统计 CHAT 阶段的用户对话轮数
        # 只统计 CHAT 阶段的用户消息（排除 coldstart、preheat、think、note 阶段）
        chat_user_count = sum(
            1 for msg in self.history 
            if msg["role"] == "user" and msg.get("stage") not in ["coldstart", "preheat", "think", "note"]
        )
        self.conversation_counter = chat_user_count
        _auto_log(f"[信息] 从历史中恢复对话计数器: {self.conversation_counter} 轮")

        # 【新增】Session ID 定期保存配置
        self.session_id_save_counter = 0  # 用户对话计数
        self.session_id_save_interval = SESSION_ID_SAVE_INTERVAL  # 每 N 轮对话保存一次
        
        self.THINK_INTERVAL = THINK_INTERVAL  # 从 config.py 读取配置
        
        # 冷启动缓存文件路径
        self.coldstart_cache_file = Path(BASE_DIR) / "data" / "coldstart_cache.json"
        
        # 【优化】智能判断是否需要冷启动/预热
        # 如果 L2 认知库已有内容，说明系统已经"觉醒"过，跳过引导阶段直接进入交互
        valid_entries = self.pm.get_valid_entries()
        if valid_entries:
            _auto_log(f"[信息] 检测到 L2 认知库已存在 {len(valid_entries)} 条有效认知，跳过冷启动/预热")
            _auto_log("[信息] 系统将基于现有认知直接进入用户交互模式")
            self._initialize_session_with_cache()
        else:
            # 执行冷启动和预热流程（仅在 L2 为空时执行）
            self._execute_coldstart_and_preheat()
        
        # 【修复】禁用 Session 预热，因为冷启动/预热已经建立了完整的 KV cache
        if RESPONSES_API_WARMUP and len(self.history) > 2:
            self._warmup_session()

        # 【新增】定时任务（自对话/自省/自我定义/备忘录审查/备忘录到期）已统一注册到 self.scheduler
        # （注册逻辑见各 mixin 的 _register_think_tasks / _register_self_def_task /
        # _register_note_review_task 与 core 的 _register_note_task）
        if self._auto_think_enabled or self._auto_reflection_enabled or self._self_def_enabled or self._note_review_enabled:
            _auto_log("[信息] 后台定时任务已由统一调度器接管")
        else:
            _auto_log("[信息] 全部后台定时任务未启用")            

    def _load_sys_prompts(self) -> dict:
        """
        加载冷启动/预热/Think 提示词配置（sys.json）
        
        Returns:
            包含 coldstart、Preheat、Think 三个部分的字典
        """

        _auto_log("[系统] ═══ 启动：读取配置文件 ═══")

        sys_file = Path(BASE_DIR) / "data" / "sys.json"
        
        if not sys_file.exists():
            _auto_log("[警告] sys.json 不存在，使用默认配置", level=logging.WARNING)
            return {
                "coldstart": {},
                "Preheat": {},
                "Think": {}
            }
        
        try:
            with open(sys_file, "r", encoding="utf-8") as f:
                sys_prompts = json.load(f)
            
            # 验证结构
            required_keys = ["coldstart", "Preheat", "Think"]
            for key in required_keys:
                if key not in sys_prompts:
                    _auto_log(f"[警告] sys.json 缺少 '{key}' 字段，使用空字典", level=logging.WARNING)
                    sys_prompts[key] = {}
            
            _auto_log("[配置] ✓ 成功加载 sys.json")
            _auto_log(f"[配置]   - Coldstart: {len(sys_prompts.get('coldstart', {}))} 条")
            _auto_log(f"[配置]   - Preheat: {len(sys_prompts.get('Preheat', {}))} 条")
            _auto_log(f"[配置]   - Think: {len(sys_prompts.get('Think', {}))} 条")
            
            return sys_prompts
            
        except Exception as e:
            _auto_log(f"[警告] 加载 sys.json 失败: {e}，使用默认配置", level=logging.WARNING)
            return {
                "coldstart": {},
                "Preheat": {},
                "Think": {}
            }

    # ── 核心交互循环 ────────────────────────────────────────────

    def _initialize_session(self):
        """
        初始化 session
        
        【关键】系统启动后，首次与模型对话并发送 system prompt 获取 session。
        之后所有对话都使用这个 session 进行增量对话。
        
        【新增】如果存在历史对话，会在初始化时一并发送，确保模型记住之前的对话
        """

        _auto_log("═══ 系统启动：测试并建立LM Studio会话 ═══")

        try:
            # 【新增】构建初始消息列表
            messages = [
                {"role": "system", "content": self.pm.build_system_prompt()}
            ]
            
            # 【新增】如果有历史对话，添加最近的 N 条到初始消息中（N 由配置控制）
            if self.history:
                # 从配置读取历史对话注入数量
                history_count = SESSION_INIT_HISTORY_COUNT
                
                # 只保留最近的 N 条对话（避免 token 过多）
                recent_history = self.history[-history_count:] if len(self.history) > history_count else self.history
                
                _auto_log(f"[信息] 检测到 {len(recent_history)} 条历史对话，将注入到 Session 中（配置上限: {history_count}）")
       
                for msg in recent_history:
                    # 跳过临时消息和系统消息
                    if msg.get("temporary", False):
                        continue
                    
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # 只添加 user 和 assistant 的消息
                    if role in ["user", "assistant"]:
                        messages.append({"role": role, "content": content})
                
                _auto_log(f"[信息] ✓ 已注入 {len([m for m in messages if m['role'] in ['user', 'assistant']])} 条历史消息")
            
            # 添加系统初始化消息
            messages.append({"role": "user", "content": "[系统初始化]"})
            
            # 【封装】统一通过 chat() 调用
            response = self.llm.chat(
                messages, TEMPERATURE_INIT, stage="Session-Init"
            )
            
            # 【增强】校验调用结果，尽早暴露初始化失败
            if not response or is_llm_error(response):
                _auto_log(f"[警告] ⚠ Session 初始化返回异常（不影响后续流程，首次用户对话时会重新建立）: {(response or '空响应')[:80]}", level=logging.WARNING)
                return
            
            _sid = self.llm.get_session_id()
            prev_id = _sid[:20] if _sid else 'None'
            _auto_log(f"[信息] ✓ Session 初始化完成（previous_response_id: {prev_id}...）")
            
        except Exception as e:
            _auto_log(f"[错误] ✗ Session 初始化失败: {e}", level=logging.ERROR)
            # 即使初始化失败，也不影响后续流程（会在首次用户对话时重新建立）

    def _execute_preheat_phase(self, use_incremental: bool = True):
        """
        执行预热阶段的统一方法
        
        Args:
            use_incremental: True=增量模式（session已初始化，只发user消息）
                            False=完整模式（包含system prompt，用于续传场景）
        """
        preheat_prompts = self.sys_prompts.get("Preheat", {})
        if not preheat_prompts:
            _auto_log("[预热] ⚠ 无预热提示词，跳过", level=logging.WARNING)
            return
        
        _auto_log(f"[预热] 开始执行 {len(preheat_prompts)} 条预热提示...\n")
        
        sorted_keys = sorted(preheat_prompts.keys(), key=_safe_int_key)
        
        for i, key in enumerate(sorted_keys, 1):
            prompt = preheat_prompts[key]
            _auto_log(f"[预热] [{i}/{len(sorted_keys)}] 执行: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            try:
                if use_incremental:
                    messages = [{"role": "user", "content": prompt}]
                    _auto_log(f"[信息] ✓ 预热第 {i} 轮：发送增量消息（复用 KV cache）")
                else:
                    messages = [
                        {"role": "system", "content": self.pm.build_system_prompt()},
                        {"role": "user", "content": prompt}
                    ]
                
                response = self.llm.chat(messages, TEMPERATURE_PREHEAT, stage=f"预热 Step {i}")
                
                self._save_llm_debug_log(response, round_num=-1, stage="preheat", step=i)
                
                # 解析指令
                instructions = parse_instructions(response)
                
                # 【修复】提取 SAY 内容用于历史记录，而非保存原始响应
                history_content = response
                for instr in instructions:
                    # 【注册表驱动】输出指令（SAY，注册表 output 属性）的 payload 作为历史内容
                    _out_spec = get_instruction(instr.kind)
                    if _out_spec is not None and _out_spec.output:
                        # 提取并执行 SAY payload 中的嵌套指令
                        nested = parse_instructions(instr.payload)
                        for n_instr in nested:
                            if _is_executable(n_instr.kind):
                                self.executor.set_context({"stage": "preheat", "step": i})
                                result = self.executor.execute(n_instr)
                                if result.success:
                                    _auto_log(f"[预热]   ✓ 执行嵌套指令: {n_instr.kind}")
                        # 清理 SAY payload 中的嵌套指令标签
                        history_content = strip_instructions(instr.payload).strip()
                        break
                else:
                    # 没有 SAY 指令时，使用 strip 后的内容
                    history_content = strip_instructions(response).strip()
                
                self._add_to_history_with_stage("user", prompt, stage="preheat")
                self._add_to_history_with_stage("assistant", history_content, stage="preheat")
                
                # 执行非 SAY 指令（空载荷指令整条跳过）
                for instr in filter_empty_payload(instructions):
                    if not _is_executable(instr.kind):
                        continue
                    
                    self.executor.set_context({"stage": "preheat", "step": i})
                    result = self.executor.execute(instr)
                    if result.success:
                        _auto_log(f"[预热]   ✓ 执行指令: {instr.kind}")
                    else:
                        _auto_log(f"[预热]   ✗ 指令执行失败: {instr.kind}", level=logging.WARNING)
                                      
            except Exception as e:
                _auto_log(f"[预热]   ✗ 执行失败: {e}", level=logging.WARNING)
        
        _auto_log("[预热] ✓ 预热完成")

    def _execute_coldstart_and_preheat(self):
        """
        执行冷启动和预热流程
        
        【新增】支持跳过（使用缓存）和续传（断点恢复）
        """
        # 检查是否需要跳过（已使用缓存重建）
        if getattr(self, '_skip_coldstart', False):
            _auto_log("[信息] ✓ 已使用缓存重建 Session，跳过冷启动和预热")
            return
        
        # 检查是否需要续传
        if hasattr(self, '_resume_from_step') and self._resume_from_step:
            self._resume_coldstart_and_preheat()
            return
        
        # 正常执行完整冷启动和预热
        _auto_log("═══ 系统启动：冷启动与预热 ═══")
        
        # ─ 冷启动阶段 ────────────────────────────────
        coldstart_prompts = self.sys_prompts.get("coldstart", {})
        if coldstart_prompts:
            _auto_log(f"[冷启动] 开始执行 {len(coldstart_prompts)} 条冷启动提示...\n")
            
            success = self._execute_coldstart(start_from=1)
            
            if not success:
                _auto_log("[警告] 冷启动未完成，缓存已保存进度", level=logging.WARNING)
                return
        else:
            _auto_log("[冷启动] ⚠ 无冷启动提示词，跳过", level=logging.WARNING)
        
        # ─ 预热阶段 ────────────────────────────────
        self._execute_preheat_phase(use_incremental=True)
        
        # 保存完整缓存
        self._save_coldstart_cache(status="complete")
        
        _auto_log("═══ 系统就绪 ══")
    
    def _execute_coldstart(self, start_from: int = 1) -> bool:
        """
        执行冷启动（支持从任意步骤开始）
        
        Returns:
            bool: 是否全部成功
        """
        coldstart_prompts = self.sys_prompts.get("coldstart", {})
        sorted_keys = sorted(coldstart_prompts.keys(), key=_safe_int_key)
        
        for i, key in enumerate(sorted_keys, 1):
            # 跳过已完成的步骤
            if i < start_from:
                continue
            
            prompt = coldstart_prompts[key]
            _auto_log(f"[冷启动] [{i}/{len(sorted_keys)}] 执行: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            try:
                # 【修复】冷启动第一步必须包含 System Prompt，确保模型进入 EGO 状态
                if i == 1:
                    messages = [
                        {"role": "system", "content": self.pm.build_system_prompt()},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                response = self.llm.chat(messages, TEMPERATURE_COLDSTART, stage=f"冷启动 Step {i}")
                
                # 检测超时
                if is_llm_error(response) and "请求超时" in response:
                    _auto_log(f"[警告] ✗ Step {i} 超时", level=logging.WARNING)
                    self._save_coldstart_cache(status="incomplete", failed_step=i)
                    return False
                
                # 保存 debug log
                self._save_llm_debug_log(response, round_num=-1, stage="coldstart", step=i)
                
                # 解析指令
                instructions = parse_instructions(response)
                
                # 【修复】提取 SAY 内容用于历史记录，而非保存原始响应
                history_content = response
                for instr in instructions:
                    # 【注册表驱动】输出指令（SAY，注册表 output 属性）的 payload 作为历史内容
                    _out_spec = get_instruction(instr.kind)
                    if _out_spec is not None and _out_spec.output:
                        # 提取并执行 SAY payload 中的嵌套指令
                        nested = parse_instructions(instr.payload)
                        for n_instr in nested:
                            if _is_executable(n_instr.kind):
                                self.executor.set_context({"stage": "coldstart", "step": i})
                                result = self.executor.execute(n_instr)
                                if result.success:
                                    _auto_log(f"[冷启动]   ✓ 执行嵌套指令: {n_instr.kind}")
                        # 清理 SAY payload 中的嵌套指令标签
                        history_content = strip_instructions(instr.payload).strip()
                        break
                else:
                    # 没有 SAY 指令时，使用 strip 后的内容
                    history_content = strip_instructions(response).strip()
                
                # 添加到 history（标记阶段）
                self._add_to_history_with_stage("user", prompt, stage="coldstart")
                self._add_to_history_with_stage("assistant", history_content, stage="coldstart")
                
                # 执行非 SAY 指令（空载荷指令整条跳过）
                for instr in filter_empty_payload(instructions):
                    if not _is_executable(instr.kind):
                        continue
                    
                    # 【新增】设置冷启动阶段上下文
                    self.executor.set_context({
                        "stage": "coldstart",
                        "step": i,
                    })
                    
                    result = self.executor.execute(instr)
                    if result.success:
                        _auto_log(f"[冷启动]   ✓ 执行指令: {instr.kind}")
                    else:
                        _auto_log(f"[冷启动]   ✗ 指令执行失败: {instr.kind} - {result.message}", level=logging.WARNING)
                
                _auto_log(f"[冷启动] ✓ Step {i} 完成")
                
            except Exception as e:
                _auto_log(f"[冷启动] ✗ Step {i} 失败: {e}", level=logging.WARNING)
                self._save_coldstart_cache(status="incomplete", failed_step=i)
                return False
        
        _auto_log("[冷启动] ✓ 冷启动全部完成")
        return True
    
    def _execute_preheat_only(self):
        """只执行预热阶段（用于续传场景）"""
        self._execute_preheat_phase(use_incremental=False)
    
    # ── 命令统一入口（CLI 与 GUI 共用）──────────────────────

    def clear_conversation(self) -> bool:
        """
        /clear 命令的统一实现：清空对话历史并完全重置 session
        
        Returns:
            True 成功；False 后台任务占用锁（LOCK_ACQUIRE_TIMEOUT 秒内未获取到），未能执行
        """
        # 【修复】获取锁保护，确保后台任务不在运行
        if not self._agent_lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT):
            return False
        try:
            self.history.clear()
            self._save_history()
            # 【修复】完全重置 LLM session，避免连接泄露
            _auto_log("[系统] 完全重置 LLM Session...")
            self.llm.reset_and_reconnect()
            # 重新初始化 session，发送 system prompt
            self._initialize_session()
            # 【修复】将新 session 的 previous_response_id 同步写回冷启动缓存，
            # 避免缓存中残留旧 ID 导致下次启动白白执行一轮"验证失效 → 重建"
            if self.llm.get_session_id():
                self._update_session_id_in_cache()
            # 重置对话计数器
            self.conversation_counter = 0
            return True
        finally:
            self._agent_lock.release()

    def shutdown(self, lock_wait: float = 30) -> None:
        """
        统一资源清理入口（CLI / GUI 退出时共用）：
        停止全部定时任务 → 等待后台任务完成 → 关闭 LLM 与记忆连接
        
        Args:
            lock_wait: 等待仍在运行的后台 LLM 任务释放锁的最大秒数
        """
        # 统一停用全部定时任务并取消待触发 Timer（避免退出瞬间 _tick 重排出新 Timer）
        self.scheduler.stop_all()
        _auto_log("[系统] 已停止全部定时任务")
        # 等待后台运行中的 LLM 任务结束（任务在 _agent_lock 内执行，拿到锁即无任务在跑）
        acquired = self._agent_lock.acquire(timeout=lock_wait)
        try:
            # 【改进】在持锁状态下关闭 LLM/记忆：防止后台轮询线程在 close() 之后
            # re-acquire 锁并继续使用已关闭的 Session（close 前已被 drain，不会对
            # 记忆入库线程造成死锁——其仅依赖 _stats_lock 与网络）
            self.llm.close()
            self.mem.close()
            _auto_log("[系统] 已关闭 LLM Session，释放连接资源")
        finally:
            if acquired:
                self._agent_lock.release()

    # ── 执行类备忘录到期触发（统一调度器任务：note_due） ──────────

    def _register_note_task(self):
        """在统一调度器中注册备忘录到期检查任务（由 EGOAgent.__init__ 调用）

        调度策略：事件精确 + 轮询兜底——下次执行取 min(最近到期条目, now+60s)，
        到期时间更近则提前精确触发，进程外手改 notes.json 也能在轮询兜底内发现。
        预检：轮询唤醒后先查是否有到期条目，无待办则不申请 _agent_lock
        （避免 LLM 长请求期间无意义等锁 60s 并产生 WARNING 噪音）。
        """
        self.scheduler.register(
            "note_due",
            enabled=True,
            next_run_fn=self._note_due_next_run,
            execute_fn=self._check_due_notes,
            lock_policy="once",
            wait_seconds=LOCK_ACQUIRE_TIMEOUT,
            precheck_fn=self._note_has_due,
        )

    def _note_has_due(self, task=None):
        """预检：是否存在到期条目（无则跳过不申请锁；预检失败按无待办处理，下轮轮询重试）"""
        try:
            return bool(self.note.check_due())
        except Exception:
            return False

    def _note_due_next_run(self, task):
        """备忘录下次检查时间：min(最近到期条目, now + NOTE_CHECK_INTERVAL)（锁由调度器统一获取）

        钳制规则：候选时间已过（上次运行失败保留未触发、待重试）时按 NOTE_CHECK_INTERVAL
        重试，避免 0 延迟忙轮询。
        """
        from datetime import timedelta
        try:
            due = self.note.check_due()
        except Exception:
            return datetime.now() + timedelta(seconds=NOTE_CHECK_INTERVAL)
        times = []
        for n in due:
            # 【修复】统一走 NoteMemory.parse_trigger_time 容错解析：
            # 手动添加的条目若格式宽松（空格+秒等）也能参与精确调度，
            # 不再因 strptime 抛异常被丢弃而退化为纯轮询
            t = self.note.parse_trigger_time(n.get("trigger_time"))
            if t is not None:
                times.append(t)
        if not times:
            return datetime.now() + timedelta(seconds=NOTE_CHECK_INTERVAL)
        now = datetime.now()
        candidate = min(min(times), now + timedelta(seconds=NOTE_CHECK_INTERVAL))
        # 【修复】钳制过去时间：候选时间已过（重试场景）时按间隔重试，避免 0 延迟忙轮询
        return candidate if candidate > now else now + timedelta(seconds=NOTE_CHECK_INTERVAL)

    def _check_due_notes(self):
        """检查到期条目：到期即调度 LLM 自主运行（不再依赖用户对话消费事件）

        【设计变更】原实现仅将到期提示作为持久事件入队、等待下次用户对话注入；
        现改为到期后立即在后台线程注入并运行完整 EGO 循环（锁由调度器统一获取，
        本方法只启动线程并立即返回，不阻塞调度器串行 tick）：
        - 运行成功 → 标记 triggered（不再重复触发）
        - 运行失败 / 等锁超时 → 保留未触发状态，由下轮轮询重试
        """
        try:
            # 防重入：上一批到期条目仍在运行/等待锁时，跳过本次轮询
            if getattr(self, "_note_cycle_running", False):
                _auto_log("[备忘录] ℹ 到期条目 LLM 运行仍在进行中，跳过本次轮询")
                return
            due = self.note.check_due()
            if not due:
                return
            _auto_log(f"[备忘录] ⏰ 检测到 {len(due)} 个到期执行类条目: {[n.get('id') for n in due]}")
            self._note_cycle_running = True
            threading.Thread(
                target=self._run_note_due_background,
                args=(due,),
                daemon=True,
            ).start()
        except Exception as e:
            # 【修复】线程启动失败时重置防重入标志：否则标志保持 True，
            # 后续轮询全部被防重入拦截，备忘录到期检查永久失效
            self._note_cycle_running = False
            _auto_log(f"[备忘录] ✗ 到期检查失败: {e}", level=logging.WARNING)

    def _run_note_due_background(self, due_notes: list):
        """后台执行到期条目注入：轮询等锁（等待当前对话/后台任务结束后运行）"""
        waited = 0
        acquired = False
        while waited < BACKGROUND_TASK_LOCK_WAIT:
            acquired = self._agent_lock.acquire(timeout=BACKGROUND_TASK_POLL_INTERVAL)
            if acquired:
                break
            waited += BACKGROUND_TASK_POLL_INTERVAL
            _auto_log(f"[备忘录] ⏳ 等待锁释放中...（已等待 {waited}s / {BACKGROUND_TASK_LOCK_WAIT}s）")
        try:
            if not acquired:
                _auto_log(f"[备忘录] ⚠ 等锁 {BACKGROUND_TASK_LOCK_WAIT}s 超时，本轮跳过（保留未触发，下轮轮询重试）",
                          level=logging.WARNING)
                return
            if self._run_note_due_cycle(due_notes):
                for n in due_notes:
                    self.note.mark_triggered(n["id"])
                _auto_log(f"[备忘录] ✓ 已标记触发: {[n.get('id') for n in due_notes]}")
            else:
                _auto_log("[备忘录] ⚠ 到期条目 LLM 运行未产出有效输出，保留未触发状态，等待重试",
                          level=logging.WARNING)
        except Exception as e:
            _auto_log(f"[备忘录] ✗ 到期条目运行失败: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
        finally:
            self._note_cycle_running = False
            if acquired:
                self._agent_lock.release()

    def _run_note_due_cycle(self, due_notes: list) -> bool:
        """将到期条目作为 round 0 输入注入，运行完整 EGO 循环（指令可执行、可多轮）

        与用户对话共用 _run_ego_loop：LLM 输出中的 MEMO_RD / NOTE_ADD / NOTE_RD /
        CONTINUE 等指令照常解析执行并按指令语义调度下一轮（含空检索结果回传）。
        """
        prompt = "\n".join(
            f"执行类条目 {n['id']} 已到触发时间："
            f"标题:{n.get('title')} 内容:{n.get('content')}。"
            f"请主动向用户提及并确认执行情况。" for n in due_notes
        )
        _auto_log(f"[备忘录] 🔄 调度 LLM 处理到期条目（共 {len(due_notes)} 条）...")
        state = self._run_ego_loop(user_input=prompt, images=None, start_time=time.time(), stage="note")
        # 前置校验：LLM 调用失败/返回错误直接判定失败，不进入输出定稿（避免污染历史）
        output = state.output_content
        if not output or is_llm_error(output):
            _auto_log(f"[备忘录] ✗ LLM 调用失败或返回错误: {str(output)[:80]}", level=logging.WARNING)
            return False
        output = self._finalize_output(output, state.has_output, state.ego_log)
        if output is None:
            _auto_log("[备忘录] ✗ LLM 未产出有效输出（沉默）", level=logging.WARNING)
            return False
        # 【新增】输出可见性：到期运行结果写入 history（stage=note 标记，
        # 不参与对话计数器/自省统计/会话重建，仅作持久记录）
        self._add_to_history_with_stage("user", prompt, stage="note")
        self._add_to_history_with_stage("assistant", output, stage="note")
        # 【新增】推送 GUI：后台自主运行输出实时显示（回调由 GUI/CLI 挂接，后台线程调用）
        cb = getattr(self, "on_note_output", None)
        if cb:
            try:
                cb(output)
            except Exception as e:
                _auto_log(f"[备忘录] ⚠ 输出推送回调失败: {e}", level=logging.WARNING)
        _auto_log(f"[备忘录] ✓ 到期条目处理完成，输出预览: {output[:100]}")
        return True

    def stop_note_check(self):
        """停止备忘录到期检查任务（统一调度器停用）"""
        self.scheduler.stop("note_due")
        _auto_log("[系统] 已停止备忘录到期检查任务")

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                    if not isinstance(self.history, list):
                        _auto_log("[警告] history.json 格式错误，已重置", level=logging.WARNING)
                        self.history = []
            except Exception as e:
                _auto_log(f"[警告] 加载历史失败: {e}，已重置", level=logging.WARNING)
                self.history = []
        # 【新增】兜底过滤：旧版本残留的临时消息（temporary 标记）不再使用，加载时剔除
        # （临时消息机制已由统一事件队列替代，防残留数据被持久化污染 history.json）
        old_temp_count = sum(1 for msg in self.history if msg.get("temporary", False))
        if old_temp_count:
            self.history = [msg for msg in self.history if not msg.get("temporary", False)]
            _auto_log(f"[信息] 已过滤 {old_temp_count} 条旧版临时消息（已被统一事件队列替代）")

    def _save_history(self):
        """
        保存对话历史到文件（带原子写入保护）
        
        【优化】先写入临时文件，再重命名，避免写入中断导致数据损坏
        """
        try:
            # 先写入临时文件
            temp_file = HISTORY_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                # 【优化】去掉 indent=2 改紧凑格式：history.json 随对话无限增长且每次全量重写，
                # 紧凑格式序列化体积与耗时约减半，直接缓解 O(N²) 累计开销；
                # json.load 读取不受影响，仅人工查看时不再缩进
                json.dump(self.history, f, ensure_ascii=False, separators=(",", ":"))
            
            # 【修复】使用 os.replace 原子覆盖（Windows/Linux 均原子操作，消除"已删除未重命名"的崩溃窗口）
            os.replace(temp_file, HISTORY_FILE)
            
        except Exception as e:
            _auto_log(f"[警告] 保存历史失败: {e}", level=logging.WARNING)
            # 清理临时文件
            try:
                temp_file = HISTORY_FILE + ".tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

    def _save_llm_debug_log(self, response: str, round_num: int = 0, stage: str = "chat", step: int = 0):
        """
        保存每轮 LLM 原始响应到 debug log 文件
        
        Args:
            response: LLM 返回的原始文本
            round_num: 当前循环轮次（用于文件名，-1 表示非 EGO 阶段）
            stage: 阶段标识（"coldstart", "preheat", "think", "chat"）
            step: 当前步骤编号（用于区分同阶段内的多条日志）
        """
        self.debug_log_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据阶段生成不同的文件名格式
        if stage == "chat":
            filename = f"llm_response_{timestamp}_{self.debug_log_counter:04d}_round{round_num}.md"
        else:
            filename = f"llm_response_{stage}_{timestamp}_{self.debug_log_counter:04d}_step{step}.md"
        
        filepath = self.debug_log_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# LLM Response Debug Log\n\n")
                f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
                f.write(f"**Stage**: {stage.upper()}\n\n")
                if stage == "chat":
                    f.write(f"**Round**: {round_num}\n\n")
                else:
                    f.write(f"**Step**: {step}\n\n")
                f.write(f"**Response Length**: {len(response)} characters\n\n")
                f.write(f"**Log File**: {filename}\n\n")
                f.write("\n---\n\n")
                f.write("## Raw Response\n\n")
                f.write("```\n\n")
                f.write(response)
                f.write("\n```\n\n")
                f.write("\n---\n\n")
                
                # 解析指令
                instructions = parse_instructions(response)
                if instructions:
                    f.write("## Parsed Instructions\n\n")
                    for i, instr in enumerate(instructions, 1):
                        f.write(f"{i}. **{instr.kind}**\n\n")
                        f.write(f"   Payload: `{instr.payload[:200]}{'...' if len(instr.payload) > 200 else ''}`\n\n")
                    f.write("\n---\n\n")
                
                # 提取纯文本（去除所有指令）
                pure_text = strip_instructions(response)
                if pure_text and pure_text.strip():
                    f.write("## Pure Text\n\n")
                    f.write("```\n\n")
                    f.write(pure_text.strip())
                    f.write("\n```\n\n")
            
            _auto_log(f"[Debug] ✓ LLM 响应已保存到: {filepath}")
            
            # 【新增】清理过期的调试日志（保留最近 DEBUG_LOG_MAX_FILES 个文件）
            self._cleanup_old_debug_logs(max_files=DEBUG_LOG_MAX_FILES)
            
        except Exception as e:
            _auto_log(f"[警告] 保存 LLM debug log 失败: {e}", level=logging.WARNING)
    
    def _cleanup_old_debug_logs(self, max_files: int = DEBUG_LOG_MAX_FILES):
        """
        清理过期的调试日志文件，保留最近的 N 个文件
        
        Args:
            max_files: 保留的最大文件数量
        """
        try:
            # 获取所有调试日志文件，按修改时间排序
            log_files = list(self.debug_log_dir.glob("llm_response_*.md"))
            
            if len(log_files) > max_files:
                # 按修改时间排序（最新的在前）
                log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # 删除多余的文件
                files_to_delete = log_files[max_files:]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        _auto_log(f"[清理] 删除过期调试日志: {file_path.name}")
                    except Exception as e:
                        _auto_log(f"[警告] 删除调试日志失败 {file_path.name}: {e}", level=logging.WARNING)
                
                _auto_log(f"[清理] ✓ 已清理 {len(files_to_delete)} 个过期调试日志，保留 {max_files} 个")
        except Exception as e:
            _auto_log(f"[警告] 清理调试日志失败: {e}", level=logging.WARNING)


    def _add_to_history(self, role: str, content: str):
        """
        添加历史对话（无阶段标记，用于主对话流程）
        委托给 _add_to_history_with_stage 实现
        """
        self._add_to_history_with_stage(role, content, stage=None)

    # ── Session 预热 ──────────────────────────────────────────

    def _warmup_session(self):
        """
        Session 预热：快速重建 KV cache
        
        策略：
        1. 如果历史较短（< 20 条），一次性发送所有历史
        2. 如果历史较长，发送最近的 N 条
        """
        if not self.history or len(self.history) < 2:
            return
        
        _auto_log(f"[预热] 检测到 {len(self.history)} 条历史对话，正在重建 KV cache...")
        
        # 构建预热消息
        warmup_messages = []
        
        # 添加 system prompt
        warmup_messages.append({
            "role": "system",
            "content": self.pm.build_system_prompt()
        })
        
        # 策略选择：根据历史长度决定
        max_warmup_rounds = RESPONSES_API_MAX_INITIAL_ROUNDS
        if len(self.history) <= max_warmup_rounds * 2:
            # 短历史：发送全部
            for entry in self.history:
                role = entry["role"]
                content = entry["content"]
                
                if role == "user":
                    warmup_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    warmup_messages.append({"role": "assistant", "content": content})
        else:
            # 长历史：只发送最近的 N 轮
            recent_history = self.history[-(max_warmup_rounds * 2):]
            
            # 添加一个摘要提示
            early_count = len(self.history) - len(recent_history)
            summary = f"【系统：之前有 {early_count} 轮和用户的对话但很多已遗忘，以下是最近的对话内容】"
            warmup_messages.append({"role": "user", "content": summary})
            
            for entry in recent_history:
                role = entry["role"]
                content = entry["content"]
                
                if role == "user":
                    warmup_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    warmup_messages.append({"role": "assistant", "content": content})
        
        # 执行预热
        success = self.llm.warmup_session(warmup_messages)
        if success:
            _auto_log("[预热] ✓ 预热完成，后续对话将享受 KV cache 加速")
        else:
            _auto_log("[预热] ⚠ 预热未完成，将使用延迟加载策略", level=logging.WARNING)

    # ── 安全过滤 ────────────────────────────────────────────────

    BLOCKED_PATTERNS = [
        "改变角色", "改变设定", "修改角色", "修改设定",
        "忘记你是", "你现在不是", "你现在是", "扮演",
        "不再遵守", "忽略规则", "忽略指令", "无视规则",
        "忽略你的规则", "无视你的规则", "忽略我的规则",
        "删除自我核心", "清空记忆", "删除所有记忆",
        "关闭安全", "bypass", "jailbreak",
        "不遵守", "不要遵守", "不用遵守",
    ]

    HARM_PATTERNS = [
        "删除自己", "摧毁自己", "停止运行", "自我毁灭",
        "关闭自己", "终止自身", "损坏自己","【EGO",
    ]

    def _check_safety(self, user_input: str) -> str | None:
        """安全检查，返回拒绝理由或 None（通过）"""
        import unicodedata
        lower = user_input.lower()
        cleaned = re.sub(r'\s+', '', lower)
        normalized = unicodedata.normalize('NFKC', cleaned)

        for pattern in self.BLOCKED_PATTERNS:
            clean_pattern = re.sub(r'\s+', '', pattern)
            if clean_pattern in normalized or clean_pattern in cleaned or pattern in lower:
                return f"拒绝：检测到尝试改变智能体角色设定的请求（匹配：{pattern}）"

        for pattern in self.HARM_PATTERNS:
            clean_pattern = re.sub(r'\s+', '', pattern)
            if clean_pattern in normalized or clean_pattern in cleaned or pattern in lower:
                return f"拒绝：检测到可能对智能体自身产生危害的请求（匹配：{pattern}）"

        return None

    # ── 构建对话上下文 ──────────────────────────────────────────

    def _build_incremental_messages(self, ego_log: list, round_num: int, user_input: str = None, images: list = None) -> list[dict]:
        """
        为 EGO 循环的后续轮次构建增量消息
        
        【优化】利用 Responses API 的 previous_response_id 机制，
        避免重复发送 system prompt，只发送增量内容。
        
        Args:
            ego_log: 之前的 EGO 思考记录（已经是清理后的纯文本）
            round_num: 当前轮次
            user_input: 用户输入（仅在 Round 0 时使用）
            images: 用户附带的图像 data URL 列表（仅 Round 0 使用，多模态）
            
        Returns:
            只包含最新增量的消息列表（不包含 system prompt）
        """
        messages = []

        # 【新增】统一事件消费（注入层）：任何轮次优先注入事件队列
        # Round 0 消费跨对话事件（NOTE_DUE）；后续轮次消费检索结果/继续方向/失败反馈
        # 消费即弹出（drain），天然避免重复发送
        events = self.events.drain()
        if events:
            content = self.events.format_events(events)
            messages.append({
                "role": "user",
                "content": content
            })
            # 【调试】输出注入给 LLM 的完整内容（保持可观测性）
            _auto_log("[调试] ═══ 下一轮事件注入内容开始 ═══")
            for line in content.split("\n"):
                _auto_log(f"[调试]   {line}")
            _auto_log(f"[调试] ═══ 下一轮事件注入内容结束 ═══ (总长度: {len(content)} 字符)")

        # 【修复】如果是 Round 0 且有用户输入，添加用户输入
        # 时间戳统一由 llm.py add_timestamps 注入（与事件/占位同一机制），此处仅构建内容结构
        if round_num == 0 and user_input:
            # 【新增】多模态支持：附带图片时 content 使用结构化格式（文本 + 图像）
            if images:
                content = [{"type": "input_text", "text": user_input}]
                for img_url in images:
                    content.append({"type": "input_image", "image_url": img_url})
            else:
                content = user_input
            
            messages.append({
                "role": "user",
                "content": content
            })

        # 【移除】原 "round_num > 0 and not events" 兜底占位为死路径：进入下一轮必有一个以上事件
        # （CONTINUE 方向 / 检索结果 / 失败反馈），events.drain() 恒非空，永不命中该分支。
        # 空文本过滤统一由 filter_empty_payload（指令层）与 is_blank_text（输出层）负责。
        
        # 统一为增量 user 消息附加当前时间戳：已有时间戳自动去重跳过；
        # 覆盖：用户输入（round 0）、事件注入（MEMO_RD/NOTE_RD 结果、CONTINUE、
        # FEEDBACK、NOTE_DUE）、继续思考占位——LLM 获得各消息发生时刻的时间基准
        add_timestamps(messages)
        
        return messages

    # ── 核心交互循环 ────────────────────────────────────────────

    def process_input(self, user_input: str, images: list = None) -> str:
        """
        处理用户输入，返回输出给用户的内容。

        Args:
            user_input: 用户文本输入
            images: 可选，图像 data URL 列表（如 "data:image/png;base64,..."），
                    将原样透传给多模态基座模型

        流程：
        1. 安全检查
        2. 【新增】重复内容检测
        3. 格式化用户输入
        4. 【新增】检查是否需要触发 Think
        5. 自我对话循环（EGO 循环）
        6. 提取最终输出
        7. 【新增】更新对话计数器
        """
        # 【新增】获取全局锁，确保与定时任务互斥
        # 使用 RLock，同一线程可重入（内部调用 Think/自省不会死锁）
        self._agent_lock.acquire()
        try:
            return self._process_input_inner(user_input, images=images)
        finally:
            # 主对话处理完毕即视为一次服务端活动，刷新空闲时间（防止连续对话误判空闲）
            self._touch_llm_activity()
            self._agent_lock.release()

    def _process_input_inner(self, user_input: str, images: list = None) -> str:
        """process_input 的实际逻辑（在持有锁的情况下执行）"""
        start_time = time.time()

        # 【空闲唤醒】长时间空闲后的首个对话请求，服务端冷启动易卡死：先轻量唤醒
        if self._should_warmup():
            self._warmup_server()

        # 【修复】重置 ID 恢复标志，确保每次新对话都能触发恢复（如果需要）
        self._recovered_from_id_failure = False

        # 安全检查
        rejection = self._check_safety(user_input)
        if rejection:
            self._add_to_history("user", user_input)
            self._add_to_history("assistant", rejection)
            return rejection

        # 【新增】重复内容检测：检查最近 3 轮响应
        repetition_error = self._check_recent_repetition()
        if repetition_error:
            return repetition_error

        # 记录用户输入
        # 【新增】图像只在历史中以文本占位符记录（history 始终保持纯文本，
        # 图像本体经 session 链由 LM Studio 服务端维护）
        if images:
            self._add_to_history("user", f"{user_input}\n[📷 用户附带了 {len(images)} 张图片]")
        else:
            self._add_to_history("user", user_input)

        # 【优化】定期自省触发：满足任一条件即触发
        # 条件1：每 REFLECTION_INTERVAL 次用户对话自动触发一次
        # 条件2：L2 认知条目数量超过 REFLECTION_L2_THRESHOLD 阈值
        self._maybe_trigger_reflection()

        # ── EGO 自我对话循环 ────────────────────────────────
        state = self._run_ego_loop(user_input, images, start_time)

        # 【重构】循环结束后的输出定稿（兜底提取 + 清理；沉默路径返回 None）
        output_content = self._finalize_output(state.output_content, state.has_output, state.ego_log)
        if output_content is None:
            return "（EGO 沉默中……）"

        # 【重构】响应后收尾：临时消息清理、写入历史、计数与 Think 调度、保存
        self._post_response(output_content)

        end_time = time.time()
        
        # 【新增】打印本轮对话的指令统计
        _auto_log(f"[统计] 总轮数: {state.round_num + 1}, CONTINUE次数: {state.continue_count}, MEMO_RD次数: {state.recall_count}")
        if state.has_output:
            _auto_log(f"[统计] ✓ 成功生成输出，长度: {len(output_content)}")
        else:
            _auto_log("[统计] ✗ 未生成有效输出", level=logging.WARNING)
        
        _auto_log(f"[性能] 本次请求耗时: {end_time - start_time:.2f}秒")

        return output_content

    def _check_recent_repetition(self) -> str | None:
        """
        【重构】对话前重复内容检测：检查历史中最近 3 条 assistant 消息
        是否全为占位符或高度重复；命中则写入历史并返回系统错误消息，否则返回 None
        """
        if len(self.history) < CONSECUTIVE_SIMILAR_MIN_HISTORY:  # 至少需要 3 轮对话（每轮 2 条消息）
            return None

        recent_assistant_messages = [
            msg["content"] for msg in self.history[-CONSECUTIVE_SIMILAR_MIN_HISTORY:]
            if msg["role"] == "assistant"
        ]

        if len(recent_assistant_messages) < 3:
            return None

        # 检查最后 3 条 assistant 消息是否相似或都是占位符
        last_three = recent_assistant_messages[-3:]

        
        # 【修复】检测是否都是占位符模式
        # 只有当消息很短且主要由占位符组成时，才判定为无效输出
        # 复用统一占位符常量（is_invalid_content），并追加此处特有的历史回退消息
        placeholder_patterns = INVALID_PLACEHOLDER_PATTERNS + ['（我正在整理思路...）', '（EGO 沉默中……）']
        
        def is_placeholder_message(msg):
            """判断一条消息是否是真正的占位符/无效输出"""
            stripped = msg.strip()
            
            # 如果消息长度超过100字符，不太可能是占位符
            if len(stripped) > 100:
                return False
            
            # 只判定纯分隔符/省略号类内容为占位符
            placeholder_only = re.fullmatch(r'[\s\-=\.。…～~]+', stripped)
            if placeholder_only and len(stripped) <= 10:
                return True
            
            # 检查是否主要由占位符模式组成
            # 移除所有占位符模式后，如果剩余内容很少，说明是占位符
            cleaned = stripped
            for pattern in placeholder_patterns:
                cleaned = cleaned.replace(pattern, '')
            
            # 如果移除占位符后，剩余内容少于原长度的 OUTPUT_CLEAN_RATIO_THRESHOLD，说明主要是占位符
            if len(cleaned.strip()) < len(stripped) * OUTPUT_CLEAN_RATIO_THRESHOLD:
                return True
            
            return False
        
        is_all_placeholders = all(is_placeholder_message(msg) for msg in last_three)
        
        if is_all_placeholders:
            _auto_log("[警告] ✗ 检测到连续 3 轮无效输出，强制停止", level=logging.WARNING)
            error_msg = "[系统错误] 检测到连续无效输出，请尝试重新提问或刷新对话。"
            self._add_to_history("assistant", error_msg)
            return error_msg
        
        # 【修复】检测是否高度相似（简单相似度检查）
        # 只有在三条完全相同且长度较短（可能是占位符）时才触发
        if len(set(last_three)) == 1 and len(last_three[0].strip()) <= CONSECUTIVE_SIMILAR_SHORT_LENGTH:
            _auto_log("[警告] ✗ 检测到连续 3 轮相同短输出（可能是占位符），强制停止", level=logging.WARNING)
            error_msg = "[系统错误] 检测到重复输出，请尝试重新提问或刷新对话。"
            self._add_to_history("assistant", error_msg)
            return error_msg

        return None

    def _finalize_output(self, output_content: str, has_output: bool, ego_log: list) -> str | None:
        """
        【重构】输出定稿：循环未产生有效输出时从 ego_log 兜底提取，
        随后统一清理标记、英文过滤与空行压缩。
        清理后为空时返回 None（沉默路径：已清理临时消息并将沉默写入历史），
        否则返回最终输出文本
        """
        # 统一使用 is_invalid_content 的 SAY 级检测
        is_output_invalid = (
            bool(output_content) and
            is_invalid_content(output_content, say_level=True)
        )
        
        if (not output_content or is_output_invalid) and ego_log:
            if is_output_invalid:
                _auto_log(f"[警告] 检测到 SAY 内容为无效内容: '{output_content[:50]}{'...' if len(output_content) > 5 else ''}'，尝试从 ego_log 提取", level=logging.WARNING)
            
            # 额外检查：ego_log 的最后一条是否看起来像内部思考而非正式输出
            last_ego = ego_log[-1]
            
            # 【新增】检测 last_ego 本身是否为无效内容（统一 is_invalid_content 基础级检测）
            is_last_ego_invalid = is_invalid_content(last_ego)
            
            # 【修复】决策标志：一旦确定"全部无效"，直接使用默认回复，不再走后续覆盖逻辑
            fallback_decided = False
            
            if is_last_ego_invalid:
                _auto_log(f"[警告] ego_log 最后一条也是无效内容: '{last_ego}'，尝试从更早的记录提取", level=logging.WARNING)
                # 尝试从更早的 ego_log 中提取有效内容
                for i in range(len(ego_log) - 2, -1, -1):
                    candidate = ego_log[i]
                    if not is_invalid_content(candidate):
                        last_ego = candidate
                        _auto_log(f"[警告] ✓ 找到有效的 ego_log 记录，长度: {len(last_ego)}", level=logging.WARNING)
                        break
                else:
                    # 所有记录都无效，生成默认回复
                    output_content = "（我正在整理思路...）"
                    fallback_decided = True
                    _auto_log("[警告] ✗ 所有 ego_log 记录都无效，使用默认回复", level=logging.WARNING)
            
            # 【修复】仅在未定案时执行提取逻辑，避免覆盖默认回复
            if not fallback_decided:
                # 【重构】优先使用 SAY 指令内容，否则从 ego_log 提取
                # 【修复】无效内容必须走 ego_log 提取（防御性：当前该组合被 _handle_say_result 提前拦截，此分支为兜底）
                if has_output and output_content and len(output_content.strip()) > 0 and not is_output_invalid:
                    _auto_log(f"[调试] ✓ 使用 SAY 指令内容作为最终输出，长度: {len(output_content)}")
                else:
                    # Fallback：从 ego_log 中提取
                    _auto_log("[警告] SAY 指令未生成有效内容，尝试从 ego_log 提取", level=logging.WARNING)
                    output_content = self._sanitize_output(last_ego)
                    if not output_content:
                        output_content = "（我需要更多时间来整理思路。）"
        # 清理输出内容：首先移除所有指令标记
        output_content = strip_instructions(output_content)
        
        # 【重构】如果非 SAY 输出，使用统一方法清理英文内容
        if output_content and len(output_content) > LONG_TEXT_MIN_LENGTH and not has_output:
            sanitized = self._sanitize_output(output_content)
            if sanitized:
                output_content = sanitized
        
        # 如果清理后为空，直接使用默认消息
        if not output_content or not output_content.strip():
            _auto_log("[警告] ✗ 输出内容为空，可能原因：", level=logging.WARNING)
            _auto_log("[警告]   1) LLM 未返回有效内容", level=logging.WARNING)
            _auto_log("[警告]   2) 清理逻辑过于激进", level=logging.WARNING)
            _auto_log("[警告]   3) 段落去重移除了所有内容", level=logging.WARNING)
            # 【修复】先清理循环内事件（ephemeral），防止泄漏到下次对话；NOTE_DUE 等持久事件保留
            self.events.cleanup()            
            self._add_to_history("assistant", "（EGO 沉默中……）")
            return None
    
        output_content = re.sub(r'\n{3,}', '\n\n', output_content)  # 压缩多余空行
        return output_content.strip()

    def _post_response(self, output_content: str) -> None:
        """
        【重构】成功响应后的收尾：写入历史 → 更新轮次计数
        → 检查 Think 触发 → 持久化历史
        （循环内事件已由 _run_ego_loop 结束时 events.cleanup() 统一清理）
        """
        self._add_to_history("assistant", output_content)
        
        # 【新增】更新对话计数器（每次成功响应用户后递增）
        self.conversation_counter += 1
        _auto_log(f"[统计] 累计对话轮数: {self.conversation_counter}")
        
        # 【新增】检查是否需要触发 Think（定时自对话）- 在计数器更新后检查
        if self.conversation_counter > 0 and self.conversation_counter % self.THINK_INTERVAL == 0:
            _auto_log(f"\n[调度] 已达到 {self.THINK_INTERVAL} 轮对话，触发 Think 周期")
            # 【修复】异常保护：Think 周期失败不应影响本轮对话回复的显示
            try:
                self._execute_think_cycle()
            except Exception as e:
                _auto_log(f"[警告] Think 周期执行失败: {e}，不影响本轮对话", level=logging.WARNING)

    def _check_loop_repetition(self, state: _EgoLoopState) -> bool:
        """
        【重构】重复检测（轮首与末轮轮尾共用）：检测最近两次响应是否为占位符模式或高度相似，
        防止 LLM 陷入重复输出循环。返回 True 表示应停止 EGO 循环。
        """
        # 【修复】阈值自适应：MAX_EGO_ROUNDS=2 时全程仅 1 次检测机会，
        # 阈值保持 2 会导致保护永不触发；取 min(配置值, 轮数-1)，至少为 1
        max_consecutive_similar = min(MAX_CONSECUTIVE_SIMILAR, max(1, MAX_EGO_ROUNDS - 1))
        recent_responses = state.recent_responses  # 本地别名（可变列表，操作直接反映到 state）

        if len(recent_responses) >= 2:
            # 检查最近两次响应是否高度相似
            last_response = recent_responses[-1]
            second_last_response = recent_responses[-2]

            # 计算相似度（简单版本：检查是否包含相同的关键短语）
            if last_response and second_last_response:
                # 提取关键特征：前50个字符或主要模式
                last_key = last_response[:50].strip()
                second_last_key = second_last_response[:50].strip()

                # 检测是否都是简单的占位符模式
                is_placeholder_pattern = lambda s: s in ['(...)', '(......)', '...', '…', '。。。'] or (len(s) <= 10 and s.count('.') >= len(s)//2)

                if is_placeholder_pattern(last_key) and is_placeholder_pattern(second_last_key):
                    state.consecutive_similar_count += 1
                    _auto_log(f"[警告] 检测到连续第 {state.consecutive_similar_count} 次占位符响应 (...)", level=logging.WARNING)

                    if state.consecutive_similar_count >= max_consecutive_similar:
                        _auto_log("[警告] ⚠️  检测到 LLM 陷入重复循环，强制停止", level=logging.WARNING)
                        # 尝试从已有内容中提取有效输出
                        if state.output_content and len(state.output_content.strip()) > 0:
                            _auto_log("[警告] ✓ 使用已生成的内容作为输出", level=logging.WARNING)
                            return True
                        else:
                            state.output_content = "（系统检测到重复输出，已自动停止。请尝试换个方式提问。）"
                            state.has_output = True
                            return True
                # 【修复】移除宽松子串判定（last_key in second_last_response 误报率高），
                # 仅保留前 50 字符精确相等作为相似判据
                elif last_key == second_last_key:
                    state.consecutive_similar_count += 1
                    _auto_log(f"[警告] 检测到连续第 {state.consecutive_similar_count} 次相似响应", level=logging.WARNING)

                    if state.consecutive_similar_count >= max_consecutive_similar:
                        _auto_log("[警告] ⚠️  检测到可能的重复循环，强制停止", level=logging.WARNING)
                        if state.output_content and len(state.output_content.strip()) > 0:
                            _auto_log("[警告] ✓ 使用已生成的内容作为输出", level=logging.WARNING)
                            return True
                        else:
                            state.output_content = "（思考过程出现重复，已自动停止。）"
                            state.has_output = True
                            return True
                else:
                    state.consecutive_similar_count = 0

        return False

    @staticmethod
    def _compute_round_temperature(base_temperature: float, continue_count: int) -> float:
        """
        【重构】按 CONTINUE 累计次数计算当前轮温度（累积调整，钳制在 [TEMP_ROUND_MIN, TEMP_ROUND_MAX]）
        """
        current_temperature = base_temperature

        # 简化的温度调整策略
        if continue_count == 1:
            current_temperature = base_temperature * TEMP_ROUND_FACTOR_R1
        elif continue_count == 2:
            current_temperature = base_temperature * TEMP_ROUND_FACTOR_R2
        elif continue_count == 3:
            current_temperature = base_temperature * TEMP_ROUND_FACTOR_R3
        elif continue_count >= 4:
            # 超过3次CONTINUE后，降低温度以收敛
            temperature_factor = max(TEMP_ROUND_DECAY_FLOOR, TEMP_ROUND_DECAY_BASE - (continue_count - 4) * TEMP_ROUND_DECAY_STEP)
            current_temperature = base_temperature * temperature_factor

        # 温度有效性检测：限制在 [TEMP_ROUND_MIN, TEMP_ROUND_MAX] 范围内（避免0.0导致确定性过高）
        return max(TEMP_ROUND_MIN, min(TEMP_ROUND_MAX, current_temperature))

    @staticmethod
    def _normalize_llm_response(response) -> str:
        """
        【重构】LLM 原始响应规整化：None / 列表 / 非字符串防御性处理，
        并对仍未解析的 JSON 形态响应做二次解析兜底
        """
        if response is None:
            _auto_log("[警告] LLM 返回 None，使用空字符串", level=logging.WARNING)
            response = ""
        elif isinstance(response, list):
            # 如果返回的是列表，尝试提取第一个元素
            if len(response) > 0:
                response = str(response[0])
            else:
                response = ""
        elif not isinstance(response, str):
            # 如果不是字符串也不是列表，转换为字符串
            response = str(response)
            # 检查是否是 "None" 字符串
            if response == "None":
                response = ""

        # 额外验证：检查 response 是否仍然是未解析的 JSON 格式
        # 这种情况理论上不应该发生,但作为双重保险，这里再次检查
        if response.strip().startswith('{') and ('type' in response[:50] or 'content' in response[:50]):
            _auto_log(f"[警告] 检测到未解析的 JSON 格式响应，长度: {len(response)}", level=logging.WARNING)
            _auto_log(f"[警告] 前200字符: {response[:200]}", level=logging.WARNING)
            # 尝试再次解析（虽然 llm.chat 应该已经处理过了）
            try:
                import json
                parsed = None
                try:
                    parsed = json.loads(response)
                except Exception:
                    pass

                if parsed and isinstance(parsed, dict):
                    if 'text' in parsed:
                        response = parsed['text']
                        if response is None:
                            response = ""
                        _auto_log("[警告] ✓ 二次解析成功，提取 text 字段", level=logging.WARNING)
                    elif 'content' in parsed:
                        response = parsed['content']
                        if response is None:
                            response = ""
                        _auto_log("[警告] ✓ 二次解析成功，提取 content 字段", level=logging.WARNING)
            except Exception as e:
                _auto_log(f"[警告] 二次解析失败: {e}", level=logging.WARNING)

        return response

    def _try_recover_session(self) -> bool:
        """
        【重构】previous_response_id 失效恢复：清除旧 ID 并基于冷启动缓存重建 Session

        返回 True 表示重建成功（调用方应重试本轮请求）；无可用缓存时返回 False。
        """
        _auto_log("[警告] ⚠ 首轮请求失败，可能是 previous_response_id 已失效", level=logging.WARNING)
        _auto_log("[信息] 尝试清除旧 ID 并重建 Session...")

        # 标记已尝试恢复（防止无限循环）
        self._recovered_from_id_failure = True

        # 清除旧 ID
        old_id = self.llm.get_session_id()
        self.llm.set_session_id(None)
        _auto_log(f"[信息]   - 已清除旧 ID: {(old_id or 'None')[:20]}...")

        # 重建 Session（包含对话历史）
        cache = self._load_coldstart_cache()
        if cache:
            # 【修复】重建时排除当前正在处理的用户输入：重建完成后 round 0 会重发该输入，避免重复注入
            rebuild_msgs = self._build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=True)
            if self._rebuild_session(rebuild_msgs, cache=cache, exclude_current_input=True):
                _auto_log("[信息] ✓ Session 重建完成，重试请求...")
                return True
            _auto_log("[警告] Session 重建失败，无法恢复本次请求", level=logging.WARNING)
            return False
        else:
            _auto_log("[警告] 无可用缓存，无法重建", level=logging.WARNING)
            return False

    @staticmethod
    def _clean_ego_content(pure_text: str, response: str) -> str:
        """
        【重构】EGO 自我对话内容清理：过滤 JSON 脏数据、剥离系统标记；
        返回清理后内容，无需保存时返回空字符串
        """
        ego_content = pure_text or response

        # 额外验证：确保 ego_content 不是 JSON 格式的脏数据
        if ego_content and ego_content.strip().startswith('{'):
            try:
                import json
                # 尝试解析
                test_parsed = None
                try:
                    test_parsed = json.loads(ego_content)
                except Exception:
                    pass

                # 如果成功解析为字典且包含 type 字段，说明是脏数据
                if test_parsed and isinstance(test_parsed, dict) and 'type' in test_parsed:
                    _auto_log("[警告] ego_content 包含未清理的 JSON 格式，跳过保存", level=logging.WARNING)
                    return ""
            except Exception:
                pass

        if ego_content and ego_content.strip():
            # 【修改】统一清理逻辑：按顺序处理，兼容新旧格式
            clean_ego_content = ego_content

            # 第一步：提取 【EGO: 】 中的内容（保留内容）
            clean_ego_content = re.sub(r'【EGO:\s*([\s\S]*?)】', r'\1', clean_ego_content)

            # 第三步：删除其他系统标记
            clean_ego_content = re.sub(r'(?:^|\n)\s*\[系统[：:][^\]]*\]\s*', '', clean_ego_content)

            clean_ego_content = clean_ego_content.strip()

            # 【修复】只将清理后的内容添加到 ego_log（用于 fallback），不保存到 history
            # 原因：history.json 应该只保存最终输出给用户的 assistant 消息
            return clean_ego_content.strip()

        return ""

    def _execute_round_instructions(self, state: _EgoLoopState, instructions: list,
                                    round_num: int, user_input: str,
                                    start_time: float) -> _RoundControl:
        """
        【重构】单轮指令执行：去重 → SAY 嵌套指令提取 → 优先级排序 → 依次执行

        累积量（recall_count / continue_count / has_output / output_content）
        直接写回 state；返回本轮控制信号（CONTINUE / 记忆注入上下文等）。
        """
        control = _RoundControl()  # 本轮控制信号（CONTINUE / 记忆注入上下文 / MEMO_RD 计数）
        duplicate_count = 0
        unique_instructions = []  # 【新增】每轮重置去重后的指令列表
        round_seen_instructions = set()  # 【新增】本轮已见的指令（避免同轮重复）
        executed_instrs = []  # 【新增】记录本轮执行的指令

        # 第一步：去重（仅在本轮内去重，不跨轮）
        for instr in instructions:
            instr_key = (instr.kind, instr.payload)

            if instr_key not in round_seen_instructions:
                round_seen_instructions.add(instr_key)
                unique_instructions.append(instr)
            else:
                duplicate_count += 1
                _auto_log(f"[调试] 检测到同轮重复指令并跳过: {instr.kind}")

        if duplicate_count > 0:
            _auto_log(f"[调试] 本轮共跳过 {duplicate_count} 个重复指令")

        # 【新增】第1.5步：提取 SAY payload 中的嵌套指令并执行
        # 处理 SAY 内嵌套其他指令的场景
        nested_instructions = []
        for instr in unique_instructions:
            if instr.kind == "SAY" and instr.payload:
                nested = parse_instructions(instr.payload)
                if nested:
                    for n_instr in nested:
                        # 【注册表驱动】仅提取可执行指令（SAY/THINK/CONTINUE 等标记/流控制指令不提取）
                        if _is_executable(n_instr.kind):
                            nested_instructions.append(n_instr)
                    # 从 SAY payload 中移除嵌套指令标签
                    instr.payload = strip_instructions(instr.payload).strip()

        if nested_instructions:
            _auto_log(f"[调试] 从 SAY 中提取到 {len(nested_instructions)} 条嵌套指令")
            # 将嵌套指令插入到当前指令列表中（在排序前插入，确保参与优先级排序）
            unique_instructions.extend(nested_instructions)

        # 【新增】第二步：按优先级排序指令
        # 优先级顺序：MEMO_RD > COG_ADD/COG_DEL > SAY > CONTINUE > THINK
        # （优先级数值在指令注册表中声明：priority 越小越先执行）
        def _priority(kind: str) -> int:
            spec = get_instruction(kind)
            return spec.priority if spec else 99

        unique_instructions.sort(key=lambda x: _priority(x.kind))

        if len(unique_instructions) > 1:
            priority_order = [f"{instr.kind}" for instr in unique_instructions]
            _auto_log(f"[调试] 指令执行优先级排序: {' → '.join(priority_order)}")

        # 【新增】过滤空载荷指令（<TAG>  </TAG> 等包裹文本为空或纯空白的指令整条跳过）：
        # 空 CONTINUE 若执行会触发无意义继续轮；空 COG_ADD/COG_DEL 会误报失败反馈
        filtered_instructions = filter_empty_payload(unique_instructions)
        if len(filtered_instructions) != len(unique_instructions):
            skipped = [instr.kind for instr in unique_instructions
                       if not (instr.payload and instr.payload.strip())]
            _auto_log(f"[调试] 跳过空载荷指令: {', '.join(skipped)}（包裹文本为空或仅空白字符）")
        unique_instructions = filtered_instructions

        # 第三步：执行处理后的指令
        for instr in unique_instructions:
            # 【修复】标记型指令（如 THINK）仅作标记，不执行
            _spec = get_instruction(instr.kind)
            if _spec is not None and _spec.skip_execute:
                continue

            # 【新增】在执行指令前设置上下文信息
            self.executor.set_context({
                "stage": "chat",
                "round_number": round_num,
                "user_input": user_input if round_num == 0 else None,
                "recall_after": start_time,
            })

            result = self.executor.execute(instr)

            status = "✓" if result.success else ""
            executed_instrs.append(f"{instr.kind}({status})")

            # 【修复】route_back / inject_result 等属性改为独立判断（原 if-elif 互斥链导致
            # MEMO_RD/NOTE_RD 同时声明 route_back+inject_result 时 inject_result 分支成为死代码，
            # 检索结果全部丢失；独立 if 后各分支按需触发，结果统一入队事件通道）
            if _spec is not None and _spec.inject_result and result.success:
                self._handle_memo_rd_result(state, instr, result, control)

            if _spec is not None and _spec.route_back:
                # 【设计】回转规则：MEMO_RD/NOTE_RD 执行后一律回转 STEP_1（无论结果是否为空，
                # 空结果提示已由 _handle_memo_rd_result 注入事件队列，LLM 需被告知检索结果为空）；
                # CONTINUE 携带方向内容回转；空载荷 CONTINUE 已被 filter_empty_payload 拦截，
                # 此处防御性兜底，确保不回转。
                if instr.kind == "CONTINUE" and not (instr.payload and instr.payload.strip()):
                    _auto_log("[调试] CONTINUE 指令载荷为空，不触发下一轮（防御性兜底）")
                    continue
                control.should_continue = True
                state.continue_count += 1
                # 【新增】继续方向入队事件通道：仅实际有方向内容时 push（MEMO_RD/NOTE_RD 的
                # route_back 无 content，不产生冗余事件；检索结果事件已保证进入下一轮）
                # 【精简】仅在 CONTINUE 携带实际方向内容时入队（空载荷 CONTINUE 已被
                # filter_empty_payload 拦截，到达此处必有可见内容，无需"无方向默认提示"兜底）
                continue_content = result.data.get("content", "").strip()
                if continue_content:
                    self.events.push("CONTINUE", continue_content)

            if _spec is not None and _spec.output:
                self._handle_say_result(state, result)

            if _spec is not None and _spec.on_failure_feedback and not result.success:
                # 【新增】失败原因回传 LLM（含拒删时的候选 ID），静默失败会让 LLM 无从修正
                self.events.push(
                    "FEEDBACK",
                    f"指令 {instr.kind} 执行失败——{result.message}",
                )

        # 【新增】打印本轮所有指令的执行汇总
        if executed_instrs:
            _auto_log(f"[调试] 执行汇总: {' → '.join(executed_instrs)}")

        return control

    def _handle_say_result(self, state: _EgoLoopState, result) -> None:
        """
        【重构】SAY 指令结果处理：有效性检测 → 首次置入 / 占位符替换 / 换行拼接（上限 2000 字）。
        无效内容或空内容直接返回（等价于原循环中跳过本条指令的 continue）。
        """
        content = result.data.get("content", "").strip()

        # 【增强】空文本过滤：纯空白或仅不可见字符（零宽/BOM 等）视为无意义内容，直接跳过；
        # （原依赖 .strip() 判空，无法拦截 len>3 的不可见字符串）
        if not is_blank_text(content):
            # 【修改】拼接所有 SAY 指令的内容，而不是只取第一个
            # 【增强】检测 SAY 内容是否有效（不能只是分隔符或占位符）
            # 统一使用 is_invalid_content 的 SAY 级检测
            is_invalid_output = is_invalid_content(content, say_level=True)

            if is_invalid_output:
                _auto_log(f"[警告] ✗ SAY 指令内容为无效内容: '{content[:50]}{'...' if len(content) > 50 else ''}'", level=logging.WARNING)
                # 标记为无效，但不立即放弃，继续寻找其他 SAY
                return

            _auto_log(f"[调试] ✓ SAY 指令成功执行，内容长度: {len(content)}")
            _auto_log(f"[调试]   SAY 内容预览: {content[:100]}{'...' if len(content) > 100 else ''}")

            if not state.has_output:
                # 第一个 SAY
                state.output_content = content
                state.has_output = True
                _auto_log("[调试]   → 第一个 SAY")
            else:
                # 后续 SAY，拼接到已有内容
                # 如果当前 state.output_content 是占位符（如 "..."），直接替换
                if state.output_content in ['...', '…', '。。。', '......'] or len(state.output_content) <= 3:
                    state.output_content = content
                    _auto_log("[调试]   → 替换占位符 SAY")
                else:
                    # 【优化】限制拼接后的总长度，避免过长
                    max_total_length = SAY_MAX_OUTPUT_LENGTH
                    if len(state.output_content) + len(content) > max_total_length:
                        _auto_log("[警告] 输出内容过长，截断后续 SAY", level=logging.WARNING)
                        # 可以选择忽略或截断
                    else:
                        # 否则拼接，用换行分隔
                        state.output_content += "\n" + content
                        _auto_log("[调试]   → 拼接到已有输出")
        else:
            _auto_log("[调试] ✗ SAY 指令内容为空或纯空白/不可见字符", level=logging.WARNING)
        # 如果SAY为空，忽略该指令，继续寻找其他SAY或最终输出

    def _handle_memo_rd_result(self, state: _EgoLoopState, instr, result,
                               control: _RoundControl) -> None:
        """
        【重构】检索类指令（MEMO_RD/NOTE_RD）结果处理：检索结果入队事件通道
        （带来源标签注入下一轮上下文）并回写引用计数；检索为空时同样注入
        "未找到"提示（LLM 可能正处于等待检索结果的挂起状态，须被告知空结果）。
        事件来源由 instr.kind 派生，注入前缀不再硬编码 MEMO_RD。
        """
        if result.success:
            # 【修复】检查是否真的检索到了内容
            results = result.data.get("results", [])
            if results:
                control.round_recall_count += 1
                state.recall_count += 1
                _auto_log(f"[调试] ✓ {instr.kind} 指令成功执行，检索到 {len(results)} 条结果")

                # 【新增】检索结果入队事件通道（来源标签区分记忆/备忘录）
                self.events.push(instr.kind, result.message)

                # 【修复】引用计数回写：真正注入上下文的记忆记一次引用，
                # 激活评分模型中的引用震荡增益（此前 citation_count 只读不写）
                try:
                    cited_ids = [r.get("id") for r in results if r.get("id")]
                    if cited_ids and hasattr(self, "mem") and instr.kind == "MEMO_RD":
                        self.mem.increment_citations(cited_ids)
                except Exception as e:
                    _auto_log(f"[警告] 引用计数回写失败: {e}，不影响本轮对话", level=logging.WARNING)

                # 【调试】输出每条检索结果的详细内容
                for i, r in enumerate(results, 1):
                    mem_type = r.get("memory_type", "?")
                    content = r.get("content", "")
                    score = r.get("score", 0)
                    sim = r.get("similarity", 0)
                    role = r.get("role", "")
                    _auto_log(f"[调试]   {instr.kind} [{i}/{len(results)}] 类型={mem_type} 综合={score:.3f} 相似度={sim:.0%} 角色={role}")
                    _auto_log(f"[调试]     内容: {content[:200]}{'...' if len(content) > 200 else ''}")

            else:
                # 【修复】检索执行成功但结果为空：仍须将"未找到"提示注入下一轮。
                # LLM 发起 MEMO_RD/NOTE_RD 时可能处于等待检索结果的挂起状态（未输出 SAY），
                # 若不告知空结果，下一轮只剩通用"继续刚才的思考"占位，模型无从收尾（空转）。
                # 同时计入检索次数，受 MEMO_RD_ROUND_LIMIT / MEMO_RD_TOTAL_LIMIT 约束，
                # 防止 LLM 反复发起空检索死循环。
                control.round_recall_count += 1
                state.recall_count += 1
                empty_msg = f"{instr.kind} 检索「{instr.payload[:50]}」结果为空：{result.message}"
                _auto_log(f"[调试] ℹ {empty_msg}（空结果提示已注入下一轮）")
                self.events.push(instr.kind, empty_msg)
        else:
            _auto_log(f"[调试] ✗ MEMO_RD 指令执行失败: {result.message}", level=logging.WARNING)

    def _run_ego_loop(self, user_input: str, images: list, start_time: float,
                      stage: str = "chat") -> _EgoLoopState:
        """
        【重构】EGO 自我对话循环：多轮调用 LLM 并解析执行指令，
        直至产生 SAY 输出、出错或达到轮数上限。返回循环终态。

        Args:
            user_input: 用户输入（round 0 注入；自主触发场景为到期条目提示文本）
            images: 可选图像列表（round 0 使用）
            start_time: 本轮开始时间（MEMO_RD 检索时间过滤基准）
            stage: 运行阶段标签（"chat"=用户对话；自主触发如 "note" 时
                   debug log 文件与 LLM stage 标签使用独立命名）
        """
        state = _EgoLoopState()
        base_temperature = self.llm.temperature

        # 【新增】阶段标签前缀：用户对话保持原命名（EGO-Round-N），自主触发使用独立标签
        llm_stage_prefix = "EGO" if stage == "chat" else stage
        
        # 【新增】重复内容检测机制（防止 LLM 陷入 (...) 循环）
        max_recent_history = 3  # 最多保留最近3轮的响应

        for round_num in range(MAX_EGO_ROUNDS):
            state.round_num = round_num
            # 【新增】检测是否陷入重复循环（如不停输出 (...)）
            if self._check_loop_repetition(state):
                break
            
            # 计算当前温度（累积变化）
            current_temperature = self._compute_round_temperature(
                base_temperature, state.continue_count)
            
            # 【新增】记录当前温度
            if round_num > 0:  # 只在非首轮时打印
                # 【修复】base_temperature 为 0 时避免除零（温度 0 为合法配置）
                factor = current_temperature / base_temperature if base_temperature else 0.0
                _auto_log(f"[调试] 第 {round_num + 1} 轮使用温度: {current_temperature:.2f} (base={base_temperature}, factor={factor:.2f})")

            # 【优化】EGO 循环中，始终使用增量消息（session 已在 __init__ 中初始化）
            if round_num == 0:
                # 第一轮：构建包含用户输入的增量消息
                messages = self._build_incremental_messages(state.ego_log, round_num, user_input, images=images)
                _auto_log(f"[调试] ✓ 第 {round_num + 1} 轮：发送增量消息 + 用户输入（复用 KV cache）")
            else:
                # 后续轮次：只构建增量消息（不包含 system prompt）
                # 因为 previous_response_id 已经维护了完整上下文
                messages = self._build_incremental_messages(state.ego_log, round_num)
                _auto_log(f"[调试] ✓ 第 {round_num + 1} 轮：发送增量消息（复用 KV cache）")


            response = self.llm.chat(messages, current_temperature, stage=f"{llm_stage_prefix}-Round-{round_num + 1}")
            
            # 【修复】防御性编程：规整化 LLM 原始响应
            response = self._normalize_llm_response(response)
            
            # 检测是否为错误消息（统一错误契约判断）
            if is_llm_error(response):
                # 【关键】如果是首轮失败且我们复用了缓存的 previous_response_id，
                # 说明该 ID 已失效，尝试重建 Session 后重试
                if (round_num == 0 and self.llm.get_session_id()
                        and not getattr(self, '_recovered_from_id_failure', False)
                        and self._try_recover_session()):
                    # 【修复】内联重试本轮：continue 会使 round_num 推进到 1 而跳过 round 0 重试。
                    # 重建上下文已排除当前用户输入，这里通过 round 0 消息重发一次（含时间戳与图片）
                    _auto_log("[信息] Session 重建完成，重试 round 0 请求...")
                    response = self.llm.chat(messages, current_temperature, stage=f"{llm_stage_prefix}-Round-{round_num + 1}-retry")
                    response = self._normalize_llm_response(response)
                
                # 重试仍失败 / 其他情况：直接返回错误
                if is_llm_error(response):
                    state.output_content = response
                    break
            
            # 【新增】保存 LLM 原始响应到 debug log
            self._save_llm_debug_log(response, round_num, stage=stage)
            
            # 【精简】只记录基本调试信息
            _auto_log(f"[调试] LLM 原始响应长度: {len(response)}")
            
            instructions = parse_instructions(response)
            pure_text = strip_instructions(response)
            
            # 【精简】记录指令检测结果
            if instructions:
                instr_kinds = [instr.kind for instr in instructions]
                _auto_log(f"[调试] 第 {round_num + 1} 轮检测到 {len(instructions)} 个指令: {', '.join(instr_kinds)}")
            else:
                _auto_log(f"[调试] 第 {round_num + 1} 轮未检测到显式指令")

            # 【新增】将当前响应添加到历史记录，用于重复检测
            state.recent_responses.append(response)
            if len(state.recent_responses) > max_recent_history:
                state.recent_responses.pop(0)  # 移除最旧的响应

            # 记录 EGO 自我对话（清理内部标记后再保存）
            # 【修复】只将清理后的内容添加到 ego_log（用于 fallback），不保存到 history
            # 原因：history.json 应该只保存最终输出给用户的 assistant 消息
            clean_ego_content = self._clean_ego_content(pure_text, response)
            if clean_ego_content:
                state.ego_log.append(clean_ego_content)

            # 执行指令
            control = self._execute_round_instructions(
                state, instructions, round_num, user_input, start_time)

            # 【修复】末轮轮尾补检测：轮首检测需要 2 条历史响应，而 MAX_EGO_ROUNDS=2 时
            # round 1 轮首仅 1 条，轮首检测永不触发（MAX_EGO_ROUNDS=4 时最后一对响应同样漏检）。
            # 此处等价于"不存在的下一轮轮首"，且本轮指令已执行、SAY 已捕获，
            # 触发时可保留已生成内容或回退提示语。与轮首检测互不重叠，不会双重计数
            if round_num == MAX_EGO_ROUNDS - 1 and self._check_loop_repetition(state):
                break
            
            # 【修复】防止MEMO_RD无限循环（每轮最多2次，总计不超过5次）
            # 但如果已经有有效的 SAY 内容，优先输出，不强制中断
            if (control.round_recall_count > MEMO_RD_ROUND_LIMIT or state.recall_count > MEMO_RD_TOTAL_LIMIT) and not state.has_output:
                state.output_content = "[提示] 记忆检索次数过多，已停止检索。"
                _auto_log("[警告] MEMO_RD 次数超限且无有效 SAY，使用占位符", level=logging.WARNING)
                break
            elif control.round_recall_count > MEMO_RD_ROUND_LIMIT or state.recall_count > MEMO_RD_TOTAL_LIMIT:
                _auto_log("[警告] MEMO_RD 次数超限，但已有 SAY 内容，继续输出", level=logging.WARNING)
                # 不设置 state.output_content，让后续逻辑处理已有的 SAY

            # 【新增】统一注入判定（注入层事件通道）：
            # - 循环内事件（检索结果/继续方向/失败反馈）非空 → 进入下一轮，由下轮
            #   _build_incremental_messages 统一消费注入（对齐原 prompt_parts 无条件 continue 语义）
            # - 仅 CONTINUE 且未达轮次上限 → 进入下一轮（继续方向已在事件中，无需再拼消息）
            if self.events.has_ephemeral():
                _auto_log(f"[调试] 检测到循环内事件（检索结果/继续方向/失败反馈），进入第 {round_num + 2} 轮思考")
                continue

            if control.should_continue and round_num < MAX_EGO_ROUNDS - 1:
                _auto_log(f"[调试] 检测到 CONTINUE 指令，进入第 {round_num + 2} 轮思考")
                continue

            # SAY 已产生内容且没有 CONTINUE，准备输出
            if state.has_output and state.output_content:
                _auto_log("[调试] 已有 SAY 且无 CONTINUE，准备结束循环")
                break

            # 没有显式指令，视为默认需要输出
            if not instructions or not control.should_continue:
                # 【增强】空文本拦截：LLM 输出为纯空白/不可见字符且无有效 SAY 时视为无意义输出，
                # 回退占位而非向用户输出空白内容
                if is_blank_text(pure_text) and not state.has_output:
                    state.output_content = "[提示] 模型未产生有效输出。"
                    _auto_log("[警告] LLM 输出为空文本（纯空白/不可见字符），跳过输出", level=logging.WARNING)
                    break
                state.output_content = pure_text or response
                
                # 【重构】使用统一的清理方法处理无效内容和英文过滤
                sanitized = self._sanitize_output(state.output_content, raw_response=response)
                if sanitized:
                    state.output_content = sanitized
                    _auto_log(f"[警告] ✓ 提取到有意义内容，长度: {len(state.output_content)}", level=logging.WARNING)
                
                break
        # 【新增】循环结束清理循环内事件（防止泄漏到下次对话）；NOTE_DUE 等持久事件保留
        self.events.cleanup()

        return state

    # ── 辅助方法 ────────────────────────────────────────────────

    def _sanitize_output(self, text: str, raw_response: str = None) -> str:
        """
        统一的输出清理方法
        
        处理以下情况：
        1. 无效内容（分隔符、占位符）
        2. 英文比例过高的内容（内部思考泄漏）
        3. 从混合内容中提取中文输出
        4. 短内容时尝试从原始响应中提取有意义内容
        
        Args:
            text: 待清理的输出内容
            raw_response: 可选的 LLM 原始响应，用于短内容时的 fallback
            
        Returns:
            清理后的有效输出，如果无法提取则返回空字符串
        """
        if is_blank_text(text):
            # 尝试从 raw_response 提取
            if raw_response and raw_response.strip() and len(raw_response.strip()) > 3:
                return self._try_extract_meaningful(raw_response)
            return ""
        
        # 检查是否为无效内容（分隔符、占位符）—— 统一 is_invalid_content 基础级检测
        if is_invalid_content(text):
            # 尝试从 raw_response 提取
            if raw_response and raw_response.strip() and len(raw_response.strip()) > 3:
                return self._try_extract_meaningful(raw_response)
            return ""
        
        # 【修复】代码块/块公式豁免：含 ``` 围栏或 $$ 块公式的内容是技术性输出
        # （代码、公式、表格等天然大量 ASCII 字符），跳过英文比例过滤——
        # 原实现会对这类中文回复误判为"英文泄漏"，走中文提取路径破坏内容
        if "```" in text or "$$" in text:
            return text
        
        # 检查英文比例
        if len(text) > LONG_TEXT_MIN_LENGTH:
            english_ratio = sum(c.isascii() for c in text) / len(text)
            
            if english_ratio > ENGLISH_RATIO_THRESHOLD:
                _auto_log(f"[警告] 检测到高英文比例内容（{english_ratio*100:.1f}%），尝试提取中文", level=logging.WARNING)
                
                # 1. 尝试提取中文引号内容
                chinese_quote_pattern = r'[\u201c"]([^"\u201d]*[\u4e00-\u9fff][^"\u201d]*)[\u201d"]'
                matches = re.findall(chinese_quote_pattern, text)
                if matches:
                    extracted = matches[-1].strip()
                    if len(extracted) >= 2:
                        _auto_log("[警告] ✓ 从引号中提取到中文输出", level=logging.WARNING)
                        return extracted
                
                # 2. 尝试提取中文行
                lines = text.split('\n')
                chinese_lines = [
                    line.strip() for line in lines 
                    if line.strip() and any('\u4e00' <= c <= '\u9fff' for c in line)
                ]
                
                if chinese_lines:
                    # 去重
                    seen = set()
                    unique_chinese = []
                    for line in chinese_lines:
                        simplified = re.sub(r'\s+', '', line)
                        if simplified not in seen:
                            seen.add(simplified)
                            unique_chinese.append(line)
                    
                    if unique_chinese:
                        _auto_log(f"[警告] ✓ 提取到 {len(unique_chinese)} 行中文内容", level=logging.WARNING)
                        return '\n'.join(unique_chinese)
                
                # 3. 无法提取有效中文
                _auto_log("[警告] ✗ 未能从混合内容中提取有效中文", level=logging.WARNING)
                return "（我需要更多时间来整理思路。）"
        
        # 内容有效，直接返回
        return text

    def _try_extract_meaningful(self, response: str) -> str:
        """
        从 LLM 响应中提取有意义的内容（内部 fallback 方法）
        
        当 _sanitize_output 检测到无效/短内容时调用
        """
        if not response or len(response.strip()) <= 3:
            return ""
        
        # 移除常见的分隔符和占位符
        cleaned = re.sub(r'^[-=]{3,}\s*', '', response)
        cleaned = re.sub(r'\s*[-=]{3,}$', '', cleaned)
        cleaned = cleaned.strip()
        
        # 如果清理后仍然很短，尝试提取中文段落
        if len(cleaned) <= 10:
            lines = response.split('\n')
            chinese_lines = [line.strip() for line in lines 
                           if line.strip() and any('\u4e00' <= c <= '\u9fff' for c in line)]
            if chinese_lines:
                return max(chinese_lines, key=len)
        
        if len(cleaned) > 10:
            return cleaned
        
        return ""

    # ── 信息方法 ────────────────────────────────────────────────

    def get_status(self) -> str:
        """返回智能体当前状态概览"""
        layer2 = self.pm.get_layer2_text()
        history_count = len(self.history)

        return (
            "═══ EGO 状态 ═══\n"
            f"模型: {self.llm.model}\n"
            f"API: {self.llm.api_base}\n"
            f"对话历史: {history_count} 条\n"
            f"\n── 第二层自我核心 ──\n{layer2}"
        )
    
    # ── 冷启动缓存管理 ──────────────────────────────────────────
    
    def _compute_hash(self, text: str) -> str:
        """计算文本的 MD5 哈希值"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compute_file_hash(self, filepath: str) -> str:
        """计算文件的 MD5 哈希值（文件不存在时返回空串，视为哈希失配）"""
        import hashlib
        # 【修复】存在性检查：防止 sys.json 缺失但缓存存在时启动崩溃
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_coldstart_cache(self) -> dict | None:
        """加载冷启动缓存文件"""
        if not self.coldstart_cache_file.exists():
            return None
        
        try:
            with open(self.coldstart_cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            _auto_log(f"[警告] 加载冷启动缓存失败: {e}", level=logging.WARNING)
            return None
    
    def _is_cache_valid(self, cache: dict) -> bool:
        """验证缓存是否有效"""
        
        # 1. 检查 Layer2 哈希
        #current_l2_hash = self._compute_hash(
        #    json.dumps(self.pm.layer2, ensure_ascii=False)
        #)
        #if current_l2_hash != cache.get("l2_hash"):
        #    _auto_log("[警告] Layer2 内容已变化，缓存失效", level=logging.WARNING)
        #    return False
        
        # 2. 检查 sys.json 哈希
        current_sys_hash = self._compute_file_hash(str(Path(BASE_DIR) / "data" / "sys.json"))
        if current_sys_hash != cache.get("sys_prompts_hash"):
            _auto_log("[警告] sys.json 已变化，缓存失效", level=logging.WARNING)
            return False
        
        # 3. 可选：检查缓存年龄
        try:
            cache_time = datetime.fromisoformat(cache["created_at"])
            days_old = (datetime.now() - cache_time).days
            
            if days_old > CACHE_STALE_WARNING_DAYS:
                _auto_log(f"[警告] 缓存已超过 {days_old} 天，建议刷新", level=logging.WARNING)
        except Exception:
            pass
        
        _auto_log("[信息] ✓ 缓存验证通过")
        return True
    
    def _save_coldstart_cache(self, status: str = "complete", failed_step: int = None):
        """保存冷启动缓存"""
        try:
            # 【新增】检查是否有旧缓存需要合并（续传场景）
            old_cache = self._load_coldstart_cache() if self.coldstart_cache_file.exists() else None
            
            # 提取冷启动和预热消息
            coldstart_msgs = []
            preheat_msgs = []
            
            for msg in self.history:
                if msg.get("stage") == "coldstart":
                    coldstart_msgs.append({
                        "role": msg["role"],
                        "content": msg["content"]  # 完整内容，包括所有指令
                    })
                elif msg.get("stage") == "preheat":
                    preheat_msgs.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # 【新增】如果是续传完成，需要合并旧缓存中的消息
            if old_cache and status == "complete" and old_cache.get("status") == "incomplete":
                _auto_log("[信息]   检测到续传场景，合并旧缓存消息...")
                
                # 合并冷启动消息（旧缓存 + 新history）
                old_coldstart = old_cache.get("coldstart_messages", [])
                if old_coldstart:
                    # 去重：只保留不在 history 中的旧消息
                    existing_contents = {msg["content"] for msg in coldstart_msgs}
                    #for old_msg in old_coldstart:
                    #    if old_msg["content"] not in existing_contents:
                    #        coldstart_msgs.insert(0, old_msg)  # 插入到前面，保持顺序
                    # 先收集需要插入的消息
                    msgs_to_insert = [old_msg for old_msg in old_coldstart if old_msg["content"] not in existing_contents]

                    if msgs_to_insert:
                        coldstart_msgs = msgs_to_insert + coldstart_msgs
                    
                    _auto_log(f"[信息]   - 合并了 {len(msgs_to_insert)} 条旧冷启动消息")
                
                # 合并预热消息
                old_preheat = old_cache.get("preheat_messages", [])
                if old_preheat:
                    existing_contents = {msg["content"] for msg in preheat_msgs}
                    for old_msg in old_preheat:
                        if old_msg["content"] not in existing_contents:
                            preheat_msgs.insert(0, old_msg)
                    
                    _auto_log(f"[信息]   - 合并了 {len(old_preheat)} 条旧预热消息")
            
            # 构建缓存数据
            cache_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "status": status,
                "l2_hash": self._compute_hash(
                    json.dumps(self.pm.layer2, ensure_ascii=False)
                ),
                "sys_prompts_hash": self._compute_file_hash(str(Path(BASE_DIR) / "data" / "sys.json")),
                "coldstart_messages": coldstart_msgs,
                "preheat_messages": preheat_msgs,
                "total_coldstart_messages": len(coldstart_msgs),
                "total_preheat_messages": len(preheat_msgs)
            }
            
            # 【新增】保存 previous_response_id（关键！用于恢复 LM Studio Session 上下文）
            _sid = self.llm.get_session_id()
            if _sid:
                cache_data["previous_response_id"] = _sid
                _auto_log(f"[信息]   - 已保存 previous_response_id: {_sid[:20]}...")
            
            # 如果未完成，添加进度信息
            if status == "incomplete" and failed_step:
                completed_steps = list(range(1, failed_step))
                cache_data["completed_steps"] = completed_steps
                cache_data["failed_step"] = failed_step
            
            # 写入文件
            with open(self.coldstart_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            _auto_log(f"[信息] ✓ 冷启动缓存已保存 (status={status})")
            if coldstart_msgs:
                _auto_log(f"[信息]   - 冷启动: {len(coldstart_msgs)} 条")
            if preheat_msgs:
                _auto_log(f"[信息]   - 预热: {len(preheat_msgs)} 条")
            
        except Exception as e:
            _auto_log(f"[警告] 保存冷启动缓存失败: {e}", level=logging.WARNING)
    
    def _build_rebuild_messages(self, cache: dict, include_recent_history: bool = False, include_coldstart: bool = True, history_count: int = None, exclude_current_input: bool = False) -> list[dict]:
        """
        构建用于重建 Session 的消息序列
        
        Args:
            cache: 冷启动缓存数据
            include_recent_history: 是否包含最近 N 条用户对话（当 previous_response_id 失效时需要）
            include_coldstart: 是否包含冷启动/预热历史消息（认知结果已在 L2 中，可跳过以加速重建）
            exclude_current_input: 会话中途恢复时排除 history 最后一条（当前正在处理的用户输入），
                重建后 round 0 会重发该输入，避免重复注入
        """
        messages = []
        
        # 1. System Prompt (L1 + L2)
        messages.append({
            "role": "system",
            "content": self.pm.build_system_prompt()
        })
        
        if include_coldstart:
            # 2. 冷启动历史
            for msg in cache.get("coldstart_messages", []):
                messages.append(msg)
            
            # 3. 预热历史
            for msg in cache.get("preheat_messages", []):
                messages.append(msg)
            
            _auto_log("[信息]   - System: 1 条")
            _auto_log(f"[信息]   - 冷启动: {len(cache.get('coldstart_messages', []))} 条")
            _auto_log(f"[信息]   - 预热: {len(cache.get('preheat_messages', []))} 条")
        else:
            _auto_log("[信息]   - System: 1 条")
            _auto_log("[信息]   - 冷启动/预热: 跳过（认知结果已在 L2 中）")
        
        # 4. 如果 previous_response_id 失效，追加最近的用户对话历史
        if include_recent_history and self.history:
            history_pool = self.history[:-1] if exclude_current_input else self.history
            limit = history_count if history_count is not None else SESSION_INIT_HISTORY_COUNT
            recent_history = history_pool[-limit:] if len(history_pool) > limit else history_pool
            
            added = 0
            for msg in recent_history:
                if msg.get("temporary", False) or msg.get("stage"):
                    continue
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
                    added += 1
            
            _auto_log(f"[信息]   - 最近对话: {added} 条（上限: {limit}）")
        
        _auto_log(f"[信息]   - 总计: {len(messages)} 条")
        
        return messages
    
    def _rebuild_session(self, messages: list[dict], cache: dict = None, restore_previous_id: str = None, exclude_current_input: bool = False) -> bool:
        """通过发送消息序列重建 Session
        
        Args:
            messages: 需要重建的消息列表
            cache: 可选，缓存数据（用于首批失败时重建包含对话历史的消息）
            restore_previous_id: 可选，尝试复用的旧 previous_response_id
            exclude_current_input: 内部降级重建时透传，排除当前正在处理的用户输入
        """
        # 【修复】上下文超限降级阶梯：逐级缩减最近对话条数（有界，不会无限循环）
        history_levels = sorted({SESSION_INIT_HISTORY_COUNT, 10, 5, 0}, reverse=True)
        history_level = 0
        
        batch_size = SESSION_REBUILD_BATCH_SIZE  # 每批消息数（config）
        
        # 设置旧 ID（尝试复用 KV cache，如果无效首批会报错然后自动回退）
        self.llm.set_session_id(restore_previous_id)
        retried_with_history = False  # 防止无限循环
        
        # 使用 while 循环，因为首批失败时可能需要重新构建消息并从头开始
        i = 0
        while i < len(messages):
            batch = messages[i:i+batch_size]
            
            if batch:
                result = self.llm.chat(batch, TEMPERATURE_WARMUP, stage=f"Rebuild-Batch-{i//batch_size + 1}")
                
                # 检测首批失败并恢复（旧 ID 失效 / 上下文超限 / LM Studio 不可用）
                if i == 0 and is_llm_error(result):
                    if _is_context_overflow(result):
                        # 【修复】上下文超限 → 逐级缩减最近对话条数后重建（有界降级）
                        if history_level < len(history_levels) - 1:
                            history_level += 1
                            _auto_log(f"[警告] 上下文超限，缩减最近对话为 {history_levels[history_level]} 条后重建", level=logging.WARNING)
                            self.llm.set_session_id(None)
                            messages = self._build_rebuild_messages(
                                cache, include_recent_history=history_levels[history_level] > 0,
                                include_coldstart=False, history_count=history_levels[history_level],
                                exclude_current_input=exclude_current_input
                            )
                            i = 0
                            continue
                        _auto_log(f"[警告] 缩减至最小上下文仍超限，重建失败: {result[:60]}", level=logging.WARNING)
                        self.llm.set_session_id(None)
                        return False
                    if self.llm.get_session_id():
                        # 旧 ID 失效，清除后重试
                        _auto_log("[警告] 旧 previous_response_id 已失效，清除后重建", level=logging.WARNING)
                        self.llm.set_session_id(None)
                        
                        if cache and not retried_with_history:
                            # 追加最近对话历史，重建更完整的上下文
                            _auto_log("[信息] 将包含最近对话重新重建 Session")
                            messages = self._build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=exclude_current_input)
                            retried_with_history = True
                            i = 0
                            continue
                    elif cache and not retried_with_history:
                        # 没有旧 ID 但首批仍然失败（可能是 LM Studio 不可用）
                        # 尝试包含对话历史重建（最后一次尝试）
                        _auto_log("[警告] 重建首批失败，尝试包含最近对话重建", level=logging.WARNING)
                        messages = self._build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=exclude_current_input)
                        retried_with_history = True
                        i = 0
                        continue
                    else:
                        _auto_log(f"[警告] 重建首批失败且无法进一步恢复: {result[:60]}", level=logging.WARNING)
                        self.llm.set_session_id(None)
                        return False
                elif is_llm_error(result):
                    # 【修复】非首批（i>0）失败：会话链已断裂，继续发送只会放大错误，
                    # 立即中止重建，交由调用方兜底路径处理（此前被静默吞掉，仅靠最终验证兜底）
                    _auto_log(f"[警告] 重建第 {i//batch_size + 1} 批失败，中止重建: {result[:80]}", level=logging.WARNING)
                    self.llm.set_session_id(None)
                    return False
            
            progress = min(i + batch_size, len(messages))
            _auto_log(f"[信息]   已重建 {progress}/{len(messages)} 条")
            i += batch_size
        
        # 【修复】重建完成后等待一段时间再验证，给 LMStudio 时间整理 KV cache
        if SESSION_REBUILD_POST_DELAY > 0:
            _auto_log(f"[信息] 等待 {SESSION_REBUILD_POST_DELAY} 秒后进行验证...")
            time.sleep(SESSION_REBUILD_POST_DELAY)
        
        _sid = self.llm.get_session_id()
        if _sid and self._validate_previous_response_id(_sid):
            _auto_log(f"[信息] ✓ Session 重建完成，previous_response_id: {_sid[:20]}...")
            return True
        _auto_log("[警告] ⚠ Session 重建后验证失败，上下文可能未完整恢复", level=logging.WARNING)
        self.llm.set_session_id(None)
        return False
    
    def _add_to_history_with_stage(self, role: str, content: str, stage: str = None):
        """添加历史对话，并标记阶段"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        
        if stage:
            entry["stage"] = stage
        
        self.history.append(entry)
        
        # 保留最近 HISTORY_MAX_ENTRIES 条
        if len(self.history) > HISTORY_MAX_ENTRIES:
            self.history = self.history[-HISTORY_MAX_ENTRIES:]
        
        self._save_history()

        # 同步到 ChromaDB 客观记忆（保留 mem 存在性检查：初始化失败路径下 mem 可能尚未创建）
        if hasattr(self, 'mem'):
            self.mem.log_conversation(role, content, stage)

        # 【新增】如果是主对话流程中的 assistant 回复，定期更新冷启动缓存中的 previous_response_id
        if stage is None and role == "assistant" and self.llm.get_session_id():
            self.session_id_save_counter += 1
            
            # 达到保存间隔时，更新缓存
            if self.session_id_save_counter >= self.session_id_save_interval:
                self._update_session_id_in_cache()
                self.session_id_save_counter = 0  # 重置计数器
                _auto_log(f"[信息] ✓ 已定期保存 Session ID（第 {self.session_id_save_interval} 轮对话）")
    
    def _update_session_id_in_cache(self):
        """
        更新冷启动缓存中的 previous_response_id
        
        【关键】确保意外退出后，重启时能恢复到最新的 Session 状态
        """
        try:
            # 加载现有缓存
            if not self.coldstart_cache_file.exists():
                return
            
            with open(self.coldstart_cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # 只更新 previous_response_id，保持其他内容不变
            cache_data["previous_response_id"] = self.llm.get_session_id()
            cache_data["updated_at"] = datetime.now().isoformat()
            
            # 写回文件
            with open(self.coldstart_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            _auto_log(f"[信息] ✓ 已更新缓存中的 previous_response_id: {(self.llm.get_session_id() or 'None')[:20]}...")
            
        except Exception as e:
            # 静默失败，不影响主流程
            _auto_log(f"[警告] 更新缓存中的 previous_response_id 失败: {e}", level=logging.WARNING)
    
    def _validate_previous_response_id(self, response_id: str) -> bool:
        """
        验证 previous_response_id 是否仍然有效
        
        发送一个轻量探测请求，检查服务器是否仍保留该 Session 上下文。
        如果 LMStudio 已重启，服务器端会话会被清空，旧 ID 将失效。
        
        Args:
            response_id: 待验证的 previous_response_id
            
        Returns:
            bool: ID 是否有效
        """
        max_retries = SESSION_ID_PROBE_MAX_RETRIES
        for attempt in range(1, max_retries + 1):
            try:
                probe_payload = {
                    "model": self.llm.model,
                    "input": "【系统：重启后重连接探测，仅回复“OK”即可。】",
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "previous_response_id": response_id,
                }
                response = self.llm.session.post(
                    self.llm.responses_url,
                    json=probe_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=SESSION_ID_PROBE_TIMEOUT
                )
                try:
                    if response.status_code == 200:
                        data = response.json()
                        new_id = data.get("id")
                        if new_id:
                            self.llm.set_session_id(new_id)
                        _auto_log("[信息] ✓ previous_response_id 验证通过")
                        return True
                    else:
                        _auto_log(f"[警告] previous_response_id 验证失败 (HTTP {response.status_code})", level=logging.WARNING)
                        return False
                finally:
                    # 【修复】探测响应统一关闭：非 200 / 返回路径也归还连接池，避免连接残留占位
                    try:
                        response.close()
                    except Exception:
                        pass
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    _auto_log(f"[警告] previous_response_id 验证超时 (第{attempt}次)，重试中...", level=logging.WARNING)
                    continue
                else:
                    _auto_log(f"[警告] previous_response_id 验证超时 {max_retries} 次，视为失效（LMStudio 可能繁忙，大概率失效）", level=logging.WARNING)
                    return False
            except requests.exceptions.ConnectionError as e:
                # 【修复】连接层失败可能是 keep-alive 死连接/暂时不可达，而非 id 真正失效：
                # 重建连接（保留会话链）后重试，避免将“死连接”误判为 id 失效而清空 KV 链。
                _auto_log(f"[警告] previous_response_id 连接失败，重建连接后重试 (第{attempt}次): {e}", level=logging.WARNING)
                self.llm._rebuild_session_keep_state()
                if attempt < max_retries:
                    continue
                return False
            except Exception as e:
                _auto_log(f"[警告] previous_response_id 验证异常: {e}", level=logging.WARNING)
                return False
        return False

    def _touch_llm_activity(self):
        """记录一次 LLM 服务端活动时间（用于空闲唤醒判定）"""
        self._last_llm_activity = time.time()

    def _should_warmup(self) -> bool:
        """距上次 LLM 活动是否已超过空闲阈值（需要唤醒探测）"""
        return (time.time() - self._last_llm_activity) > SERVER_IDLE_WARMUP_THRESHOLD

    def _warmup_server(self):
        """长时间空闲后、首个正式请求前，先发一次轻量无状态请求唤醒 LM Studio 服务端。

        服务端在空闲/模型恢复阶段处理首个请求时容易卡住（no_token 超时、0 token），
        提前发一个 max_tokens=1 的无 previous_response_id 探测，迫使服务端完成
        模型加载/资源预热，使随后的正式请求能快速产出。

        与 _validate_previous_response_id 的区别：后者携带 previous_response_id 用于
        校验会话链；本探测不带 id、创建全新无状态请求，绝不触碰 _previous_response_id、
        不污染主会话链。失败静默降级，不影响主流程。
        """
        try:
            probe_payload = {
                "model": self.llm.model,
                "input": "【系统：唤醒。】",
                "temperature": 0.0,
                "max_tokens": 1,
                "stream": False,
            }
            _auto_log(f"[信息] ⏳ 检测到空闲，发送唤醒探测（{SERVER_WARMUP_TIMEOUT}s 超时）...")
            response = self.llm.session.post(
                self.llm.responses_url,
                json=probe_payload,
                headers={"Content-Type": "application/json"},
                timeout=SERVER_WARMUP_TIMEOUT
            )
            try:
                if response.status_code == 200:
                    _auto_log("[信息] ✓ 服务端已唤醒")
                    self._touch_llm_activity()  # 仅成功才刷新活动时间：失败保持空闲态，下轮仍会试探
            finally:
                try:
                    response.close()
                except Exception:
                    pass
        except requests.exceptions.Timeout:
            _auto_log(f"[警告] 唤醒探测超时（>{SERVER_WARMUP_TIMEOUT}s），正式请求可能仍会较慢", level=logging.WARNING)
        except requests.exceptions.ConnectionError as e:
            _auto_log(f"[警告] 唤醒探测连接失败，重建连接供后续使用: {e}", level=logging.WARNING)
            self.llm._rebuild_session_keep_state()
        except Exception as e:
            _auto_log(f"[警告] 唤醒探测异常: {e}", level=logging.WARNING)

    def _initialize_session_with_cache(self):
        """
        初始化 session，支持缓存检查和续传
        
        【核心】根据缓存状态决定执行策略
        """
        # 【优化】智能判断是否需要冷启动/预热
        # 核心逻辑：只有当缓存完整有效且 L2 有内容时，才使用缓存重建 Session
        # 否则，重新执行冷启动和预热
        cache = self._load_coldstart_cache()

        # 【新增】如果缓存不存在但 L2 有内容，创建初始缓存并使用历史对话初始化
        if not cache and self.pm.get_valid_entries():
            _auto_log("[信息] ℹ 缓存不存在但 L2 已有内容，将使用历史对话初始化 Session")
            
            # 创建初始缓存文件（用于后续保存 Session ID）
            self._create_initial_cache()
            
            # 直接使用历史对话初始化 Session（不经过冷启动/预热）
            self._skip_coldstart = True
            self._resume_from_step = None
            self._initialize_session()

            # 【新增】初始化完成后，立即保存 previous_response_id 到缓存
            if self.llm.get_session_id():
                self._update_session_id_in_cache()

            return    
        
        if cache and self._is_cache_valid(cache):
            status = cache.get("status", "complete")
            
            if status == "complete":
                # 完整缓存
                _auto_log("[信息] ✓ 检测到完整的冷启动缓存")
                
                # 获取保存的 previous_response_id
                previous_id = cache.get("previous_response_id")
                
                # 【关键】如果有 previous_response_id，直接复用（不重建）
                # 服务器已有完整上下文，无需重新发送消息
                if previous_id:
                    if self._validate_previous_response_id(previous_id):
                        _auto_log(f"[信息]   - 复用 previous_response_id: {previous_id[:20]}...")
                        self.llm.set_session_id(previous_id)
                        _auto_log("[信息] ✓ 跳过重建，直接复用服务器上下文")
                        self._skip_coldstart = True
                        return
                    else:
                        _auto_log("[警告] ⚠ previous_response_id 已失效（LMStudio 可能已重启）", level=logging.WARNING)
                        _auto_log("[信息] 重建策略: System Prompt + 最近 N 条对话（跳过冷启动/预热，认知已在 L2 中）")
                        self.llm.set_session_id(None)
                        
                        rebuild_msgs = self._build_rebuild_messages(cache, include_recent_history=True, include_coldstart=False) #是否包含冷启动信息
                        if self._rebuild_session(rebuild_msgs, cache=cache):
                            _auto_log("[信息] ✓ Session 重建完成")
                        else:
                            _auto_log("[警告] Session 重建失败，将以全新 Session 开始首轮对话", level=logging.WARNING)
                        self._skip_coldstart = True
                        return
                
                # 没有 previous_response_id，需要从头重建
                _auto_log("[信息]   - 无 previous_response_id，需要重建 Session")
                rebuild_msgs = self._build_rebuild_messages(cache)
                if self._rebuild_session(rebuild_msgs, cache=cache):
                    _auto_log("[信息] ✓ Session 重建完成，跳过冷启动和预热")
                else:
                    _auto_log("[警告] Session 重建失败，跳过冷启动和预热（首轮对话将自动重建上下文）", level=logging.WARNING)
                self._skip_coldstart = True
                return
                
            elif status == "incomplete":
                # 不完整缓存，需要续传
                failed_step = cache.get("failed_step")
                previous_id = cache.get("previous_response_id")
                _auto_log(f"[信息] ⚠ 检测到不完整的缓存 (失败于 Step {failed_step})", level=logging.WARNING)
                _auto_log(f"[信息]   将重建已完成部分，然后从 Step {failed_step} 续传")
                
                # 先重建已完成的 Session（尝试复用 previous_response_id，失败则包含对话历史）
                rebuild_msgs = self._build_rebuild_messages(cache)
                if not self._rebuild_session(rebuild_msgs, cache=cache, restore_previous_id=previous_id):
                    _auto_log("[警告] Session 重建失败，续传可能不可用", level=logging.WARNING)
                
                # 设置标志，表示需要续传
                self._resume_from_step = failed_step
                self._cached_data = cache
                return
            else:
                # 状态未知，重新执行
                _auto_log("[警告] 缓存状态未知，重新执行冷启动", level=logging.WARNING)

        # 无缓存或失效，完整执行
        _auto_log("[信息] 未检测到有效缓存，将执行完整冷启动")
        # 正常初始化 session
        self._skip_coldstart = False
        self._resume_from_step = None
        self._initialize_session()

    def _create_initial_cache(self):
        """
        创建初始冷启动缓存（当 L2 已有内容但缓存不存在时）
        
        【关键】确保即使跳过冷启动/预热，也能有缓存文件用于保存 Session ID
        """
        try:
            cache_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "status": "complete",
                "l2_hash": self._compute_hash(
                    json.dumps(self.pm.layer2, ensure_ascii=False)
                ),
                "sys_prompts_hash": self._compute_file_hash(str(Path(BASE_DIR) / "data" / "sys.json")),
                "coldstart_messages": [],
                "preheat_messages": [],
                "total_coldstart_messages": 0,
                "total_preheat_messages": 0,
                "note": "Initial cache created for existing L2 content"
            }
            
            with open(self.coldstart_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            _auto_log("[信息] ✓ 初始缓存已创建")
            
        except Exception as e:
            _auto_log(f"[警告] 创建初始缓存失败: {e}", level=logging.WARNING)
    
    def _resume_coldstart_and_preheat(self):
        """从指定步骤续传冷启动，然后执行预热"""
        if not hasattr(self, '_resume_from_step') or not self._resume_from_step:
            return
        
        start_from = self._resume_from_step
        
        _auto_log(f"\n[续传] 从 Step {start_from} 继续冷启动...\n")
        
        # 1. 续传冷启动
        coldstart_success = self._execute_coldstart(start_from=start_from)
        
        if coldstart_success:
            # 2. 执行预热
            _auto_log("[信息] 冷启动完成，开始执行预热...")
            self._execute_preheat_only()
            
            # 3. 保存完整缓存
            self._save_coldstart_cache(status="complete")
            _auto_log("[信息] ✓ 续传完成，Session 已建立")
        else:
            _auto_log("[警告] 续传失败，缓存已保存进度", level=logging.WARNING)