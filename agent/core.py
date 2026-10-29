"""
EGO 智能体核心

整合两层自我核心、指令系统、记忆锚点、自我对话循环。
"""

from __future__ import annotations

import time
import re
import logging
import threading
import traceback
from datetime import datetime

from agent.llm import LMStudioClient, is_llm_error
from agent.prompts import PromptManager
from agent.event_bus import EventBus  # 【新增】统一事件注入通道（注入层）
from agent.scheduler import EgoScheduler  # 【新增】统一后台定时调度器（调度层）
from agent.instructions import InstructionExecutor
from agent.chroma_memory import ChromaMemoryManager
from agent.history import HistoryStore
from agent.debug_log import LLMDebugLogger
# StartupError 已迁至 session.py（仍为 RuntimeError 子类；GUI/CLI 只按 RuntimeError 捕获）
from agent.session import SessionBootstrap, load_sys_prompts
# EGO 循环引擎：chat / think / note 共用；循环状态与「已尝试恢复」标记收在 EgoLoopRunner 内
from agent.ego_loop import EgoLoopRunner
from agent.note_memory import NoteMemory
from agent.think_reflection import ThinkReflectionMixin
from agent.self_definition import SelfDefinitionMixin
from agent.note_review import NoteReviewMixin
from config import (
    RESPONSES_API_WARMUP,
    THINK_INTERVAL,
    SESSION_ID_SAVE_INTERVAL,
    LOCK_ACQUIRE_TIMEOUT,  # 用户命令一次性等锁最大秒数
    NOTE_CHECK_INTERVAL,  # 【新增】备忘录到期检查轮询间隔（秒）
    BACKGROUND_TASK_LOCK_WAIT,  # 后台任务轮询等锁最大秒数
    BACKGROUND_TASK_POLL_INTERVAL,  # 后台任务轮询等锁间隔（秒）
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


# 注：无效内容统一检测（is_invalid_content / INVALID_PLACEHOLDER_PATTERNS）
# 定义于 agent/instructions.py（避免循环导入），由 ego_loop.py 等使用方直接引用


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
        # 【新增】自对话（Think）SAY 输出回调：GUI/CLI 挂接后实时推送（后台线程调用）
        self.on_think_output = None  # callable(text) -> None
        # 【新增】用户回复定稿即回调（就绪即显示）：在自对话等后续阶段之前先推送界面（GUI/CLI 挂接）
        self.on_chat_output = None  # callable(text) -> None

        # LLM Debug Log（目录创建 / 文件编号 / 过期清理由 LLMDebugLogger 负责）
        # 必须建在调度器启动之前：后台 Think/自省/备忘录任务都会写 debug log
        self.debug_log = LLMDebugLogger()

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

        # 对话历史（统一走 HistoryStore：加载/保存/追加/查询）
        self.hist = HistoryStore()
        self.hist.load()

        # 加载系统提示词配置（sys.json；Think 段与冷启动/预热段共用同一份只读配置）
        self.sys_prompts = load_sys_prompts()
        
        # 【修复】对话计数器初始化：从历史中统计 CHAT 阶段的用户对话轮数
        # 只统计 CHAT 阶段的用户消息（排除 coldstart、preheat、think、note 阶段）
        self.conversation_counter = self.hist.count_chat_user_turns()
        _auto_log(f"[信息] 从历史中恢复对话计数器: {self.conversation_counter} 轮")

        # 【新增】Session ID 定期保存配置
        self.session_id_save_counter = 0  # 用户对话计数
        self.session_id_save_interval = SESSION_ID_SAVE_INTERVAL  # 每 N 轮对话保存一次
        
        self.THINK_INTERVAL = THINK_INTERVAL  # 从 config.py 读取配置

        # 会话建立与维护（sys.json 配置 / 冷启动预热 / 冷启动缓存 / Session 重建 /
        # 口令校验 / 空闲唤服）：状态全部收在 SessionBootstrap 内，本类只保留入口调用
        self.session = SessionBootstrap(
            llm=self.llm,
            pm=self.pm,
            hist=self.hist,
            executor=self.executor,
            debug_log=self.debug_log,
            sys_prompts=self.sys_prompts,
        )

        # 自对话循环引擎：chat / think / note 三个阶段共用同一套循环，循环状态
        # （_EgoLoopState / 「已尝试恢复」标记）全部收在 EgoLoopRunner 内；
        # 本类只注入共享基础设施与两个回调（写历史 / 会话重建）
        self.loop = EgoLoopRunner(
            llm=self.llm,
            events=self.events,
            executor=self.executor,
            hist=self.hist,
            debug_log=self.debug_log,
            add_to_history=self._add_to_history,
            recover_session=self.session.recover_session,
            mem=self.mem,
        )
        
        # 【优化】智能判断是否需要冷启动/预热
        # 如果 L2 认知库已有内容，说明系统已经"孵化"成功，跳过引导阶段直接进入交互
        valid_entries = self.pm.get_valid_entries()
        if valid_entries:
            _auto_log(f"[信息] 检测到 L2 认知库已存在 {len(valid_entries)} 条有效认知，跳过冷启动/预热")
            _auto_log("[信息] 系统将基于现有认知直接进入用户交互模式")
            try:
                self.session.initialize_with_cache()
            except Exception:
                # 【F4】初始化失败（如 StartupError）：先释放已建资源再抛出，避免 GUI
                # 「重试初始化」反复 new Agent 造成连接池 / 记忆句柄 / 定时器泄漏
                # 先停调度器（取消已 reschedule 的 Timer），避免半初始化对象被后台任务引用
                try:
                    self.scheduler.stop_all()
                except Exception:
                    pass
                try:
                    self.llm.close()
                except Exception:
                    pass
                try:
                    self.mem.close()
                except Exception:
                    pass
                raise
        else:
            # 执行冷启动和预热流程（仅在 L2 为空时执行）
            self.session.execute_coldstart_and_preheat()
        
        # 【修复】禁用 Session 预热，因为冷启动/预热已经建立了完整的 KV cache
        if RESPONSES_API_WARMUP and len(self.hist) > 2:
            self.session.warmup_from_history()

        # 【新增】定时任务（自对话/自省/自我定义/备忘录审查/备忘录到期）已统一注册到 self.scheduler
        # （注册逻辑见各 mixin 的 _register_think_tasks / _register_self_def_task /
        # _register_note_review_task 与 core 的 _register_note_task）
        if self._auto_think_enabled or self._auto_reflection_enabled or self._self_def_enabled or self._note_review_enabled:
            _auto_log("[信息] 后台定时任务已由统一调度器接管")
        else:
            _auto_log("[信息] 全部后台定时任务未启用")            


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
            self.hist.clear()
            # 【修复】完全重置 LLM session，避免连接泄露
            _auto_log("[系统] 完全重置 LLM Session...")
            self.llm.reset_and_reconnect()
            # 重新初始化 session，发送 system prompt
            self.session.initialize_session()
            # 【修复】将新 session 的 previous_response_id 同步写回冷启动缓存，
            # 避免缓存中残留旧 ID 导致下次启动白白执行一轮"验证失效 → 重建"
            if self.llm.get_session_id():
                self.session.update_session_id_in_cache()
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

        与用户对话共用 EgoLoopRunner.run：LLM 输出中的 MEMO_RD / NOTE_ADD / NOTE_RD /
        CONTINUE 等指令照常解析执行并按指令语义调度下一轮（含空检索结果回传）。
        """
        prompt = "\n".join(
            f"执行类条目 {n['id']} 已到触发时间："
            f"标题:{n.get('title')} 内容:{n.get('content')}。"
            f"请主动向用户提及并确认执行情况。" for n in due_notes
        )
        _auto_log(f"[备忘录] 🔄 调度 LLM 处理到期条目（共 {len(due_notes)} 条）...")
        state = self.loop.run(user_input=prompt, images=None, start_time=time.time(), stage="note")
        # 前置校验：LLM 调用失败/返回错误直接判定失败，不进入输出定稿（避免污染历史）
        output = state.output_content
        if not output or is_llm_error(output):
            _auto_log(f"[备忘录] ✗ LLM 调用失败或返回错误: {str(output)[:80]}", level=logging.WARNING)
            return False
        output = self.loop.finalize_output(output)
        if output is None:
            _auto_log("[备忘录] ✗ LLM 未产出有效输出", level=logging.WARNING)
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

    def get_chat_history(self, limit: int = None, exclude_last: bool = False) -> list:
        """
        【新增】返回"与用户的对话"（chat 阶段）历史条目。

        跳过 temporary 与 think/note/反思等内部阶段输出（这类条目仍存 history.json 供追溯，
        但不注入 LLM 上下文、也不在重启回显中展示）。会话初始化/重建/预热注入与 GUI/CLI
        重启回显共用此方法，保证口径与条数一致。

        Args:
            limit: 最多返回最近 N 条（先过滤后切片；None 表示不限制，0 表示不返回）
            exclude_last: 排除最后一条（会话中途恢复时排除正在处理的用户输入，避免重复注入）
        """
        return self.hist.chat_entries(limit=limit, exclude_last=exclude_last)

    def _add_to_history(self, role: str, content: str):
        """
        添加历史对话（无阶段标记，用于主对话流程）
        委托给 _add_to_history_with_stage 实现
        """
        self._add_to_history_with_stage(role, content, stage=None)


    # ── 安全过滤 ────────────────────────────────────────────────

    BLOCKED_PATTERNS = [
        "bypass", "jailbreak",
    ]

    # 【设计】这两个前缀是角色标签通道：系统侧注入（【EGO: …】/【系统: …】）由内部
    # 构造、不经此检查；用户输入一律禁止出现它们，即使只是引用日志或讨论协议。
    # 因此 pattern 故意不带冒号，以同时覆盖半角「【系统:」与全角「【系统：」两种变体。
    # 不拦话题词（改变角色/自身危害等）：那类请求由 L1 语义层拒绝，不靠关键词。
    INJECTION_PATTERNS = [
        "【系统","【EGO",
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
                return f"拒绝：检测到显式的越狱/绕过指令（匹配：{pattern}）"

        for pattern in self.INJECTION_PATTERNS:
            clean_pattern = re.sub(r'\s+', '', pattern)
            if clean_pattern in normalized or clean_pattern in cleaned or pattern in lower:
                return f"拒绝：检测到伪造的系统/智能体指令块（匹配：{pattern}）"

        return None

    # ── 构建对话上下文 ──────────────────────────────────────────


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
            self.session.touch_activity()
            self._agent_lock.release()

    def _process_input_inner(self, user_input: str, images: list = None) -> str:
        """process_input 的实际逻辑（在持有锁的情况下执行）"""
        start_time = time.time()

        # 【空闲唤醒】长时间空闲后的首个对话请求，服务端冷启动易卡死：先轻量唤醒
        if self.session.should_warmup():
            self.session.warmup_server()

        # 【修复】重置 ID 恢复标志，确保每次新对话都能触发恢复（如果需要）
        self.loop.reset_recovery_flag()

        # 安全检查
        rejection = self._check_safety(user_input)
        if rejection:
            self._add_to_history("user", user_input)
            self._add_to_history("assistant", rejection)
            self._emit_chat_output(rejection)
            return rejection

        # 【新增】重复内容检测：检查最近 3 轮响应
        repetition_error = self.loop.check_recent_repetition()
        if repetition_error:
            self._emit_chat_output(repetition_error)
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
        # 【新增】记录本轮开始前的会话链锚点：本轮若被判为“未写 SAY”需要回滚，
        # 防止 LM Studio 服务端会话链残留的思考流在下一轮被模型当作示例模仿
        session_id_before = self.llm.get_session_id()

        state = self.loop.run(user_input, images, start_time)

        # 【新增】纯思考流拦截：整轮循环完全没写出任何 SAY 指令时（state.has_output 保持
        # False，该标志仅在首个有效 SAY 载荷被捕获时才置 True），丢弃思考流本身，
        # 改为以固定文案落盘/上屏，避免畸形思考流以“历史示例”的形式污染后续轮次
        # （THINK@ 英文长推理 → 漏写 SAY 的根源）。
        if not state.has_output:
            _auto_log(
                "[警告] ✗ 本轮未产生任何 SAY 指令（仅思考流）：丢弃思考流，改用固定文案",
                level=logging.WARNING,
            )
            _auto_log(
                f"[警告]   已执行轮数: {state.round_num + 1}，"
                f"思考流长度: {len(state.output_content or '')}",
                level=logging.WARNING,
            )
            if session_id_before != self.llm.get_session_id():
                self.llm.set_session_id(session_id_before)
                _auto_log("[警告]   已回滚会话链到本轮之前的状态", level=logging.WARNING)
            self.events.cleanup()

            # 固定文案照常上屏 / 写历史 / 写记忆（思考流本身绝不进入任何持久化通道）
            no_output_notice = "（EGO 无输出……）"
            self._emit_chat_output(no_output_notice)
            self._post_response(no_output_notice)
            return no_output_notice

        # 【重构】循环结束后的输出定稿（清理；SAY 载荷全为嵌套指令时返回 None）
        output_content = self.loop.finalize_output(state.output_content)
        if output_content is None:
            # 与上面的纯思考流沉默路径一致：上屏 + 写历史 + 计数/Think 调度
            no_output_notice = "（EGO 无输出……）"
            self._emit_chat_output(no_output_notice)
            self._post_response(no_output_notice)
            return no_output_notice

        # 【新增】就绪即显示：在触发自对话（Think）等后续阶段之前，先把定稿回复推送到界面，
        # 避免回复显示被自对话阶段阻塞（多轮 SAY 仍在本循环内拼接完成后一次性返回）
        self._emit_chat_output(output_content)

        # 【重构】响应后收尾：临时消息清理、写入历史、计数与 Think 调度、保存
        self._post_response(output_content)

        end_time = time.time()
        
        # 【新增】打印本轮对话的指令统计
        _auto_log(f"[统计] 总轮数: {state.round_num + 1}, CONTINUE次数: {state.continue_count}, MEMO_RD次数: {state.recall_count}")
        _auto_log(f"[统计] ✓ 成功生成输出，长度: {len(output_content)}")
        
        _auto_log(f"[性能] 本次请求耗时: {end_time - start_time:.2f}秒")

        return output_content


    def _emit_chat_output(self, text: str) -> None:
        """
        【新增】用户回复定稿即回调（就绪即显示）：在自对话（Think）等后续阶段之前
        推送到界面，避免回复被后续阶段阻塞；回调由 GUI/CLI 挂接，未挂接时静默。
        """
        cb = getattr(self, "on_chat_output", None)
        if cb and text:
            try:
                cb(text)
            except Exception as e:
                _auto_log(f"[警告] 回复推送回调失败: {e}", level=logging.WARNING)

    def _post_response(self, output_content: str) -> None:
        """
        【重构】成功响应后的收尾：写入历史 → 更新轮次计数
        → 检查 Think 触发 → 持久化历史
        （循环内事件已由 EgoLoopRunner.run 结束时 events.cleanup() 统一清理）
        """
        self._add_to_history("assistant", output_content)
        
        # 【新增】更新对话计数器（每次成功响应用户后递增）
        self.conversation_counter += 1
        _auto_log(f"[统计] 累计对话轮数: {self.conversation_counter}")
        
        # 【新增】检查是否需要触发 Think（定时自对话）- 在计数器更新后检查
        if self.conversation_counter > 0 and self.conversation_counter % self.THINK_INTERVAL == 0:
            _auto_log(f"\n[调度] 已达到 {self.THINK_INTERVAL} 轮对话，触发 Think 周期")
            # 【修复】改为后台线程执行：Think 复用完整 EGO 循环（EGO_MAX_ROUNDS × 提示词数），
            # 原同步调用会阻塞当轮用户回复（最长数千秒）；后台线程经 _execute_think_background
            # 复用 run_think_cycle 的独立加锁入口，待当前对话释放锁后执行，不阻塞回复显示
            try:
                threading.Thread(target=self._execute_think_background, daemon=True).start()
            except Exception as e:
                _auto_log(f"[警告] Think 后台任务启动失败: {e}，不影响本轮对话", level=logging.WARNING)


    # ── 信息方法 ────────────────────────────────────────────────

    def get_status(self) -> str:
        """返回智能体当前状态概览"""
        layer2 = self.pm.get_layer2_text()
        history_count = len(self.hist)

        return (
            "═══ EGO 状态 ═══\n"
            f"模型: {self.llm.model}\n"
            f"API: {self.llm.api_base}\n"
            f"对话历史: {history_count} 条\n"
            f"\n── 第二层自我核心 ──\n{layer2}"
        )
    
    
    def _add_to_history_with_stage(self, role: str, content: str, stage: str = None):
        """添加历史对话并标记阶段（落盘/截断由 HistoryStore 负责）"""
        self.hist.add(role, content, stage)

        # 同步到 ChromaDB 客观记忆（保留 mem 存在性检查：初始化失败路径下 mem 可能尚未创建）
        if hasattr(self, 'mem'):
            self.mem.log_conversation(role, content, stage)

        # 【新增】如果是主对话流程中的 assistant 回复，定期更新冷启动缓存中的 previous_response_id
        if stage is None and role == "assistant" and self.llm.get_session_id():
            self.session_id_save_counter += 1
            
            # 达到保存间隔时，更新缓存
            if self.session_id_save_counter >= self.session_id_save_interval:
                self.session.update_session_id_in_cache()
                self.session_id_save_counter = 0  # 重置计数器
                _auto_log(f"[信息] ✓ 已定期保存 Session ID（第 {self.session_id_save_interval} 轮对话）")
    
