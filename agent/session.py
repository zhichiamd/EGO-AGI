# -*- coding: utf-8 -*-
"""
会话建立与维护（SessionBootstrap）

从 core.py 抽出的会话生命周期：sys.json 配置加载 → 冷启动/预热 → Session 初始化 →
冷启动缓存读写 → previous_response_id 口令校验 → 失效恢复 → 空闲唤服探测。

设计约束：
- 不持有 EGOAgent：llm / pm / hist / executor / debug_log / sys_prompts 全部构造注入，
  因此可脱离主控类单独构造与测试。
- logger 沿用 "EGOAgent"：EGO_GUI 初始化期按该名字挂日志透传，启动日志必须继续可见。
- 历史读取统一走 HistoryStore（hist.chat_entries），不再回落到 EGOAgent.get_chat_history。
- StartupError 随本模块迁出：GUI/CLI 不按名引用它，只按 RuntimeError 捕获，
  因此「初始化失败 / 重试初始化」的语义不变。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.llm import is_llm_error
from agent.history import CHAT_HISTORY_STAGES
# 冷启动缓存层：文件读写 / 有效性校验 / MD5 指纹 / previous_response_id 回写
from agent.coldstart_cache import ColdstartCache
# 冷启动/预热执行层 与 Session 重建层（本类只做编排与对外透传，不实现细节）
from agent.coldstart_runner import ColdstartRunner
from agent.session_rebuilder import SessionRebuilder
from config import (
    BASE_DIR,
    TEMPERATURE_INIT,  # Session 初始化（首轮 system prompt）
    SESSION_INIT_HISTORY_COUNT,  # 启动注入的最近对话条数
    RESPONSES_API_MAX_INITIAL_ROUNDS,  # KV cache 预热最大轮数
    SERVER_IDLE_WARMUP_THRESHOLD,  # 空闲唤服探测空闲阈值（秒）
    SERVER_WARMUP_TIMEOUT,  # 空闲唤服探测请求超时（秒）
)


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """记录日志（级别由调用方显式指定，默认 INFO）"""
    logger.log(level, message)


class StartupError(RuntimeError):
    """会话上下文重建后口令校验仍未通过 → 启动失败。

    由 GUI/CLI 顶层捕获：GUI 显示「初始化失败 / 重试初始化」，CLI 打印错误并提示。
    继承 RuntimeError 以复用 CLI 既有的「请确认 LM Studio 已启动」提示分支。
    """
    pass


def load_sys_prompts() -> dict:
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


class SessionBootstrap:
    """会话建立与维护：冷启动/预热、冷启动缓存、Session 重建、口令校验、空闲唤服。"""

    def __init__(self, llm, pm, hist, executor, debug_log, sys_prompts, cache_path=None):
        """
        Args:
            llm: LMStudioClient（会话唯一出口：chat / session_id / warmup_session）
            pm: PromptManager（system prompt 与 L2 认知）
            hist: HistoryStore（重建时注入最近 N 条 chat 历史）
            executor: InstructionExecutor（冷启动/预热阶段的指令执行）
            debug_log: LLMDebugLogger（原始响应落盘）
            sys_prompts: sys.json 配置（与 EGOAgent.sys_prompts 为同一只读对象）
            cache_path: 冷启动缓存文件路径（默认 data/coldstart_cache.json，便于测试注入）
        """
        self.llm = llm
        self.pm = pm
        self.hist = hist
        self.executor = executor
        self.debug_log = debug_log
        self.sys_prompts = sys_prompts

        # 冷启动缓存层：文件读写 / 有效性校验 / MD5 指纹 / previous_response_id 回写，
        # 以及冷启动/预热消息缓冲区（待落盘数据，只有 save() 消费）
        self.cache = ColdstartCache(llm=self.llm, pm=self.pm, cache_path=cache_path)

        # 冷启动/预热执行层 与 Session 重建层：只注入共享基础设施，不持有本类
        self.coldstart = ColdstartRunner(
            llm=self.llm,
            pm=self.pm,
            executor=self.executor,
            debug_log=self.debug_log,
            sys_prompts=self.sys_prompts,
            cache=self.cache,
        )
        self.rebuilder = SessionRebuilder(
            llm=self.llm,
            pm=self.pm,
            hist=self.hist,
            cache=self.cache,
        )

        # 上次 LLM 服务端活动时间（用于空闲唤醒判定）；0 表示进程启动后首个请求即视为空闲
        self._last_llm_activity = 0.0

    # ── 启动入口 ──────────────────────────────────────────────

    def initialize_with_cache(self):
        """
        初始化 session，支持缓存检查和续传
    
        【核心】根据缓存状态决定执行策略
        """
        # 【优化】智能判断是否需要冷启动/预热
        # 核心逻辑：只有当缓存完整有效且 L2 有内容时，才使用缓存重建 Session
        # 否则，重新执行冷启动和预热
        cache = self.cache.load()

        # 【新增】如果缓存不存在但 L2 有内容，创建初始缓存并使用历史对话初始化
        if not cache and self.pm.get_valid_entries():
            _auto_log("[信息] ℹ 缓存不存在但 L2 已有内容，将使用历史对话初始化 Session")
        
            # 创建初始缓存文件（用于后续保存 Session ID）
            self.cache.create_initial()
        
            # 直接使用历史对话初始化 Session（不经过冷启动/预热）
            self.initialize_session()

            # 【新增】初始化完成后，立即保存 previous_response_id 到缓存
            if self.llm.get_session_id():
                self.cache.update_session_id()

            return    
    
        if cache and self.cache.is_valid(cache):
            status = cache.get("status", "complete")
        
            if status == "complete":
                # 完整缓存
                _auto_log("[信息] ✓ 检测到完整的冷启动缓存")
            
                # 获取保存的 previous_response_id
                previous_id = cache.get("previous_response_id")
            
                # 【关键】如果有 previous_response_id，直接复用（不重建）
                # 服务器已有完整上下文，无需重新发送消息
                if previous_id:
                    if self.rebuilder.validate_previous_response_id(previous_id):
                        # 【统一调用】校验探针走 chat() 并在通过时接管新链头；
                        # 此处不得再把链头回退到 previous_id——回退会让主会话与服务端最新
                        # 链尾分叉，探针已推进的那段 KV 前缀随即作废（下次请求需回滚重算），
                        # 也违背“启动拿到 response_id 后主会话都走这条链”的约定
                        _live = self.llm.get_session_id()
                        _auto_log(f"[信息]   - 复用 previous_response_id: {(_live or previous_id)[:20]}...")
                        # 链头已是探针产生的新 ID → 立即回写缓存，下次启动直接走这条
                        self.cache.update_session_id()
                        _auto_log("[信息] ✓ 跳过重建，直接复用服务器上下文")
                        return
                    else:
                        _auto_log("[警告] ⚠ previous_response_id 已失效（LMStudio 可能已重启）", level=logging.WARNING)
                        _auto_log("[信息] 重建策略: System Prompt + 最近 N 条对话（跳过冷启动/预热，认知已在 L2 中）")
                        self.llm.set_session_id(None)
                    
                        rebuild_msgs = self.rebuilder.build_rebuild_messages(cache, include_recent_history=True, include_coldstart=False) #是否包含冷启动信息
                        if self.rebuilder.rebuild_session(rebuild_msgs, cache=cache):
                            _auto_log("[信息] ✓ Session 重建完成")
                        else:
                            # 【启动失败】重建后口令校验仍未通过 → 不再静默降级，交由用户重试/重启
                            raise StartupError(
                                "会话上下文重建后口令校验未通过：LM Studio 未保留系统提示词/历史。"
                                "请重启 LM Studio 或重载模型后点击「重试初始化」；若反复失败请检查口令配置。"
                            )
                        return
            
                # 没有 previous_response_id，需要从头重建
                _auto_log("[信息]   - 无 previous_response_id，需要重建 Session")
                rebuild_msgs = self.rebuilder.build_rebuild_messages(cache, include_recent_history=True)
                if self.rebuilder.rebuild_session(rebuild_msgs, cache=cache):
                    _auto_log("[信息] ✓ Session 重建完成，跳过冷启动和预热")
                else:
                    # 【启动失败】重建后口令校验仍未通过 → 交由用户重试/重启
                    raise StartupError(
                        "会话上下文重建后口令校验未通过：LM Studio 未保留系统提示词/历史。"
                        "请重启 LM Studio 或重载模型后点击「重试初始化」；若反复失败请检查口令配置。"
                    )
                return
            
            elif status == "incomplete":
                # 不完整缓存，需要续传
                failed_step = cache.get("failed_step")
                previous_id = cache.get("previous_response_id")
                _auto_log(f"[信息] ⚠ 检测到不完整的缓存 (失败于 Step {failed_step})", level=logging.WARNING)
                _auto_log(f"[信息]   将重建已完成部分，然后从 Step {failed_step} 续传")
            
                # 先重建已完成的 Session（续传属初始化未完成场景，需回灌冷启动/预热上下文，故显式 True）
                rebuild_msgs = self.rebuilder.build_rebuild_messages(cache, include_coldstart=True)
                if not self.rebuilder.rebuild_session(rebuild_msgs, cache=cache, restore_previous_id=previous_id):
                    # 【启动失败】不完整缓存续传的重建也走同一闸门 → 交由用户重试/重启
                    raise StartupError(
                        "会话上下文重建后口令校验未通过（不完整缓存续传）：LM Studio 未保留系统提示词/历史。"
                        "请重启 LM Studio 或重载模型后点击「重试初始化」；若反复失败请检查口令配置。"
                    )
            
                # 重建成功后立即补跑剩余步骤并收口缓存。
                # 不再经由实例标志交给 execute_coldstart_and_preheat：core 的两个启动入口互斥
                # （L2 非空只调 initialize_with_cache、L2 为空只调 execute_coldstart_and_preheat），
                # 标志永远读不到非默认值，续传因此长期不可达。
                self.resume_coldstart_and_preheat(failed_step)
                return
            else:
                # 状态未知，重新执行
                _auto_log("[警告] 缓存状态未知，重新执行冷启动", level=logging.WARNING)

        # 无缓存或失效，完整执行
        _auto_log("[信息] 未检测到有效缓存，将执行完整冷启动")
        # 正常初始化 session
        self.initialize_session()

    def initialize_session(self):
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
            if len(self.hist):
                # 从配置读取历史对话注入数量
                history_count = SESSION_INIT_HISTORY_COUNT
            
                # 【修复】仅"与用户的对话"（chat 阶段），且先过滤后切片（N = 最近 N 条用户对话）
                recent_history = self.hist.chat_entries(limit=history_count)
            
                _auto_log(f"[信息] 检测到 {len(recent_history)} 条历史对话，将注入到 Session 中（配置上限: {history_count}）")
   
                for msg in recent_history:
                    # 跳过临时消息和系统消息
                    if msg.get("temporary", False):
                        continue
                    # 【新增】仅注入与用户的对话（chat）阶段，跳过自对话/备忘录执行/反思等内部阶段输出
                    if msg.get("stage") not in CHAT_HISTORY_STAGES:
                        continue
                
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                
                    # 只添加 user 和 assistant 的消息
                    if role in ["user", "assistant"]:
                        messages.append({"role": role, "content": content})
            
                _auto_log(f"[信息] ✓ 已注入 {len([m for m in messages if m['role'] in ['user', 'assistant']])} 条历史消息")
        
            # 添加系统初始化消息
            # 【格式】用【系统: …】自述角色包裹：命中 llm 的前缀白名单，序列化时不再叠加
            # "[用户]" 前缀，与 L1“被【系统: 内容】包裹的内容视为系统信息提示”的约定一致。
            messages.append({"role": "user", "content": "【系统: 工作上下文重建完成，请检索备忘录（LIST所有条目，然后读取最新状态快照和感兴趣条目）接续重建前状态】"})
        
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

    def execute_coldstart_and_preheat(self):
        """
        执行冷启动和预热流程
    
        调用方（core）仅在 L2 认知库为空时进入本入口；使用缓存的场景走
        initialize_with_cache，两者互斥，因此这里不做跳过/续传判定。
        """
        # 正常执行完整冷启动和预热
        _auto_log("═══ 系统启动：冷启动与预热 ═══")
    
        # ─ 冷启动阶段 ────────────────────────────────
        coldstart_prompts = self.sys_prompts.get("coldstart", {})
        if coldstart_prompts:
            _auto_log(f"[冷启动] 开始执行 {len(coldstart_prompts)} 条冷启动提示...\n")
        
            success = self.coldstart.run_coldstart(start_from=1)
        
            if not success:
                _auto_log("[警告] 冷启动未完成，缓存已保存进度", level=logging.WARNING)
                return
        else:
            _auto_log("[冷启动] ⚠ 无冷启动提示词，跳过", level=logging.WARNING)
    
        # ─ 预热阶段 ────────────────────────────────
        self.coldstart.run_preheat(use_incremental=True)
    
        # 保存完整缓存
        self.cache.save(status="complete")
    
        _auto_log("═══ 系统就绪 ══")

    def resume_coldstart_and_preheat(self, start_from):
        """从指定步骤续传冷启动，再跑预热并收口缓存（incomplete 缓存的续传出口）

        仅在 initialize_with_cache 的 incomplete 分支调用：该分支已完成「重建已走过的
        上下文」，本方法负责把剩下的冷启动步骤补齐，使缓存回到 complete。

        Args:
            start_from: 续传起始步号（1 基，即缓存中的 failed_step）；
                缺失/非法时归一化为 1（重跑全部冷启动，避免静默丢步骤）

        失败语义：冷启动续传再次失败时缓存保持 incomplete（执行层已落盘最新进度），
        本方法不抛异常——与 execute_coldstart_and_preheat 的失败处理一致，
        启动继续，下次启动按新的 failed_step 再续。
        """
        try:
            start_from = int(start_from)
        except (TypeError, ValueError):
            _auto_log(
                f"[警告] 缓存中的 failed_step 不可用（{start_from!r}），将重跑全部冷启动",
                level=logging.WARNING,
            )
            start_from = 1
        if start_from < 1:
            start_from = 1

        _auto_log(f"\n[续传] 从 Step {start_from} 继续冷启动...\n")

        if not self.coldstart.run_coldstart(start_from=start_from):
            _auto_log("[警告] 续传失败，缓存已保存进度", level=logging.WARNING)
            return

        _auto_log("[信息] 冷启动完成，开始执行预热...")
        # 完整模式（含 system prompt）：与原续传设计一致，预热不依赖重建链的前缀是否完整
        self.coldstart.run_preheat(use_incremental=False)

        # resume_from 告诉缓存层：旧缓存中前 start_from-1 步需要保留并与本轮缓冲拼接
        self.cache.save(status="complete", resume_from=start_from)
        _auto_log("[信息] ✓ 续传完成，Session 已建立")

    # ── KV cache 预热（按最近对话重建） ──────────────────────

    def warmup_from_history(self):
        """
        Session 预热：快速重建 KV cache
    
        策略：
        1. 如果历史较短（< 20 条），一次性发送所有历史
        2. 如果历史较长，发送最近的 N 条
        """
        # 【新增】仅使用与用户的对话（chat）阶段重建 KV cache（先过滤后切片，统一走 get_chat_history）
        chat_history = self.hist.chat_entries()
        if len(chat_history) < 2:
            return
    
        _auto_log(f"[预热] 检测到 {len(chat_history)} 条历史对话，正在重建 KV cache...")
    
        # 构建预热消息
        warmup_messages = []
    
        # 添加 system prompt
        warmup_messages.append({
            "role": "system",
            "content": self.pm.build_system_prompt()
        })
    
        # 策略选择：根据历史长度决定
        max_warmup_rounds = RESPONSES_API_MAX_INITIAL_ROUNDS
        if len(chat_history) <= max_warmup_rounds * 2:
            # 短历史：发送全部
            for entry in chat_history:
                role = entry["role"]
                content = entry["content"]
            
                if role == "user":
                    warmup_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    warmup_messages.append({"role": "assistant", "content": content})
        else:
            # 长历史：只发送最近的 N 轮
            recent_history = chat_history[-(max_warmup_rounds * 2):]
        
            # 添加一个摘要提示
            early_count = len(chat_history) - len(recent_history)
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

    # ── 运行期失效恢复 ────────────────────────────────────────

    def recover_session(self) -> bool:
        """previous_response_id 失效恢复（实现在 SessionRebuilder）

        签名固定为 () -> bool：EgoLoopRunner 以回调方式注入，返回 True 表示重建成功、
        调用方应重试本轮请求。
        """
        return self.rebuilder.recover_session()

    # ── 缓存回写（对外入口）──────────────────────────────────

    def update_session_id_in_cache(self):
        """把当前 previous_response_id 写回冷启动缓存（缓存层实现在 ColdstartCache）

        保留此公共入口：core 在 /clear 重建与定期保存时调用；内部一律直接用 self.cache。
        """
        self.cache.update_session_id()

    # ── 空闲唤服探测 ──────────────────────────────────────────

    def touch_activity(self):
        """记录一次 LLM 服务端活动时间（用于空闲唤醒判定）"""
        self._last_llm_activity = time.time()

    def should_warmup(self) -> bool:
        """距上次 LLM 活动是否已超过空闲阈值（需要唤醒探测）"""
        return (time.time() - self._last_llm_activity) > SERVER_IDLE_WARMUP_THRESHOLD

    def warmup_server(self):
        """长时间空闲后、首个正式请求前，先发一次轻量"链内"请求唤醒 LM Studio 服务端。

        服务端在空闲/模型恢复阶段处理首个请求时容易卡住（no_token 超时、0 token），
        提前发一个 max_tokens=1 的探测，迫使服务端完成模型加载/资源预热，使随后的
        正式请求能快速产出。

        【修复】探测改为"链内探测"：存在 previous_response_id 时将其一并携带，并从响应
        同步新的会话链头（与 _validate_previous_response_id 同构）。原实现发的是"无
        previous_response_id 的全新无状态请求"，在单 slot（LRU 换入）下会顶掉主会话已
        缓存的 KV 前缀，使紧随其后的正式请求 t_last=-1、从 0% 重算整段上下文（空闲越久
        上下文越大，代价越高）。链内探测只延长、不替换 KV 前缀，兼得"唤醒模型"与
        "保留会话缓存"。无会话链（冷启动）时仍按无状态探测处理。失败静默降级。
        """
        cur_id = self.llm.get_session_id()
        try:
            _auto_log(f"[信息] ⏳ 检测到空闲，发送唤醒探测（{SERVER_WARMUP_TIMEOUT}s 超时）...")
            # 【统一调用】走主会话唯一出口：存在 previous_response_id 时自动携带（链内探测），
            # 只延长、不替换 KV 前缀；apply_session=bool(cur_id) 保持原语义——
            # 仅在原本就有链时接管探测产生的新链头，冷启动无链时不接管
            response = self.llm.chat(
                [{"role": "user", "content": "【系统：唤醒探测，仅回复“OK”即可。】"}],
                temperature=0.0,
                stage="warmup",
                timeout=SERVER_WARMUP_TIMEOUT,
                max_tokens=1,
                apply_session=bool(cur_id),
            )
            if is_llm_error(response):
                _auto_log(f"[警告] 唤醒探测失败，正式请求可能仍会较慢: {str(response)[:80]}",
                          level=logging.WARNING)
                return
            _auto_log("[信息] ✓ 服务端已唤醒")
            self.touch_activity()  # 仅成功才刷新活动时间：失败保持空闲态，下轮仍会试探
        except Exception as e:
            _auto_log(f"[警告] 唤醒探测异常: {e}", level=logging.WARNING)
