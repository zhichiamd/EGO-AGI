# -*- coding: utf-8 -*-
"""
会话重建与链头校验（SessionRebuilder）

从 session.py 抽出：previous_response_id 失效恢复、重建消息构建、分批重建与上下文降级、
口令 canary 校验。

设计约束：
- 不持有 SessionBootstrap：llm / pm / hist / cache 全部构造注入。
- 链头只归 llm：本模块只用 llm.get_session_id() / set_session_id() 读写，不缓存副本；
  口令校验走 apply_session=False，只有校验通过才接管新链头，失败不接管、交调用方清空。
- logger 沿用 "EGOAgent"：GUI 初始化期按该名字挂日志透传，启动日志必须继续可见。
- recover_session 是 EgoLoopRunner 的注入回调，签名固定为 () -> bool，不得改动。
"""

from __future__ import annotations

import logging
import time

from agent.history import CHAT_HISTORY_STAGES
from agent.llm import is_llm_error
from config import (
    TEMPERATURE_WARMUP,  # 系统侧低温度调用（重建）
    SESSION_INIT_HISTORY_COUNT,  # 重建注入的最近对话条数
    SESSION_REBUILD_BATCH_SIZE,  # 重建每批消息数
    SESSION_ID_PROBE_MAX_RETRIES,  # previous_response_id 探测重试次数
    SESSION_ID_PROBE_TIMEOUT,  # 探测请求超时（秒）
    SESSION_ID_PROBE_MAX_TOKENS,  # 探测输出 token 上限
    SESSION_CANARY_TOKEN,  # 口令探测期望应答口令
    SESSION_CANARY_ENABLED,  # 口令探测总开关
    SESSION_REBUILD_POST_DELAY,  # 重建后等待间隔（秒）
)


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """记录日志（级别由调用方显式指定，默认 INFO）"""
    logger.log(level, message)


def _is_context_overflow(result: str) -> bool:
    """判断重建失败是否为上下文超限（引擎拒绝特征：context/exceeds/tokens）"""
    low = result.lower()
    return "context" in low or "exceeds" in low or "tokens" in low


class SessionRebuilder:
    """Session 重建、上下文超限降级与 previous_response_id 口令校验。"""

    def __init__(self, llm, pm, hist, cache):
        """
        Args:
            llm: LMStudioClient（chat / 链头读写）
            pm: PromptManager（system prompt）
            hist: HistoryStore（重建时注入最近 N 条 chat 历史）
            cache: ColdstartCache（读取冷启动缓存 + 回写新链头）
        """
        self.llm = llm
        self.pm = pm
        self.hist = hist
        self.cache = cache

    # ── 运行期失效恢复 ────────────────────────────────────────
    def recover_session(self) -> bool:
        """
        【重构】previous_response_id 失效恢复：清除旧 ID 并基于冷启动缓存重建 Session

        返回 True 表示重建成功（调用方应重试本轮请求）；无可用缓存时返回 False。
        调用方（EGOAgent）负责置位“已尝试恢复”标记：该标记属于对话循环状态。
        """
        _auto_log("[警告] ⚠ 首轮请求失败，可能是 previous_response_id 已失效", level=logging.WARNING)
        _auto_log("[信息] 尝试清除旧 ID 并重建 Session...")

        # 清除旧 ID
        old_id = self.llm.get_session_id()
        self.llm.set_session_id(None)
        _auto_log(f"[信息]   - 已清除旧 ID: {(old_id or 'None')[:20]}...")

        # 重建 Session（包含对话历史）
        cache = self.cache.load()
        if cache:
            # 【修复】重建时排除当前正在处理的用户输入：重建完成后 round 0 会重发该输入，避免重复注入
            rebuild_msgs = self.build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=True)
            if self.rebuild_session(rebuild_msgs, cache=cache, exclude_current_input=True):
                _auto_log("[信息] ✓ Session 重建完成，重试请求...")
                return True
            _auto_log("[警告] Session 重建失败，无法恢复本次请求", level=logging.WARNING)
            return False
        else:
            _auto_log("[警告] 无可用缓存，无法重建", level=logging.WARNING)
            return False

    def build_rebuild_messages(self, cache: dict, include_recent_history: bool = False, include_coldstart: bool = False, history_count: int = None, exclude_current_input: bool = False) -> list[dict]:
        """
        构建用于重建 Session 的消息序列
    
        Args:
            cache: 冷启动缓存数据
            include_recent_history: 是否包含最近 N 条用户对话（当 previous_response_id 失效时需要）
            include_coldstart: 是否包含冷启动/预热历史消息（默认 False；其结晶已在 L2 中，重启重建无需回灌）
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
        if include_recent_history and len(self.hist):
            limit = history_count if history_count is not None else SESSION_INIT_HISTORY_COUNT
            # 【修复】先过滤（仅 chat 阶段）后切片：上限表示"最近 N 条用户对话"，不被内部阶段挤占额度
            recent_history = self.hist.chat_entries(limit=limit, exclude_last=exclude_current_input)
        
            added = 0
            for msg in recent_history:
                if msg.get("temporary", False):
                    continue
                # 【调整】仅注入与用户的对话（chat）阶段，与启动注入口径统一；
                # 自对话（think）、备忘录执行（note）、反思等内部阶段输出不注入；
                # 冷启动/预热由上方 cache 通道注入（include_coldstart=True）时一并跳过，避免重复。
                if msg.get("stage") not in CHAT_HISTORY_STAGES:
                    continue
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
                    added += 1
        
            _auto_log(f"[信息]   - 最近对话: {added} 条（上限: {limit}）")
    
        _auto_log(f"[信息]   - 总计: {len(messages)} 条")
    
        return messages

    def rebuild_session(self, messages: list[dict], cache: dict = None, restore_previous_id: str = None, exclude_current_input: bool = False) -> bool:
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
                            messages = self.build_rebuild_messages(
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
                            messages = self.build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=exclude_current_input)
                            retried_with_history = True
                            i = 0
                            continue
                    elif cache and not retried_with_history:
                        # 没有旧 ID 但首批仍然失败（可能是 LM Studio 不可用）
                        # 尝试包含对话历史重建（最后一次尝试）
                        _auto_log("[警告] 重建首批失败，尝试包含最近对话重建", level=logging.WARNING)
                        messages = self.build_rebuild_messages(cache, include_recent_history=True, exclude_current_input=exclude_current_input)
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
        if _sid and self.validate_previous_response_id(_sid):
            # 【统一调用】校验探针走 chat() 并已推进链头：日志与后续请求都必须用
            # 实际生效的链头（_sid 只是探针发出前的旧 ID）
            _live = self.llm.get_session_id() or _sid
            _auto_log(f"[信息] ✓ Session 重建完成，previous_response_id: {_live[:20]}...")
            # 链头已更新 → 立即回写缓存，避免下次启动用已失效的旧 ID 重新验证/重建
            self.cache.update_session_id()
            return True
        _auto_log("[警告] ⚠ Session 重建后验证失败，上下文可能未完整恢复", level=logging.WARNING)
        self.llm.set_session_id(None)
        return False

    def validate_previous_response_id(self, response_id: str) -> bool:
        """
        验证 previous_response_id 是否仍然有效（口令 canary 语义校验）
    
        发送一个轻量探测请求，检查服务器是否仍保留该 Session 上下文。
        如果 LMStudio 已重启，服务器端会话会被清空：此时可能返回 200 但上下文已丢失（静默失效）。
        改用 L1 约定的“口令”作 canary：探测输入不含答案，应答与 SESSION_CANARY_TOKEN 一致
        （即答对 L1 口令，改配置即同步改判据）即等价于上下文仍可达；
        关闭 canary（SESSION_CANARY_ENABLED=false）时回落为仅判请求是否可用。

        【统一调用】必须走 LMStudioClient.chat（主会话唯一出口）：探测前先把待验证的
        ID 设为当前链头，探测才会携带它；apply_session=False 保证未通过校验前不接管
        新链头（静默失效场景下新链不可信），通过后再显式接管 last_response_id。
    
        Args:
            response_id: 待验证的 previous_response_id
        
        Returns:
            bool: ID 是否有效
        """
        # 探测必须跑在待验证的链上：先接管为当前链头，chat() 才会携带它
        # （失败由调用方清空，本方法不额外接管任何新链头）
        self.llm.set_session_id(response_id)

        canary_messages = [{"role": "user", "content": "【系统: 口令？】"}]
        max_retries = SESSION_ID_PROBE_MAX_RETRIES

        for attempt in range(1, max_retries + 1):
            # 【统一调用】走主会话唯一出口：自动携带 previous_response_id；
            # apply_session=False 保证“未通过校验前不接管新链头”
            response = self.llm.chat(
                canary_messages,
                temperature=0.0,
                stage="canary",
                timeout=SESSION_ID_PROBE_TIMEOUT,
                max_tokens=SESSION_ID_PROBE_MAX_TOKENS,  # 【F2】旧实现=1 会截断口令
                apply_session=False,
            )

            if is_llm_error(response):
                if attempt < max_retries:
                    _auto_log(f"[警告] previous_response_id 验证失败 (第{attempt}次)，重试中: {str(response)[:60]}",
                              level=logging.WARNING)
                    continue
                _auto_log(f"[警告] previous_response_id 验证失败 {max_retries} 次，视为失效: {str(response)[:60]}",
                          level=logging.WARNING)
                return False

            new_id = self.llm.last_response_id
            # 逃生开关：关闭 canary 时回落为原“仅判可用”逻辑
            if not SESSION_CANARY_ENABLED:
                if new_id:
                    self.llm.set_session_id(new_id)
                _auto_log("[信息] ✓ previous_response_id 验证通过（canary 已关闭）")
                return True
            # 口令 canary 语义校验：模型答出口令，才认定上下文仍可达
            # （流式路径累积所有 delta，reasoning 段文本同样会被扫到，不会因此漏判）
            if new_id and SESSION_CANARY_TOKEN in (response or ""):
                self.llm.set_session_id(new_id)  # 仅通过时才接管新链头
                _auto_log("[信息] ✓ previous_response_id 验证通过（口令探测）")
                return True
            _auto_log("[警告] previous_response_id 口令探测未通过（上下文可能已清空）", level=logging.WARNING)
            return False  # 不接管 id，交调用方清空后重建

        return False
