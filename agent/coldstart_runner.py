# -*- coding: utf-8 -*-
"""
冷启动/预热执行（ColdstartRunner）

从 session.py 抽出的「执行层」：按 sys.json 的步骤逐条调用 LLM、解析指令、执行记忆类指令、
把消息对追加进缓存缓冲区。

职责边界：
- 本模块只负责「跑第 i 条」；「是否执行冷启动 / 预热」的决策留在 SessionBootstrap（编排层），不下沉。
- 缓冲区不归本模块：统一通过 cache.append_step() 写入，由 ColdstartCache.save() 落盘。
- 冷启动与预热共用应答处理（解析指令 → 执行 → 取历史内容 → 写缓冲区），
  差异项（提示词来源 / 日志前缀 / 起始偏移 / 超时语义 / messages 组装）由调用方参数化。

设计约束：
- 不持有 SessionBootstrap：llm / pm / executor / debug_log / sys_prompts / cache 全部构造注入。
- logger 沿用 "EGOAgent"：EGO_GUI 初始化期按该名字挂日志透传，启动日志必须继续可见。
- 失败语义不变：单步异常记 WARNING 并中止（冷启动会把进度写成 incomplete 供续传）。
"""

from __future__ import annotations

import logging

from agent.instructions import (
    parse_instructions, strip_instructions, filter_empty_payload,
    get_instruction, is_executable as _is_executable,
)
from agent.llm import is_llm_error
from config import TEMPERATURE_COLDSTART, TEMPERATURE_PREHEAT


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """记录日志（级别由调用方显式指定，默认 INFO）"""
    logger.log(level, message)


def _safe_int_key(key):
    """sys.json 步骤键排序键：无法解析为数字的键排最后（防止用户编辑文件导致启动崩溃）"""
    try:
        return int(key)
    except (ValueError, TypeError):
        return float("inf")


class ColdstartRunner:
    """冷启动与预热阶段的逐条执行（sys.json 驱动的 LLM 步骤）。"""

    def __init__(self, llm, pm, executor, debug_log, sys_prompts, cache):
        """
        Args:
            llm: LMStudioClient（唯一出口：chat）
            pm: PromptManager（system prompt 与 L2 认知）
            executor: InstructionExecutor（记忆类指令执行，带阶段门控）
            debug_log: LLMDebugLogger（原始响应落盘）
            sys_prompts: sys.json 配置（coldstart / Preheat 两段）
            cache: ColdstartCache（缓冲区写入 + incomplete 进度落盘）
        """
        self.llm = llm
        self.pm = pm
        self.executor = executor
        self.debug_log = debug_log
        self.sys_prompts = sys_prompts
        self.cache = cache

    # ── 冷启动 / 预热 ─────────────────────────────────────────
    def run_coldstart(self, start_from: int = 1) -> bool:
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
                    self.cache.save(status="incomplete", failed_step=i, resume_from=start_from)
                    return False
            
                self._consume_response(
                    response, prompt, stage="coldstart", step=i,
                    tag="[冷启动]", with_failure_message=True,
                )
                _auto_log(f"[冷启动] ✓ Step {i} 完成")
            
            except Exception as e:
                _auto_log(f"[冷启动] ✗ Step {i} 失败: {e}", level=logging.WARNING)
                self.cache.save(status="incomplete", failed_step=i, resume_from=start_from)
                return False
    
        _auto_log("[冷启动] ✓ 冷启动全部完成")
        return True

    def run_preheat(self, use_incremental: bool = True):
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
            
                self._consume_response(response, prompt, stage="preheat", step=i,
                                       tag="[预热]")
            except Exception as e:
                _auto_log(f"[预热]   ✗ 执行失败: {e}", level=logging.WARNING)
    
        _auto_log("[预热] ✓ 预热完成")

    # ── 单步应答处理（冷启动/预热共用）─────────────────────────
    def _consume_response(self, response: str, prompt: str, *, stage: str, step: int,
                          tag: str, with_failure_message: bool = False):
        """解析单步应答：落 debug → 执行嵌套/其余指令 → 写入缓存缓冲区

        Args:
            response: LLM 原始响应
            prompt: 本步提示词（与应答构成写入缓存的消息对）
            stage: "coldstart" / "preheat"（debug 归档名 + 执行上下文 + 缓冲区归属）
            step: 步骤号（1 基）
            tag: 日志前缀（"[冷启动]" / "[预热]"），保证两条路径日志逐字一致
            with_failure_message: 指令失败时是否附带 result.message（冷启动记录、预热不记录；
                属既有差异，保留以免改变线上日志文本）
        """
        # 保存 debug log
        self.debug_log.save(response, round_num=-1, stage=stage, step=step)

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
                        self.executor.set_context({"stage": stage, "step": step})
                        result = self.executor.execute(n_instr)
                        if result.skipped:
                            continue  # 阶段门控：检索类指令本阶段不执行，静默跳过
                        if result.success:
                            _auto_log(f"{tag}   ✓ 执行嵌套指令: {n_instr.kind}")
                # 清理 SAY payload 中的嵌套指令标签
                history_content = strip_instructions(instr.payload).strip()
                break
        else:
            # 没有 SAY 指令时，使用 strip 后的内容
            history_content = strip_instructions(response).strip()

        # 【调整】冷启动/预热内容只写 coldstart_cache.json，不写 history.json
        self.cache.append_step(stage, prompt, history_content)

        # 执行非 SAY 指令（空载荷指令整条跳过）
        for instr in filter_empty_payload(instructions):
            if not _is_executable(instr.kind):
                continue

            self.executor.set_context({"stage": stage, "step": step})

            result = self.executor.execute(instr)
            if result.skipped:
                continue  # 阶段门控：检索类指令本阶段不执行，静默跳过
            if result.success:
                _auto_log(f"{tag}   ✓ 执行指令: {instr.kind}")
            else:
                detail = f" - {result.message}" if with_failure_message else ""
                _auto_log(f"{tag}   ✗ 指令执行失败: {instr.kind}{detail}", level=logging.WARNING)
