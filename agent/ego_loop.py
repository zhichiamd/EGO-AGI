# -*- coding: utf-8 -*-
"""
EGO 自我对话循环（EgoLoopRunner）

从 core.py 抽出的循环引擎：增量消息构建 → 多轮 LLM 调用 → 指令解析/排序/执行 →
SAY 拼接与去重检测 → 输出定稿。chat / think / note 三个阶段共用同一套循环。

设计约束：
- 不持有 EGOAgent：llm / events / executor / hist / debug_log 及两个回调
  （add_to_history / recover_session）全部构造注入，因此可脱离主控类单独构造与测试。
- logger 沿用 "EGOAgent"：EGO_GUI 初始化期按该名字挂日志透传，循环日志必须继续可见。
- 循环状态（_EgoLoopState / _RoundControl）来自 agent.ego_state，纯数据、无业务逻辑。
- 「已尝试恢复」标记随循环走（属于循环内部状态）；调用方在每次新对话开始前调用
  reset_recovery_flag()，与 think/note 路径不重置的既有行为保持一致。
"""

from __future__ import annotations

import logging
import re

from agent.ego_state import _EgoLoopState, _RoundControl
from agent.instructions import (
    parse_instructions, strip_instructions,
    is_invalid_content, is_blank_text, INVALID_PLACEHOLDER_PATTERNS,
    filter_empty_payload, get_instruction, is_executable,
)
from agent.llm import add_timestamps, is_llm_error
from config import (
    MAX_EGO_ROUNDS,  # 循环最大轮数
    MAX_CONSECUTIVE_SIMILAR,  # 连续相似响应重复循环阈值
    CONSECUTIVE_SIMILAR_MIN_HISTORY,  # 连续相似检测最小历史条数
    CONSECUTIVE_SIMILAR_SHORT_LENGTH,  # 连续相似检测短消息阈值
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
)


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """记录日志（级别由调用方显式指定，默认 INFO）"""
    logger.log(level, message)


class EgoLoopRunner:
    """EGO 自我对话循环：多轮调用 LLM、执行指令、拼接 SAY、去重与输出定稿。"""

    def __init__(self, llm, events, executor, hist, debug_log,
                 add_to_history, recover_session, mem=None):
        """
        Args:
            llm: LMStudioClient（chat / session_id / temperature）
            events: EventBus（注入层事件通道：检索结果 / 继续方向 / 失败反馈）
            executor: InstructionExecutor（单条指令执行）
            hist: HistoryStore（对话前重复检测读取最近条目）
            debug_log: LLMDebugLogger（原始响应落盘）
            add_to_history: callable(role, content)，重复检测命中时写错误消息
            recover_session: callable() -> bool，previous_response_id 失效时重建会话
            mem: ChromaMemoryManager（检索引用计数回写；None 表示不可用）
        """
        self.llm = llm
        self.events = events
        self.executor = executor
        self.hist = hist
        self.debug_log = debug_log
        self.add_to_history = add_to_history
        self.recover_session = recover_session
        self.mem = mem

        # 本次对话是否已尝试过 ID 失效恢复（防止同一轮反复重建）
        self._recovered_from_id_failure = False

    # ── 失效恢复标记 ──────────────────────────────────────────

    def reset_recovery_flag(self) -> None:
        """复位「已尝试恢复」标记：由调用方在每次新对话开始前调用"""
        self._recovered_from_id_failure = False

    def _recover(self) -> bool:
        """
        previous_response_id 失效恢复：置位标记后委托重建。

        返回 True 表示重建成功（调用方应重试本轮请求）。标记随循环状态保存在
        本类：chat 路径每轮复位，think/note 路径不复位（沿用既有行为）。
        """
        self._recovered_from_id_failure = True
        return self.recover_session()

    # ── 循环体 ────────────────────────────────────────────────
    def run(self, user_input: str, images: list, start_time: float,
                      stage: str = "chat",
                      base_temperature_override: float | None = None,
                      step: int = 0) -> _EgoLoopState:
        """
        【重构】EGO 自我对话循环：多轮调用 LLM 并解析执行指令，
        直至产生 SAY 输出、出错或达到轮数上限。返回循环终态。

        Args:
            user_input: 用户输入（round 0 注入；自主触发场景为到期条目提示文本）
            images: 可选图像列表（round 0 使用）
            start_time: 本轮开始时间（MEMO_RD 检索时间过滤基准）
            stage: 运行阶段标签（"chat"=用户对话；"think"/"note"=自主触发）。
                   同时作为指令执行的阶段上下文，检索类指令按其 allowed_stages 门控
            base_temperature_override: 【新增】覆盖温度基准（如 Think 传 TEMPERATURE_THINK）；
                   None=沿用 self.llm.temperature（用户对话温度）
            step: 【新增】步骤序号（写入非 chat 阶段 debug 日志文件名，如 Think 的提示词序号）
        """
        state = _EgoLoopState()
        base_temperature = (
            base_temperature_override
            if base_temperature_override is not None
            else self.llm.temperature
        )

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
                        and not self._recovered_from_id_failure
                        and self._recover()):
                    # 【修复】内联重试本轮：continue 会使 round_num 推进到 1 而跳过 round 0 重试。
                    # 重建上下文已排除当前用户输入，这里通过 round 0 消息重发一次（含时间戳与图片）
                    _auto_log("[信息] Session 重建完成，重试 round 0 请求...")
                    response = self.llm.chat(messages, current_temperature, stage=f"{llm_stage_prefix}-Round-{round_num + 1}-retry")
                    response = self._normalize_llm_response(response)

                # 重试仍失败 / 其他情况：直接返回错误
                if is_llm_error(response):
                    state.output_content = response
                    break

            # 【新增】保存 LLM 原始响应到 debug log（step 透传，供 Think 等按提示词序号命名）
            self.debug_log.save(response, round_num, stage=stage, step=step)

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

            # 记录 EGO 自我对话（清理内部标记后留作循环上下文，不写入 history）
            clean_ego_content = self._clean_ego_content(pure_text, response)
            if clean_ego_content:
                state.ego_log.append(clean_ego_content)

            # 执行指令
            control = self._execute_round_instructions(
                state, instructions, round_num, user_input, start_time, stage=stage)

            # 【修复】末轮轮尾补检测：轮首检测需要 2 条历史响应，而 MAX_EGO_ROUNDS=2 时
            # round 1 轮首仅 1 条，轮首检测永不触发（MAX_EGO_ROUNDS=4 时最后一对响应同样漏检）。
            # 此处等价于"不存在的下一轮轮首"，且本轮指令已执行、SAY 已捕获，
            # 触发时可保留已生成内容或回退提示语。与轮首检测互不重叠，不会双重计数
            if round_num == MAX_EGO_ROUNDS - 1 and self._check_loop_repetition(state):
                break

            # 【修复】防止检索类指令无限循环（MEMO_RD / NOTE_RD / WEB_SRCH 共用同一预算：
            # 单轮上限 MEMO_RD_ROUND_LIMIT、全程上限 MEMO_RD_TOTAL_LIMIT）
            # 但如果已经有有效的 SAY 内容，优先输出，不强制中断
            if (control.round_recall_count > MEMO_RD_ROUND_LIMIT or state.recall_count > MEMO_RD_TOTAL_LIMIT) and not state.has_output:
                state.output_content = "[提示] 检索次数过多，已停止检索。"
                _auto_log("[警告] 检索次数超限且无有效 SAY，使用占位符", level=logging.WARNING)
                break
            elif control.round_recall_count > MEMO_RD_ROUND_LIMIT or state.recall_count > MEMO_RD_TOTAL_LIMIT:
                _auto_log("[警告] 检索次数超限，但已有 SAY 内容，继续输出", level=logging.WARNING)
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
                # 【精简】无 SAY 时不再从裸文本兜底提取：直接结束循环，由调用方按
                # has_output 判定——主对话判为无输出（回滚会话链 + 固定文案），
                # note 路径判失败（保留未触发，下轮重试），Think 路径静默跳过。
                # 思考流只作为循环上下文（ego_log），绝不充当用户回复。
                break
        # 【新增】循环结束清理循环内事件（防止泄漏到下次对话）；NOTE_DUE 等持久事件保留
        self.events.cleanup()

        return state

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

    def _execute_round_instructions(self, state: _EgoLoopState, instructions: list,
                                    round_num: int, user_input: str,
                                    start_time: float, stage: str = "chat") -> _RoundControl:
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
                        if is_executable(n_instr.kind):
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
                # 【重构】阶段由调用方透传（chat/think/note），使检索类指令的
                # allowed_stages 门控名副其实（此前硬编码 "chat"，Think 无法回注）
                "stage": stage,
                "round_number": round_num,
                "user_input": user_input if round_num == 0 else None,
                "recall_after": start_time,
            })

            result = self.executor.execute(instr)

            status = "✓" if result.success and not result.skipped else ""
            executed_instrs.append(f"{instr.kind}({status})")

            # 【修复】route_back / inject_result 等属性改为独立判断（原 if-elif 互斥链导致
            # MEMO_RD/NOTE_RD 同时声明 route_back+inject_result 时 inject_result 分支成为死代码，
            # 检索结果全部丢失；独立 if 后各分支按需触发，结果统一入队事件通道）
            if _spec is not None and _spec.inject_result and result.success and not result.skipped:
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
        【重构】检索类指令（MEMO_RD/NOTE_RD/WEB_SRCH）结果处理：检索结果入队事件通道
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
                    if cited_ids and self.mem is not None and instr.kind == "MEMO_RD":
                        self.mem.increment_citations(cited_ids)
                except Exception as e:
                    _auto_log(f"[警告] 引用计数回写失败: {e}，不影响本轮对话", level=logging.WARNING)

                # 【调试】输出每条检索结果的详细内容
                # （WEB_SRCH 外部结果字段可能与记忆不同，数值缺失时降级为原样输出，避免格式化异常）
                for i, r in enumerate(results, 1):
                    mem_type = r.get("memory_type", "?")
                    content = r.get("content", "")
                    score = r.get("score", 0)
                    sim = r.get("similarity", 0)
                    role = r.get("role", "")
                    score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                    sim_txt = f"{sim:.0%}" if isinstance(sim, (int, float)) else str(sim)
                    _auto_log(f"[调试]   {instr.kind} [{i}/{len(results)}] 类型={mem_type} 综合={score_txt} 相似度={sim_txt} 角色={role}")
                    _auto_log(f"[调试]     内容: {content[:200]}{'...' if len(content) > 200 else ''}")

            else:
                # 【修复】检索执行成功但结果为空：仍须将"未找到"提示注入下一轮。
                # LLM 发起 MEMO_RD/NOTE_RD 时可能处于等待检索结果的挂起状态（未输出 SAY），
                # 若不告知空结果，下一轮只剩通用"继续刚才的思考"占位，模型无从收尾（空转）。
                # 同时计入检索次数，受 MEMO_RD_ROUND_LIMIT / MEMO_RD_TOTAL_LIMIT 约束，
                # 防止 LLM 反复发起空检索死循环。
                control.round_recall_count += 1
                state.recall_count += 1
                # WEB_SRCH 的 message 本身已含"未找到与「X」相关的联网结果"，再套前缀会重复关键词；
                # 其余检索指令沿用"类型 + 关键词 + 原因"的拼装格式
                if instr.kind == "WEB_SRCH":
                    empty_msg = result.message
                else:
                    empty_msg = f"{instr.kind} 检索「{instr.payload[:50]}」结果为空：{result.message}"
                _auto_log(f"[调试] ℹ {empty_msg}（空结果提示已注入下一轮）")
                self.events.push(instr.kind, empty_msg)
        else:
            _auto_log(f"[调试] ✗ {instr.kind} 指令执行失败: {result.message}", level=logging.WARNING)

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

    def check_recent_repetition(self) -> str | None:
        """
        【重构】对话前重复内容检测：检查历史中最近 3 条 assistant 消息
        是否全为占位符或高度重复；命中则写入历史并返回系统错误消息，否则返回 None
        """
        if len(self.hist) < CONSECUTIVE_SIMILAR_MIN_HISTORY:  # 至少需要 3 轮对话（每轮 2 条消息）
            return None

        # 【修复】只统计真实对话输出（排除 think/note 等自主阶段），避免其正常文本
        # 混入窗口稀释占位符/重复检测的灵敏度
        recent_assistant_messages = [
            msg["content"] for msg in self.hist.tail(CONSECUTIVE_SIMILAR_MIN_HISTORY)
            if msg["role"] == "assistant" and msg.get("stage") in (None, "chat")
        ]

        if len(recent_assistant_messages) < 3:
            return None

        # 检查最后 3 条 assistant 消息是否相似或都是占位符
        last_three = recent_assistant_messages[-3:]


        # 【修复】检测是否都是占位符模式
        # 只有当消息很短且主要由占位符组成时，才判定为无效输出
        # 复用统一占位符常量（is_invalid_content），并追加此处特有的历史回退消息
        placeholder_patterns = INVALID_PLACEHOLDER_PATTERNS + ['（我正在整理思路...）', '（EGO 无输出……）']

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
            self.add_to_history("assistant", error_msg)
            return error_msg

        # 【修复】检测是否高度相似（简单相似度检查）
        # 只有在三条完全相同且长度较短（可能是占位符）时才触发
        if len(set(last_three)) == 1 and len(last_three[0].strip()) <= CONSECUTIVE_SIMILAR_SHORT_LENGTH:
            _auto_log("[警告] ✗ 检测到连续 3 轮相同短输出（可能是占位符），强制停止", level=logging.WARNING)
            error_msg = "[系统错误] 检测到重复输出，请尝试重新提问或刷新对话。"
            self.add_to_history("assistant", error_msg)
            return error_msg

        return None

    def finalize_output(self, output_content: str) -> str | None:
        """
        【重构】输出定稿：移除指令标记并压缩空行。
        清理后为空时返回 None，由调用方按各自语义收尾（纯计算，无副作用）。

        【精简】不再从 ego_log / 裸响应兜底提取对话内容：主对话无 SAY 时由
        `_process_input_inner` 的 has_output 闸门统一判为无输出并回滚会话链，
        note 路径据此判失败并在下轮重试；内部思考只留在会话链上，不作为回复。
        """
        # 清理输出内容：移除指令标记（SAY 载荷内可能嵌有嵌套指令）
        output_content = strip_instructions(output_content)

        if not output_content or not output_content.strip():
            # 走到这里只可能是 SAY 载荷内全是嵌套指令，清理后无可见内容
            _auto_log("[警告] ✗ SAY 载荷清理后为空（内容全为嵌套指令标记）", level=logging.WARNING)
            return None

        output_content = re.sub(r'\n{3,}', '\n\n', output_content)  # 压缩多余空行
        return output_content.strip()

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

            return clean_ego_content.strip()

        return ""
