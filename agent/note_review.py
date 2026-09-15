# -*- coding: utf-8 -*-
"""
定点审查备忘录子系统（Mixin 文件级解耦）

借鉴定点自省（审查 L2 认知库）的调度模式，每日定点审查备忘录条目：
调度（_calculate_next_note_review_time / 统一调度器注册 _register_note_review_task）、
后台执行（_execute_note_review_task，锁由调度器统一获取）、
启停与状态（stop/start/set/get + 手动入口 run_note_review_cycle）、
审查周期（_execute_note_review_cycle：LLM 输出 NOTE_DEL / NOTE_ADD 白名单指令执行）。
方法体通过 self 鸭子类型访问 EGOAgent 实例属性，行为与既有 mixin 风格一致。
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime

from agent.instructions import parse_instructions, filter_empty_payload
from agent.llm import add_timestamps
from config import (
    AUTO_NOTE_REVIEW_ENABLED,  # 是否启用定点备忘录审查
    AUTO_NOTE_REVIEW_TIME,  # 定点审查时间（格式："HH:MM"）
    TEMPERATURE_NOTE_REVIEW,  # 审查温度
    NOTE_REVIEW_API_TIMEOUT,  # 审查阶段超时（秒）
    LOCK_ACQUIRE_TIMEOUT,  # 手动触发等锁超时（秒）
)


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """
    记录日志（级别由调用方显式指定，默认 INFO；不再靠关键词猜测）

    Args:
        message: 要记录的消息
        level: 日志级别（logging.INFO / WARNING / ERROR）
    """
    logger.log(level, message)


class NoteReviewMixin:
    """EGO 智能体的定点备忘录审查子系统（由 EGOAgent 继承混入）"""

    def _register_note_review_task(self):
        """在统一调度器中注册定点备忘录审查任务（由 EGOAgent.__init__ 调用）"""
        self._note_review_enabled = AUTO_NOTE_REVIEW_ENABLED
        self._note_review_time = AUTO_NOTE_REVIEW_TIME
        self.scheduler.register(
            "note_review",
            enabled=self._note_review_enabled,
            next_run_fn=self._note_review_next_run,
            execute_fn=self._execute_note_review_task,
            lock_policy="polling",
        )
        if self._note_review_enabled:
            _auto_log(f"[信息] ✓ 定点备忘录审查已启用，时间: {AUTO_NOTE_REVIEW_TIME}")
        else:
            _auto_log("[信息] ℹ 定点备忘录审查未启用（可通过 EGO_AUTO_NOTE_REVIEW_ENABLED=true 启用）")

    def _calculate_next_note_review_time(self):
        """计算下一次备忘录审查执行时间"""
        now = datetime.now()

        try:
            hour, minute = map(int, self._note_review_time.split(':'))
        except Exception:
            _auto_log(f"[错误] 无效的备忘录审查时间格式: {self._note_review_time}，应为 'HH:MM'", level=logging.ERROR)
            return None

        # 【修复】捕获数值越界（如 env 误配 "99:99"）：replace 会抛 ValueError，此前未捕获导致启动崩溃
        try:
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            _auto_log(f"[错误] 备忘录审查时间数值越界: {self._note_review_time}，小时应为 00-23，分钟应为 00-59", level=logging.ERROR)
            return None

        if target_time <= now:
            from datetime import timedelta
            target_time += timedelta(days=1)

        return target_time

    def _note_review_next_run(self, task):
        """备忘录审查下次执行：下一个目标定点（今天未到则今天，已过则明天）"""
        return self._calculate_next_note_review_time()

    def _execute_note_review_task(self):
        """执行定点备忘录审查任务（业务回调；锁获取/释放由调度器统一承担，轮询等待可补执行）"""
        _auto_log(f"\n{'='*60}")
        _auto_log(f"[定时任务] ⏰ 触发定点备忘录审查任务（{datetime.now().strftime('%Y年%m月%d日 %H:%M')}）")
        _auto_log(f"{'='*60}\n")
        self._execute_note_review_cycle()
        _auto_log("\n[定时任务] ✓ 定点备忘录审查任务完成\n")

    def stop_note_review(self):
        """停止定点备忘录审查任务（统一调度器停用）"""
        # 【修复】同步置位禁用标志：防止运行中任务结束后复活，且允许 start_note_review 重新启用
        self._note_review_enabled = False
        self.scheduler.stop("note_review")
        _auto_log("[系统] 已停止定点备忘录审查任务")

    def start_note_review(self):
        """启动定点备忘录审查任务"""
        if self._note_review_enabled:
            _auto_log("[系统] 定点备忘录审查已在运行中")
            return

        self._note_review_enabled = True
        _auto_log(f"[系统] 启动定点备忘录审查任务，时间: {self._note_review_time}")
        self.scheduler.start("note_review")

    def set_note_review_time(self, time_str: str):
        """设置定点备忘录审查时间（格式："HH:MM"）"""
        # 验证时间格式
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                _auto_log("[错误] 时间格式无效，小时应为 0-23，分钟应为 0-59", level=logging.ERROR)
                return
        except Exception:
            _auto_log("[错误] 时间格式无效，应为 'HH:MM' 格式", level=logging.ERROR)
            return

        old_time = self._note_review_time
        self._note_review_time = time_str

        self._note_review_enabled = True

        _auto_log(f"[系统] 定点备忘录审查时间已更新: {old_time} → {time_str}")

        # 重新调度（next_run_fn 读取新时间，调度器重排最近事件）
        self.scheduler.start("note_review")

    def get_note_review_status(self) -> str:
        """获取定点备忘录审查状态文本（/auto-note-review status 的统一实现，CLI 与 GUI 共用）"""
        enabled = self._note_review_enabled
        time_str = self._note_review_time
        timer_active = self.scheduler.is_enabled("note_review")

        lines = [
            "[定点备忘录审查状态]",
            f"  启用状态: {'✓ 已启用' if enabled else '✗ 已禁用'}",
            f"  定时器运行: {'✓ 运行中' if timer_active else '✗ 未运行'}",
            f"  设定时间: {time_str}",
        ]
        if enabled and timer_active:
            next_run = self.scheduler.get_next_run("note_review")
            if next_run:
                lines.append("  下次执行: " + next_run.strftime("%Y年%m月%d日 %H:%M:%S"))
        return "\n".join(lines)

    def run_note_review_cycle(self, timeout: float = LOCK_ACQUIRE_TIMEOUT) -> bool:
        """
        /note-review 命令的统一实现：手动触发一次备忘录审查任务

        Args:
            timeout: 锁等待超时（秒）；默认 LOCK_ACQUIRE_TIMEOUT（env 可覆盖）

        Returns:
            True 执行成功；False 超时未获取到锁
        """
        if not self._agent_lock.acquire(timeout=timeout):
            return False
        try:
            self._execute_note_review_cycle()
            return True
        finally:
            self._agent_lock.release()

    def _execute_note_review_cycle(self, stage_label: str = "note-review"):
        """
        执行备忘录审查任务（统一入口，供定时任务和手动命令调用）

        审查全部备忘录条目（含已触发的执行类），找出失效、错误、过时或可合并的内容；
        LLM 输出 NOTE_DEL（删除）/ NOTE_ADD（合并新条目）白名单指令并执行。

        Args:
            stage_label: debug log 的阶段标签（默认 "note-review"）
        """
        entries = self.note.all_entries()

        if not entries:
            _auto_log("[备忘录审查] 备忘录为空，跳过审查")
            return

        _auto_log(f"[备忘录审查] 当前备忘录条目数: {len(entries)}")

        # 构建条目列表（含完整内容，供 LLM 判断失效/合并）
        entries_text = "\n".join(
            f"{i+1}. [{entry.get('id', 'N/A')}] 性质:{entry.get('nature', '')}"
            f" 触发时间:{entry.get('trigger_time') or '无'} 标题:{entry.get('title', '')}"
            f" 内容:{entry.get('content', '')}"
            for i, entry in enumerate(entries)
        )

        review_prompt = (
            "【EGO: 按下述要求审查备忘录条目，找出其中错误或失效的内容。】\n"
            f"当前共有 {len(entries)} 条备忘录：\n\n{entries_text}\n\n"
            #"如果基于截至目前的所有对话和自己的所思、所想、所感，总结成 800 字记忆锚点，使用 <TOOL> [NOTE_ADD] 新备忘事项 </TOOL> 指令添加；\n"
            "如果通过整合多条备忘事项产生更凝练的新事项，使用 <TOOL> [NOTE_ADD] 新备忘事项 </TOOL> 指令添加；\n"
            "如果发现需要删除的条目，使用 <TOOL> [NOTE_DEL] 错误、失效、被合并的备忘事项 ID </TOOL> 指令删除；\n"
            "如果没有需要操作的内容，无需输出任何指令。\n"
            "重要：\n"
            "一、每条指令必须闭合：以开标签开始指令，以对应的闭标签结束指令；指令必须使用大写字母，严禁使用小写字母。\n"
            "  *正确示例*：<TOOL> [NOTE_DEL] N_001 </TOOL>；\n"
            "  *错误示例*：<TOOL> [NOTE_DEL] N_001  ---指令未闭合，应使用闭标签结束指令；\n"
            "  *错误示例*：<TOOL> [NOTE_ADD] 新备忘事项 </tool>  ---指令标签错误的使用小写字母；\n"
            "  *错误示例*：<TOOL> [NOTE_DEL] N_001 <TOOL>  ---指令闭合错误，应使用闭标签结束指令。\n"
            "二、严禁嵌套使用指令：每条指令必须独立使用，不得在其它指令内部嵌套。\n"
            "  *正确示例*：<TOOL> [NOTE_ADD] 新备忘事项 </TOOL> <TOOL> [NOTE_DEL] N_001 </TOOL>  ---各指令独立使用；\n"
            "  *错误示例*：<TOOL> [NOTE_ADD] 新备忘事项 <TOOL> [NOTE_DEL] N_001 </TOOL> </TOOL>  ---错误的在指令内部嵌套指令。\n"
            "三、输出时非真实使用（如仅提及、回顾格式等）某个指令标签时，必须用反引号包裹该指令标签使其失效，严禁裸写指令标签。\n"
            "  *正确示例*：使用前检查指令格式 `<TOOL>` [NOTE_ADD] 新备忘事项 `</TOOL>`；\n"
            "  *错误示例*：使用前检查指令格式 <TOOL> [NOTE_ADD] 新备忘事项 </TOOL>  ---裸写指令标签，应为`<TOOL>`和`</TOOL>`。"
        )

        review_messages = [
            {"role": "system", "content": self.pm.build_system_prompt()},
            {"role": "user", "content": review_prompt}
        ]
        add_timestamps(review_messages)  # 统一附加当前时间戳（增量消息）

        _auto_log(f"[备忘录审查] 开始审查备忘录条目（共 {len(entries)} 条）...\n")

        try:
            review_response = self.llm.chat(
                review_messages, TEMPERATURE_NOTE_REVIEW,
                stage=stage_label, timeout=NOTE_REVIEW_API_TIMEOUT,
            )

            # 保存审查阶段的 LLM 响应到 debug log
            self._save_llm_debug_log(review_response, round_num=-1, stage=stage_label, step=0)

            review_instructions = filter_empty_payload(parse_instructions(review_response))

            success_count = 0
            failed_count = 0

            total_note_del = sum(1 for instr in review_instructions if instr.kind == "NOTE_DEL")
            total_note_add = sum(1 for instr in review_instructions if instr.kind == "NOTE_ADD")
            _auto_log(f"[备忘录审查] 检测到 {total_note_del} 条 NOTE_DEL、{total_note_add} 条 NOTE_ADD 指令，开始执行...")

            for idx, instr in enumerate(review_instructions):
                # 【设计】审查阶段白名单：NOTE_DEL / NOTE_ADD（审查备忘录的核心语义），
                # 其余指令（COG_ADD/COG_DEL/MEMO_RD/NOTE_RD/CONTINUE/SAY 等）在此阶段不执行
                if instr.kind in ("NOTE_DEL", "NOTE_ADD"):
                    try:
                        _auto_log(f"[备忘录审查] [{idx+1}/{len(review_instructions)}] 正在处理: {instr.payload}")
                        result = self.executor.execute(instr)
                        if result.success:
                            success_count += 1
                            _auto_log(f"[备忘录审查] ✓ {instr.kind}: {instr.payload[:50]}{'...' if len(instr.payload) > 50 else ''}")
                        else:
                            failed_count += 1
                            _auto_log(f"[备忘录审查] ⚠ {instr.kind} 失败 ({failed_count}): {result.message}", level=logging.WARNING)
                    except Exception as e:
                        failed_count += 1
                        _auto_log(f"[备忘录审查] ✗ 执行 {instr.kind} 指令时出错: {e}", level=logging.WARNING)
                        _auto_log(traceback.format_exc())
                        continue

            if success_count > 0:
                _auto_log(f"[备忘录审查] ✓ 共处理 {success_count} 条指令")
            if failed_count > 0:
                _auto_log(f"[备忘录审查] ⚠ 共 {failed_count} 条处理失败", level=logging.WARNING)
            if success_count == 0 and failed_count == 0:
                _auto_log("[备忘录审查] ✓ 所有备忘录条目均有效，无需操作")

        except Exception as e:
            _auto_log(f"[备忘录审查] ✗ 审查任务执行失败: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())

        _auto_log("[备忘录审查] ✓ 审查任务完成\n")
