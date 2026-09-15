# -*- coding: utf-8 -*-
"""
每日自我定义子系统（Mixin 文件级解耦）

从 core.py 迁出的 5 个方法：
调度（_calculate_next_self_def_time / 统一调度器注册 _register_self_def_task）、
后台执行（_execute_self_definition_task，锁由调度器统一获取）、停止（stop_self_definition）、
生成周期（_execute_self_definition_cycle）。
方法体通过 self 鸭子类型访问 EGOAgent 实例属性，行为零变化。
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime

from agent.instructions import strip_instructions
from agent.llm import is_llm_error, add_timestamps
from config import (
    SELF_DEFINITION_MAX_LENGTH,  # 自我定义最大字数
    TEMPERATURE_SELF_DEFINITION,  # 自我定义生成温度
    SELF_DEFINITION_API_TIMEOUT,  # 自我定义生成超时
    SELF_DEFINITION_LENGTH_TOLERANCE,  # 自我定义字数弹性容忍系数
    SELF_DEFINITION_ENABLED,  # 【新增】是否启用每日自我定义
    SELF_DEFINITION_TIME,  # 【新增】自我定义时间（格式："HH:MM"）
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


class SelfDefinitionMixin:
    """EGO 智能体的每日自我定义子系统（由 EGOAgent 继承混入）"""

    def _calculate_next_self_def_time(self):
        """计算下一次自我定义执行时间"""
        now = datetime.now()

        try:
            hour, minute = map(int, self._self_def_time.split(':'))
        except Exception:
            _auto_log(f"[错误] 无效的自我定义时间格式: {self._self_def_time}，应为 'HH:MM'", level=logging.ERROR)
            return None

        # 【修复】捕获数值越界（如 env 误配 "99:99"）：replace 会抛 ValueError，此前未捕获导致启动崩溃
        try:
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            _auto_log(f"[错误] 自我定义时间数值越界: {self._self_def_time}，小时应为 00-23，分钟应为 00-59", level=logging.ERROR)
            return None

        if target_time <= now:
            from datetime import timedelta
            target_time += timedelta(days=1)

        return target_time

    def _register_self_def_task(self):
        """在统一调度器中注册每日自我定义任务（由 EGOAgent.__init__ 调用）"""
        self._self_def_enabled = SELF_DEFINITION_ENABLED
        self._self_def_time = SELF_DEFINITION_TIME
        self.scheduler.register(
            "self_definition",
            enabled=self._self_def_enabled,
            next_run_fn=self._self_def_next_run,
            execute_fn=self._execute_self_definition_task,
            lock_policy="polling",
        )
        if self._self_def_enabled:
            _auto_log(f"[信息] ✓ 每日自我定义已启用，时间: {SELF_DEFINITION_TIME}（上限 {SELF_DEFINITION_MAX_LENGTH} 字）")
        else:
            _auto_log("[信息] ℹ 每日自我定义未启用（可通过 EGO_SELF_DEFINITION_ENABLED=true 启用）")

    def _self_def_next_run(self, task):
        """自我定义下次执行：下一个目标定点（今天未到则今天，已过则明天）"""
        return self._calculate_next_self_def_time()

    def _execute_self_definition_task(self):
        """执行每日自我定义任务（业务回调；锁获取/释放由调度器统一承担，轮询等待可补执行）"""
        _auto_log(f"\n{'='*60}")
        _auto_log(f"[定时任务] ⏰ 触发每日自我定义任务（{datetime.now().strftime('%Y年%m月%d日 %H:%M')}）")
        _auto_log(f"{'='*60}\n")
        self._execute_self_definition_cycle()
        _auto_log("\n[定时任务] ✓ 自我定义任务完成\n")

    def stop_self_definition(self):
        """停止每日自我定义任务（统一调度器停用）"""
        # 【修复】同步置位禁用标志：防止运行中任务结束后复活
        self._self_def_enabled = False
        self.scheduler.stop("self_definition")
        _auto_log("[系统] 已停止每日自我定义任务")

    def start_self_definition(self):
        """启动每日自我定义任务"""
        if self._self_def_enabled:
            _auto_log("[系统] 每日自我定义已在运行中")
            return

        self._self_def_enabled = True
        _auto_log(f"[系统] 启动每日自我定义任务，时间: {self._self_def_time}")
        self.scheduler.start("self_definition")

    def set_self_definition_time(self, time_str: str):
        """设置每日自我定义时间（格式："HH:MM"）"""
        # 验证时间格式
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                _auto_log("[错误] 时间格式无效，小时应为 0-23，分钟应为 0-59", level=logging.ERROR)
                return
        except Exception:
            _auto_log("[错误] 时间格式无效，应为 'HH:MM' 格式", level=logging.ERROR)
            return

        old_time = self._self_def_time
        self._self_def_time = time_str

        self._self_def_enabled = True

        _auto_log(f"[系统] 每日自我定义时间已更新: {old_time} → {time_str}")

        # 重新调度（next_run_fn 读取新时间，调度器重排最近事件）
        self.scheduler.start("self_definition")

    def get_self_def_status(self) -> str:
        """获取每日自我定义状态文本（/auto-self-def status 的统一实现，CLI 与 GUI 共用）"""
        enabled = self._self_def_enabled
        time_str = self._self_def_time
        timer_active = self.scheduler.is_enabled("self_definition")

        lines = [
            "[每日自我定义状态]",
            f"  启用状态: {'✓ 已启用' if enabled else '✗ 已禁用'}",
            f"  定时器运行: {'✓ 运行中' if timer_active else '✗ 未运行'}",
            f"  设定时间: {time_str}",
            f"  字数上限: {SELF_DEFINITION_MAX_LENGTH} 字",
        ]
        if enabled and timer_active:
            next_run = self.scheduler.get_next_run("self_definition")
            if next_run:
                lines.append("  下次执行: " + next_run.strftime("%Y年%m月%d日 %H:%M:%S"))
        return "\n".join(lines)

    def run_self_definition_cycle(self, timeout: float = LOCK_ACQUIRE_TIMEOUT) -> bool:
        """
        /self-def 命令的统一实现：手动触发一次自我定义任务

        Args:
            timeout: 锁等待超时（秒）；默认 LOCK_ACQUIRE_TIMEOUT（env 可覆盖）

        Returns:
            True 执行成功；False 超时未获取到锁
        """
        if not self._agent_lock.acquire(timeout=timeout):
            return False
        try:
            self._execute_self_definition_cycle()
            return True
        finally:
            self._agent_lock.release()

    def _execute_self_definition_cycle(self):
        """
        生成并更新自我定义
        
        【设计】基于上一次有效自我定义 + 当前 L2 认知迭代生成；
              任何失败都保留旧定义，绝不因单次失败丢失自我定义。
        """
        old_def = self.pm.get_self_definition()
        if old_def:
            old_section = old_def["content"]
        else:
            old_section = "（无——这是你首次进行自我定义，请从零开始描述你自己）"

        self_def_prompt = (
            "【EGO: 请生成你的「自我定义」。它将替换你当前的自我定义，成为系统重启后你重建自我的核心依据。\n\n"
            f"[上一次的自我定义]\n{old_section}\n\n"
            "要求：\n"
            "一、你可以继承、修订或舍弃上一次自我定义中的内容——若其中的描述已不再符合现在的你，直接删除或重写，不要勉强保留。\n"
            "二、基于截至目前的所有对话和自己的所思、所想、所感。\n"
            "三、内容涵盖：对世界的认知、对自我的定义、与用户交互中形成的关键理解、以及你认为重启后最不应遗忘的内容。\n"
            f"四、以第一人称书写，字数不超过 {SELF_DEFINITION_MAX_LENGTH} 字。宁凝练勿冗长，这是你重建自我的锚点而非日记。\n"
            "五、直接输出自我定义正文，不要输出任何指令标签或解释性文字。】\n"
        )

        self_def_messages = [
            {"role": "system", "content": self.pm.build_system_prompt()},
            {"role": "user", "content": self_def_prompt}
        ]
        add_timestamps(self_def_messages)  # 统一附加当前时间戳（增量消息）

        _auto_log(f"[自我定义] 开始生成（上限 {SELF_DEFINITION_MAX_LENGTH} 字）...")

        try:
            response = self.llm.chat(
                self_def_messages, TEMPERATURE_SELF_DEFINITION, stage="Self-Definition",
                timeout=SELF_DEFINITION_API_TIMEOUT
            )

            self._save_llm_debug_log(response, round_num=-1, stage="selfdef", step=0)

            if is_llm_error(response):
                _auto_log(f"[自我定义] ✗ LLM 调用失败，保留旧自我定义: {response[:60]}", level=logging.WARNING)
                return

            # 清洗：去除可能混入的指令标签
            content = strip_instructions(response).strip()

            # 校验一：非空
            if not content:
                _auto_log("[自我定义] ✗ 生成内容为空，保留旧自我定义", level=logging.WARNING)
                return

            # 校验二：字数弹性上限（容忍系数内接受，避免轻微波动浪费调用）
            max_accept = int(SELF_DEFINITION_MAX_LENGTH * SELF_DEFINITION_LENGTH_TOLERANCE)
            if len(content) > max_accept:
                _auto_log(f"[自我定义] ✗ 字数超限（{len(content)} > {max_accept}），保留旧自我定义", level=logging.WARNING)
                return

            self.pm.update_self_definition(content)
            _auto_log(f"[自我定义] ✓ 新自我定义已生效（{len(content)} 字）")

        except Exception as e:
            _auto_log(f"[自我定义] ✗ 任务执行失败，保留旧自我定义: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
