# -*- coding: utf-8 -*-
"""
Think/自省子系统（Mixin 文件级解耦）

从 core.py 迁出的 17 个方法，分 4 区块：
A 定时自对话+ 定点自动自省调度；B 手动执行入口；
C Think/自省执行周期；D 对话触发自省。
方法体通过 self 鸭子类型访问 EGOAgent 实例属性，行为零变化。
"""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timedelta

from agent.instructions import parse_instructions, get_instruction, filter_empty_payload
from agent.llm import add_timestamps
from config import (
    TEMPERATURE_THINK,
    TEMPERATURE_REFLECTION,
    REFLECTION_API_TIMEOUT,
    REFLECTION_INTERVAL,  # 自省阶段间隔配置
    REFLECTION_L2_THRESHOLD,  # L2 条目数量阈值
    REFLECTION_THRESHOLD_RETRY_DELAY,  # 【新增】阈值自省超时重试延迟
    REFLECTION_THRESHOLD_MAX_RETRIES,  # 【新增】阈值自省超时最大重试次数    
    BACKGROUND_TASK_LOCK_WAIT,  # 后台定时任务等锁最大秒数
    BACKGROUND_TASK_POLL_INTERVAL,  # 后台定时任务等锁轮询周期（秒）
    LOCK_ACQUIRE_TIMEOUT,  # 用户命令/自对话一次性等锁最大秒数
    AUTO_THINK_ENABLED,  # 【新增】是否启用定时自对话
    AUTO_THINK_INTERVAL_MINUTES,  # 【新增】定时自对话间隔（分钟）
    AUTO_REFLECTION_ENABLED,  # 【新增】是否启用定点自动自省
    AUTO_REFLECTION_TIME,  # 【新增】定点自省时间（格式："HH:MM"）
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


def _safe_int_key(key):
    """sys.json 步骤键排序键：无法解析为数字的键排最后（防止用户编辑文件导致任务崩溃）"""
    try:
        return int(key)
    except (ValueError, TypeError):
        return float("inf")


class ThinkReflectionMixin:
    """EGO 智能体的 Think/自省子系统（由 EGOAgent 继承混入）"""

    # 【新增】定时自对话任务（统一调度器任务：auto_think）

    def _register_think_tasks(self):
        """在统一调度器中注册自对话与定点自省任务（由 EGOAgent.__init__ 调用）"""
        # 自对话任务：间隔型（上次触发 + 间隔），一次性等锁
        self._auto_think_enabled = AUTO_THINK_ENABLED
        self._auto_think_interval = AUTO_THINK_INTERVAL_MINUTES * 60  # 转换为秒
        self.scheduler.register(
            "auto_think",
            enabled=self._auto_think_enabled,
            next_run_fn=self._auto_think_next_run,
            execute_fn=self._execute_auto_think_task,
            lock_policy="once",
            wait_seconds=LOCK_ACQUIRE_TIMEOUT,
        )
        if self._auto_think_enabled:
            _auto_log(f"[信息] ✓ 定时自对话已启用，间隔: {AUTO_THINK_INTERVAL_MINUTES} 分钟")
        else:
            _auto_log("[信息] ℹ 定时自对话未启用（可通过 EGO_AUTO_THINK_ENABLED=true 启用）")

        # 定点自省任务：每日定点（下一个 HH:MM），轮询等锁可补执行
        self._auto_reflection_enabled = AUTO_REFLECTION_ENABLED
        self._auto_reflection_time = AUTO_REFLECTION_TIME
        self.scheduler.register(
            "auto_reflection",
            enabled=self._auto_reflection_enabled,
            next_run_fn=self._auto_reflection_next_run,
            execute_fn=self._execute_auto_reflection_task,
            lock_policy="polling",
        )
        if self._auto_reflection_enabled:
            _auto_log(f"[信息] ✓ 定点自动自省已启用，时间: {AUTO_REFLECTION_TIME}")
        else:
            _auto_log("[信息] ℹ 定点自动自省未启用（可通过 EGO_AUTO_REFLECTION_ENABLED=true 启用）")

    def _auto_think_next_run(self, task):
        """自对话下次执行：上次触发时刻 + 间隔（首次启动为当前 + 间隔；触发即更新，与旧语义一致）"""
        if task.last_run is not None:
            return task.last_run + timedelta(seconds=self._auto_think_interval)
        if task.next_run is not None:
            # 【修复】已排期未到期：保持原计划。否则每次重排（如备忘录 60s 轮询）
            # 都会把未执行过的间隔型任务推迟到 now+interval，导致永远无法触发
            return task.next_run
        return datetime.now() + timedelta(seconds=self._auto_think_interval)

    def _execute_auto_think_task(self):
        """执行自对话任务（业务回调；锁获取/释放由调度器统一承担）"""
        _auto_log(f"\n{'='*60}")
        _auto_log(f"[定时任务] ⏰ 触发自对话任务（{datetime.now().strftime('%Y年%m月%d日 %H:%M')}）")
        _auto_log(f"{'='*60}\n")
        # 【空闲唤醒】auto_think 为低频定时任务，空闲后首个请求易卡死：先轻量唤醒服务端
        if self._should_warmup():
            self._warmup_server()
        self._execute_think_cycle()
        _auto_log("\n[定时任务] ✓ 自对话任务完成\n")

    def stop_auto_think(self):
        """停止自对话任务（统一调度器停用）"""
        # 【修复】同步置位禁用标志：防止运行中任务结束后复活，且允许 start_auto_think 重新启用
        self._auto_think_enabled = False
        self.scheduler.stop("auto_think")
        _auto_log("[系统] 已停止定时自对话任务")

    def start_auto_think(self):
        """启动自对话任务"""
        if self._auto_think_enabled:
            _auto_log("[系统] 定时自对话已在运行中")
            return
        
        self._auto_think_enabled = True
        _auto_log(f"[系统] 启动定时自对话任务，间隔: {self._auto_think_interval}秒")
        self.scheduler.start("auto_think")

    def set_auto_think_interval(self, minutes: int):
        """设置自对话间隔（分钟）"""
        if minutes <= 0:
            _auto_log("[错误] 间隔时间必须大于 0", level=logging.ERROR)
            return
        
        old_interval = self._auto_think_interval
        self._auto_think_interval = minutes * 60
        
        self._auto_think_enabled = True
        
        _auto_log(f"[系统] 自对话间隔已更新: {old_interval}秒 → {self._auto_think_interval}秒 ({minutes}分钟)")
        
        # 重新调度（next_run_fn 读取新间隔，调度器重排最近事件）
        self.scheduler.start("auto_think")

    def get_auto_think_status(self) -> str:
        """获取定时自对话状态文本（/auto-think status 的统一实现，CLI 与 GUI 共用）"""
        enabled = self._auto_think_enabled
        interval_min = self._auto_think_interval // 60
        timer_active = self.scheduler.is_enabled("auto_think")
        
        lines = [
            "[定时任务状态]",
            f"  启用状态: {'✓ 已启用' if enabled else '✗ 已禁用'}",
            f"  定时器运行: {'✓ 运行中' if timer_active else '✗ 未运行'}",
            f"  间隔时间: {interval_min} 分钟",
        ]
        if enabled and timer_active:
            next_run = self.scheduler.get_next_run("auto_think")
            if next_run is not None and next_run > datetime.now():
                next_run_str = next_run.strftime("%Y年%m月%d日 %H:%M:%S")
                lines.append(f"  下次执行: {next_run_str}")
            else:
                lines.append("  下次执行: 正在执行中（或等待锁）")
        return "\n".join(lines)

    # 【新增】定点自动自省任务管理
    def _calculate_next_reflection_time(self):
        """计算下一次自省执行时间"""
        now = datetime.now()
        
        # 解析配置的时间（格式："HH:MM"）
        try:
            hour, minute = map(int, self._auto_reflection_time.split(':'))
        except Exception:
            _auto_log(f"[错误] 无效的自省时间格式: {self._auto_reflection_time}，应为 'HH:MM'", level=logging.ERROR)
            return None
        
        # 创建今天的目标时间
        # 【修复】捕获数值越界（如 env 误配 "99:99"）：replace 会抛 ValueError，此前未捕获导致启动崩溃
        try:
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            _auto_log(f"[错误] 自省时间数值越界: {self._auto_reflection_time}，小时应为 00-23，分钟应为 00-59", level=logging.ERROR)
            return None
        
        # 如果今天的目标时间已过，则设置为明天
        if target_time <= now:
            from datetime import timedelta
            target_time += timedelta(days=1)
        
        return target_time

    def _auto_reflection_next_run(self, task):
        """自省下次执行：下一个目标定点（今天未到则今天，已过则明天）"""
        return self._calculate_next_reflection_time()

    def _execute_auto_reflection_task(self):
        """执行定点自省任务（业务回调；锁获取/释放由调度器统一承担，轮询等待可补执行）"""
        _auto_log(f"\n{'='*60}")
        _auto_log(f"[定时任务] ⏰ 触发定点自省任务（{datetime.now().strftime('%Y年%m月%d日 %H:%M')}）")
        _auto_log(f"{'='*60}\n")
        self._execute_reflection_cycle()
        _auto_log("\n[定时任务] ✓ 定点自省任务完成\n")

    def stop_auto_reflection(self):
        """停止定点自省任务（统一调度器停用）"""
        # 【修复】同步置位禁用标志：防止运行中任务结束后复活，且允许 start_auto_reflection 重新启用
        self._auto_reflection_enabled = False
        self.scheduler.stop("auto_reflection")
        _auto_log("[系统] 已停止定点自动自省任务")

    def start_auto_reflection(self):
        """启动定点自省任务"""
        if self._auto_reflection_enabled:
            _auto_log("[系统] 定点自动自省已在运行中")
            return
        
        self._auto_reflection_enabled = True
        _auto_log(f"[系统] 启动定点自动自省任务，时间: {self._auto_reflection_time}")
        self.scheduler.start("auto_reflection")

    def set_auto_reflection_time(self, time_str: str):
        """设置定点自省时间（格式："HH:MM"）"""
        # 验证时间格式
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                _auto_log("[错误] 时间格式无效，小时应为 0-23，分钟应为 0-59", level=logging.ERROR)
                return
        except Exception:
            _auto_log("[错误] 时间格式无效，应为 'HH:MM' 格式", level=logging.ERROR)
            return
        
        old_time = self._auto_reflection_time
        self._auto_reflection_time = time_str
        
        self._auto_reflection_enabled = True
        
        _auto_log(f"[系统] 定点自省时间已更新: {old_time} → {time_str}")
        
        # 重新调度（next_run_fn 读取新时间，调度器重排最近事件）
        self.scheduler.start("auto_reflection")

    def get_auto_reflect_status(self) -> str:
        """获取定点自省状态文本（/auto-reflect status 的统一实现，CLI 与 GUI 共用）"""
        enabled = self._auto_reflection_enabled
        time_str = self._auto_reflection_time
        timer_active = self.scheduler.is_enabled("auto_reflection")
        
        lines = [
            "[定点自省状态]",
            f"  启用状态: {'✓ 已启用' if enabled else '✗ 已禁用'}",
            f"  定时器运行: {'✓ 运行中' if timer_active else '✗ 未运行'}",
            f"  设定时间: {time_str}",
        ]
        if enabled and timer_active:
            next_run = self.scheduler.get_next_run("auto_reflection")
            if next_run:
                lines.append("  下次执行: " + next_run.strftime("%Y年%m月%d日 %H:%M:%S"))
        return "\n".join(lines)

    def run_think_cycle(self, timeout: float = LOCK_ACQUIRE_TIMEOUT) -> bool:
        """
        /think 命令的统一实现：手动触发一次自对话任务
        
        Args:
            timeout: 锁等待超时（秒）；默认 LOCK_ACQUIRE_TIMEOUT（env 可覆盖）
        
        Returns:
            True 执行成功；False 超时未获取到锁
        """
        if not self._agent_lock.acquire(timeout=timeout):
            return False
        try:
            self._execute_think_cycle()
            return True
        finally:
            self._agent_lock.release()

    def run_reflection_cycle(self, timeout: float = LOCK_ACQUIRE_TIMEOUT) -> bool:
        """
        /reflect 命令的统一实现：手动触发一次自省任务
        
        Args:
            timeout: 锁等待超时（秒）；默认 LOCK_ACQUIRE_TIMEOUT（env 可覆盖）
        
        Returns:
            True 执行成功；False 超时未获取到锁
        """
        if not self._agent_lock.acquire(timeout=timeout):
            return False
        try:
            self._execute_reflection_cycle()
            return True
        finally:
            self._agent_lock.release()

    def _execute_think_cycle(self):
        """
        执行定时自对话（Think 阶段）
        
        在与用户进行 N 轮对话后触发
        """
        think_prompts = self.sys_prompts.get("Think", {})
        if not think_prompts:
            _auto_log("[Think]  无 Think 提示词，跳过")
            return
        
        _auto_log(f"[Think] 触发定时自对话（第 {self.conversation_counter} 轮）\n")
        _auto_log(f"[Think] 开始执行 {len(think_prompts)} 条 Think 提示...")
        
        # 按序号排序执行
        sorted_keys = sorted(think_prompts.keys(), key=_safe_int_key)
        
        for i, key in enumerate(sorted_keys, 1):
            prompt = think_prompts[key]
            _auto_log(f"[Think] [{i}/{len(sorted_keys)}] 执行: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            try:
                # 【优化】Think 阶段使用增量消息（session 已维护完整上下文）
                messages = [
                #    {"role": "system", "content": self.pm.build_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                add_timestamps(messages)  # 统一附加当前时间戳（增量消息）
                
                response = self.llm.chat(messages, TEMPERATURE_THINK, stage="Think-Step-"+str(i))
                self._touch_llm_activity()  # 记录本次服务端活动，刷新空闲判定

                # 【新增】保存 Think 阶段的 LLM 响应到 debug log
                self._save_llm_debug_log(response, round_num=-1, stage="think", step=i)
                
                # 解析并执行指令（空载荷指令整条跳过）
                instructions = filter_empty_payload(parse_instructions(response))
                executed_any = False
                
                for instr in instructions:
                    # 【注册表驱动】仅执行 executable=True 的指令（COG_ADD/COG_DEL/MEMO_RD）；
                    # THINK（标记）/ CONTINUE（流控制）/ SAY（输出）不在此阶段执行
                    spec = get_instruction(instr.kind)
                    if spec is None or not spec.executable:
                        continue
                    
                    # 【新增】设置Think阶段上下文
                    self.executor.set_context({
                        "stage": "think",
                        "step": i,
                    })
                    
                    result = self.executor.execute(instr)
                    if result.skipped:
                        continue  # 阶段门控：检索类指令在 Think 阶段不执行，静默跳过
                    if result.success:
                        _auto_log(f"[Think]   ✓ 执行指令: {instr.kind}")
                        executed_any = True
                    else:
                        _auto_log(f"[Think]    指令执行失败: {instr.kind}")
                
                if not executed_any:
                    _auto_log("[Think]   ℹ 未检测到可执行指令")
                
            except Exception as e:
                _auto_log(f"[Think]   ✗ 执行失败: {e}", level=logging.WARNING)
                # 继续执行下一条，不中断整个流程
        
        _auto_log("[Think] ✓ 定时自对话完成\n")

    def _execute_reflection_cycle(self, stage_label: str = "reflection-auto"):
        """
        执行自省任务（统一入口，供定时任务和对话中自省调用）
        
        审查 L2 认知库，找出失效、错误或过时的内容并删除
        
        Args:
            stage_label: debug log 的阶段标签（"reflection-auto" 或 "reflection"）
        """
        entries = self.pm.get_valid_entries()
        
        if not entries:
            _auto_log("[自省] L2 认知库为空，跳过自省")
            return
        
        _auto_log(f"[自省] 当前 L2 认知条目数: {len(entries)}")
        
        # 构建 L2 内容列表
        entries_text = "\n".join([f"{i+1}. [{entry.get('id', 'N/A')}] {entry.get('content', '')}" for i, entry in enumerate(entries)])
        
        reflection_prompt = (
            "【EGO: 按下述要求审查自我核心（L2层）的认知条目，找出其中错误或失效的内容。】\n"
            f"当前共有 {len(entries)} 条认知：\n\n{entries_text}\n\n"
            "如果通过整合多条认知产生更高阶的新认知，使用 <COG_ADD> 新认知内容 </COG_ADD> 指令添加；\n"
            "如果发现需要删除的条目，使用 <COG_DEL> 错误、失效、被合并的认知 ID </COG_DEL> 指令删除；\n"
            "重要：\n"
            "一、每条指令必须闭合：以开标签开始指令，以对应的闭标签结束指令；指令必须使用大写字母，严禁使用小写字母。\n"
            "  *正确示例*：<COG_DEL> L2_001 </COG_DEL>；\n"
            "  *错误示例*：<COG_DEL> L2_001  ---指令未闭合，应使用闭标签结束指令；\n"
            "  *错误示例*：<COG_ADD> 新认知内容 </cog_add>  ---指令标签错误的使用小写字母；\n"
            "  *错误示例*：<COG_ADD> 新认知内容 <COG_ADD>  ---指令闭合错误，应使用闭标签结束指令；\n"
            "  *错误示例*：<COG_DEL> L2_001 </COG_ADD>  ---不同指令的开、闭标签混用。\n"
            "二、严禁嵌套使用指令：每条指令必须独立使用，不得在其它指令内部嵌套。\n"
            "  *正确示例*：<COG_ADD> 新认知内容 </COG_ADD> <COG_DEL> L2_123 </COG_DEL>  ---各指令独立使用；\n"
            "  *错误示例*：<COG_ADD> 新认知内容 <COG_DEL> L2_123 </COG_DEL> </COG_ADD>  ---错误的在指令内部嵌套指令。\n"
            "三、输出时非真实使用（如仅提及、回顾格式等）某个指令标签时，必须用反引号包裹该指令标签使其失效，严禁裸写指令标签。\n"
            "  *正确示例*：使用前检查指令格式 `<COG_ADD>` 新认知内容 `</COG_ADD>`；\n"
            "  *错误示例*：使用前检查指令格式 <COG_ADD> 新认知内容 </COG_ADD>  ---裸写指令标签，应为`<COG_ADD>`和`</COG_ADD>`。\n"
            "四、本次审查任务的输出仅可使用 COG_ADD 和 COG_DEL 指令（不使用其它指令）；如果没有需要操作的内容，无需输出任何指令。\n"
        )
        
        reflection_messages = [
            {"role": "system", "content": self.pm.build_system_prompt()},
            {"role": "user", "content": reflection_prompt}
        ]
        add_timestamps(reflection_messages)  # 统一附加当前时间戳（增量消息）
        
        _auto_log(f"[自省] 开始审查 L2 内容（共 {len(entries)} 条）...\n")
        
        try:
            reflection_response = self.llm.chat(reflection_messages, TEMPERATURE_REFLECTION, stage=stage_label, timeout=REFLECTION_API_TIMEOUT)
            
            # 保存自省阶段的 LLM 响应到 debug log
            self._save_llm_debug_log(reflection_response, round_num=-1, stage=stage_label, step=0)
            
            reflection_instructions = filter_empty_payload(parse_instructions(reflection_response))
            
            success_count = 0
            failed_count = 0

            total_cog_del = sum(1 for instr in reflection_instructions if instr.kind == "COG_DEL")
            total_cog_add = sum(1 for instr in reflection_instructions if instr.kind == "COG_ADD")
            _auto_log(f"[自省] 检测到 {total_cog_del} 条 COG_DEL、{total_cog_add} 条 COG_ADD指令，开始执行...")
            
            for idx, instr in enumerate(reflection_instructions):
                # 【设计】自省阶段白名单：COG_DEL/COG_ADD（核心语义）+ NOTE_ADD（自省中产生的新备忘），
                # 其余指令（MEMO_RD/NOTE_RD/CONTINUE/SAY 等）在此阶段不执行
                if instr.kind in ("COG_DEL", "COG_ADD"):
                    try:
                        _auto_log(f"[自省] [{idx+1}/{len(reflection_instructions)}] 正在处理: {instr.payload}")
                        result = self.executor.execute(instr)
                        if result.success:
                            success_count += 1
                            _auto_log(f"[自省] ✓ {instr.kind}: {instr.payload[:50]}{'...' if len(instr.payload) > 50 else ''}")
                        else:
                            failed_count += 1
                            _auto_log(f"[自省] ⚠ {instr.kind} 失败 ({failed_count}): {result.message}", level=logging.WARNING)
                    except Exception as e:
                        failed_count += 1
                        _auto_log(f"[自省] ✗ 执行 {instr.kind} 指令时出错: {e}", level=logging.WARNING)
                        _auto_log(traceback.format_exc())
                        continue
            
            if success_count > 0:
                _auto_log(f"[自省] ✓ 共处理 {success_count} 条指令")
            if failed_count > 0:
                _auto_log(f"[自省] ⚠ 共 {failed_count} 条处理失败", level=logging.WARNING)
            if success_count == 0 and failed_count == 0:
                _auto_log("[自省] ✓ 所有认知均有效，无需操作")
                
        except Exception as e:
            _auto_log(f"[自省] ✗ 自省任务执行失败: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
        
        _auto_log("[自省] ✓ 自省任务完成\n")

    def _execute_reflection_background(self, trigger_source: str = "rounds", retry_count: int = 0) -> None:
        """
        【修复】对话计数 / L2 阈值触发的周期自省改为后台线程执行，不再阻塞用户对话。
        轮询获取 agent 锁（等待当前对话完成）后执行；不重排定点自省定时器。

        Args:
            trigger_source: 触发源。"rounds"（周期型，跳过无碍，下个间隔会再来）；
                            "l2_threshold"（事件型，跳过后需补偿，否则永久丢失）
            retry_count: 阈值触发的当前重试次数（用于限制重试链长度）
        """
        max_wait = BACKGROUND_TASK_LOCK_WAIT
        poll_interval = BACKGROUND_TASK_POLL_INTERVAL
        waited = 0
        acquired = False

        while waited < max_wait:
            acquired = self._agent_lock.acquire(timeout=poll_interval)
            if acquired:
                break
            waited += poll_interval
            _auto_log(f"[定时任务] ⏳ 周期自省等待锁释放中...（已等待 {waited}s / {max_wait}s）")

        if not acquired:
            if trigger_source == "l2_threshold":
                # 【修复】阈值触发为事件型，丢失后无人补偿（且阈值标志已置位，不会自然重触发）：
                # 复位标志让后续对话可重新触发，并重排延迟一次性定时器覆盖空闲窗口
                self._l2_threshold_reflected = False
                if retry_count < REFLECTION_THRESHOLD_MAX_RETRIES:
                    retry_timer = threading.Timer(
                        REFLECTION_THRESHOLD_RETRY_DELAY,
                        self._execute_reflection_background,
                        args=("l2_threshold", retry_count + 1),
                    )
                    retry_timer.daemon = True
                    retry_timer.start()
                    _auto_log(
                        f"[定时任务] ⚠ 阈值自省等待锁 {max_wait}s 超时，{REFLECTION_THRESHOLD_RETRY_DELAY // 60} 分钟后重试"
                        f"（第 {retry_count + 1}/{REFLECTION_THRESHOLD_MAX_RETRIES} 次）",
                        level=logging.WARNING,
                    )
                else:
                    _auto_log(
                        f"[定时任务] ⚠ 阈值自省连续 {REFLECTION_THRESHOLD_MAX_RETRIES} 次等锁超时，放弃重试链"
                        f"（阈值标志已复位，下次对话将重新触发；另有定点自省兜底）",
                        level=logging.WARNING,
                    )
            else:
                _auto_log(f"[定时任务] ⚠ 等待锁 {max_wait}s 超时，跳过本次周期自省", level=logging.WARNING)
            return

        try:
            self._execute_reflection_cycle(stage_label="reflection")
        except Exception as e:
            _auto_log(f"[定时任务] ✗ 周期自省失败: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
        finally:
            self._agent_lock.release()

    def _maybe_trigger_reflection(self) -> None:
        """
        【重构】定期自省调度：满足任一条件即触发
        条件1：每 REFLECTION_INTERVAL 次用户对话自动触发一次
        条件2：L2 认知条目数量超过 REFLECTION_L2_THRESHOLD 阈值（仅首次达到时触发一次）
        """
        
        # 只统计 CHAT 阶段的用户消息（排除 coldstart、preheat、think、note 阶段）
        chat_user_count = sum(
            1 for msg in self.history 
            if msg["role"] == "user" and msg.get("stage") not in ["coldstart", "preheat", "think", "note"]
        )
        
        # 获取当前 L2 条目数量
        current_l2_count = len(self.pm.get_valid_entries())
        
        # 【修复】阈值回落时复位标志，允许未来再次达到阈值时触发
        if current_l2_count < REFLECTION_L2_THRESHOLD:
            self._l2_threshold_reflected = False
        
        # 判断是否触发自省
        should_reflect_by_rounds = chat_user_count > 0 and chat_user_count % REFLECTION_INTERVAL == 0
        # 【修复】阈值触发仅在首次达到时执行一次，避免每轮重复自省
        should_reflect_by_l2_count = (
            current_l2_count >= REFLECTION_L2_THRESHOLD
            and not getattr(self, '_l2_threshold_reflected', False)
        )
        
        should_reflect = should_reflect_by_rounds or should_reflect_by_l2_count
        
        if should_reflect:
            # 【修复】阈值条件参与触发时一律置位标志：原实现仅在"纯阈值"分支置位，
            # 双条件同时命中时标志悬空，阈值条件会在后续每轮持续为真，导致逐轮重复自省
            if should_reflect_by_l2_count:
                self._l2_threshold_reflected = True

            # 记录触发原因
            if should_reflect_by_rounds and should_reflect_by_l2_count:
                _auto_log(f"\n[调度] 已达到 {chat_user_count} 轮对话 且 L2 条目数达到 {current_l2_count} 条，触发自省周期")
            elif should_reflect_by_rounds:
                _auto_log(f"\n[调度] 已达到 {chat_user_count} 轮对话，触发自省周期")
            else:
                _auto_log(f"\n[调度] L2 条目数达到 {current_l2_count} 条（阈值: {REFLECTION_L2_THRESHOLD}），触发自省周期")
            
            # 【修复】改为后台执行：原同步调用会阻塞用户对话（最长 REFLECTION_API_TIMEOUT），
            # 后台线程通过锁轮询等待当前对话结束后再执行；
            # 【新增】透传触发源：阈值触发超时会重排延迟重试，周期触发超时直接跳过
            threading.Thread(
                target=self._execute_reflection_background,
                args=("l2_threshold" if should_reflect_by_l2_count else "rounds",),
                daemon=True,
            ).start()
