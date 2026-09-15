"""
统一后台定时调度器（EGO 三层架构：调度层）

以单个 Timer 驱动全部后台定时任务（自对话 / 自省 / 自我定义 / 备忘录到期），
替代原先 4 套并行的 Timer + 锁等待 + 重调度重复实现。

任务注册模型：
- next_run_fn(task): () -> datetime | None    每次重排时重新计算（支持动态事件，如备忘录到期）
- execute_fn:       () -> None                业务回调（锁获取/释放由调度器统一承担）
- precheck_fn:      (task) -> bool | None     可选预检：到点后先确认本次确有工作要执行，
                                              无待办则不申请锁直接跳过（如备忘录轮询唤醒但无到期条目）
- lock_policy:      "once"    一次性等锁（LOCK_ACQUIRE_TIMEOUT，超时跳过）
                    "polling" 轮询等锁（BACKGROUND_TASK_LOCK_WAIT，可补执行）

执行模型：_tick（Timer 线程）取出所有到期任务 → 串行执行（共用 _agent_lock，
与改造前多 Timer 争同一把锁语义等价）→ 重排最近事件设定下一个 Timer。

停止模型：stop 置禁用标志（防运行中任务结束后复活，沿用既有模式）。
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime


def _auto_log(message: str, level: int = logging.INFO):
    logger = logging.getLogger("EgoScheduler")
    logger.log(level, message)


@dataclass
class _Task:
    name: str
    enabled: bool
    next_run_fn: callable  # (task) -> datetime | None
    execute_fn: callable   # () -> None
    lock_policy: str       # "once" / "polling"
    wait_seconds: int
    precheck_fn: callable = None  # 可选预检：(task) -> bool；False 则不申请锁静默跳过
    last_run: datetime = None  # 最近一次触发时刻（间隔型任务据此计算下次时间）
    next_run: datetime = None  # 最近一次重排计算出的下次执行时间


class EgoScheduler:
    """单 Timer 驱动的统一后台调度器"""

    def __init__(self, agent, lock_acquire_timeout: int = 60,
                 background_wait: int = 900, poll_interval: int = 15):
        self._agent = agent
        self._tasks: dict[str, _Task] = {}
        self._timer = None
        self._timer_lock = threading.Lock()
        self._lock_acquire_timeout = lock_acquire_timeout
        self._background_wait = background_wait
        self._poll_interval = poll_interval

    # ── 注册与查询 ────────────────────────────────────────────

    def register(self, name: str, enabled: bool, next_run_fn, execute_fn,
                 lock_policy: str = "once", wait_seconds: int = None,
                 precheck_fn=None) -> "_Task":
        """注册一个定时任务（next_run_fn 接收 task 参数；
        precheck_fn 可选：到点后先预检，返回 False 则不申请锁直接跳过）"""
        if wait_seconds is None:
            wait_seconds = (self._lock_acquire_timeout if lock_policy == "once"
                            else self._background_wait)
        task = _Task(name, enabled, next_run_fn, execute_fn, lock_policy,
                     wait_seconds, precheck_fn)
        self._tasks[name] = task
        return task

    def is_enabled(self, name: str) -> bool:
        task = self._tasks.get(name)
        return task is not None and task.enabled

    def get_next_run(self, name: str):
        """下次执行时间（datetime | None），供状态查询"""
        task = self._tasks.get(name)
        return task.next_run if task else None

    # ── 启停与重排 ────────────────────────────────────────────

    def start(self, name: str):
        task = self._tasks.get(name)
        if task is not None:
            task.enabled = True
        self.reschedule()

    def stop(self, name: str):
        task = self._tasks.get(name)
        if task is not None:
            task.enabled = False
        self.reschedule()

    def stop_all(self):
        """统一停用全部任务并取消待触发 Timer（退出/整体停用场景使用）。

        原子地 disable 全部任务 + cancel 待触发 Timer：与逐个 stop() 相比，
        避免因任务名拼写不一致残留启用任务，也不会在退出瞬间由残留 _tick 重排出新 Timer。
        """
        with self._timer_lock:
            for task in self._tasks.values():
                task.enabled = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def reschedule(self):
        """重算并重排最近事件（配置热更新 / 任务启停后调用）"""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._schedule_next()

    # ── 核心循环 ──────────────────────────────────────────────

    def _schedule_next(self):
        """计算所有启用任务的下次执行时间，取最近者设定单个 Timer"""
        now = datetime.now()
        best = None
        for task in self._tasks.values():
            if not task.enabled:
                continue
            try:
                nxt = task.next_run_fn(task)
            except Exception as e:
                _auto_log(f"[定时任务] ✗ 任务 {task.name} 计算下次执行时间失败: {e}",
                          level=logging.WARNING)
                continue
            if nxt is None:
                continue
            task.next_run = nxt
            if best is None or nxt < best:
                best = nxt
        if best is None:
            return
        delay = max(0.0, (best - now).total_seconds())
        with self._timer_lock:
            # 【修复】创建新 Timer 前先取消旧 Timer：否则在并发场景下（主线程 reschedule
            # 与运行中的 _tick 同时调用本方法）旧 Timer 引用被覆盖但未被 cancel，
            # 残留 Timer 到期后仍会触发 _tick，导致任务被重复执行
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(delay, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self):
        """Timer 到期：执行所有到期任务（串行），随后重排"""
        now = datetime.now()
        due = [t for t in self._tasks.values()
               if t.enabled and t.next_run is not None and t.next_run <= now]
        for task in due:
            task.last_run = now  # 触发时刻记录（间隔型任务从触发时刻起算，与旧语义一致）
            self._run_task(task)
        self._schedule_next()

    def _run_task(self, task):
        """按任务等锁策略执行（once 一次性超时跳过；polling 轮询等待可补执行）

        预检在前：无待办的任务（如备忘录轮询唤醒但无到期条目）不申请锁，
        避免 LLM 长请求期间无意义等锁超时并产生 WARNING 噪音。
        """
        # 预检：确认本次确有工作要执行（无待办则静默跳过，下轮轮询再来）
        if task.precheck_fn is not None:
            try:
                if not task.precheck_fn(task):
                    return
            except Exception as e:
                _auto_log(f"[定时任务] ✗ 任务 {task.name} 预检失败，跳过本次执行: {e}",
                          level=logging.WARNING)
                return
        acquired = False
        if task.lock_policy == "polling":
            waited = 0
            while waited < task.wait_seconds:
                acquired = self._agent._agent_lock.acquire(timeout=self._poll_interval)
                if acquired:
                    break
                waited += self._poll_interval
                _auto_log(f"[定时任务] ⏳ 任务 {task.name} 等待锁释放中...（已等待 {waited}s / {task.wait_seconds}s）")
        else:
            acquired = self._agent._agent_lock.acquire(timeout=task.wait_seconds)

        if not acquired:
            _auto_log(f"[定时任务] ⚠ 任务 {task.name} 等锁超时，跳过本次执行", level=logging.WARNING)
            return
        try:
            task.execute_fn()
        except Exception as e:
            _auto_log(f"[定时任务] ✗ 任务 {task.name} 执行失败: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
        finally:
            self._agent._agent_lock.release()
