"""
统一事件注入通道（EGO 三层架构：调度层 / 注入层 / 执行层）

注入层职责：承载所有"需要 LLM 感知"的事件，供 EGO 循环每轮消费。
事件源：
- 执行层（同步）：MEMO_RD / NOTE_RD 检索结果、CONTINUE 继续方向、指令失败反馈
- 调度层（异步）：备忘录到期触发（NOTE_DUE）

事件属性：
- persistent=False（默认）：仅存活于当前 EGO 循环，循环结束时被 cleanup 清理
- persistent=True：跨对话保留（如 NOTE_DUE），等待下一次对话 Round 0 消费

相比旧的"临时消息单槽位（history[-1]）+ 字符串合并"方案：
- 多事件按序消费（drain 即弹出），不再互相挤压或拼接
- Round 0 / 后续轮次消费逻辑统一，不再探测 history 末尾
- 来源带标签注入（SOURCE_LABELS），LLM 不会混淆记忆/备忘录/继续方向
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


# 事件来源 → 注入 LLM 的标签（替代原硬编码 MEMO_RD 前缀）
SOURCE_LABELS = {
    "MEMO_RD": "记忆检索",
    "NOTE_RD": "备忘录检索",
    "CONTINUE": "继续方向",
    "FEEDBACK": "指令失败反馈",
    "NOTE_DUE": "备忘录触发",
}


@dataclass
class EgoEvent:
    source: str
    content: str
    persistent: bool = False


class EventBus:
    """线程安全事件队列（调度线程 push / 主线程 drain 消费）"""

    def __init__(self):
        self._events = deque()
        self._lock = threading.Lock()

    def push(self, source: str, content: str, persistent: bool = False):
        """入队一条事件"""
        with self._lock:
            self._events.append(EgoEvent(source, content, persistent))

    def drain(self) -> list:
        """取出全部事件并清空（消费即弹出，天然避免重复注入）"""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def has_ephemeral(self) -> bool:
        """是否存在非持久事件（EGO 循环据此决定是否继续下一轮）"""
        with self._lock:
            return any(not e.persistent for e in self._events)

    def cleanup(self):
        """清理非持久事件（EGO 循环结束调用；NOTE_DUE 等持久事件保留给下次对话）"""
        with self._lock:
            self._events = deque(e for e in self._events if e.persistent)

    def format_events(self, events: list) -> str:
        """拼装为 LLM 可读文本（每条带来源标签，来源互不混淆）"""
        lines = []
        for e in events:
            label = SOURCE_LABELS.get(e.source, e.source)
            lines.append(f"【{label}】\n{e.content}")
        return "\n".join(lines)
