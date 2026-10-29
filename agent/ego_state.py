"""
EGO 自我对话循环的跨轮运行状态

纯数据结构：只描述状态字段，不含任何业务逻辑，也不依赖 agent 包内其它模块
（供 core.py 与后续拆分出的循环模块共用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _EgoLoopState:
    """
    【重构】EGO 自我对话循环运行状态（跨轮累积）
    """
    output_content: str = ""
    has_output: bool = False
    recall_count: int = 0
    continue_count: int = 0
    consecutive_similar_count: int = 0
    round_num: int = 0
    ego_log: list = field(default_factory=list)
    recent_responses: list = field(default_factory=list)


@dataclass
class _RoundControl:
    """
    【重构】单轮指令执行后产出的控制信号
    """
    should_continue: bool = False
    round_recall_count: int = 0
