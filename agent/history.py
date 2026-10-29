"""
对话历史持久化

从 core.py 抽出的 HistoryStore：加载（含历史迁移过滤）/ 原子保存 / 追加 / 查询。
统一历史写入口，避免出现「后台任务写成用户对话」这类 stage 误标。

注：logger 沿用 core 的 "EGOAgent" 名 —— EGO_GUI 初始化期按该名字挂日志透传，
加载/迁移告警需要保持原有可见性。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from config import HISTORY_FILE, HISTORY_MAX_ENTRIES

logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    logger.log(level, message)


# 会话重建/初始化注入历史的阶段白名单：仅"与用户的对话"（stage 为 None 或 "chat"）。
# 自对话（think）、备忘录执行（note）、反思（reflection）等内部阶段的输出虽保留在
# history.json 中供追溯，但不注入 LLM 上下文，避免重启/断连重建后内部思考污染对话。
CHAT_HISTORY_STAGES: tuple = (None, "chat")


class HistoryStore:
    """对话历史存储：内存列表 + JSON 文件持久化。

    写入口只有 add() / clear()，两者都立即落盘；stage 为假值时不写该键，
    与既有 history.json 格式保持一致（chat 条目无 stage 字段）。
    读取走 chat_entries / count_chat_user_turns / tail，避免调用方直接操作底层列表。
    """

    def __init__(self, path: str | Path = HISTORY_FILE,
                 max_entries: int = HISTORY_MAX_ENTRIES):
        self._path = Path(path)
        self._max_entries = int(max_entries)
        self._entries: list[dict] = []

    @property
    def path(self) -> Path:
        """历史文件路径（只读）。"""
        return self._path

    # ── 加载 / 保存 ──────────────────────────────────────────────

    def load(self) -> None:
        """从磁盘加载历史，并做两段历史迁移过滤。

        迁移结果不在此处落盘（与原实现一致），下次 add() 时一并写入。
        """
        self._entries = []
        if not self._path.exists():
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = data
            else:
                _auto_log("[警告] history.json 格式错误，已重置", level=logging.WARNING)
        except Exception as e:
            _auto_log(f"[警告] 加载历史失败: {e}，已重置", level=logging.WARNING)
            self._entries = []

        # 兜底过滤：旧版本残留的临时消息（temporary 标记）不再使用，加载时剔除
        # （临时消息机制已由统一事件队列替代，防残留数据被持久化污染 history.json）
        old_temp_count = sum(1 for msg in self._entries if msg.get("temporary", False))
        if old_temp_count:
            self._entries = [m for m in self._entries if not m.get("temporary", False)]
            _auto_log(f"[信息] 已过滤 {old_temp_count} 条旧版临时消息（已被统一事件队列替代）")

        # 迁移：冷启动/预热内容不再写入 history（仅存 coldstart_cache.json），
        # 剔除历史文件中可能残留的该阶段条目，确保重启重建只基于真实对话
        boot_count = sum(1 for m in self._entries if m.get("stage") in ("coldstart", "preheat"))
        if boot_count:
            self._entries = [m for m in self._entries if m.get("stage") not in ("coldstart", "preheat")]
            _auto_log(f"[信息] 已剔除 {boot_count} 条冷启动/预热历史条目（仅保留于 coldstart_cache.json）")

    def save(self) -> None:
        """原子写入历史文件（先写 .tmp 再 os.replace，写入中断不损坏原文件）。"""
        tmp_path = str(self._path) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # 紧凑格式：history.json 随对话增长且每次全量重写，紧凑序列化体积/耗时约减半
                json.dump(self._entries, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, str(self._path))
        except Exception as e:
            _auto_log(f"[警告] 保存历史失败: {e}", level=logging.WARNING)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    # ── 写入口 ──────────────────────────────────────────────────

    def add(self, role: str, content: str, stage: str | None = None) -> dict:
        """追加一条历史并立即落盘；超过上限时丢弃最旧条目。"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if stage:
            entry["stage"] = stage

        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        self.save()
        return entry

    def clear(self) -> None:
        """清空全部历史并立即落盘。"""
        self._entries.clear()
        self.save()

    # ── 读接口 ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def tail(self, n: int) -> list[dict]:
        """返回最近 n 条原始条目（不做阶段过滤，由调用方自行判定）。"""
        if n <= 0:
            return []
        return self._entries[-n:]

    def chat_entries(self, limit: int | None = None,
                     exclude_last: bool = False) -> list[dict]:
        """返回"与用户的对话"（chat 阶段）历史条目。

        跳过 temporary 与 think/note/反思等内部阶段输出（这类条目仍存 history.json
        供追溯，但不注入 LLM 上下文、也不在重启回显中展示）。

        Args:
            limit: 最多返回最近 N 条（先过滤后切片；None 表示不限制，0 表示不返回）
            exclude_last: 排除最后一条（会话中途恢复时排除正在处理的用户输入）
        """
        src = self._entries[:-1] if exclude_last else self._entries
        entries = [
            m for m in src
            if not m.get("temporary", False)
            and m.get("stage") in CHAT_HISTORY_STAGES
            and m.get("role") in ("user", "assistant")
        ]
        if limit is not None:
            entries = entries[-limit:] if limit > 0 else []
        return entries

    def count_chat_user_turns(self) -> int:
        """统计"与用户的对话"轮数（chat 阶段 user 条目数）。

        口径与 chat_entries 一致（白名单 CHAT_HISTORY_STAGES），
        用于 conversation_counter 恢复与自省触发判定。
        """
        return sum(
            1 for m in self._entries
            if m.get("role") == "user" and m.get("stage") in CHAT_HISTORY_STAGES
        )
