"""
备忘录存储（NOTE 指令的后端）

支持 NOTE_ADD / NOTE_RD / NOTE_DEL 三条 LLM 自主指令：
- 条目分两类：执行类（指定时间触发）/ 备忘类（不自动触发，需主动读取）
- 存储：data/notes.json（运行时自动创建，与 history.json / layer2.json 同级）
- 线程安全；软删除；ID 自增（N_001 格式）

协议文案见 agent/prompts.py LAYER1_DEFAULT.rules「（三）备忘录」，
本模块的字段解析与格式校验与该协议保持一致。
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from config import BASE_DIR

# NOTE_ADD payload 字段解析：[性质：执行] [触发时间：2026-08-19-12:00] [标题:xxx] [内容:xxx]
# 冒号兼容全角（：）与半角（:）；内容允许跨行
# 【修复】字段值前瞻匹配：原 [^\]]* 在 content 含 ] 时截断（如"参考 [任务A] 的清单"），
# 改为匹配到"下一个字段标记或结尾"为止——content 可含任意字符（含 ]）；
# 仅在内容中出现"字段标记形态"文本时截断（原方案遇 ] 必截断，此方案显著改善）
_FIELD_RE = re.compile(
    r"\[(性质|触发时间|标题|内容)\s*[:：]\s*(.*?)(?=\s*\]?\s*\[(?:性质|触发时间|标题|内容)\s*[:：]|\s*\]?\s*$)",
    re.DOTALL,
)
TRIGGER_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}$")


class NoteMemory:
    """备忘录存储（线程安全，软删除，ID 自增 N_001 格式）"""

    def __init__(self, file_path: str = None):
        self.file_path = file_path or str(Path(BASE_DIR) / "data" / "notes.json")
        self._lock = threading.Lock()
        self._notes: list[dict] = []
        self._next_seq = 1
        self._load()

    # ── 持久化 ──────────────────────────────────────────────
    def _load(self):
        """重新加载文件；解析失败时保留现有内存数据（避免读到半写文件清空内存）"""
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            notes = data.get("notes", []) if isinstance(data, dict) else []
            # 只有成功解析才替换内存数据
            self._notes = notes
            # 恢复 ID 自增计数：取现存最大序号
            max_seq = 0
            for n in self._notes:
                m = re.match(r"^N_(\d+)$", n.get("id", ""))
                if m:
                    max_seq = max(max_seq, int(m.group(1)))
            self._next_seq = max_seq + 1
        except Exception:
            # 解析失败保留旧数据（首次加载失败时 _notes 保持初始空列表）
            pass

    def _refresh(self):
        """【修复】外部修改热加载：无条件重新加载文件（notes.json 极小，json.load
        开销可忽略）。不依赖 mtime 检测——Windows 文件系统时间戳存在粒度对齐，
        毫秒级连续写入会得到相同 mtime 而漏检。
        调用方必须已持有 _lock。"""
        self._load()

    def _save(self):
        """原子写入（先写临时文件再替换），避免外部读取到半写文件"""
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        temp_file = self.file_path + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump({"notes": self._notes}, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, self.file_path)

    # ── 工具 ────────────────────────────────────────────────
    @staticmethod
    def parse_trigger_time(text: str) -> datetime | None:
        """解析触发时间为 datetime（容错：标准 YYYY-MM-DD-HH:MM，兼容空格+秒等宽松格式）；
        无法解析返回 None"""
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        for fmt in ("%Y-%m-%d-%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d-%H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def parse_payload(payload: str) -> dict:
        """解析 NOTE_ADD payload → {nature, trigger_time, title, content}"""
        fields = {"性质": "", "触发时间": "", "标题": "", "内容": ""}
        for m in _FIELD_RE.finditer(payload):
            fields[m.group(1)] = m.group(2).strip()
        return {
            "nature": fields["性质"],
            "trigger_time": fields["触发时间"],
            "title": fields["标题"],
            "content": fields["内容"],
        }

    def _valid_notes(self):
        return [n for n in self._notes if n.get("valid", True)]

    @staticmethod
    def _partial(n: dict) -> dict:
        """部分字段（ID、性质、触发时间、标题）——NOTE_RD 关键词/LIST 查询返回"""
        return {k: n.get(k, "") for k in ("id", "nature", "trigger_time", "title")}

    # ── 增删查 ──────────────────────────────────────────────
    def add(self, nature: str, title: str, content: str, trigger_time: str = "") -> dict:
        with self._lock:
            # 【修复】写前刷新：合并外部手改条目后再追加，避免覆盖丢失；
            # _next_seq 同步重算，避免外部新条目与本条目 ID 冲突
            self._refresh()
            note = {
                "id": f"N_{self._next_seq:03d}",
                "nature": nature,
                "trigger_time": trigger_time,
                "title": title,
                "content": content,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "valid": True,
            }
            self._next_seq += 1
            self._notes.append(note)
            self._save()
            return note

    def delete(self, note_id: str) -> tuple:
        """软删除：返回 (是否成功, 原因)"""
        with self._lock:
            # 【修复】写前刷新：先合并外部手改条目，避免 _save 覆盖丢失
            self._refresh()
            for n in self._notes:
                if n.get("id", "").lower() == note_id.strip().lower() and n.get("valid", True):
                    n["valid"] = False
                    self._save()
                    return True, f"已删除备忘录 {n['id']}"
            return False, f"未找到备忘录条目: {note_id}"

    def get(self, note_id: str):
        """按 ID 查完整条目（无效条目返回 None）"""
        with self._lock:
            self._refresh()
            for n in self._notes:
                if n.get("id", "").lower() == note_id.strip().lower() and n.get("valid", True):
                    return dict(n)
            return None

    def search(self, keyword: str) -> list:
        """关键词匹配标题/内容，返回部分字段"""
        kw = keyword.strip().lower()
        with self._lock:
            self._refresh()
            return [self._partial(n) for n in self._valid_notes()
                    if kw in n.get("title", "").lower() or kw in n.get("content", "").lower()]

    def list_all(self) -> list:
        with self._lock:
            self._refresh()
            return [self._partial(n) for n in self._valid_notes()]

    def all_entries(self) -> list:
        """全部有效条目完整字段（定点审查备忘录使用，需含内容供 LLM 判断失效/合并）"""
        with self._lock:
            self._refresh()
            return [dict(n) for n in self._valid_notes()]

    def format(self, notes: list, full: bool = False) -> str:
        """格式化为 LLM 可读文本（full=True 时含内容）"""
        if not notes:
            return "（无备忘录条目）"
        lines = []
        for n in notes:
            t = f" 触发时间:{n.get('trigger_time', '') or '无'}"
            head = f"[{n.get('id', '?')}] 性质:{n.get('nature', '')}{t} 标题:{n.get('title', '')}"
            lines.append(head + (f" 内容:{n.get('content', '')}" if full else ""))
        return "\n".join(lines)

    # ── 执行类到期检查（供定时调度调用）──────────────────────
    def check_due(self, now: datetime = None) -> list:
        """返回已到触发时间、未触发过的执行类条目（时间按 datetime 精确比较，
        格式容错：标准 YYYY-MM-DD-HH:MM 与空格+秒等宽松格式均可正确比较；
        完全无法解析的触发时间按无效条目忽略，避免字符串字典序误判提前触发）"""
        now = now or datetime.now()
        with self._lock:
            self._refresh()
            due = []
            for n in self._valid_notes():
                if n.get("nature") != "执行" or not n.get("trigger_time") or n.get("triggered"):
                    continue
                t = self.parse_trigger_time(n["trigger_time"])
                if t is not None and t <= now:
                    due.append(dict(n))
            return due

    def mark_triggered(self, note_id: str):
        """标记执行类条目已触发（触发后不再进入 check_due，条目仍保留可见）"""
        with self._lock:
            # 【修复】写前刷新：合并外部手改条目后标记，避免 _save 覆盖丢失
            self._refresh()
            for n in self._notes:
                if n.get("id") == note_id:
                    n["triggered"] = True
                    self._save()
                    return
