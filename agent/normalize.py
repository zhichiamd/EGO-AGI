# -*- coding: utf-8 -*-
"""
指令归一化层：将自然语言 payload 转译为标准协议文本（FLM e4b，失败透传）

设计要点：
- 幂等：looks_standard() 判定已是标准协议格式的 payload 直接透传，不触发转译
- 降级：服务不可达 / JSON 无效 / 字段校验失败 → 返回 None，调用方透传原 payload（行为=现状）
- 熔断：连续失败 NORM_CIRCUIT_BREAK 次后冷却 NORM_CIRCUIT_COOLDOWN 秒，跳过转译
- prompt：与实测 100% 的强化 prompt v2 逐字一致；few-shot 时间基准随当前时间动态生成
- 时间事实：每次调用动态注入当前时间（YYYY-MM-DD HH:MM + 星期），无需缓存/跨天重建
"""

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta

import requests

from agent.note_memory import TRIGGER_TIME_RE
from config import (
    NORM_ENABLED,
    NORM_API_BASE,
    NORM_MODEL,
    NORM_TEMPERATURE,
    NORM_MAX_TOKENS,
    NORM_TIMEOUT,
    NORM_MIN_LENGTH,
    NORM_CIRCUIT_BREAK,
    NORM_CIRCUIT_COOLDOWN,
)

logger = logging.getLogger("Normalize")

# ── 熔断状态（模块级，execute 可能被多线程调用）────────────────
_circuit = {"fails": 0, "until": 0.0}
_circuit_lock = threading.Lock()

# 【复用】转译用长连接 Session：避免每次请求新建 TCP 连接（连接池复用）。
# 进程存活期间复用；随进程退出回收，不构成泄漏。
_session = requests.Session()


def _circuit_open() -> bool:
    """熔断打开：连续失败达阈值且在冷却期内 → 跳过转译"""
    with _circuit_lock:
        return _circuit["fails"] >= NORM_CIRCUIT_BREAK and time.time() < _circuit["until"]


def _circuit_fail():
    with _circuit_lock:
        _circuit["fails"] += 1
        if _circuit["fails"] >= NORM_CIRCUIT_BREAK:
            _circuit["until"] = time.time() + NORM_CIRCUIT_COOLDOWN
            logger.warning("[归一化] 连续 %d 次失败，熔断 %ds", NORM_CIRCUIT_BREAK, NORM_CIRCUIT_COOLDOWN)


def _circuit_success():
    with _circuit_lock:
        _circuit["fails"] = 0


# ── SYSTEM prompt（与实测 100% 的强化 prompt v2 逐字一致）────────
def _build_system() -> str:
    now = datetime.now()
    weekday = ("一", "二", "三", "四", "五", "六", "日")[now.weekday()]
    tomorrow_9 = (now + timedelta(days=1)).strftime("%Y-%m-%d-09:00")
    half_hour = (now + timedelta(minutes=30)).strftime("%Y-%m-%d-%H:%M")
    return f"""你是指令转译器。当前时间：{now.strftime('%Y-%m-%d %H:%M')}（星期{weekday}）。
你负责把用户的自然语言指令转译为严格的 JSON 输出。只输出 JSON，禁止输出任何解释、注释或 Markdown 围栏。

指令类型与输出格式：
1. MEMO_RD（检索历史对话）：{{"query": "从指令中提取的核心检索词（保留主题词，去掉语气词）", "role": "user"或"ego"}}
   - role=user 表示检索用户说过的话（"用户之前说的/用户说过/用户上次吐槽/用户与我分享/用户和我说过"等）；role=ego 表示检索 EGO 自己说过的话（"我说过的/我曾/我写过/自己说过/EGO上次说的"等）；不限定则省略
2. NOTE_RD（读取备忘录）：{{"target": "备忘录ID（如N_003）、或检索关键词、或 LIST（列出全部/所有）"}}
3. NOTE_ADD（添加备忘录）：{{"nature": "执行"或"备忘", "trigger_time": "YYYY-MM-DD-HH:MM", "title": "简短标题", "content": "详细内容"}}
   - 执行类必须给出 trigger_time，必须是晚于当前时间的绝对时间；相对时间（明早/明天/后天/半小时后/一小时后/周五等）必须基于当前时间精确换算
   - 备忘类 nature 为"备忘"，省略 trigger_time

示例：
输入：指令类型=NOTE_ADD，内容=明早9点提醒我开周会
输出：{{"nature": "执行", "trigger_time": "{tomorrow_9}", "title": "开周会", "content": "明早9点提醒我开周会"}}
输入：指令类型=NOTE_ADD，内容=半小时后提醒我喝水
输出：{{"nature": "执行", "trigger_time": "{half_hour}", "title": "喝水提醒", "content": "半小时后提醒我喝水"}}
输入：指令类型=MEMO_RD，内容=检索用户之前分享过的诗，名字叫《秋夜》
输出：{{"query": "诗，秋夜", "role": "user"}}
输入：指令类型=MEMO_RD，内容=回忆一下我关于诗《秋夜》的理解
输出：{{"query": "诗，秋夜，理解", "role": "ego"}}"""


# ── 幂等判定：标准协议格式直接透传，不触发转译 ────────────────
def looks_standard(kind: str, payload: str) -> bool:
    """
    判定 payload 是否已是标准协议格式。
    - note_add：协议固定带 [性质 字段标记
    - memo_rd：@user/@ego 前缀是协议格式；短文本（< NORM_MIN_LENGTH）视为已归一化
      关键词（如"健身计划"）直通——MEMO_RD 走 embedding 语义检索，短句可模糊命中
    - note_rd：仅纯 ID / LIST 直通；短文本一律转译（note.search 为纯子串匹配，
      "检索秋夜" 等自然语言短句直通必然失配，转译后才可命中）
    """
    s = payload.strip()
    if not s:
        return True
    if kind == "note_add":
        return "[性质" in s
    if kind == "memo_rd":
        return bool(re.match(r"^@(user|ego)\s", s)) or len(s) < NORM_MIN_LENGTH
    if kind == "note_rd":
        return bool(re.match(r"^N_\d+$", s, re.IGNORECASE)) or s.upper() == "LIST"
    return True


# ── JSON 容错提取 ─────────────────────────────────────
def _extract_json(text: str) -> dict:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except Exception:
        return None


# ── 回填：JSON → 标准协议文本（字段校验失败返回 None → 透传）────
def _fill_note_add(obj: dict) -> str:
    nature = obj.get("nature")
    if nature not in ("执行", "备忘"):
        return None
    title = str(obj.get("title") or "").strip()
    content = str(obj.get("content") or "").strip()
    if not title or not content:
        return None
    if nature == "执行":
        trigger = str(obj.get("trigger_time") or "").strip()
        if not TRIGGER_TIME_RE.match(trigger):
            return None
        return f"[性质：执行] [触发时间：{trigger}] [标题:{title}] [内容:{content}]"
    return f"[性质：备忘] [标题:{title}] [内容:{content}]"


def _fill_memo_rd(obj: dict) -> str:
    query = str(obj.get("query") or "").strip()
    if not query:
        return None
    role = obj.get("role")
    if role in ("user", "ego"):
        return f"@{role} {query}"
    return query


def _fill_note_rd(obj: dict) -> str:
    target = str(obj.get("target") or "").strip()
    if not target:
        return None
    return target


_FILLERS = {
    "note_add": _fill_note_add,
    "memo_rd": _fill_memo_rd,
    "note_rd": _fill_note_rd,
}


# ── 转译：自然语言 payload → 标准协议文本（None = 透传原样）────
def translate(kind: str, payload: str) -> str:
    """调用 FLM e4b 将自然语言 payload 转译为标准协议文本；失败返回 None"""
    if not NORM_ENABLED:
        return None
    if _circuit_open():
        logger.warning("[归一化] 熔断冷却中，跳过转译（透传）")
        return None
    try:
        resp = _session.post(
            f"{NORM_API_BASE}/chat/completions",
            json={
                "model": NORM_MODEL,
                "messages": [
                    {"role": "system", "content": _build_system()},
                    {"role": "user", "content": f"指令类型：{kind}\n自然语言指令：{payload}"},
                ],
                "temperature": NORM_TEMPERATURE,
                "max_tokens": NORM_MAX_TOKENS,
            },
            timeout=NORM_TIMEOUT,
        )
        resp.raise_for_status()
        obj = _extract_json(resp.json()["choices"][0]["message"]["content"])
        if obj is None:
            _circuit_fail()
            return None
        filled = _FILLERS[kind](obj)
        if filled is None:
            # 【修复】字段校验失败是模型输出质量问题而非服务故障，不计熔断：
            # 熔断本意是保护不可用服务（网络 / JSON 结构错误已计）；字段问题
            # 会在下次调用由模型自然修正，计入熔断会误伤冷却期内的全部转译
            logger.warning("[归一化] %s 字段校验失败，透传: %r", kind, obj)
            return None
        _circuit_success()
        return filled
    except Exception as e:
        logger.warning("[归一化] %s 转译失败: %s，透传原样", kind, e)
        _circuit_fail()
        return None


def normalize_payload(kind: str, payload: str) -> str:
    """归一化入口：非标准 payload 才转译；失败/未启用返回 None（调用方透传）"""
    if not NORM_ENABLED:
        return None
    if looks_standard(kind, payload):
        return None
    return translate(kind, payload)
