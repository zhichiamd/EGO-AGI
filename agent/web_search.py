# -*- coding: utf-8 -*-
"""
联网检索层：WEB_SRCH 指令的 Tavily REST 实现（requests 直连，零额外依赖）

设计要点：
- 降级：未启用 / 未配 Key / 鉴权失败 / 配额耗尽 / 网络异常 → 返回 (False, 人话原因, [])，
        由指令层以失败反馈回注 LLM 供其自愈，不抛异常、不影响主流程
- 形态：与 agent/normalize.py 对齐——模块级长连接 Session 复用；条数/深度/超时走 config
- 输出：message 为"供 LLM 阅读的格式化文本"（可选综述 + 标题/来源/摘要），摘要按
        WEB_SRCH_SNIPPET_MAX_LENGTH 截断，且 include_raw_content=False，避免上下文 token 爆炸
- 综述：由 WEB_SRCH_INCLUDE_ANSWER 控制（默认 false 不请求；basic/advanced 才请求），
        该字段由 Tavily 侧 LLM 生成，会额外占用注入 token，故默认关闭
"""

import logging

import requests

from config import (
    WEB_SRCH_ENABLED,
    TAVILY_API_KEY,
    WEB_SRCH_API_BASE,
    WEB_SRCH_MAX_RESULTS,
    WEB_SRCH_SEARCH_DEPTH,
    WEB_SRCH_INCLUDE_ANSWER,
    WEB_SRCH_TIMEOUT,
    WEB_SRCH_SNIPPET_MAX_LENGTH,
)

logger = logging.getLogger("WebSearch")

# 【复用】长连接 Session：避免每次检索新建 TCP/TLS 连接（与 normalize 层保持一致）
_session = requests.Session()


def _truncate(text: str, limit: int) -> str:
    """按字符数截断长文本（超长追加省略号）；limit<=0 表示不截断"""
    text = (text or "").strip()
    if limit > 0 and len(text) > limit:
        return text[:limit] + "……"
    return text


def search(query: str):
    """执行一次联网检索。

    Returns:
        (ok, message, results)
        - ok=True  : message 为供 LLM 阅读的格式化结果（无命中时说明"未找到"），
                     results 为结构化条目列表（无命中则为空列表）
        - ok=False : message 为失败原因（人话），results 恒为空列表
        结构化条目字段与记忆检索结果对齐（id/content/score/similarity/role/memory_type），
        便于复用 core 侧统一的检索结果事件处理路径。
    """
    query = (query or "").strip()
    if not query:
        return False, "联网检索需要提供检索关键词。", []

    if not WEB_SRCH_ENABLED:
        return False, "联网检索功能未启用（EGO_WEB_SRCH_ENABLED=false）。", []
    if not TAVILY_API_KEY:
        return False, "联网检索未配置 API Key（请在 untitled.env 设置 EGO_TAVILY_API_KEY）。", []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": WEB_SRCH_MAX_RESULTS,
        "search_depth": WEB_SRCH_SEARCH_DEPTH,
        # 官方三态：False / "basic" / "advanced"；默认 false 不请求综述（省 token 与延迟）
        "include_answer": WEB_SRCH_INCLUDE_ANSWER,
        "include_raw_content": False,  # 原始正文体积巨大，禁止拉取，防止上下文爆炸
    }

    try:
        resp = _session.post(WEB_SRCH_API_BASE, json=payload, timeout=WEB_SRCH_TIMEOUT)
    except requests.exceptions.Timeout:
        logger.warning("[联网检索] 超时（%ss）：%s", WEB_SRCH_TIMEOUT, query)
        return False, f"联网检索超时（超过 {WEB_SRCH_TIMEOUT} 秒），可稍后重试或更换关键词。", []
    except requests.exceptions.RequestException as e:
        logger.warning("[联网检索] 请求失败：%s", e)
        return False, f"联网检索请求失败（网络或服务不可达）：{e}", []

    if resp.status_code == 401:
        return False, "联网检索鉴权失败（401），请检查 EGO_TAVILY_API_KEY 是否正确。", []
    if resp.status_code == 429:
        return False, "联网检索配额已用尽或请求过于频繁（429），请稍后再试。", []
    if resp.status_code != 200:
        return False, f"联网检索服务返回异常状态码 {resp.status_code}。", []

    try:
        data = resp.json()
    except ValueError:
        return False, "联网检索返回内容不是合法 JSON，无法解析。", []
    if not isinstance(data, dict):
        return False, "联网检索返回结构异常，无法解析。", []

    items = data.get("results") or []
    if not items:
        logger.info("[联网检索] 无命中：%s", query)
        return True, f"未找到与「{query}」相关的联网结果。", []

    answer = (data.get("answer") or "").strip()
    results = []
    entries = []
    for it in items:
        # 防御：合同外结构（非 dict 条目）直接跳过，避免个别脏数据令整次检索判定失败
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "无标题").strip()
        url = (it.get("url") or "").strip()
        snippet = _truncate(it.get("content") or "", WEB_SRCH_SNIPPET_MAX_LENGTH)
        score = it.get("score", 0.0)

        entries.append(f"{len(results) + 1}. {title}\n   来源：{url}\n   摘要：{snippet}")
        results.append({
            "id": url,
            "content": f"{title}｜{snippet}",
            "score": score,
            "similarity": score,
            "role": "web",
            "memory_type": "web",
        })

    # 所有命中条目均为非法结构 → 等价于无命中（避免报“共 N 条”却无任何条目）
    if not results:
        logger.info("[联网检索] 命中条目均为非法结构，视为无结果：%s", query)
        return True, f"未找到与「{query}」相关的联网结果。", []

    lines = [f"【系统：联网检索结果 - 共 {len(results)} 条】"]
    if answer:
        lines.append(f"综述：{_truncate(answer, WEB_SRCH_SNIPPET_MAX_LENGTH)}")
    lines.extend(entries)
    lines.append("注意：以上为外部检索结果，需自行判断可信度后再作答。")
    logger.info("[联网检索] 命中 %d 条：%s", len(results), query)
    return True, "\n".join(lines), results
