"""
ChromaDB 记忆管理模块

使用 ChromaDB 作为向量数据库，Ollama BGE-M3 做向量化，FLM 做摘要。

记忆类型：
- 会话记录：用户与系统的全部对话记录

MEMO_RD 检索时查询会话记录，返回结果。

依赖：
    pip install chromadb requests
"""

import math
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from datetime import datetime

import chromadb
from chromadb import Documents, Embeddings
from chromadb.api.types import EmbeddingFunction

from config import (
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    FLM_API_BASE,
    FLM_MODEL,
    FLM_SUMMARY_TEMPERATURE,
    FLM_SUMMARY_MAX_TOKENS,
    FLM_SUMMARY_TIMEOUT,
    CHROMA_OBJECTIVE_COLLECTION,
    CHROMA_RECALL_TOP_K,
    MEMORY_CITATION_CYCLE_LENGTH,
    MEMORY_CITATION_AMPLITUDE,
    MEMORY_CITATION_DECAY_RATE,
    MEMORY_CITATION_MIN_BOOST,
    MEMORY_CITATION_MAX_BOOST,
    MEMORY_SIMILARITY_THRESHOLD,  # MEMO_RD 相似度过滤阈值
    MEMORY_TIME_DECAY_SCALE_DAYS,  # 时间衰减半衰期（天）
    MEMORY_TIME_DECAY_FLOOR,  # 时间衰减权重下限
    FLM_SUMMARY_TRIGGER_LENGTH,  # 超过该字符数触发 FLM 摘要
    FLM_SUMMARY_TARGET_LENGTH,  # FLM 摘要目标字符数
    OLLAMA_DETECT_TIMEOUT,  # Ollama 模型检测超时（秒）
    OLLAMA_EMBED_TIMEOUT,  # Ollama 嵌入生成超时（秒）
    MEMORY_MAX_PENDING_STORES,  # 会话入库待执行任务上限（防单 worker 队列无限积压）
)

logger = logging.getLogger("ChromaMemory")

# 【复用】FLM 摘要用长连接 Session：避免每次请求新建 TCP 连接（单 worker 串行，无并发）。
_flm_session = requests.Session()


def _auto_log(message: str, level: int = logging.INFO):
    # 记录日志（级别由调用方显式指定，默认 INFO；不再靠关键词猜测）
    logger.log(level, message)


# ═══════════════════════════════════════════════════════════════
#  Ollama Embedding Function（ChromaDB 兼容接口）
# ═══════════════════════════════════════════════════════════════

class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    基于 Ollama /api/embed 接口的 ChromaDB 嵌入函数。

    使用 BGE-M3 模型，向量维度 1024。
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session = requests.Session()
        self._verify_model()

    def _verify_model(self):
        """启动时验证 Ollama 中是否有目标模型"""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=OLLAMA_DETECT_TIMEOUT)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # 支持模糊匹配（如 "bge-m3:latest" 匹配 "bge-m3"）
            matched = [m for m in models if self.model in m]
            if matched:
                _auto_log(f"[配置] ✓ Ollama 模型可用: {matched[0]}")
            else:
                _auto_log(f"[警告] ⚠ 未找到模型 '{self.model}'，已安装: {models}", level=logging.WARNING)
        except Exception as e:
            _auto_log(f"[警告] ⚠ 无法连接 Ollama: {e}", level=logging.WARNING)

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB 调用此方法获取嵌入向量"""
        resp = self._session.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": input},
            timeout=OLLAMA_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ══════════════════════════════════════════════════════════════
#  FLM 摘要工具
# ═══════════════════════════════════════════════════════════════

def _flm_summarize(text: str, max_length: int = FLM_SUMMARY_TARGET_LENGTH) -> str:
    """
    使用 FLM 小模型对长文本做摘要。

    Args:
        text: 原始文本
        max_length: 摘要最大字符数

    Returns:
        摘要文本（如果摘要失败则返回原文前 max_length 字符）
    """
    if len(text) <= max_length:
        return text

    try:
        resp = _flm_session.post(
            f"{FLM_API_BASE}/chat/completions",
            json={
                "model": FLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是信息摘要助手。任务是将输入文本压缩成高度凝练的信息，严格遵循以下规则：\n\n  1. 高密度信息保持原文：如诗歌、数学公式、精确数据等。\n  2. 低密度信息极限压缩：如寒暄、重复、解释性、情感抒发等内容，用最凝练的语言提取其核心意图、结论或关键动作。\n  3. 忠于原文：只允许使用对话原文已经出现的信息，禁止推断、润色添加原文不存在的观点。\n  4、输出格式：按时间顺序叙述；保留的高密度原文置于引号内，尽量醒目；整个摘要内容用一个段落表示，不要分段，不加多余标题。"},
                    {"role": "user", "content": f"请摘要（{max_length}字以内）：\n{text}"}
                ],
                "temperature": FLM_SUMMARY_TEMPERATURE,
                "max_tokens": FLM_SUMMARY_MAX_TOKENS,
            },
            timeout=FLM_SUMMARY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        summary = data["choices"][0]["message"]["content"].strip()
        if summary:
            return summary
    except Exception as e:
        _auto_log(f"[警告] ⚠ FLM 摘要失败: {e}，使用截断", level=logging.WARNING)

    # 降级：直接截断
    return text[:max_length] + "..."


# ═══════════════════════════════════════════════════════════════
#  ChromaMemoryManager
# ═══════════════════════════════════════════════════════════════

class ChromaMemoryManager:
    """
    基于 ChromaDB 的记忆管理器

    历史会话（objective）：用户与系统的完整对话记录

    对外接口：
    - recall()          → 检索历史会话
    - format_anchors()  → 格式化分类结果
    - log_conversation() → 存入会话记录（由 core.py 调用）
    """

    def __init__(self):
        # 初始化 ChromaDB 持久化客户端
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # 初始化 Ollama 嵌入函数
        self._embed_fn = OllamaEmbeddingFunction()

        # 创建/获取 collection
        self._objective = self._client.get_or_create_collection(
            name=CHROMA_OBJECTIVE_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        _auto_log(f"[信息] ✓ 会话记录: {self._objective.count()} 条")

        self._stats = {
            "total_recalls": 0,
            "total_conversation_stores": 0,
            "last_recall_time": None,
        }
        # 【异步化】统计自增锁（后台入库线程与主线程 recall 并发访问）
        self._stats_lock = threading.Lock()
        # 【异步化】会话记录入库走后台单线程队列：FLM 摘要 / Ollama embedding 为慢 IO，
        # 不阻塞主对话流程；单 worker 保证 FIFO 入库顺序与限并发
        self._store_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-store")
        # 【修复】待执行入库任务计数：服务变慢时单 worker 队列本会无界积压占内存，
        # 超出 MEMORY_MAX_PENDING_STORES 即丢弃新任务（记忆为辅助功能，失败不影响主流程）
        self._pending_stores = 0

    # ─ 会话记录 ──────────────────────────────────

    def log_conversation(self, role: str, content: str, stage: str = None):
        """
        存入对话记录（异步）

        仅存储用户对话阶段（stage=None 或 "chat"），
        内部阶段（coldstart/preheat/think/reflection）不存入。
        长文本会自动通过 FLM 做摘要后存储。

        注意：摘要与入库在后台单线程队列中执行，立即返回，不阻塞主对话流程；
        程序退出时已提交的任务由 close() 等待完成（失败不影响主流程）。

        Args:
            role: "user" 或 "assistant"
            content: 消息内容
            stage: 阶段标识（chat/think/coldstart/preheat 等）
        """
        # 过滤内部阶段，只记录真实用户对话
        if stage and stage not in ("chat",):
            return

        # 过滤空内容
        if not content or not content.strip():
            return

        doc_id = f"obj_{uuid.uuid4().hex[:12]}"
        # 【修复】积压防护：待执行任务超过上限则丢弃本任务（保护内存），
        # 记忆为辅助功能，失败不影响主对话流程（与既有设计声明一致）
        with self._stats_lock:
            if self._pending_stores >= MEMORY_MAX_PENDING_STORES:
                _auto_log(f"[警告] 会话入库队列积压 {self._pending_stores} 条（>= {MEMORY_MAX_PENDING_STORES}），丢弃本次写入", level=logging.WARNING)
                return
            self._pending_stores += 1
        self._store_executor.submit(self._store_conversation_async, role, content, stage, doc_id)

    def _store_conversation_async(self, role: str, content: str, stage: str, doc_id: str):
        """后台线程执行：FLM 摘要（长文本）→ 构造文档 → ChromaDB 入库（embedding 自动完成）

        【修复】异常保护：记忆写入是辅助功能，失败（Ollama/ChromaDB 不可用）不应击穿主对话流程
        """
        try:
            # 长文本摘要（超过 FLM_SUMMARY_TRIGGER_LENGTH 字符时）
            store_text = content
            original_length = len(content)
            if original_length > FLM_SUMMARY_TRIGGER_LENGTH:
                store_text = _flm_summarize(content, max_length=FLM_SUMMARY_TARGET_LENGTH)
                _auto_log(f"[信息] 📝 会话记录已摘要: {original_length} → {len(store_text)} 字符")

            role_cn = "用户" if role == "user" else "EGO"
            document = f"[{role_cn}] {store_text}"

            metadata = {
                "memory_type": "objective",
                "role": role,
                "stage": stage or "chat",
                "timestamp": datetime.now().isoformat(),
                "original_length": str(original_length),
            }

            self._objective.add(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )

            with self._stats_lock:
                self._stats["total_conversation_stores"] += 1
            _auto_log(f"[信息] ✓ 会话记录已存入 [{doc_id}]：[{role_cn}] {store_text[:50]}{'...' if len(store_text) > 50 else ''}")
        except Exception as e:
            _auto_log(f"[警告] 会话记录存入失败（ChromaDB/Ollama）: {e}，不影响本轮对话", level=logging.WARNING)
        finally:
            # 【修复】任务结束（含异常）时递减积压计数，与 log_conversation 的 +1 配对
            with self._stats_lock:
                self._pending_stores = max(0, self._pending_stores - 1)

    # ── 记忆检索（MEMO_RD） ─────────────────────────────────

    def recall(self, keyword: str, max_results: int = None, role: str = None,
               recall_after: float = None) -> list[dict]:
        """
        检索会话记录

        流程：
        1. ChromaDB cosine 相似度初筛（扩大候选池）
        2. 综合评分重排序：similarity × time_decay × citation_boost

        Args:
            keyword: 检索关键词
            max_results: 最大返回数量
            role: 可选，按角色过滤（"user" 或 "assistant"）
            recall_after: 可选，过滤掉该 Unix 时间戳之后写入的结果
                          （传入 process_input 的 start_time，防止当前问题被检索回来）

        Returns:
            检索结果列表
        """
        if max_results is None:
            max_results = CHROMA_RECALL_TOP_K

        results = self._search_collection(
            self._objective, keyword, max_results, "objective", role=role
        )

        # 过滤掉相似度过低的结果（阈值由 config 提供）
        SIMILARITY_THRESHOLD = MEMORY_SIMILARITY_THRESHOLD
        before_count = len(results)
        results = [r for r in results if r.get("similarity", 0) >= SIMILARITY_THRESHOLD]
        filtered_count = before_count - len(results)
        if filtered_count > 0:
            _auto_log(f"[警告] ⚠ MEMO_RD 过滤掉 {filtered_count} 条低相似度记忆（阈值: {SIMILARITY_THRESHOLD:.0%}）", level=logging.WARNING)

        # 【修复】过滤本轮对话开始后写入的结果（防止当前问题被检索回来）
        if recall_after and results:
            before_ts_count = len(results)
            filtered_results = []
            for r in results:
                ts = r.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt.timestamp() >= recall_after:
                            continue
                    except Exception:
                        pass
                filtered_results.append(r)
            excluded_ts_count = before_ts_count - len(filtered_results)
            if excluded_ts_count > 0:
                _auto_log(f"[警告] ⚠ MEMO_RD 过滤掉 {excluded_ts_count} 条本轮对话写入的记忆（时间戳过滤）", level=logging.WARNING)
            results = filtered_results

        with self._stats_lock:
            self._stats["total_recalls"] += 1
            self._stats["last_recall_time"] = datetime.now().isoformat()

        if results:
            _auto_log(f"[信息] ✓ MEMO_RD: 历史会话 {len(results)} 条（查询: {keyword[:50]}）")
        else:
            _auto_log(f"[信息] ℹ MEMO_RD 未找到匹配记忆（查询: {keyword[:50]}）")

        return results

    def increment_citations(self, doc_ids: list) -> None:
        """
        【修复】为指定记忆条目的引用计数 +1（激活引用震荡增益）

        调用时机：MEMO_RD 结果真正注入上下文时（citation_count 此前只读不写，震荡增益从未生效）。
        仅更新 metadata，不重新嵌入；失败静默，不影响主对话流程。
        """
        if not doc_ids:
            return
        try:
            existing = self._objective.get(ids=doc_ids, include=["metadatas"])
            ids = existing.get("ids", [])
            metadatas = existing.get("metadatas", [])
            if not ids:
                return
            updated = []
            for meta in metadatas:
                meta = dict(meta) if meta else {}
                meta["citation_count"] = int(meta.get("citation_count", 0)) + 1
                updated.append(meta)
            self._objective.update(ids=ids, metadatas=updated)
        except Exception as e:
            _auto_log(f"[警告] 更新记忆引用计数失败: {e}，不影响本轮对话", level=logging.WARNING)

    # ── 综合评分（向量相似度 × 时间衰减 × 引用震荡） ────────────

    @staticmethod
    def _compute_score(similarity: float, timestamp: str, citation_count: int) -> float:
        """
        综合评分 = 向量相似度 × 时间衰减 × 引用震荡增益

        - 向量相似度：BGE-M3 cosine similarity (0~1)
        - 时间衰减：7天内权重 1.0，30天后降至 ~0.5（指数衰减）
        - 引用震荡：正弦波周期增益 + 指数衰减包络（模拟"熟视无睹-重新发现"）
        """
        # 时间衰减
        time_weight = 1.0
        try:
            dt = datetime.fromisoformat(timestamp)
            days_diff = (datetime.now() - dt).days
            time_weight = max(MEMORY_TIME_DECAY_FLOOR, 1.0 / (1 + days_diff / MEMORY_TIME_DECAY_SCALE_DAYS))
        except Exception:
            pass

        # 引用震荡增益
        if citation_count <= 0:
            citation_boost = 1.0
        else:
            base_wave = math.sin(math.pi * citation_count / (MEMORY_CITATION_CYCLE_LENGTH / 2.0))
            envelope = math.exp(-citation_count / MEMORY_CITATION_DECAY_RATE)
            oscillation = base_wave * envelope
            citation_boost = 1.0 + MEMORY_CITATION_AMPLITUDE * oscillation
            citation_boost = max(MEMORY_CITATION_MIN_BOOST, min(citation_boost, MEMORY_CITATION_MAX_BOOST))

        return similarity * time_weight * citation_boost

    def _search_collection(self, collection, query: str, n_results: int,
                           memory_type: str, role: str = None) -> list[dict]:
        """搜索单个 collection（cosine 初筛 + 综合评分重排序）"""
        try:
            total = collection.count()
            if total == 0:
                return []

            # 扩大候选池用于重排序（3倍，最少 10 条）
            fetch_n = min(total, max(n_results * 3, 10))

            # 构建 where 过滤条件
            where_filter = None
            if role:
                where_filter = {"role": role}

            results = collection.query(
                query_texts=[query],
                n_results=fetch_n,
                where=where_filter,
            )

            candidates = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # 【修复】元数据可能为 None（无 metadata 的历史文档），统一兜底为空字典
                meta = meta or {}
                similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
                citation_count = int(meta.get("citation_count", 0))
                timestamp = meta.get("timestamp", "")

                # 综合评分 = 向量相似度 × 时间衰减 × 引用震荡
                score = self._compute_score(similarity, timestamp, citation_count)

                anchor = {
                    "id": doc_id,
                    "content": doc,
                    "memory_type": memory_type,
                    "tags": meta.get("tags", "").split(", ") if meta.get("tags") else [],
                    "timestamp": timestamp,
                    "source_stage": meta.get("stage", ""),
                    "role": meta.get("role", ""),
                    "similarity": round(similarity, 4),
                    "score": round(score, 4),
                    "_citation_count": citation_count,  # 内部字段，用于更新引用计数
                    "_raw_meta": dict(meta),  # 原始 metadata 快照，用于 update 时保留全部字段
                }
                candidates.append(anchor)

            # 按综合得分降序排序
            candidates.sort(key=lambda x: x["score"], reverse=True)

            # 截取 top n_results
            return candidates[:n_results]

        except Exception as e:
            _auto_log(f"[警告] ⚠ {memory_type} 检索失败: {e}", level=logging.WARNING)
            return []

    # ── 格式化输出 ─────────────────────────────────────────

    def format_anchors(self, anchors: list[dict], for_llm: bool = True) -> str:
        """
        格式化检索结果
        """
        if not anchors:
            return "（未找到匹配的记忆）"

        if not for_llm:
            lines = []
            for a in anchors:
                lines.append(f"[{a.get('id', '')}] {a.get('content', '')}")
            return "\n".join(lines)

        lines = [
            f"【系统：记忆检索结果 - 共找到 {len(anchors)} 条历史对话片段】",
            ""
        ]

        if anchors:
            lines.append(f"═══ 根据关键字检索到历史会话片段（{len(anchors)} 条）═══")
            lines.append("提示：综合评分、相似度越高的历史对话片段与查询内容越相关。")
            for idx, a in enumerate(anchors, 1):
                content = a.get("content", "")
                role = a.get("role", "")
                role_cn = "用户" if role == "user" else "EGO"
                role_icon = "👤" if role == "user" else "🤖"
                timestamp = self._format_time(a.get("timestamp", ""))
                similarity = a.get("similarity", 0)

#                lines.append(f"📌 历史会话 #{idx} [{a.get('id', '')}] [{role_cn}] (综合评分: {a.get('score', 0):.2f} | 相似度: {similarity:.0%})")
                lines.append(f"📌 历史会话 #{idx} [{a.get('id', '')}] {role_icon}[{role_cn}] (综合评分: {a.get('score', 0):.2f} | 相似度: {similarity:.0%})")

                lines.append(f"   内容：{content}")
                if timestamp:
                    lines.append(f"   时间：{timestamp}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_time(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return timestamp

    # ─ 兼容接口 ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._stats_lock:
            return self._stats.copy()

    def reset_stats(self):
        with self._stats_lock:
            self._stats = {
                "total_recalls": 0,
                "total_conversation_stores": 0,
                "last_recall_time": None,
            }

    def close(self):
        """关闭连接（ChromaDB 持久化客户端无需特别关闭）"""
        # 【修复】等待已提交任务完成再关闭：原 wait=False + cancel_futures=True 会
        # 静默丢弃排队中的会话入库（已提交的任务应尽力完成）；每个任务内部有异常保护
        # 且有界超时（FLM 摘要 FLM_SUMMARY_TIMEOUT / embedding OLLAMA_EMBED_TIMEOUT），
        # wait=True 的阻塞时间有界，不会挂死退出流程
        if hasattr(self, "_store_executor"):
            self._store_executor.shutdown(wait=True)
        if hasattr(self._embed_fn, '_session'):
            self._embed_fn._session.close()
        # 【复用】关闭 FLM 摘要长连接 Session（入库线程已 drain，安全）。
        try:
            _flm_session.close()
        except Exception:
            pass
