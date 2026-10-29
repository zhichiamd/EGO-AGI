# -*- coding: utf-8 -*-
"""
冷启动缓存（ColdstartCache）

从 session.py 抽出的冷启动缓存层：data/coldstart_cache.json 的读写、有效性校验、
MD5 指纹、previous_response_id 回写，以及冷启动/预热消息缓冲区。

设计约束：
- 不持有 SessionBootstrap：llm / pm 与缓存路径全部构造注入（cache_path 便于用临时文件
  测试），因此可脱离主流程单独构造与验证。
- 缓冲区随本模块走：_coldstart_msgs / _preheat_msgs 是「待落盘数据」而非流程状态，
  只有 save() 消费它们；冷启动/预热执行层通过 append_step() 写入，不直接碰列表。
- logger 沿用 "EGOAgent"：EGO_GUI 初始化期按该名字挂日志透传，启动日志必须继续可见。
- 本模块不含任何 LLM 调用：全部是文件 I/O + 哈希，失败一律降级为 WARNING，不抛异常。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, CACHE_STALE_WARNING_DAYS


# 获取 logger 实例（与 core.py 同名，日志归属不变）
logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    """记录日志（级别由调用方显式指定，默认 INFO）"""
    logger.log(level, message)


class ColdstartCache:
    """冷启动缓存文件的读写与校验（无 LLM 调用）。"""

    def __init__(self, llm, pm, cache_path=None):
        """
        Args:
            llm: LMStudioClient（读取当前 previous_response_id 用于回写缓存）
            pm: PromptManager（读取 layer2 计算认知库指纹）
            cache_path: 冷启动缓存文件路径（默认 data/coldstart_cache.json，便于测试注入）
        """
        self.llm = llm
        self.pm = pm

        # 冷启动缓存文件路径
        self.cache_file = Path(cache_path) if cache_path else Path(BASE_DIR) / "data" / "coldstart_cache.json"
        # 冷启动/预热提示词与输出缓冲区：仅用于写入 coldstart_cache.json，
        # 不写入 history.json（其结晶已体现在 L2，重启重建无需回灌）
        self._coldstart_msgs: list[dict] = []
        self._preheat_msgs: list[dict] = []

    # ── 冷启动缓存管理 ────────────────────────────────────────

    def load(self) -> dict | None:
        """加载冷启动缓存文件"""
        if not self.cache_file.exists():
            return None
    
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            _auto_log(f"[警告] 加载冷启动缓存失败: {e}", level=logging.WARNING)
            return None

    def is_valid(self, cache: dict) -> bool:
        """验证缓存是否有效"""
    
        # 1. 检查 sys.json 哈希
        current_sys_hash = self._file_md5(str(Path(BASE_DIR) / "data" / "sys.json"))
        if current_sys_hash != cache.get("sys_prompts_hash"):
            _auto_log("[警告] sys.json 已变化，缓存失效", level=logging.WARNING)
            return False
    
        # 2. 可选：检查缓存年龄
        try:
            cache_time = datetime.fromisoformat(cache["created_at"])
            days_old = (datetime.now() - cache_time).days
        
            if days_old > CACHE_STALE_WARNING_DAYS:
                _auto_log(f"[警告] 缓存已超过 {days_old} 天，建议刷新", level=logging.WARNING)
        except Exception:
            pass
    
        _auto_log("[信息] ✓ 缓存验证通过")
        return True

    def save(self, status: str = "complete", failed_step: int = None, resume_from: int = None):
        """保存冷启动缓存

        Args:
            status: "complete" / "incomplete"
            failed_step: 冷启动失败步号（status=incomplete 时写入进度）
            resume_from: 本次执行的起始步号（1 基）。旧缓存为 incomplete 时按「步」对齐：
                只保留前 resume_from-1 步的旧消息，再接上本轮缓冲区。不传则视为
                全新执行（丢弃旧消息）。
        """
        try:
            # 【新增】检查是否有旧缓存需要合并（续传场景）
            old_cache = self.load() if self.cache_file.exists() else None
        
            # 【调整】冷启动/预热内容直接从缓冲区取（history 已不再保存该阶段）
            coldstart_msgs = [dict(m) for m in self._coldstart_msgs]
            preheat_msgs = [dict(m) for m in self._preheat_msgs]
        
            # 【续传对齐】旧缓存为 incomplete 时必须合并此前已完成步骤的消息：
            # 解耦后冷启动/预热内容只存于缓冲区，续传仅把本轮执行的步写入缓冲区，
            # 旧缓存中已完成的步骤消息必须补入，否则再次失败时会永久丢失。
            # 对齐基准是「步号」而非消息正文：按 content 去重时，只要两步的 assistant
            # 输出文本恰好相同，就会把真实的旧消息误删，导致 completed_steps 与
            # 实际消息数账实不符（续传路径最早暴露的问题）。
            if old_cache and old_cache.get("status") == "incomplete":
                old_coldstart = old_cache.get("coldstart_messages") or []
                if resume_from and resume_from > 1:
                    keep = (resume_from - 1) * 2
                    coldstart_msgs = old_coldstart[:keep] + coldstart_msgs
                    _auto_log(
                        f"[信息]   续传对齐：保留旧冷启动前 {resume_from - 1} 步 ({len(old_coldstart[:keep])} 条)"
                    )
                else:
                    _auto_log("[信息]   旧缓存未完成但无续传起点，按全新执行处理（丢弃旧消息）")
                # 预热：incomplete 缓存的预热不可能已完整跑过（冷启动未完成就不会进预热），
                # 本轮预热是完整重跑，故旧预热一律不参与合并。
        
            # 构建缓存数据
            cache_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "status": status,
                "l2_hash": self._md5(
                    json.dumps(self.pm.layer2, ensure_ascii=False)
                ),
                "sys_prompts_hash": self._file_md5(str(Path(BASE_DIR) / "data" / "sys.json")),
                "coldstart_messages": coldstart_msgs,
                "preheat_messages": preheat_msgs,
                "total_coldstart_messages": len(coldstart_msgs),
                "total_preheat_messages": len(preheat_msgs)
            }
        
            # 【新增】保存 previous_response_id（关键！用于恢复 LM Studio Session 上下文）
            _sid = self.llm.get_session_id()
            if _sid:
                cache_data["previous_response_id"] = _sid
                _auto_log(f"[信息]   - 已保存 previous_response_id: {_sid[:20]}...")
        
            # 如果未完成，添加进度信息
            if status == "incomplete" and failed_step:
                completed_steps = list(range(1, failed_step))
                cache_data["completed_steps"] = completed_steps
                cache_data["failed_step"] = failed_step
        
            # 写入文件
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
            _auto_log(f"[信息] ✓ 冷启动缓存已保存 (status={status})")
            if coldstart_msgs:
                _auto_log(f"[信息]   - 冷启动: {len(coldstart_msgs)} 条")
            if preheat_msgs:
                _auto_log(f"[信息]   - 预热: {len(preheat_msgs)} 条")
        
        except Exception as e:
            _auto_log(f"[警告] 保存冷启动缓存失败: {e}", level=logging.WARNING)

    def create_initial(self):
        """
        创建初始冷启动缓存（当 L2 已有内容但缓存不存在时）
    
        【关键】确保即使跳过冷启动/预热，也能有缓存文件用于保存 Session ID
        """
        try:
            cache_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "status": "complete",
                "l2_hash": self._md5(
                    json.dumps(self.pm.layer2, ensure_ascii=False)
                ),
                "sys_prompts_hash": self._file_md5(str(Path(BASE_DIR) / "data" / "sys.json")),
                "coldstart_messages": [],
                "preheat_messages": [],
                "total_coldstart_messages": 0,
                "total_preheat_messages": 0,
                "note": "Initial cache created for existing L2 content"
            }
        
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
            _auto_log("[信息] ✓ 初始缓存已创建")
        
        except Exception as e:
            _auto_log(f"[警告] 创建初始缓存失败: {e}", level=logging.WARNING)

    def _md5(self, text: str) -> str:
        """计算文本的 MD5 哈希值"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _file_md5(self, filepath: str) -> str:
        """计算文件的 MD5 哈希值（文件不存在时返回空串，视为哈希失配）"""
        import hashlib
        # 【修复】存在性检查：防止 sys.json 缺失但缓存存在时启动崩溃
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def update_session_id(self):
        """
        更新冷启动缓存中的 previous_response_id
    
        【关键】确保意外退出后，重启时能恢复到最新的 Session 状态
        """
        try:
            # 加载现有缓存
            if not self.cache_file.exists():
                return
        
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        
            # 只更新 previous_response_id，保持其他内容不变
            cache_data["previous_response_id"] = self.llm.get_session_id()
            cache_data["updated_at"] = datetime.now().isoformat()
        
            # 写回文件
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
            _auto_log(f"[信息] ✓ 已更新缓存中的 previous_response_id: {(self.llm.get_session_id() or 'None')[:20]}...")
        
        except Exception as e:
            # 静默失败，不影响主流程
            _auto_log(f"[警告] 更新缓存中的 previous_response_id 失败: {e}", level=logging.WARNING)

    # ── 缓冲区（供冷启动/预热执行层写入）────────────────────

    def append_step(self, stage: str, prompt: str, content: str):
        """记录一步冷启动/预热消息对（只进缓存，不落 history）

        Args:
            stage: "coldstart" 或 "preheat"
            prompt: 该步的提示词（作为 user 消息）
            content: 从响应中提取的 SAY 正文（作为 assistant 消息）
        """
        buf = self._coldstart_msgs if stage == "coldstart" else self._preheat_msgs
        buf.append({"role": "user", "content": prompt})
        buf.append({"role": "assistant", "content": content})
