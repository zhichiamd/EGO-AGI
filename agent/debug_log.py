"""
LLM 原始响应 debug 日志

从 core.py 抽出的 LLMDebugLogger：每轮 LLM 响应落一个 .md 文件（含原始响应、
解析出的指令、剥离指令后的纯文本），并按数量上限清理过期文件。

只依赖 config 与 instructions，不持有 agent 的其它状态 —— 可独立构造、独立测试。

注：logger 沿用 core 的 "EGOAgent" 名 —— EGO_GUI 初始化期按该名字挂日志透传，
保存/清理提示需要保持原有可见性。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, DEBUG_LOG_MAX_FILES
from agent.instructions import parse_instructions, strip_instructions

logger = logging.getLogger("EGOAgent")


def _auto_log(message: str, level: int = logging.INFO):
    logger.log(level, message)


class LLMDebugLogger:
    """LLM 响应 debug 日志写入器（文件名/编号语义与旧实现完全一致）。"""

    def __init__(self, directory: str | Path | None = None,
                 max_files: int = DEBUG_LOG_MAX_FILES):
        """
        Args:
            directory: 日志目录，默认 data/debug_logs（由 BASE_DIR 推导，兼容打包路径）
            max_files: 保留的最大文件数量（超过时按修改时间清理最旧的）
        """
        self._dir = Path(directory) if directory else Path(BASE_DIR) / "data" / "debug_logs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_files = int(max_files)
        self._counter = 0  # 用于生成唯一的日志文件名（进程内单调递增）

    @property
    def directory(self) -> Path:
        """日志目录（只读）。"""
        return self._dir

    def save(self, response: str, round_num: int = 0, stage: str = "chat", step: int = 0) -> Path | None:
        """
        保存每轮 LLM 原始响应到 debug log 文件

        Args:
            response: LLM 返回的原始文本
            round_num: 当前循环轮次（用于文件名，-1 表示非 EGO 阶段）
            stage: 阶段标识（"coldstart", "preheat", "think", "chat"）
            step: 当前步骤编号（用于区分同阶段内的多条日志）

        Returns:
            写入的文件路径；写入失败时返回 None
        """
        self._counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根据阶段生成不同的文件名格式
        if stage == "chat":
            filename = f"llm_response_{timestamp}_{self._counter:04d}_round{round_num}.md"
        else:
            # EGO 循环类阶段（think/note）附带循环轮次，命名形如
            # llm_response_think_{ts}_{n:04d}_step{step}_round{round_num}.md；
            # 非循环阶段（preheat/coldstart/reflection/note-review/selfdef，round_num=-1）
            # 保持原命名 ..._step{step}.md，避免影响既有日志解析
            round_suffix = f"_round{round_num}" if round_num >= 0 else ""
            filename = f"llm_response_{stage}_{timestamp}_{self._counter:04d}_step{step}{round_suffix}.md"

        filepath = self._dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# LLM Response Debug Log\n\n")
                f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
                f.write(f"**Stage**: {stage.upper()}\n\n")
                if stage == "chat":
                    f.write(f"**Round**: {round_num}\n\n")
                else:
                    f.write(f"**Step**: {step}\n\n")
                f.write(f"**Response Length**: {len(response)} characters\n\n")
                f.write(f"**Log File**: {filename}\n\n")
                f.write("\n---\n\n")
                f.write("## Raw Response\n\n")
                f.write("```\n\n")
                f.write(response)
                f.write("\n```\n\n")
                f.write("\n---\n\n")

                # 解析指令
                instructions = parse_instructions(response)
                if instructions:
                    f.write("## Parsed Instructions\n\n")
                    for i, instr in enumerate(instructions, 1):
                        f.write(f"{i}. **{instr.kind}**\n\n")
                        f.write(f"   Payload: `{instr.payload[:200]}{'...' if len(instr.payload) > 200 else ''}`\n\n")
                    f.write("\n---\n\n")

                # 提取纯文本（去除所有指令）
                pure_text = strip_instructions(response)
                if pure_text and pure_text.strip():
                    f.write("## Pure Text\n\n")
                    f.write("```\n\n")
                    f.write(pure_text.strip())
                    f.write("\n```\n\n")

            _auto_log(f"[Debug] ✓ LLM 响应已保存到: {filepath}")

            # 清理过期的调试日志（保留最近 max_files 个文件）
            self._cleanup_old(self._max_files)
            return filepath

        except Exception as e:
            _auto_log(f"[警告] 保存 LLM debug log 失败: {e}", level=logging.WARNING)
            return None

    def _cleanup_old(self, max_files: int):
        """
        清理过期的调试日志文件，保留最近的 N 个文件

        Args:
            max_files: 保留的最大文件数量
        """
        try:
            # 获取所有调试日志文件，按修改时间排序
            log_files = list(self._dir.glob("llm_response_*.md"))

            if len(log_files) > max_files:
                # 按修改时间排序（最新的在前）
                log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                # 删除多余的文件
                files_to_delete = log_files[max_files:]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        _auto_log(f"[清理] 删除过期调试日志: {file_path.name}")
                    except Exception as e:
                        _auto_log(f"[警告] 删除调试日志失败 {file_path.name}: {e}", level=logging.WARNING)

                _auto_log(f"[清理] ✓ 已清理 {len(files_to_delete)} 个过期调试日志，保留 {max_files} 个")
        except Exception as e:
            _auto_log(f"[警告] 清理调试日志失败: {e}", level=logging.WARNING)
