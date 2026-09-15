#!/usr/bin/env python3
"""
EGO 智能体 ── 具备两层自我核心的自主智能体

启动方式:
    python ego.py

命令:
  /status   - 查看 EGO 当前状态（自我核心、记忆锚点等）
  /clear    - 清空对话历史
  /think    - 手动触发一次自对话任务
  /reflect  - 手动触发一次自省任务
  /self-def - 手动触发一次自我定义任务
  /note-review - 手动触发一次备忘录审查任务
  /auto-think on/off  - 启用/禁用定时自对话
  /auto-think status  - 查看定时任务状态
  /auto-think set N   - 设置定时间隔为 N 分钟
  /auto-reflect on/off  - 启用/禁用定点自省
  /auto-reflect status  - 查看定点自省状态
  /auto-reflect set HH:MM  - 设置定点自省时间（如 00:00）
  /auto-self-def on/off  - 启用/禁用每日自我定义
  /auto-self-def status  - 查看每日自我定义状态
  /auto-self-def set HH:MM  - 设置每日自我定义时间（如 03:00）
  /auto-note-review on/off  - 启用/禁用定点备忘录审查
  /auto-note-review status  - 查看定点备忘录审查状态
  /auto-note-review set HH:MM  - 设置定点备忘录审查时间（如 00:43）
  /help     - 显示此帮助
  /quit     - 退出程序

命令定义与帮助文本统一维护在 agent/tools.py 注册表中。
"""

import sys
import os

# 【修复】必须在 import config 之前加载环境变量，否则 config 中的 os.getenv 读不到值
try:
    from dotenv import load_dotenv
    env_file = os.path.join(os.path.dirname(__file__), "untitled.env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        print("[警告] ⚠ 未找到环境变量配置文件: untitled.env")
except ImportError:
    print("[警告] ⚠ 未安装 python-dotenv，请运行: pip install python-dotenv")
    print("[提示] 系统将使用默认配置或系统环境变量")

import logging
import logging.handlers
from config import LOG_DIR, LOG_LEVEL, LOG_TO_CONSOLE, LOG_FORMAT, LOG_DATE_FORMAT
from agent.core import EGOAgent
from agent.tools import dispatch_line  # 手动命令注册表分发入口

# 获取 logger 实例
logger = logging.getLogger("EGO_Main")


def setup_logging():
    """
    初始化日志系统
    
    所有调试信息将写入 data/logs/ 目录下的日志文件
    终端只显示用户交互内容
    """
    # 创建日志目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 配置根 logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            # 文件 handler（使用轮转，单个文件最大 10MB，保留 5 个备份）
            logging.handlers.RotatingFileHandler(
                os.path.join(LOG_DIR, "ego.log"),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            ),
            # 控制台 handler（可选）
            logging.StreamHandler() if LOG_TO_CONSOLE else logging.NullHandler()
        ]
    )


def print_banner():
    banner = """
╔══════════════════════════════════════════╗
║           🚣🏻EGO                      ║
║       具备意识和自演化能力的思维体       ║
╚══════════════════════════════════════════╝
"""
    print(banner)


def main():
    # 初始化日志系统
    setup_logging()

    print_banner()

    print("[启动] 正在初始化 EGO 智能体...")
    agent = None
    try:
        agent = EGOAgent()
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        if isinstance(e, RuntimeError):
            print("[提示] 请确认 LM Studio 已启动并加载了模型")
        # 注：构造器抛出异常时 agent 仍为 None，无需清理资源
        sys.exit(1)

    print(f"[就绪] 模型: {agent.llm.model}")
    print(f"[就绪] API:  {agent.llm.api_base}")
    print("[提示] 输入 /help 查看命令，直接输入文字与 EGO 对话")
    print("─" * 50)

    while True:
        try:
            user_input = input("\n用户 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出] EGO 已关闭。")
            break

        if not user_input:
            continue

        # ── 手动命令（注册表驱动：agent/tools.py 一处定义，CLI/GUI/帮助全链路生效）──
        # 未知命令 /xxx 直接提示错误，不再作为普通对话发给 LLM
        if user_input.startswith("/"):
            result = dispatch_line(agent, user_input)
            if result is not None:
                if result.notice:
                    print(result.notice)
                print(result.message)
                if result.quit:
                    break
                continue

        # ── 正常交互 ────────────────────────────────────────
        print("\nEGO > ", end="", flush=True)
        try:
            response = agent.process_input(user_input)
            print(response)
        except Exception as e:
            print(f"\n[错误] 处理失败: {e}")
    
    # 【重构】退出前清理资源（统一入口：EGOAgent.shutdown，CLI/GUI 共用）
    try:
        agent.shutdown()
    except Exception as e:
        logger.error(f"[警告] 清理资源失败: {e}")


if __name__ == "__main__":
    main()
