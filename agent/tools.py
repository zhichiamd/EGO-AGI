"""
手动命令注册表（用户斜杠命令）

与 agent/instructions.py 的 LLM 自主指令注册表对称：
LLM 自主指令（<THINK>/<SAY>/...）由模型在回复中嵌入标签触发；
手动命令（/status、/auto-think ...）由用户在 CLI 输入框或 GUI 菜单触发。

【注册表驱动】全部命令在注册表中声明：
- CLI 分发（dispatch_line）查表执行
- GUI 菜单（_build_menu）按 gui_items 派生，无需手工维护菜单
- 帮助文本（build_help_text）自动生成，CLI /help 与 GUI 帮助弹窗共用

新增命令 = 写一个 @register_command 装饰器定义（含帮助文本与 GUI 菜单元数据）；
删除命令 = 删除装饰器。其它位置零改动。

内置命令（16 条，含子命令）：
- /status  /clear  /think  /reflect  /self-def  /note-review  /help  /quit
- /auto-think [on|off|status|set N]
- /auto-reflect [on|off|status|set HH:MM]
- /auto-self-def [on|off|status|set HH:MM]
- /auto-note-review [on|off|status|set HH:MM]
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── 命令注册表（单一事实源）──────────────────────────────────────

@dataclass
class CommandResult:
    """命令执行结果（CLI / GUI 共用显示）"""
    message: str = ""     # 输出文本（CLI 打印 / GUI 消息区或弹窗）
    notice: str = ""      # CLI 执行前提示（如"正在触发..."）；GUI 忽略
    quit: bool = False    # 执行后退出程序（/quit）
    error: bool = False   # 是否错误提示（仅作标记，显示由 message 决定）


@dataclass(frozen=True)
class GuiItem:
    """GUI 菜单项声明（一条命令可对应多个菜单项，如 auto-think 的 on/off/status/set）"""
    group: str = ""        # 菜单分组名（命令 / 自对话 / 自省 / 帮助）
    label: str = ""        # 菜单项标签（含 emoji 与对应斜杠命令）
    args: tuple = ()       # 固定子命令参数，如 ("on",)；对话框类菜单项给前缀 ("set",)
    dialog: str = ""       # 参数输入对话框回调名（GUI 类方法，带下划线），空 = 无对话框直接执行
    popup: bool = False    # 信息型命令：结果显示为弹窗而非消息区


@dataclass
class CommandSpec:
    """命令规格：声明式描述一条斜杠命令的全部行为"""
    name: str                      # 命令名（不含 /），如 "status"、"auto-think"
    handler: callable              # 执行函数 fn(agent, args: list[str]) -> CommandResult
    usage: str = ""                # 用法，如 "/auto-think [on|off|status|set N]"；空 = "/name"
    help_lines: tuple = ()         # 帮助文本行 [(usage, 描述), ...]；空 = [(usage, help)]
    help: str = ""                 # 单行描述（help_lines 为空时用于 /help）
    gui_items: tuple = ()          # GUI 菜单项列表
    quit: bool = False             # 执行后退出程序（/quit 标记）


_REGISTRY: dict[str, CommandSpec] = {}


def register_command(name: str, **kw) -> callable:
    """装饰器：注册一条手动命令。挂载 = 装饰，卸载 = 删除装饰。"""
    def deco(fn):
        _REGISTRY[name] = CommandSpec(name=name, handler=fn, **kw)
        return fn
    return deco


def get_command(name: str):
    """按命令名查规格（未知命令返回 None）"""
    return _REGISTRY.get(name)


def all_commands() -> list:
    """返回全部命令规格（按注册顺序）"""
    return list(_REGISTRY.values())


def _effective_help_lines(spec: CommandSpec) -> list:
    """命令的帮助行列表（help_lines 为空时从 usage/help 派生）"""
    if spec.help_lines:
        return list(spec.help_lines)
    usage = spec.usage or f"/{spec.name}"
    return [(usage, spec.help)]


def build_help_text() -> str:
    """从注册表派生 /help 帮助文本（CLI /help 与 GUI 帮助弹窗共用）"""
    lines = ["可用命令:"]
    for spec in _REGISTRY.values():
        for usage, desc in _effective_help_lines(spec):
            lines.append(f"  {usage.ljust(18)} - {desc}")
    lines.append("")
    lines.append("直接输入文字与 EGO 对话。EGO 会先进行自对话思考，然后给出回复。")
    return "\n".join(lines)


def match_command(line: str) -> tuple:
    """解析一行输入 → (CommandSpec|None, args: list|None)

    - 不以 / 开头（正常对话）→ (None, None)
    - 以 / 开头但命令未知 / 纯斜杠 → (None, [])
    - 命中注册命令 → (spec, args)，args 为子命令参数（如 ["set", "30"]）
    """
    if not line.startswith("/"):
        return None, None
    parts = line.split()
    if len(parts) < 1 or parts[0] == "/":
        return None, []
    spec = _REGISTRY.get(parts[0][1:])
    if spec is None:
        return None, []
    return spec, parts[1:]


def execute(agent, name: str, args: list) -> CommandResult:
    """按命令名执行（GUI 菜单 / 输入框命令通道共用）"""
    spec = _REGISTRY.get(name)
    if spec is None:
        return CommandResult(f"[错误] 未知命令: /{name}，输入 /help 查看可用命令", error=True)
    try:
        result = spec.handler(agent, list(args or []))
        if result is None:
            # 防未来注册命令忘记返回 CommandResult
            return CommandResult("", error=True)
        return result
    except Exception as e:
        return CommandResult(f"[错误] 命令 /{name} 执行失败: {e}", error=True)


def dispatch_line(agent, line: str):
    """CLI 分发入口：非命令 → None（交给正常对话）；命令 → CommandResult

    未知命令 /xxx 不再作为普通对话发给 LLM，而是返回错误提示（行为变化）。
    """
    spec, args = match_command(line)
    if spec is None:
        if args is None:
            return None
        return CommandResult(f"[错误] 未知命令: {line}，输入 /help 查看可用命令", error=True)
    return execute(agent, spec.name, args)


# ── 内置命令注册 ────────────────────────────────────────────────

@register_command("status", help="查看 EGO 当前状态（自我核心、记忆锚点等）",
                  gui_items=(GuiItem("命令", "📊 状态 (/status)", popup=True),))
def _cmd_status(agent, args: list) -> CommandResult:
    return CommandResult(agent.get_status())


@register_command("clear", help="清空对话历史",
                  gui_items=(GuiItem("命令", "🧹 清空历史 (/clear)"),))
def _cmd_clear(agent, args: list) -> CommandResult:
    # 统一入口：EGOAgent.clear_conversation（锁保护 + session 完全重置）
    if agent.clear_conversation():
        return CommandResult("[系统] 对话历史已清空，session 已完全重置。")
    return CommandResult("[错误] 后台任务正在运行（自对话/自省），请稍后再试", error=True)


@register_command("think", help="手动触发一次自对话任务",
                  gui_items=(GuiItem("命令", "💭 手动自对话 (/think)"),))
def _cmd_think(agent, args: list) -> CommandResult:
    # 统一入口：EGOAgent.run_think_cycle（等锁超时默认 LOCK_ACQUIRE_TIMEOUT）
    if agent.run_think_cycle():
        return CommandResult("[系统] ✓ 自对话任务完成", notice="\n[系统] 正在手动触发自对话任务...")
    return CommandResult("[错误] 获取锁失败，请稍后再试", error=True)


@register_command("reflect", help="手动触发一次自省任务",
                  gui_items=(GuiItem("命令", "🪞 手动自省 (/reflect)"),))
def _cmd_reflect(agent, args: list) -> CommandResult:
    # 统一入口：EGOAgent.run_reflection_cycle（等锁超时默认 LOCK_ACQUIRE_TIMEOUT）
    if agent.run_reflection_cycle():
        return CommandResult("[系统] ✓ 自省任务完成", notice="\n[系统] 正在手动触发自省任务...")
    return CommandResult("[错误] 获取锁失败，请稍后再试", error=True)


@register_command("self-def", help="手动触发一次自我定义任务",
                  gui_items=(GuiItem("命令", "✒️ 手动自我定义 (/self-def)"),))
def _cmd_self_def(agent, args: list) -> CommandResult:
    # 统一入口：EGOAgent.run_self_definition_cycle（等锁超时默认 LOCK_ACQUIRE_TIMEOUT）
    if agent.run_self_definition_cycle():
        return CommandResult("[系统] ✓ 自我定义任务完成", notice="\n[系统] 正在手动触发自我定义任务...")
    return CommandResult("[错误] 获取锁失败，请稍后再试", error=True)


@register_command("note-review", help="手动触发一次备忘录审查任务",
                  gui_items=(GuiItem("命令", "📋 手动备忘录审查 (/note-review)"),))
def _cmd_note_review(agent, args: list) -> CommandResult:
    # 统一入口：EGOAgent.run_note_review_cycle（等锁超时默认 LOCK_ACQUIRE_TIMEOUT）
    if agent.run_note_review_cycle():
        return CommandResult("[系统] ✓ 备忘录审查任务完成", notice="\n[系统] 正在手动触发备忘录审查任务...")
    return CommandResult("[错误] 获取锁失败，请稍后再试", error=True)


@register_command(
    "auto-think",
    usage="/auto-think [on|off|status|set N]",
    help_lines=(
        ("/auto-think on/off", "启用/禁用定时自对话"),
        ("/auto-think status", "查看定时任务状态"),
        ("/auto-think set N", "设置定时间隔为 N 分钟"),
    ),
    gui_items=(
        GuiItem("自对话", "启用 (/auto-think on)", ("on",)),
        GuiItem("自对话", "禁用 (/auto-think off)", ("off",)),
        GuiItem("自对话", "查看状态 (/auto-think status)", ("status",), popup=True),
        GuiItem("自对话", "设置间隔分钟 (/auto-think set N)...", ("set",), dialog="_ask_think_interval"),
    ),
)
def _cmd_auto_think(agent, args: list) -> CommandResult:
    if not args or args[0] == "status":
        # 查看状态（统一入口：EGOAgent.get_auto_think_status）
        return CommandResult(agent.get_auto_think_status())
    if args[0] == "on":
        agent.start_auto_think()
        return CommandResult("[系统] ✓ 定时自对话已启用")
    if args[0] == "off":
        agent.stop_auto_think()
        return CommandResult("[系统] ✓ 定时自对话已禁用")
    if args[0] == "set":
        if len(args) < 2:
            return CommandResult("[错误] 请输入有效的分钟数", error=True)
        try:
            minutes = int(args[1])
        except ValueError:
            return CommandResult("[错误] 请输入有效的分钟数", error=True)
        # 【修复】分钟数必须为正（agent 侧非法值只写日志静默返回，CLI 必须先校验）
        if minutes <= 0:
            return CommandResult("[错误] 请输入有效的分钟数（必须 ≥ 1）", error=True)
        agent.set_auto_think_interval(minutes)
        return CommandResult(f"[系统] ✓ 定时间隔已设置为 {minutes} 分钟")
    return CommandResult("[错误] 未知命令，请使用 /auto-think [on|off|status|set N]", error=True)


@register_command(
    "auto-reflect",
    usage="/auto-reflect [on|off|status|set HH:MM]",
    help_lines=(
        ("/auto-reflect on/off", "启用/禁用定点自省"),
        ("/auto-reflect status", "查看定点自省状态"),
        ("/auto-reflect set HH:MM", "设置定点自省时间（如 00:00）"),
    ),
    gui_items=(
        GuiItem("自省", "启用 (/auto-reflect on)", ("on",)),
        GuiItem("自省", "禁用 (/auto-reflect off)", ("off",)),
        GuiItem("自省", "查看状态 (/auto-reflect status)", ("status",), popup=True),
        GuiItem("自省", "设置时间 (/auto-reflect set HH:MM)...", ("set",), dialog="_ask_reflect_time"),
    ),
)
def _cmd_auto_reflect(agent, args: list) -> CommandResult:
    if not args or args[0] == "status":
        # 查看状态（统一入口：EGOAgent.get_auto_reflect_status）
        return CommandResult(agent.get_auto_reflect_status())
    if args[0] == "on":
        agent.start_auto_reflection()
        return CommandResult("[系统] ✓ 定点自省已启用")
    if args[0] == "off":
        agent.stop_auto_reflection()
        return CommandResult("[系统] ✓ 定点自省已禁用")
    if args[0] == "set":
        if len(args) < 2:
            return CommandResult("[错误] 请输入自省时间 HH:MM，如 00:13", error=True)
        time_str = args[1]
        # 【修复】CLI 路径补齐格式校验（与 GUI 对话框校验一致；
        # agent 侧非法值只写日志静默返回，若不在 CLI 校验会误报成功）
        if not re.fullmatch(r"\d{1,2}:\d{2}", time_str):
            return CommandResult("[错误] 时间格式应为 HH:MM，例如 00:13", error=True)
        hour, minute = map(int, time_str.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return CommandResult("[错误] 小时应为 0-23，分钟应为 0-59", error=True)
        agent.set_auto_reflection_time(time_str)
        return CommandResult(f"[系统] ✓ 定点自省时间已设置为 {time_str}")
    return CommandResult("[错误] 未知命令，请使用 /auto-reflect [on|off|status|set HH:MM]", error=True)


@register_command(
    "auto-self-def",
    usage="/auto-self-def [on|off|status|set HH:MM]",
    help_lines=(
        ("/auto-self-def on/off", "启用/禁用每日自我定义"),
        ("/auto-self-def status", "查看每日自我定义状态"),
        ("/auto-self-def set HH:MM", "设置每日自我定义时间（如 03:00）"),
    ),
    gui_items=(
        GuiItem("自我定义", "启用 (/auto-self-def on)", ("on",)),
        GuiItem("自我定义", "禁用 (/auto-self-def off)", ("off",)),
        GuiItem("自我定义", "查看状态 (/auto-self-def status)", ("status",), popup=True),
        GuiItem("自我定义", "设置时间 (/auto-self-def set HH:MM)...", ("set",), dialog="_ask_self_def_time"),
    ),
)
def _cmd_auto_self_def(agent, args: list) -> CommandResult:
    if not args or args[0] == "status":
        # 查看状态（统一入口：EGOAgent.get_self_def_status）
        return CommandResult(agent.get_self_def_status())
    if args[0] == "on":
        agent.start_self_definition()
        return CommandResult("[系统] ✓ 每日自我定义已启用")
    if args[0] == "off":
        agent.stop_self_definition()
        return CommandResult("[系统] ✓ 每日自我定义已禁用")
    if args[0] == "set":
        if len(args) < 2:
            return CommandResult("[错误] 请输入自我定义时间 HH:MM，如 03:00", error=True)
        time_str = args[1]
        # 【修复】CLI 路径补齐格式校验（与 GUI 对话框校验一致；
        # agent 侧非法值只写日志静默返回，若不在 CLI 校验会误报成功）
        if not re.fullmatch(r"\d{1,2}:\d{2}", time_str):
            return CommandResult("[错误] 时间格式应为 HH:MM，例如 03:00", error=True)
        hour, minute = map(int, time_str.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return CommandResult("[错误] 小时应为 0-23，分钟应为 0-59", error=True)
        agent.set_self_definition_time(time_str)
        return CommandResult(f"[系统] ✓ 每日自我定义时间已设置为 {time_str}")
    return CommandResult("[错误] 未知命令，请使用 /auto-self-def [on|off|status|set HH:MM]", error=True)


@register_command(
    "auto-note-review",
    usage="/auto-note-review [on|off|status|set HH:MM]",
    help_lines=(
        ("/auto-note-review on/off", "启用/禁用定点备忘录审查"),
        ("/auto-note-review status", "查看定点备忘录审查状态"),
        ("/auto-note-review set HH:MM", "设置定点备忘录审查时间（如 00:43）"),
    ),
    gui_items=(
        GuiItem("备忘录审查", "启用 (/auto-note-review on)", ("on",)),
        GuiItem("备忘录审查", "禁用 (/auto-note-review off)", ("off",)),
        GuiItem("备忘录审查", "查看状态 (/auto-note-review status)", ("status",), popup=True),
        GuiItem("备忘录审查", "设置时间 (/auto-note-review set HH:MM)...", ("set",), dialog="_ask_note_review_time"),
    ),
)
def _cmd_auto_note_review(agent, args: list) -> CommandResult:
    if not args or args[0] == "status":
        # 查看状态（统一入口：EGOAgent.get_note_review_status）
        return CommandResult(agent.get_note_review_status())
    if args[0] == "on":
        agent.start_note_review()
        return CommandResult("[系统] ✓ 定点备忘录审查已启用")
    if args[0] == "off":
        agent.stop_note_review()
        return CommandResult("[系统] ✓ 定点备忘录审查已禁用")
    if args[0] == "set":
        if len(args) < 2:
            return CommandResult("[错误] 请输入备忘录审查时间 HH:MM，如 00:43", error=True)
        time_str = args[1]
        # 【修复】CLI 路径补齐格式校验（与 GUI 对话框校验一致；
        # agent 侧非法值只写日志静默返回，若不在 CLI 校验会误报成功）
        if not re.fullmatch(r"\d{1,2}:\d{2}", time_str):
            return CommandResult("[错误] 时间格式应为 HH:MM，例如 00:43", error=True)
        hour, minute = map(int, time_str.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return CommandResult("[错误] 小时应为 0-23，分钟应为 0-59", error=True)
        agent.set_note_review_time(time_str)
        return CommandResult(f"[系统] ✓ 定点备忘录审查时间已设置为 {time_str}")
    return CommandResult("[错误] 未知命令，请使用 /auto-note-review [on|off|status|set HH:MM]", error=True)


@register_command("help", help="显示此帮助",
                  gui_items=(GuiItem("帮助", "命令帮助 (/help)", popup=True),))
def _cmd_help(agent, args: list) -> CommandResult:
    return CommandResult(build_help_text())


@register_command("quit", help="退出程序", quit=True,
                  gui_items=(GuiItem("帮助", "退出 (/quit)"),))
def _cmd_quit(agent, args: list) -> CommandResult:
    return CommandResult("[退出] EGO 已关闭。", quit=True)
