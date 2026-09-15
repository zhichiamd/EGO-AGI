#!/usr/bin/env python3
"""
EGO 图形界面（GUI）—— 独立入口，不改动 CLI（ego.py）

功能：
- 与 EGO 对话（复用 agent.process_input）
- 支持附带图片（多模态，图片以 base64 data URL 透传给基座模型）
- 菜单栏提供 ego.py 全部命令（status/clear/think/reflect/auto-think/auto-reflect/help/quit）
- 信息型命令（status、auto-think status、auto-reflect status、help）以弹窗显示

启动方式：
    python EGO_GUI.py
"""

import os
import re
import base64
import queue
import logging
import mimetypes
import threading

# 【关键】必须在 import config/agent 之前加载环境变量（与 ego.py 保持一致的顺序）
try:
    from dotenv import load_dotenv
    # Nuitka --onefile 下 __file__ 指向临时目录，需用 exe 所在目录定位 env 文件
    _env_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(_env_dir, "untitled.env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    print("[Warning] python-dotenv not installed, skipping .env loading")
except Exception:
    pass

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog

from EGO_CLI import setup_logging      # 复用 ego.py 的日志初始化逻辑
from agent.core import EGOAgent
from agent import tools                # 手动命令注册表（菜单/分发/帮助均由此派生）
from config import (
    AUTO_THINK_INTERVAL_MINUTES,    # 自对话 间隔默认值（fallback 与 config 保持一致）
    SELF_DEFINITION_TIME,           # 每日自我定义时间默认值（fallback 与 config 保持一致）
    AUTO_NOTE_REVIEW_TIME,          # 定点备忘录审查时间默认值（fallback 与 config 保持一致）
    GUI_MIN_WIDTH,                  # GUI 窗口最小宽度
    GUI_MIN_HEIGHT,                 # GUI 窗口最小高度
)


# 帮助文本由 agent/tools.py 注册表派生（build_help_text），CLI /help 与 GUI 帮助弹窗共用


# ══ 初始化进度透传（纯 GUI 层：把 agent 关键日志实时显示到状态栏，不动 agent 代码）═
class _InitLogForwarder(logging.Handler):
    """初始化期间把 EGOAgent / LMStudioClient 的日志转发到 GUI 队列，显示在状态栏。

    大模型加载/生成较慢，初始化可能持续数分钟甚至更久；透传日志让用户
    实时看到当前阶段（验证会话 / 重建批次 / token 进度），而不是干等。
    """

    def __init__(self, q: "queue.Queue"):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record).strip()
            # 过滤调试噪声（SSE 原始数据等），只保留阶段/进度/警告信息
            if not msg or "[调试]" in msg:
                return
            self.q.put(("status", msg))
        except Exception:
            pass


# ══ 对话内容排版渲染（纯显示层：不改动发送/历史/日志中的任何内容）══════
# LaTeX 常用符号 → Unicode 直观符号（未收录的命令退化为去反斜杠文本）
LATEX_SYMBOLS = {
    # 箭头
    "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "longrightarrow": "⟶", "longleftarrow": "⟵", "mapsto": "↦", "uparrow": "↑",
    "downarrow": "↓",
    # 关系/运算
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝",
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·", "ast": "∗",
    "mid": "∣", "nmid": "∤",  # 整除/不整除（\mid / \nmid）
    "oplus": "⊕", "otimes": "⊗",
    # 集合/逻辑
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
    "supseteq": "⊇", "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "neg": "¬", "lnot": "¬", "land": "∧",
    "lor": "∨", "vdash": "⊢", "models": "⊨", "top": "⊤", "bot": "⊥",
    # 微积分/杂项
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏",
    "int": "∫", "sqrt": "√", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "degree": "°", "circ": "∘", "prime": "′", "dots": "…", "cdots": "⋯",
    "ldots": "…", "star": "⋆", "bullet": "•", "checkmark": "✓",
    # 希腊字母
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",
}


# 数学下标占位符（控制字符，正常文本不会出现；渲染层解析为小字号下标样式）
_SUB_L, _SUB_R = "\x02", "\x03"


def _convert_latex_piece(piece: str) -> str:
    """将单个 $...$ 内部内容转为直观文本（不识别的部分保留可读形态）"""
    s = piece.strip()
    # \text{...} / \mathrm{...}：行内逐处剥离为内容文本（可处理多段共存）
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", s)
    # 简单上下标：x^2 → x²、x_0 → x₀（数字直接用 Unicode 上下标字符）
    s = re.sub(r"\^(\d)", lambda m: "⁰¹²³⁴⁵⁶⁷⁸⁹"[int(m.group(1))], s)
    s = re.sub(r"_(\d)", lambda m: "₀₁₂₃₄₅₆₇₈₉"[int(m.group(1))], s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    # 字母下标：x_{new} → x + 下标占位（渲染层以小字号模拟真实下标）
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _SUB_L + m.group(1) + _SUB_R, s)
    # 已知符号替换（长名优先，避免 ightarrow 被 \r 截断）
    for name in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        s = s.replace("\\" + name, LATEX_SYMBOLS[name])
    # 剩余未知命令退化为去反斜杠文本；花括号仅包裹单段时剥离
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    m_brace = re.fullmatch(r"\{([^{}]*)\}", s)
    return m_brace.group(1) if m_brace else s


def _split_sub_markers(text: str):
    """把含下标占位符的文本切成 [(文本, 是否下标), ...]；无占位符时原样单段"""
    if _SUB_L not in text:
        return [(text, False)]
    out = []
    for part in text.split(_SUB_L):
        if _SUB_R in part:
            sub_text, rest = part.split(_SUB_R, 1)
            if sub_text:
                out.append((sub_text, True))
            if rest:
                out.append((rest, False))
        elif part:
            out.append((part, False))
    return out or [("", False)]


def render_inline_math(text: str) -> str:
    """将 $...$ 内联 / $$...$$ 块公式替换为 Unicode 直观形式；不成对的 $ 保持原样"""
    # 块公式 $$...$$ 先处理，避免被行内规则吞掉一半后残留外层 $
    text = re.sub(r"\$\$([^$\n]+?)\$\$",
                  lambda m: _convert_latex_piece(m.group(1)), text)
    return re.sub(r"\$([^$\n]+?)\$",
                  lambda m: _convert_latex_piece(m.group(1)), text)


# Markdown 结构识别（块级）
_RE_H = re.compile(r"^(#{1,6})\s+(.*)$")
_RE_UL = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_RE_OL = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_RE_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_RE_HR = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_RE_FENCE = re.compile(r"^\s*```(.*)$")
# 行内：代码/加粗/斜体（按此顺序匹配，避免相互干扰）
_RE_INLINE = re.compile(
    r"(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)|(__[^_\n]+__)")


def _is_indented_continuation(line: str) -> bool:
    """判断一行是否为列表项的缩进续行：以空白开头，且不是标题/列表/引用/分隔线/代码围栏等独立结构行"""
    return (bool(line) and line[0] in (" ", "\t")
            and not any(r.match(line) for r in (_RE_H, _RE_UL, _RE_OL, _RE_QUOTE, _RE_HR, _RE_FENCE)))


def render_inline(line: str):
    """行内样式切分：返回 [(文本, 样式标签元组), ...]；仅去标记不改字面

    样式标签：()=普通、("bold",)/("italic",)/("code_inline",)；
    数学下标叠加 ("sub",)，如 ("bold", "sub")。
    """
    line = render_inline_math(line)
    segs = []
    pos = 0
    for m in _RE_INLINE.finditer(line):
        if m.start() > pos:
            segs.append((line[pos:m.start()], ()))
        tok = m.group(0)
        if tok.startswith("`"):
            segs.append((tok[1:-1], ("code_inline",)))
        elif tok.startswith(("**", "__")):
            segs.append((tok[2:-2], ("bold",)))
        else:
            segs.append((tok[1:-1], ("italic",)))
        pos = m.end()
    if pos < len(line):
        segs.append((line[pos:], ()))

    # 展开数学下标占位符：下标段叠加 ("sub",) 小字号样式
    final = []
    for text, styles in (segs or [("", ())]):
        for piece, is_sub in _split_sub_markers(text):
            final.append((piece, styles + (("sub",) if is_sub else ())))
    return final or [("", ())]


def render_markdown_segments(text: str):
    """整段消息 → [(文本, 排版标签), ...]（含行内样式细分）。

    支持：标题 #、粗体/斜体、行内代码、``` 代码块、无序/有序列表（含缩进）、
    引用 >、分隔线。不识别的结构按普通段落原样显示；内容字面量不被修改。
    """
    out = []

    def emit(content, tag):
        for piece, styles in render_inline(content):
            # 标题本身已是加粗大字：行内加粗降级，避免字号被 bold(11pt) 覆盖
            if tag.startswith("h") and "bold" in styles:
                styles = tuple(s for s in styles if s != "bold")
            out.append((piece, styles or (tag,)))
        out.append(("\n", (tag,)))

    lines = text.split("\n")
    i, n = 0, len(lines)
    para = []

    def flush_para():
        if para:
            emit("\n".join(para), "body")
            para.clear()

    while i < n:
        line = lines[i]
        m_fence = _RE_FENCE.match(line)
        if m_fence:
            flush_para()
            body = []
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # 跳过闭合围栏（未闭合则自然到文末）
            code = "\n".join(body)
            # 代码块内部不做行内解析，字面保留
            out.append((code + ("\n" if code else ""), ("code_block",)))
            continue
        if line.strip():
            m_h = _RE_H.match(line)
            m_ul = _RE_UL.match(line)
            m_ol = _RE_OL.match(line)
            m_q = _RE_QUOTE.match(line)
            if m_h:
                flush_para()
                level = min(len(m_h.group(1)), 3)
                emit(m_h.group(2), "h%d" % level)
            elif m_ul:
                flush_para()
                indent = len(m_ul.group(1).expandtabs(4))
                li_tag = "li_sub" if indent >= 2 else "li"
                segs = render_inline(m_ul.group(2))
                out.append(("• ", (li_tag,)))
                for piece, styles in segs:
                    out.append((piece, styles or (li_tag,)))
                out.append(("\n", (li_tag,)))
                # 【修复】列表项的缩进续行归属同一列表项，避免被拆成独立 body 段落
                while i + 1 < n and _is_indented_continuation(lines[i + 1]):
                    i += 1
                    for piece, styles in render_inline(lines[i].lstrip()):
                        out.append((piece, styles or (li_tag,)))
                    out.append(("\n", (li_tag,)))
            elif m_ol:
                flush_para()
                num, content = m_ol.group(2), m_ol.group(3)
                segs = render_inline(content)
                out.append(("%s. " % num, ("li",)))
                for piece, styles in segs:
                    out.append((piece, styles or ("li",)))
                out.append(("\n", ("li",)))
                # 【修复】列表项的缩进续行归属同一列表项，避免被拆成独立 body 段落
                while i + 1 < n and _is_indented_continuation(lines[i + 1]):
                    i += 1
                    for piece, styles in render_inline(lines[i].lstrip()):
                        out.append((piece, styles or ("li",)))
                    out.append(("\n", ("li",)))
            elif m_q:
                flush_para()
                emit(m_q.group(1), "quote")
            elif _RE_HR.match(line):
                flush_para()
                emit("─" * 24, "hr")
            else:
                para.append(line)
        else:
            flush_para()
        i += 1
    flush_para()
    return out


class EGOApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.agent = None
        self.ready = False
        self._init_failed = False   # 初始化失败标记：失败后发送按钮复用为「重试初始化」入口
        self.pending_images: list = []   # 待发送的图片路径
        self.q: queue.Queue = queue.Queue()
        # 双独立锁：聊天和菜单命令互不干扰
        # _chat_busy 只挡新聊天，_cmd_busy 只挡并发命令
        self._chat_busy = False
        self._cmd_busy = False

        root.title("🚣🏻EGO (自演化智能体)")
        root.geometry("920x700")
        root.minsize(GUI_MIN_WIDTH, GUI_MIN_HEIGHT)

        self._build_ui()
        # 初始化完成前禁用发送按钮与菜单栏（init_ok / init_fail 时恢复），
        # 避免按钮/菜单可点击却只能提示"尚未初始化完成"
        self.send_btn.configure(state="disabled")
        self._set_menu_enabled(False)
        threading.Thread(target=self._init_agent, daemon=True).start()
        self.root.after(120, self._poll)

    # ── UI 构建 ────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu()

        self.status_var = tk.StringVar(value="正在初始化 EGO，请稍候 ...")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 relief="sunken").pack(side=tk.BOTTOM, fill=tk.X)

        input_frame = tk.Frame(self.root)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self.entry = tk.Text(input_frame, height=3, wrap="word",
                             font=("Microsoft YaHei UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self._on_enter)          # Enter 发送
        self.entry.bind("<Shift-Return>", lambda e: None)    # Shift+Enter 换行
        tk.Button(input_frame, text="📷 图片", command=self._pick_images).pack(side=tk.LEFT, padx=4)
        self.send_btn = tk.Button(input_frame, text="发送", command=self._send, width=8)
        self.send_btn.pack(side=tk.LEFT)

        self.img_bar = tk.Frame(self.root)
        self.img_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=6)

        self.chat = scrolledtext.ScrolledText(self.root, wrap="word", state="disabled",
                                              font=("Microsoft YaHei UI", 11))
        self.chat.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)
        # 基础节奏 tag：统一行距（所有插入内容均携带，保证全局一致）
        self.chat.tag_config("base", spacing1=2, spacing3=6)
        # 名字行自带消息间距：消息之间 = 上行 spacing3(6) + 名字行 spacing1(10)
        # 约等于段落内间距的 2 倍，紧凑分隔消息而不产生整行空白
        self.chat.tag_config("name_user", foreground="#1a6df2",
                             font=("Microsoft YaHei UI", 10, "bold"),
                             spacing1=10, spacing3=2)
        self.chat.tag_config("name_ego", foreground="#8e2f8e",
                             font=("Microsoft YaHei UI", 10, "bold"),
                             spacing1=10, spacing3=2)
        self.chat.tag_config("name_system", foreground="#888888",
                             font=("Microsoft YaHei UI", 10, "bold"),
                             spacing1=10, spacing3=2)
        # 排版样式 tag（与渲染层标签一一对应）
        self.chat.tag_config("h1", font=("Microsoft YaHei UI", 14, "bold"),
                             spacing1=8, spacing3=4)
        self.chat.tag_config("h2", font=("Microsoft YaHei UI", 13, "bold"),
                             spacing1=8, spacing3=4)
        self.chat.tag_config("h3", font=("Microsoft YaHei UI", 12, "bold"),
                             spacing1=6, spacing3=2)
        self.chat.tag_config("bold", font=("Microsoft YaHei UI", 11, "bold"))
        self.chat.tag_config("italic", font=("Microsoft YaHei UI", 11, "italic"))
        self.chat.tag_config("code_inline", font=("Consolas", 11),
                             background="#f2f2f2", foreground="#b03030")
        # 数学下标样式：小字号 + 轻微下沉模拟真实下标（render_inline 输出的 ("sub",)）
        self.chat.tag_config("sub", font=("Microsoft YaHei UI", 8), offset=-2)
        self.chat.tag_config("code_block", font=("Consolas", 10),
                             background="#f6f6f6",
                             lmargin1=10, lmargin2=10, spacing1=4, spacing3=6)
        self.chat.tag_config("li", lmargin1=16, lmargin2=30)
        self.chat.tag_config("li_sub", lmargin1=38, lmargin2=52)
        self.chat.tag_config("quote", lmargin1=14, lmargin2=14,
                             foreground="#666666")
        self.chat.tag_config("hr", foreground="#aaaaaa",
                             spacing1=4, spacing3=4)

    # 【注册表驱动】菜单栏：全部命令从 agent/tools.py 注册表派生
    # 新增命令只需在注册表声明 gui_items，菜单自动生成，无需改 GUI
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        menus = {}  # 分组名 → tk.Menu（分组顺序 = 注册表顺序）
        for spec in tools.all_commands():
            for item in spec.gui_items:
                if item.group not in menus:
                    menus[item.group] = tk.Menu(menubar, tearoff=0)
                    menubar.add_cascade(label=item.group, menu=menus[item.group])
                if item.dialog:
                    # 带参数对话框的菜单项：点击先弹框，再以用户输入执行
                    cmd = lambda d=item.dialog: getattr(self, d)()
                else:
                    cmd = lambda s=spec.name, a=item.args: self._run_command(s, list(a))
                menus[item.group].add_command(label=item.label, command=cmd)

        self.root.config(menu=menubar)
        self.menubar = menubar   # 保存引用，供初始化期间整体禁用/启用

    def _set_menu_enabled(self, enabled: bool):
        """禁用/启用整个菜单栏（命令 / 自动自对话 / 自动自省 / 帮助）"""
        state = "normal" if enabled else "disabled"
        try:
            for i in range(self.menubar.index("end") + 1):
                self.menubar.entryconfig(i, state=state)
        except tk.TclError:
            pass

    def _append(self, who: str, text: str):
        label = {"user": "用户", "ego": "EGO", "system": "系统"}[who]
        self.chat.configure(state="normal")
        # 不再前置空行：消息间距由名字行 spacing1 + 上行 spacing3 控制，
        # 避免相邻消息之间出现整行空白
        self.chat.insert(tk.END, f"{label} ▸ ", (f"name_{who}", "base"))
        if who in ("user", "ego"):
            # 排版渲染（纯显示层：仅标记解析与样式，不改动内容语义）
            segs = render_markdown_segments(text)
            for piece, styles in segs:
                names = tuple(s for s in styles if s)
                tags = ("base",) + names if names else ("base",)
                self.chat.insert(tk.END, piece, tags)
            # 渲染结果已自带换行结尾；空内容时补一个换行避免与下一条消息粘连
            if not segs or segs[-1][0] != "\n":
                self.chat.insert(tk.END, "\n", ("base",))
        else:
            self.chat.insert(tk.END, text + "\n", ("base",))
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

    # 【新增】信息型输出弹窗显示（不再堆进对话窗口）
    def _show_popup(self, title: str, text: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("600x520")
        win.transient(self.root)

        txt = scrolledtext.ScrolledText(win, wrap="word",
                                        font=("Microsoft YaHei UI", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        tk.Button(win, text="关闭", width=10, command=win.destroy).pack(pady=6)
        win.focus_set()

    # ── 初始化 / 事件循环 ──────────────────────────────────────
    def _init_agent(self):
        # 挂接日志透传：初始化期间把关键日志实时显示到状态栏（结束后移除）
        forwarder = _InitLogForwarder(self.q)
        for name in ("EGOAgent", "LMStudioClient"):
            logging.getLogger(name).addHandler(forwarder)
        try:
            agent = EGOAgent()
            self.agent = agent
            # 【新增】备忘录到期自主运行输出：推送至对话窗口显示（EGO 消息）
            agent.on_note_output = lambda text: self.q.put(("msg", ("ego", f"【备忘录到期】{text}")))
            self.ready = True
            self.q.put(("init_ok", f"模型: {agent.llm.model} @ {agent.llm.api_base}"))
        except Exception as e:
            self.q.put(("init_fail", str(e)))
        finally:
            for name in ("EGOAgent", "LMStudioClient"):
                logging.getLogger(name).removeHandler(forwarder)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "init_ok":
                    self._init_failed = False
                    self.status_var.set(f"就绪 | {payload}")
                    self.send_btn.configure(text="发送", state="normal")
                    self._set_menu_enabled(True)
                    self._append("system", "EGO 已就绪。可直接输入文字对话，"
                                           "或点击 📷 附带图片（需要多模态模型）。"
                                           "输入 /help 查看命令（斜杠命令与菜单栏一致）。")
                elif kind == "status":
                    # 初始化进度实时显示（截断超长消息）
                    self.status_var.set(payload if len(payload) <= 110 else payload[:107] + "...")
                elif kind == "init_fail":
                    self._init_failed = True
                    self.status_var.set("初始化失败")
                    self._append("system", f"初始化失败: {payload}\n"
                                           "请确认 LM Studio 已启动并加载了模型，"
                                           "然后点击「重试初始化」按钮。")
                    self.send_btn.configure(text="重试初始化", state="normal")
                    self._set_menu_enabled(True)
                elif kind == "msg":
                    self._append(payload[0], payload[1])
                elif kind == "popup":
                    self._show_popup(payload[0], payload[1])
                elif kind == "done":
                    # 仅聊天完成时解锁聊天状态
                    self._chat_busy = False
                    self.send_btn.configure(state="normal")
                    self.status_var.set("就绪")
                elif kind == "cmd_done":
                    # 仅命令完成时解锁命令状态（不影响聊天锁）
                    self._cmd_busy = False
                    if not self._chat_busy:
                        self.status_var.set("就绪")
                elif kind == "quit":
                    # 命令通道退出（/quit）：统一走窗口关闭清理流程
                    self._on_close()
                    return
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    # ── 图片选择与编码 ─────────────────────────────────────────
    def _pick_images(self):
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"),
                       ("所有文件", "*.*")],
        )
        self.pending_images.extend(paths)
        self._refresh_img_bar()

    def _refresh_img_bar(self):
        for child in self.img_bar.winfo_children():
            child.destroy()
        for i, p in enumerate(self.pending_images):
            chip = tk.Frame(self.img_bar, relief="groove", bd=1)
            chip.pack(side=tk.LEFT, padx=3, pady=2)
            tk.Label(chip, text=f"🖼 {os.path.basename(p)}").pack(side=tk.LEFT, padx=4)
            tk.Button(chip, text="✕", relief="flat",
                      command=lambda idx=i: self._remove_image(idx)).pack(side=tk.LEFT, padx=2)

    def _remove_image(self, idx: int):
        if idx < len(self.pending_images):
            self.pending_images.pop(idx)
            self._refresh_img_bar()

    @staticmethod
    def _to_data_url(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # ── 发送 ───────────────────────────────────────────────────
    def _on_enter(self, event):
        self._send()
        return "break"

    def _send(self):
        if self._chat_busy:
            return
        if not self.ready:
            if self._init_failed:
                # 初始化失败后，发送按钮复用为「重试初始化」入口
                self._init_failed = False
                self.send_btn.configure(text="发送", state="disabled")
                self._set_menu_enabled(False)
                self.status_var.set("正在重新初始化 EGO ...")
                self._append("system", "正在重新初始化 EGO ...")
                threading.Thread(target=self._init_agent, daemon=True).start()
            else:
                self._append("system", "EGO 尚未初始化完成，请稍候。")
            return

        text = self.entry.get("1.0", tk.END).strip()
        self.entry.delete("1.0", tk.END)

        if not text and not self.pending_images:
            return

        # 【注册表驱动】斜杠命令通道：与菜单栏共用同一分发（不占用聊天锁）
        if text.startswith("/"):
            self.pending_images.clear()
            self._refresh_img_bar()
            if not self.ready:
                self._append("system", "EGO 尚未初始化完成，请稍候。")
                return
            self._run_command_line(text)
            return

        if not text:
            text = "根据图片，说说你看到了什么？"

        images, self.pending_images = self.pending_images, []
        self._refresh_img_bar()

        display = text + (f"\n[📷 附带 {len(images)} 张图片]" if images else "")
        self._append("user", display)

        self._chat_busy = True
        self.send_btn.configure(state="disabled")
        self.status_var.set("EGO 正在思考（含自我对话循环，可能需要较长时间）...")
        threading.Thread(target=self._chat_worker, args=(text, images), daemon=True).start()

    def _chat_worker(self, text: str, image_paths: list):
        try:
            data_urls = [self._to_data_url(p) for p in image_paths]
            reply = self.agent.process_input(text, images=data_urls or None)
            self.q.put(("msg", ("ego", reply)))
        except Exception as e:
            self.q.put(("msg", ("system", f"处理失败: {e}")))
        finally:
            self.q.put(("done", None))

    # ── 菜单命令：参数输入对话框（UI 线程）─────────────────────
    def _ask_think_interval(self):
        if not self.ready:
            self._append("system", "EGO 尚未初始化完成。")
            return
        minutes = simpledialog.askinteger(
            "设置自对话间隔", "请输入定时间隔（分钟，≥1）:",
            parent=self.root, minvalue=1,
            initialvalue=getattr(self.agent, "_auto_think_interval", AUTO_THINK_INTERVAL_MINUTES * 60) // 60,
        )
        if minutes is not None:
            self._run_command("auto-think", ["set", str(minutes)])

    def _ask_reflect_time(self):
        if not self.ready:
            self._append("system", "EGO 尚未初始化完成。")
            return
        t = simpledialog.askstring(
            "设置自省时间", "请输入定点自省时间（HH:MM，如 00:13）:",
            parent=self.root,
            initialvalue=getattr(self.agent, "_auto_reflection_time", "00:13"),
        )
        if t is None:
            return
        t = t.strip()
        # 【校验】agent 侧对非法输入只写日志静默返回，GUI 必须先校验
        if not re.fullmatch(r"\d{1,2}:\d{2}", t):
            messagebox.showerror("格式错误", "时间格式应为 HH:MM，例如 00:13", parent=self.root)
            return
        hour, minute = map(int, t.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            messagebox.showerror("格式错误", "小时应为 0-23，分钟应为 0-59", parent=self.root)
            return
        self._run_command("auto-reflect", ["set", t])

    def _ask_self_def_time(self):
        if not self.ready:
            self._append("system", "EGO 尚未初始化完成。")
            return
        t = simpledialog.askstring(
            "设置自我定义时间", "请输入每日自我定义时间（HH:MM，如 03:00）:",
            parent=self.root,
            initialvalue=getattr(self.agent, "_self_def_time", SELF_DEFINITION_TIME),
        )
        if t is None:
            return
        t = t.strip()
        # 【校验】agent 侧对非法输入只写日志静默返回，GUI 必须先校验
        if not re.fullmatch(r"\d{1,2}:\d{2}", t):
            messagebox.showerror("格式错误", "时间格式应为 HH:MM，例如 03:00", parent=self.root)
            return
        hour, minute = map(int, t.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            messagebox.showerror("格式错误", "小时应为 0-23，分钟应为 0-59", parent=self.root)
            return
        self._run_command("auto-self-def", ["set", t])

    def _ask_note_review_time(self):
        if not self.ready:
            self._append("system", "EGO 尚未初始化完成。")
            return
        t = simpledialog.askstring(
            "设置备忘录审查时间", "请输入定点备忘录审查时间（HH:MM，如 00:43）:",
            parent=self.root,
            initialvalue=getattr(self.agent, "_note_review_time", AUTO_NOTE_REVIEW_TIME),
        )
        if t is None:
            return
        t = t.strip()
        # 【校验】agent 侧对非法输入只写日志静默返回，GUI 必须先校验
        if not re.fullmatch(r"\d{1,2}:\d{2}", t):
            messagebox.showerror("格式错误", "时间格式应为 HH:MM，例如 00:43", parent=self.root)
            return
        hour, minute = map(int, t.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            messagebox.showerror("格式错误", "小时应为 0-23，分钟应为 0-59", parent=self.root)
            return
        self._run_command("auto-note-review", ["set", t])

    # ── 输入框斜杠命令：注册表匹配（未知命令直接提示，不发给 LLM）──
    def _run_command_line(self, line: str):
        spec, args = tools.match_command(line)
        if spec is None:
            self._append("system", f"[错误] 未知命令: {line}，输入 /help 查看可用命令")
            return
        self._run_command(spec.name, args)

    # ── 菜单命令：执行（后台线程）──────────────────────────────
    def _run_command(self, name: str, args: list = None):
        if not self.ready:
            self._append("system", "EGO 尚未初始化完成。")
            return
        if self._cmd_busy:
            self._append("system", "上一个命令正在执行中，请稍后再试。")
            return
        self._cmd_busy = True
        self.status_var.set(f"正在执行 /{name} ...")
        threading.Thread(target=self._cmd_worker, args=(name, args or []), daemon=True).start()

    def _cmd_worker(self, name: str, args: list):
        agent = self.agent
        try:
            result = tools.execute(agent, name, args)
            if result.quit:
                # /quit：统一走窗口关闭清理流程（在 UI 线程 _poll 中执行）
                self.q.put(("quit", None))
                return
            # 信息型命令 → 弹窗（由注册表 gui_items 的 popup 标志派生）
            popup_title = None
            spec = tools.get_command(name)
            if spec is not None:
                for item in spec.gui_items:
                    if list(args)[:len(item.args)] == list(item.args) and item.popup:
                        popup_title = item.label
                        break
            if popup_title:
                self.q.put(("popup", (popup_title, result.message)))
            else:
                self.q.put(("msg", ("system", result.message)))
        except Exception as e:
            self.q.put(("msg", ("system", f"命令执行失败: {e}")))
        finally:
            self.q.put(("cmd_done", None))

    # ── 退出清理（统一入口：EGOAgent.shutdown，与 ego.py 共用）────────
    def _on_close(self):
        if self.agent is not None:
            try:
                # 统一清理入口：EGOAgent.shutdown（与 ego.py 共用）
                self.agent.shutdown()
            except Exception:
                pass
        self.root.destroy()


def main():
    setup_logging()
    root = tk.Tk()
    app = EGOApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()