"""
指令解析与执行系统

从 LLM 输出中提取指令并执行，返回执行结果。
指令格式：<TAG> 内容 </TAG>；工具类指令统一为 <TOOL> [名称] 内容 </TOOL>。

【注册表驱动】全部指令用 @register_instruction 装饰器注册（内置清单见 INSTRUCTION_TAGS）；
解析正则、纠错循环、执行优先级、系统提示词协议均由注册表派生。
新增指令只需写一个注册函数，无需修改任何其它位置。

内置指令：
- THINK：思考流标记（不执行）
- CONTINUE：继续自我对话
- SAY：输出内容给用户
- COG_ADD：向第二层自我核心添加认知
- COG_DEL：从第二层自我核心删除认知
- MEMO_RD：检索与用户的历史会话记录（包括输入和输出）
- NOTE_ADD：添加备忘录
- NOTE_RD：检索备忘录
- NOTE_DEL：删除备忘录
"""

from __future__ import annotations

import re
import logging
import traceback
from dataclasses import dataclass, field, replace

from agent.note_memory import NoteMemory, TRIGGER_TIME_RE
from agent.normalize import normalize_payload

# 获取 logger 实例
logger = logging.getLogger("InstructionExecutor")


def _auto_log(message: str, level: int = logging.INFO):
    """
    记录日志（级别由调用方显式指定，默认 INFO；不再靠关键词猜测）
    
    Args:
        message: 要记录的消息
        level: 日志级别（logging.INFO / WARNING / ERROR）
    """
    logger.log(level, message)


# ── 无效内容检测（统一实现，core 与 instructions 共用）─────────────
# 所有检测点共用的常见占位符/分隔符模式
INVALID_PLACEHOLDER_PATTERNS = ['---', '===', '...', '…', '。。。', '......']

# SAY 输出场景的额外模板残留占位符
_SAY_EXTRA_PLACEHOLDERS = frozenset(INVALID_PLACEHOLDER_PATTERNS) | {
    'Chinese output content', '[The summary/reflection]', '中文输出内容'
}


def is_invalid_content(text: str, say_level: bool = False) -> bool:
    """
    统一的无效内容检测（替代原先多处重复且不一致的内联实现）

    基础检测（所有场景）：
    - 内容为空，或 strip 后过短（<= 3 字符）
    - 命中常见占位符/分隔符（INVALID_PLACEHOLDER_PATTERNS）
    - 纯分隔符（^[-=]{3,}$）或纯点号（^.+$）

    say_level=True 时追加检测（SAY 输出场景）：
    - 命中模板残留占位符（Chinese output content 等）
    - 系统提示语泄漏（照搬系统提示词内容）
    - 短方括号包裹内容（<= 20 字符，典型占位符特征；
      不误伤 "[叹气]我不知道该怎么说[笑]" 等含方括号语气的正常回复）
    """
    if not text:
        return True
    if is_blank_text(text):
        return True
    stripped = text.strip()
    if len(stripped) <= 3:
        return True
    if stripped in INVALID_PLACEHOLDER_PATTERNS:
        return True
    if re.match(r'^[-=]{3,}$', stripped) or re.match(r'^\.+$', stripped):
        return True
    if say_level:
        if stripped in _SAY_EXTRA_PLACEHOLDERS:
            return True
        if "必须在内心进行自我对话思考" in stripped:
            return True
        if "自主决定使用" in stripped and "继续思考" in stripped and "停止思考" in stripped:
            return True
        if stripped.startswith('[') and stripped.endswith(']') and len(stripped) <= 20:
            return True
    return False


@dataclass
class ParsedInstruction:
    kind: str          # 指令类型
    payload: str       # 指令载荷（可能为空）
    raw: str           # 原始


@dataclass
class InstructionResult:
    """指令执行结果"""
    instruction: ParsedInstruction
    success: bool
    message: str
    data: dict = field(default_factory=dict)


# ── 指令注册表（单一事实源）──────────────────────────────────────
# 全部指令标签在此声明；解析正则、纠错循环、执行优先级、协议注入全部由此派生。
# 新增指令：用 @register_instruction 装饰执行函数即可，其它位置零改动。
INSTRUCTION_TAGS = ("THINK", "SAY", "TOOL", "CONTINUE", "MEMO_RD", "COG_ADD", "COG_DEL",
                    "NOTE_ADD", "NOTE_RD", "NOTE_DEL")

# 需经 <TOOL>[NAME] 解包的工具子指令（不作为顶层标签直接解析）
_TOOL_ONLY_INSTRUCTION_TAGS = ("CONTINUE", "MEMO_RD", "NOTE_ADD", "NOTE_RD", "NOTE_DEL")

# 匹配 <TOOL> 内首个 [名称] 前缀（工具子指令解包用）
_TOOL_PREFIX_RE = re.compile(r'^\s*\[([A-Za-z0-9_]+)\]\s*(.*)$', re.DOTALL)


@dataclass
class InstructionSpec:
    """指令规格：声明式描述一条指令的全部行为（借鉴 PI-Agent 注册表设计）"""
    tag: str                     # 指令标签名
    description: str = ""        # 用途说明（日志 / 帮助）
    handler: callable = None     # 执行函数 fn(self, instr) -> InstructionResult
    executable: bool = False     # 是否真正执行（嵌套提取/冷启动循环只执行 executable 指令）
    skip_execute: bool = False   # 纯标记，执行循环中直接跳过（THINK）
    output: bool = False         # 输出指令：payload 作为给用户的内容（SAY）
    route_back: bool = False     # 执行后转回 STEP_1 回路（CONTINUE / MEMO_RD）
    inject_result: bool = False  # 执行结果注入下一轮上下文（MEMO_RD）
    on_failure_feedback: bool = False  # 失败原因回注下一轮供 LLM 修正（COG_ADD / COG_DEL）
    priority: int = 99           # 执行优先级（小者先执行，替代原硬编码优先级表）
    protocol: str = ""           # 注入系统提示词的协议说明（新增指令时填写）
    normalize: dict = None       # 【归一化层】自然语言转译规则（如 {"kind": "note_add"}；None=不转译）


_REGISTRY: dict[str, InstructionSpec] = {}


def register_instruction(tag: str, **kw) -> callable:
    """装饰器：注册一条 LLM 自主指令。挂载 = 装饰，卸载 = 删除装饰。"""
    def deco(fn):
        _REGISTRY[tag] = InstructionSpec(tag=tag, handler=fn, **kw)
        return fn
    return deco


def get_instruction(tag: str):
    """按标签名查指令规格（未知标签返回 None）"""
    return _REGISTRY.get(tag)


def all_instructions() -> list:
    """返回全部指令规格（按注册顺序）"""
    return list(_REGISTRY.values())


# 【新格式】匹配 <TAG> 内容 </TAG> 的 XML 风格指令
# 【注册表驱动】标签列表从 _REGISTRY 实时派生：运行时注册的新指令（如 NOTE）
# 立即进入解析正则 / 引号保护 / 纠错循环，无需修改本文件任何常量
def _tag_names() -> str:
    """当前可解析的顶层指令标签（排除需经 <TOOL>[NAME] 解包的工具子指令）"""
    return "|".join(k for k in _REGISTRY if k not in _TOOL_ONLY_INSTRUCTION_TAGS)


def _normalize_tag_case(text: str) -> str:
    """将指令标签统一归一化为大写注册键（兼容 LLM 的大小写变体）。

    LLM 偶发将标签写成小写或混用大小写（如 <think>、</Think>、<Say>），
    导致闭合标签与开标签大小写不一致而解析失败（如 <THINK>...</think>）。
    解析/清理前统一将已注册标签（开/闭）转为注册表对应的大写形式，
    使后续精确匹配的各阶段（纠错/嵌套解析/清理）能正确配对。
    仅匹配 _tag_names()（不含需经 <TOOL>[NAME] 解包的工具子指令），
    不触碰载荷中的普通尖括号/引用文本。
    """
    tag_alt = _tag_names()
    return re.sub(
        rf'<(/?)({tag_alt})>',
        lambda m: f"<{m.group(1)}{m.group(2).upper()}>",
        text,
        flags=re.IGNORECASE,
    )


# ── 引号/反引号包裹的指令标记保护（parse_instructions 与 strip_instructions 共用）──
def _build_quote_protection_patterns(tag_alt: str) -> list:
    """构建“被引号包裹的指令标记”保护正则（标签列表实时派生自注册表）。

    所有引号/括号内的通配符都限制不跨行，防止吞噬其他指令标签；
    配对标签额外限制中间不能出现同类分隔符（避免误匹配）。

    同时保护开标签 `<TAG>` 与闭合标签 `</TAG>` 两类引用态标记：
    LLM 常在正文里用反引号/引号裸写指令标签作为“引用/示例”（如
    “对于 `<SAY>` 和 `</SAY>` 这两个特殊标记”）。若只保护开标签，
    闭合标签 `</TAG>` 会被误当作真实指令闭合，导致该指令载荷被截断。
    """
    return [
        # 反引号
        rf'`{{1,2}}([^`\u000a\u000d]*<({tag_alt})>[^<`\u000a\u000d]*</\2>[^`\u000a\u000d]*)`{{1,2}}',  # 配对
        rf'`{{1,2}}([^`\u000a\u000d]*<({tag_alt})>[^`\u000a\u000d]*)`{{1,2}}',  # 单开
        rf'`{{1,2}}([^`\u000a\u000d]*</({tag_alt})>[^`\u000a\u000d]*)`{{1,2}}',  # 闭合引用

        # 英文双引号
        rf'"([^"\u000a\u000d]*<({tag_alt})>[^<"\u000a\u000d]*</\2>[^"\u000a\u000d]*)"',  # 配对
        rf'"([^"\u000a\u000d]*<({tag_alt})>[^"\u000a\u000d]*)"',  # 单开
        rf'"([^"\u000a\u000d]*</({tag_alt})>[^"\u000a\u000d]*)"',  # 闭合引用

        # 中文双引号
        rf'\u201c([^\u201d\u000a\u000d]*<({tag_alt})>[^<\u201c\u201d\u000a\u000d]*</\2>[^\u201d\u000a\u000d]*)\u201d',  # 配对
        rf'\u201c([^\u201d\u000a\u000d]*<({tag_alt})>[^\u201d\u000a\u000d]*)\u201d',  # 单开
        rf'\u201c([^\u201d\u000a\u000d]*</({tag_alt})>[^\u201d\u000a\u000d]*)\u201d',  # 闭合引用

        # 方括号
        rf'\[([^\[\]\u000a\u000d]*<({tag_alt})>[^<\[\]\u000a\u000d]*</\2>[^\[\]\u000a\u000d]*)\]',  # 配对
        rf'\[([^\[\]\u000a\u000d]*<({tag_alt})>[^\[\]\u000a\u000d]*)\]',  # 单开
        rf'\[([^\[\]\u000a\u000d]*</({tag_alt})>[^\[\]\u000a\u000d]*)\]',  # 闭合引用
    ]


def _protect_quoted_tags(text: str, tag_alt: str):
    """保护被引号/反引号包裹的指令标记为占位符，返回 (受保护文本, 占位符映射)。"""
    placeholder_map = {}
    counter = [0]

    def replace_quoted(match):
        placeholder = f"__QUOTED_{counter[0]}__"
        placeholder_map[placeholder] = match.group(0)
        counter[0] += 1
        return placeholder

    for pattern in _build_quote_protection_patterns(tag_alt):
        text = re.sub(pattern, replace_quoted, text)
    return text, placeholder_map


def _restore_placeholders(text: str, placeholder_map: dict) -> str:
    """将占位符恢复为原始文本。"""
    for placeholder, original in placeholder_map.items():
        text = text.replace(placeholder, original)
    return text


def _is_placeholder_payload(payload: str) -> bool:
    """判断指令载荷是否为"格式示例占位符"（如 <TAG> ... </TAG> 中的 '...'）。

    LLM 偶发在输出中裸写指令标签作为格式清单（如"检查格式：- <COG_ADD> ... </COG_ADD>"），
    此类标签的载荷是纯占位符，应视为格式示例而非真实指令，解析时直接丢弃，
    避免产生伪指令（虚假 COG_ADD/SAY）及由其触发的虚假失败反馈污染下一轮。
    该检测刻意窄于 is_invalid_content：不比较长度，避免误伤正常短查询（如 MEMO_RD 关键词）。
    """
    stripped = payload.strip()
    return (
        stripped in INVALID_PLACEHOLDER_PATTERNS or
        re.match(r'^[-=]{3,}$', stripped) is not None or
        re.match(r'^\.+$', stripped) is not None
    )


def parse_instructions(text: str) -> list[ParsedInstruction]:
    """从文本中提取所有指令（XML标签格式：<TAG> 内容 </TAG>）"""
    results = []
    tag_names = _tag_names()
    # 【修复】标签大小写归一化：LLM 偶发将闭合标签写成小写（如 </think>），
    # 先统一为注册表大写形式，避免开/闭标签大小写不一致导致指令解析失败。
    text = _normalize_tag_case(text)
    
    # 【修复】两阶段预处理 + 嵌套感知解析
    
    # ===== 阶段1：保护被引号/反引号包裹的指令标记 =====
    temp_text, placeholder_map = _protect_quoted_tags(text, tag_names)
    
    # ===== 阶段1.5：自动纠正常见的标签错误 =====
    # 修复 LLM 常见的错误：
    # 1. 用 <TAG> 代替 </TAG> 作为闭合标签（如 <SAY> 内容 <SAY> → <SAY> 内容 </SAY>）
    # 2. THINK 内裸引用指令标签导致计数异常（通过栈匹配精确识别多余标签）
    for tag_name in list(_REGISTRY):
        open_tag = f"<{tag_name}>"
        close_tag = f"</{tag_name}>"
        
        # 使用栈匹配法找出所有未匹配的开标签位置
        open_positions = [m.start() for m in re.finditer(re.escape(open_tag), temp_text)]
        close_positions = [m.start() for m in re.finditer(re.escape(close_tag), temp_text)]
        
        if not open_positions:
            continue
        
        # 栈匹配：模拟标签配对过程
        # 未匹配的开标签 = 要么是 THINK 内的裸引用（应被忽略），要么是误用的闭标签（应替换为 </TAG>）
        stack = list(open_positions)  # 复制一份，从前往后处理
        for cp in close_positions:
            # 找到栈中在 cp 之前的最后一个开标签（它被这个闭标签匹配）
            candidates = [op for op in stack if op < cp]
            if candidates:
                stack.remove(candidates[-1])  # 移除已匹配的开标签
        
        # stack 中剩余的就是未匹配的开标签
        if not stack:
            continue
        
        if len(open_positions) > len(close_positions):
            excess = len(open_positions) - len(close_positions)
            
            # 关键判断：
            # - 如果 len(stack) > excess：说明有 THINK 内裸引用（如“使用 <SAY> 表达”），
            #   裸引用不影响解析，无需纠错
            # - 如果 len(stack) == excess：说明所有未匹配开标签都是多余的，
            #   即 LLM 用开标签代替了闭标签（如 <SAY> 内容 <SAY>）
            #   此时保留第一个开标签（真正的开标签），将其余替换为闭标签
            if len(stack) == excess and len(stack) >= 2:
                to_replace = stack[1:]  # 保留第一个，替换其余
                for pos in sorted(to_replace, reverse=True):
                    temp_text = temp_text[:pos] + close_tag + temp_text[pos + len(open_tag):]

    # ===== 阶段1.6：修复闭合标签张冠李戴 =====
    # 处理 <COG_DEL> xxx </COG_ADD> 这类闭合标签写错的情况
    # 策略：对于每个开标签 <TAG>，如果其后最近的闭合标签是 </OTHER_TAG> 而非 </TAG>，
    #       且该 </OTHER_TAG> 没有对应的 <OTHER_TAG> 开标签，则将其替换为 </TAG>
    TAG_LIST = list(_REGISTRY)
    for tag_name in TAG_LIST:
        open_tag = f"<{tag_name}>"
        close_tag = f"</{tag_name}>"
        
        found_mismatch = True
        while found_mismatch:
            found_mismatch = False
            for open_match in re.finditer(re.escape(open_tag), temp_text):
                nearest_close = None
                nearest_close_name = None
                for other_tag in TAG_LIST:
                    other_close = f"</{other_tag}>"
                    m = re.search(re.escape(other_close), temp_text[open_match.end():])
                    if m:
                        abs_pos = open_match.end() + m.start()
                        if nearest_close is None or abs_pos < nearest_close:
                            nearest_close = abs_pos
                            nearest_close_name = other_tag
                
                if nearest_close_name and nearest_close_name != tag_name:
                    other_opens = len(re.findall(rf'<{nearest_close_name}>', temp_text))
                    other_closes = len(re.findall(rf'</{nearest_close_name}>', temp_text))
                    if other_closes > other_opens:
                        wrong_close = f"</{nearest_close_name}>"
                        temp_text = temp_text[:nearest_close] + close_tag + temp_text[nearest_close + len(wrong_close):]
                        found_mismatch = True
                        break
    
    # ===== 阶段2：基于嵌套层级的指令解析 =====
    instructions_found = []
    
    # 查找所有可能的指令起始位置（标签列表实时派生自注册表）
    instruction_start_pattern = re.compile(rf'<({tag_names})>')
    
    for start_match in instruction_start_pattern.finditer(temp_text):
        instr_kind = start_match.group(1)
        start_pos = start_match.start()
        content_start = start_match.end()
        
        # 查找对应的闭合标签 </TAG>
        close_tag = f"</{instr_kind}>"
        end_pos = None
        
        # 【修复】两阶段查找：先严格匹配（内部指令全部闭合），再宽松匹配（引用模式）
        for pass_mode in ('strict', 'lenient'):
            search_from = content_start
            while search_from < len(temp_text):
                end_marker = temp_text.find(close_tag, search_from)
                if end_marker == -1:
                    break
                
                candidate_end = end_marker + len(close_tag)
                
                if pass_mode == 'strict':
                    # 严格模式：检查内部指令是否全部闭合
                    content_between = temp_text[content_start:end_marker]
                    has_unclosed_inner = False
                    for tag_name in list(_REGISTRY):
                        if tag_name == instr_kind:
                            continue
                        if content_between.count(f'<{tag_name}>') > content_between.count(f'</{tag_name}>'):
                            has_unclosed_inner = True
                            break
                    
                    if has_unclosed_inner:
                        search_from = candidate_end
                    else:
                        end_pos = candidate_end
                        break
                else:
                    end_pos = candidate_end
                    break
            
            if end_pos is not None:
                break  # 严格模式已找到，不再尝试宽松模式
        
        if end_pos is not None:
            # 成功找到完整的指令
            raw_instruction = temp_text[start_pos:end_pos]
            # 提取 payload（开标签和闭标签之间的内容）
            payload = temp_text[content_start:end_pos - len(close_tag)].strip()
            
            instructions_found.append({
                'start': start_pos,
                'end': end_pos,
                'kind': instr_kind,
                'payload': payload,
                'raw': raw_instruction
            })
    
    # ===== 阶段2.5：解包统一工具标签 <TOOL>[名称] =====
    # LLM 以 <TOOL> [NAME] 内容 </TOOL> 输出工具指令（CONTINUE/MEMO_RD/NOTE_ADD/NOTE_RD/NOTE_DEL），
    # 解包为真实指令使下游（注册表分发/归一化/核心循环）零改动；[名称] 命中工具集合才解包，
    # 否则保留为 TOOL 交由容器 handler 报错回馈（不静默丢失），供 LLM 下一轮自愈。
    for instr in instructions_found:
        if instr['kind'] == 'TOOL':
            m = _TOOL_PREFIX_RE.match(instr['payload'])
            if m and m.group(1).upper() in _TOOL_ONLY_INSTRUCTION_TAGS:
                instr['kind'] = m.group(1).upper()
                instr['payload'] = m.group(2)

    # ===== 阶段3：去重和过滤重叠指令 =====
    instructions_found.sort(key=lambda x: x['start'])
    
    filtered_instructions = []
    last_end = -1
    
    for instr in instructions_found:
        if instr['start'] >= last_end:
            filtered_instructions.append(instr)
            last_end = instr['end']
    
    # ===== 阶段4：恢复占位符并构建结果 =====
    for instr in filtered_instructions:
        # 恢复 payload 中的占位符
        payload = instr['payload']
        for placeholder, original in placeholder_map.items():
            payload = payload.replace(placeholder, original)
        
        # 恢复 raw 中的占位符
        raw = instr['raw']
        for placeholder, original in placeholder_map.items():
            raw = raw.replace(placeholder, original)
        
        # 【修复】丢弃"格式示例占位符"指令：LLM 有时在输出中裸写指令标签作为格式清单
        # （如"检查格式：- <COG_ADD> ... </COG_ADD>"），其载荷为纯占位符（.../…/--- 等），
        # 应视为格式示例而非真实指令。若不丢弃，会解析出伪 COG_ADD/SAY：
        # 伪 COG_ADD 因 on_failure_feedback 注入虚假"执行失败"反馈，污染下一轮上下文。
        if _is_placeholder_payload(payload):
            _auto_log(f"[指令] 丢弃格式示例占位符指令: <{instr['kind']}> 载荷={payload!r}", level=logging.DEBUG)
            continue
        
        results.append(ParsedInstruction(
            kind=instr['kind'],
            payload=payload,
            raw=raw
        ))
    
    # ===== 阶段5：同内容重复指令去重（保留最后一次） =====
    # LLM 常在思考/回顾段裸写指令清单（如"最终指令生成"/"格式检查"小节）,
    # 随后再在"输出"小节真正输出同一指令；即使未遵守"非真实使用必须用反引号包裹"的
    # 提示词约束（prompts.py 规则六），相同 (kind, payload) 的重复指令也仅应保留末尾一次
    # （真实执行），前面出现的裸写回顾按无效化处理，避免重复副作用及 on_failure_feedback
    # 注入"伪失败反馈"污染下一轮（如重复 COG_DEL 已删 ID 触发失败回注）。
    # 仅当同一 (kind, payload) 在该响应中多次出现时才去重（单次出现不受影响）。
    if len(results) > 1:
        last_occurrence = {}
        for idx, instr in enumerate(results):
            last_occurrence[(instr.kind, instr.payload)] = idx
        deduped = []
        for idx, instr in enumerate(results):
            if last_occurrence[(instr.kind, instr.payload)] == idx:
                deduped.append(instr)
            else:
                _auto_log(
                    f"[指令] 丢弃重复指令(无效化回顾展示): <{instr.kind}> 载荷={instr.payload[:40]!r}",
                    level=logging.DEBUG,
                )
        results = deduped
    
    return results


# 不可见字符（零宽空格/零宽关联符、行/段分隔符、BOM 等）：.strip() 无法去除，需显式视为空
# 例：LLM 偶发输出 <CONTINUE>\u200b</CONTINUE>，payload.strip() 后仍含零宽字符而被误判非空执行
_INVISIBLE_CHARS_RE = re.compile(r'[\s\u200b-\u200f\u2028\u2029\u202f\u205f\ufeff]')


def is_blank_text(text: str) -> bool:
    """文本是否"实质为空"：纯空白或仅含不可见字符（零宽空格/零宽关联符/BOM 等）视为空。

    与 is_invalid_content 的"过短/占位符"判定不同，is_blank_text 只回答"是否存在可见字符"。
    供指令空载荷过滤（filter_empty_payload）与输出清洗（_sanitize_output/_handle_say_result）统一判空。
    """
    return not (text and _INVISIBLE_CHARS_RE.sub("", text))


def filter_empty_payload(instructions: list) -> list:
    """过滤空载荷指令：<TAG>  </TAG> 等包裹文本为空、纯空白或仅含不可见字符的指令整条跳过。

    空 CONTINUE 若执行会触发无意义继续轮，空 COG_ADD/COG_DEL 会误报失败反馈。
    所有执行路径（主递归环/冷启动/预热/Think/反思/自省）在解析后统一调用。

    严谨判定：除常规空白（strip）外，还将零宽空格/零宽关联符/BOM 等不可见字符视为空，
    避免带不可见字符的空壳指令被误判为非空而执行。
    """
    return [instr for instr in instructions
            if instr.payload and not is_blank_text(instr.payload)]


def strip_instructions(text: str) -> str:
    """移除文本中的指令标记，返回纯文本（标签列表实时派生自注册表）。

    与 parse_instructions 一致：先保护被引号/反引号包裹的指令标记（引用态），
    避免剥离用户可见内容里对指令标签的引用/示例（如 SAY 输出里的 `<THINK>...</THINK>`）。
    """
    text = _normalize_tag_case(text)
    # 保护引用态标签（反引号/引号包裹）→ 剥离真实指令 → 恢复引用
    protected, placeholder_map = _protect_quoted_tags(text, _tag_names())
    instr_re = re.compile(rf"<({_tag_names()})>\s*([\s\S]*?)\s*</\1>")
    protected = instr_re.sub("", protected)
    return _restore_placeholders(protected, placeholder_map).strip()


class InstructionExecutor:
    """指令执行器，依赖 PromptManager 和 ChromaMemoryManager"""

    def __init__(self, prompt_manager, memory_system, note_system=None):
        self.pm = prompt_manager
        self.mem = memory_system
        self.note = note_system        # 备忘录存储（NOTE 指令依赖；None 时 NOTE 指令返回错误）
        # 【新增】存储当前执行的上下文信息
        self._current_context = {}

    def set_context(self, context: dict):
        """
        设置当前执行的上下文信息
        
        Args:
            context: 上下文字典，可包含：
                - stage: 阶段标识（coldstart/preheat/think/chat）
                - round_number: 轮次编号
                - step: 步骤编号
                - user_input: 用户输入
                - conversation_index: 对话索引
        """
        self._current_context = context or {}

    def execute(self, instr: ParsedInstruction) -> InstructionResult:
        """执行单条指令（按注册表分发）"""
        spec = _REGISTRY.get(instr.kind)
        if spec is None or spec.handler is None:
            return InstructionResult(instr, False, f"未知指令: {instr.kind}")
        # 【归一化层】自然语言 payload → 标准协议文本（幂等；失败/熔断/未启用 → 透传原样）
        # 用 replace 生成新指令对象，不污染调用方持有的原 instr
        # 【修复】归一化独立保护：转译异常时降级透传原 payload，不影响指令执行（与降级哲学一致）
        if spec.normalize and instr.payload:
            try:
                normalized = normalize_payload(spec.normalize["kind"], instr.payload)
                if normalized is not None and normalized != instr.payload:
                    _auto_log(f"[归一化] {instr.kind} 转译: {instr.payload[:60]!r} → {normalized[:80]!r}")
                    instr = replace(instr, payload=normalized)
            except Exception as e:
                _auto_log(f"[归一化] {instr.kind} 转译异常，透传原 payload: {e}", level=logging.WARNING)
        # 【修复】执行异常兜底：handler 抛异常时转失败结果（而非向上传播导致整轮输出丢失），
        # 失败消息经 on_failure_feedback / inject_result 机制注入下一轮，LLM 可感知并自愈
        try:
            return spec.handler(self, instr)
        except Exception as e:
            _auto_log(f"[指令] ✗ {instr.kind} 执行异常: {e}", level=logging.WARNING)
            _auto_log(traceback.format_exc())
            return InstructionResult(instr, False, f"{instr.kind} 执行异常: {e}")

    def execute_all(self, instructions: list[ParsedInstruction]) -> list[InstructionResult]:
        """按序执行所有指令"""
        return [self.execute(i) for i in instructions]

    # ── 自我控制指令 ────────────────────────────────────────────

    @register_instruction("THINK", description="思考流标记（不执行）",
                          skip_execute=True, priority=5)
    def _cmd_think(self, instr: ParsedInstruction) -> InstructionResult:
        return InstructionResult(instr, True, "思考流标记，不执行")

    @register_instruction("TOOL", description="统一工具标签容器（需 [名称] 前缀解包为具体工具）",
                          executable=False, on_failure_feedback=True, priority=99)
    def _cmd_tool_container(self, instr: ParsedInstruction) -> InstructionResult:
        return InstructionResult(instr, False,
            "TOOL 是容器标签，需形如 <TOOL> [MEMO_RD] 内容 </TOOL>；缺少或未知 [名称] 前缀无法解析")

    @register_instruction("CONTINUE", description="继续自我对话",
                          route_back=True, priority=4)
    def _cmd_continue(self, instr: ParsedInstruction) -> InstructionResult:
        return InstructionResult(instr, True, "继续自我对话", {"action": "continue", "content": instr.payload})


    @register_instruction("COG_ADD", description="向第二层自我核心添加认知",
                          executable=True, on_failure_feedback=True, priority=1)
    def _cmd_update_l2_add(self, instr: ParsedInstruction) -> InstructionResult:
        """进行内容有效性检测，防止占位符被添加"""
        if not instr.payload:
            return InstructionResult(instr, False, "COG_ADD 需要提供添加内容")
        
        # 检测内容是否为无效占位符或分隔符
        # 统一 is_invalid_content 基础检测 + COG_ADD 特有的首尾省略号检查
        _payload_invalid = (
            is_invalid_content(instr.payload) or
            instr.payload.startswith('...') or
            instr.payload.endswith('...')
        )
        
        if _payload_invalid:
            _auto_log(f"[警告] ✗ COG_ADD 内容为无效占位符: '{instr.payload}'", level=logging.WARNING)
            return InstructionResult(instr, False, f"COG_ADD 内容无效（占位符）: {instr.payload}")
        
        self.pm.add_layer2(instr.payload)
        return InstructionResult(instr, True, f"已向第二层自我核心添加：{instr.payload[:80]}")

    @register_instruction("COG_DEL", description="从第二层自我核心删除认知",
                          executable=True, on_failure_feedback=True, priority=2)
    def _cmd_update_l2_del(self, instr: ParsedInstruction) -> InstructionResult:
        if not instr.payload:
            return InstructionResult(instr, False, "COG_DEL 需要提供删除关键词")
        
        # 【修复】检查删除操作是否真正成功，失败时将具体原因（含拒删时的候选 ID）透传给 LLM
        deleted, reason = self.pm.del_layer2(instr.payload)
        
        if deleted:
            return InstructionResult(instr, True, f"已从第二层自我核心删除匹配项：{instr.payload[:80]}")
        else:
            return InstructionResult(instr, False, reason)

    # ── 信息操作指令 ────────────────────────────────────────────
    @register_instruction("MEMO_RD", description="检索历史会话",
                          executable=True, route_back=True, inject_result=True, priority=0,
                          normalize={"kind": "memo_rd"})
    def _cmd_recall(self, instr: ParsedInstruction) -> InstructionResult:
        if not instr.payload:
            return InstructionResult(instr, False, "MEMO_RD 需要提供检索关键词")
        # 【修复】与 NOTE 系列一致：记忆系统未初始化（Chroma 故障）时返回明确错误而非异常
        if self.mem is None:
            return InstructionResult(instr, False, "记忆系统未初始化")
        
        # 解析角色过滤：@user 或 @ego 前缀
        payload = instr.payload.strip()
        role_filter = None
        m = re.match(r'^@(user|ego)\s+(.+)$', payload, re.DOTALL)
        if m:
            role_filter = "user" if m.group(1) == "user" else "assistant"
            payload = m.group(2).strip()
        
        if not payload:
            return InstructionResult(instr, False, "MEMO_RD 需要提供检索关键词")
        
        # 【修复】传入本轮对话开始时间，过滤掉当前问题本身（刚写入就被检索回来）
        recall_after = self._current_context.get("recall_after")
        results = self.mem.recall(payload, role=role_filter, recall_after=recall_after)
        formatted = self.mem.format_anchors(results, for_llm=True)
        return InstructionResult(instr, True, formatted, {"results": results})      

    @register_instruction("SAY", description="输出内容给用户",
                          output=True, priority=3)
    def _cmd_output(self, instr: ParsedInstruction) -> InstructionResult:
        return InstructionResult(instr, True, instr.payload, {"action": "output", "content": instr.payload})


    # ── 备忘录指令 ────────────────────────────────────────────
    # 协议文案见 prompts.py LAYER1_DEFAULT.rules「（三）备忘录」，已在系统提示词中，
    # 注册时 protocol 留空避免重复注入；加入 INSTRUCTION_TAGS 保持内置清单语义完整。
    # 优先级：NOTE_RD=0（检索优先，与 MEMO_RD 同级）/ NOTE_ADD=1 / NOTE_DEL=2

    # 【降级自愈】归一化层不可用时（52625 故障/熔断），payload 以自然语言透传到 handler，
    # 失败消息携带完整协议格式，供 LLM 下一轮直接按协议输出，避免反复失败
    _NOTE_ADD_FORMAT_HINT = "正确格式：[性质：执行] [触发时间：YYYY-MM-DD-HH:MM] [标题:xxx] [内容:xxx]（备忘类省略触发时间）"

    @register_instruction("NOTE_ADD", description="添加备忘录条目",
                          executable=True, on_failure_feedback=True, priority=1,
                          normalize={"kind": "note_add"})
    def _cmd_note_add(self, instr: ParsedInstruction) -> InstructionResult:
        """解析 [性质：执行/备忘] [触发时间：] [标题:] [内容:] 并存储"""
        if self.note is None:
            return InstructionResult(instr, False, "备忘录系统未初始化")
        if not instr.payload:
            return InstructionResult(instr, False, f"NOTE_ADD 需要提供 [性质] [标题] [内容] 等字段；{self._NOTE_ADD_FORMAT_HINT}")
        fields = NoteMemory.parse_payload(instr.payload)
        nature = fields["nature"] or "备忘"
        if nature not in ("执行", "备忘"):
            return InstructionResult(instr, False, f"NOTE_ADD 性质无效: {nature}（应为 执行/备忘）")
        if not fields["title"] or not fields["content"]:
            return InstructionResult(instr, False, f"NOTE_ADD 需要提供 [标题] 和 [内容]；{self._NOTE_ADD_FORMAT_HINT}")
        trigger_time = fields["trigger_time"]
        if nature == "执行":
            if not trigger_time:
                return InstructionResult(instr, False, "执行类备忘录需要提供 [触发时间]（格式 YYYY-MM-DD-HH:MM）")
            if not TRIGGER_TIME_RE.match(trigger_time):
                return InstructionResult(instr, False, f"触发时间格式无效: {trigger_time}（应为 YYYY-MM-DD-HH:MM）")
            # 【新增】拒绝过去时间：LLM 误填已过期时间会造成立即意外触发（补录场景不存在，
            # NOTE_ADD 唯一来源是 LLM 自主创建）；失败消息携带实时时间供 LLM 下一轮修正
            from datetime import datetime
            now_str = datetime.now().strftime("%Y-%m-%d-%H:%M")
            if trigger_time <= now_str:
                return InstructionResult(instr, False,
                    f"触发时间 {trigger_time} 早于当前时间 {now_str}，请填写之后的时间")
        # 备忘类不自动触发：LLM 误填 [触发时间]（含格式错误）时置空存储，
        # 避免 NOTE_RD 显示误导性的“触发时间”字段，也杜绝任何触发可能
        ignored_time = trigger_time if nature == "备忘" else ""
        if nature == "备忘":
            trigger_time = ""
        note = self.note.add(nature, fields["title"], fields["content"], trigger_time)
        msg = f"已添加备忘录 {note['id']}（{nature}类）"
        if ignored_time:
            msg += f"；已忽略触发时间 {ignored_time}（备忘类不自动触发）"
        return InstructionResult(instr, True, msg)

    @register_instruction("NOTE_RD", description="读取备忘录条目",
                          executable=True, route_back=True, inject_result=True, priority=0,
                          normalize={"kind": "note_rd"})
    def _cmd_note_rd(self, instr: ParsedInstruction) -> InstructionResult:
        """ID → 完整内容；关键词 → 部分字段；LIST → 全部部分字段"""
        if self.note is None:
            return InstructionResult(instr, False, "备忘录系统未初始化")
        if not instr.payload:
            return InstructionResult(instr, False, "NOTE_RD 需要提供 备忘录ID / 关键词 / LIST")
        keyword = instr.payload.strip()
        if keyword.upper() == "LIST":
            notes = self.note.list_all()
            return InstructionResult(instr, True, self.note.format(notes), {"results": notes})
        if re.match(r"^N_\d+$", keyword, re.IGNORECASE):
            note = self.note.get(keyword)
            if note is None:
                # 【设计】查无此 ID：视为"空检索结果"而非失败（success=True），
                # 由核心循环将"未找到"提示注入下一轮，告知 LLM 检索结果为空
                return InstructionResult(instr, True, f"未找到备忘录条目: {keyword}", {"results": []})
            return InstructionResult(instr, True, self.note.format([note], full=True), {"results": [note]})
        notes = self.note.search(keyword)
        if not notes:
            # 【设计】检索执行成功但结果为空：success=True + "未找到"消息 + 空 results，
            # 核心循环据此将空结果提示注入下一轮——LLM 发起 NOTE_RD 时可能正处于
            # 等待检索结果的挂起状态（未输出 SAY），必须被告知检索结果为空
            return InstructionResult(instr, True, f"未找到与「{keyword}」匹配的备忘录", {"results": []})
        return InstructionResult(instr, True, self.note.format(notes), {"results": notes})

    @register_instruction("NOTE_DEL", description="删除备忘录条目",
                          executable=True, on_failure_feedback=True, priority=2)
    def _cmd_note_del(self, instr: ParsedInstruction) -> InstructionResult:
        """仅允许纯 ID（协议约束：不携带注解文本）"""
        if self.note is None:
            return InstructionResult(instr, False, "备忘录系统未初始化")
        if not instr.payload:
            return InstructionResult(instr, False, "NOTE_DEL 需要提供备忘录 ID")
        note_id = instr.payload.strip()
        if not re.match(r"^N_\d+$", note_id, re.IGNORECASE):
            return InstructionResult(instr, False, "NOTE_DEL 只允许使用条目 ID（如 N_001），不要添加注解文本")
        deleted, reason = self.note.delete(note_id)
        return InstructionResult(instr, deleted, reason)