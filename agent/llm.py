"""
LM Studio API 客户端

支持模式：新版 Responses API (/v1/responses) - 有状态，支持 KV cache 复用（推荐）
"""

from __future__ import annotations

import requests
import logging
import re
from datetime import datetime
from typing import Optional
from config import (
    LM_API_BASE,
    LM_MODEL,
    LM_TEMPERATURE,
    LM_MAX_TOKENS,
    RESPONSES_API_MAX_INITIAL_ROUNDS,
    MODEL_DETECT_TIMEOUT,     # 系统启动模型检测超时
    LLM_API_TIMEOUT,          # 统一的 LLM API 连接超时
    STREAM_NO_TOKEN_TIMEOUT,  # 流式调用无token超时
    REASONING_EFFORT,       # Reasoning 模式配置
    LLM_STOP_SEQUENCES,
    HTTP_POOL_CONNECTIONS,  # 连接池大小
    HTTP_POOL_MAXSIZE,      # 最大连接数
    TEMPERATURE_WARMUP      # Session 预热低温度
)

# 获取 logger 实例
logger = logging.getLogger("LMStudioClient")

# ── LLM 错误契约（统一常量，避免魔法字符串散落各模块）──────────────
# LLM 调用失败时返回的错误消息前缀；调用方请用 is_llm_error() 判断，
# 勿再直接硬编码该字符串
ERROR_PREFIX = "[EGO 错误]"


def is_llm_error(response) -> bool:
    """判断 LLM 返回值是否为错误消息（错误契约的统一判断入口）"""
    return isinstance(response, str) and response.startswith(ERROR_PREFIX)


def _auto_log(message: str, level: int = logging.INFO):
    """
    记录日志（级别由调用方显式指定，默认 INFO；不再靠关键词猜测）
    
    Args:
        message: 要记录的消息
        level: 日志级别（logging.INFO / WARNING / ERROR）
    """
    logger.log(level, message)


# ── 消息时间戳统一注入（增量消息专用）────────────────────────────
# 用户消息格式："（YYYY年MM月DD日HH:MM）内容"，由 add_timestamps 统一生成；
# 正则用于去重检测（已有时间戳则跳过，防重复调用双前缀）
TS_PREFIX_RE = re.compile(r"^\s*（\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{1,2}）")


def add_timestamps(messages: list) -> list:
    """为 user 消息统一附加当前时间戳前缀（已有则跳过；多模态只改 input_text 部分）

    覆盖增量消息的全部 user 内容：用户输入（round 0）、事件注入、继续思考占位、
    Think / 自省 / 自我定义提示——统一提供时间基准；
    勿用于会话重建完整模式——历史消息重发会被误标为当前时间。
    """
    prefix = f"（{datetime.now().strftime('%Y年%m月%d日%H:%M')}）"
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if TS_PREFIX_RE.match(content):
                continue
            msg["content"] = prefix + content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    text = part.get("text", "")
                    if not TS_PREFIX_RE.match(text):
                        part["text"] = prefix + text
    return messages


# ── 消息角色前缀（[用户]/[自己]）────────────────────────────
# 自述角色包裹（见 prompts.py 角色契约）：
#   【EGO: ...】= 与自我的对话；【系统: ...】= 系统提示 / 指令执行反馈
# 这类内容已自带角色标签，序列化时不再叠加 "[用户]" 前缀，避免出现
# "[用户]\n（时间）【系统: ...】" 的双重角色标注，使模型误判为普通用户输入
SELF_DECLARED_ROLE_RE = re.compile(r"^\s*【\s*(?:EGO|系统)\s*[:：]")


def _has_self_declared_role(content) -> bool:
    """user 消息内容是否自带 【EGO:】/【系统:】 角色标签（自动剥离前置时间戳）"""
    if not isinstance(content, str):
        return False
    return bool(SELF_DECLARED_ROLE_RE.match(TS_PREFIX_RE.sub("", content, count=1)))


def _apply_role_prefix(role: str, text: str) -> str:
    """按角色为纯文本消息附加 [用户]/[自己] 前缀

    例外：user 消息内容已自带 【EGO:】/【系统:】 角色标签时保持原样——
    覆盖 L2 自省、备忘录审查、自我定义、Think、冷启动等系统触发的任务提示。
    """
    if role == "assistant":
        return f"[自己]\n{text}"
    if role == "user":
        return text if _has_self_declared_role(text) else f"[用户]\n{text}"
    return text


class LMStudioClient:
    """LM Studio API 客户端，支持 Responses 模式"""

    def __init__(self):
        self.api_base = LM_API_BASE.rstrip("/")
        self.model = LM_MODEL
        self.temperature = LM_TEMPERATURE
        self.max_tokens = LM_MAX_TOKENS
        
        # 使用 Session 对象实现连接池，避免连接泄露
        self.session = requests.Session()
        # 配置连接池参数
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=HTTP_POOL_CONNECTIONS,  # 连接池大小
            pool_maxsize=HTTP_POOL_MAXSIZE,          # 最大连接数
            max_retries=0           # 不自动重试，由业务层处理
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Responses API 相关
        self.responses_url = f"{self.api_base}/responses"
        self._previous_response_id = None
        
        # 如果模型是 default-model，尝试从 LM Studio 获取实际模型
        _auto_log("[信息] 开始加载模型.........")
        if self.model == "default-model":
            self._detect_actual_model()
        else:
            _auto_log(f"[配置] 使用默认模型: {self.model}")
    # ── Session ID 公共访问 API（外部模块勿直接读写 _previous_response_id）──
    
    def get_session_id(self) -> Optional[str]:
        """获取当前 Responses API 会话链的 previous_response_id（无则返回 None）"""
        return self._previous_response_id
    
    def set_session_id(self, response_id: Optional[str]):
        """设置/清除 previous_response_id（传 None 表示清除）"""
        self._previous_response_id = response_id
    
    def _detect_actual_model(self):
        """
        尝试从 LM Studio 获取实际加载的模型名称
        """
        try:
            models_url = f"{self.api_base}/models"
            # 使用 session 对象，复用连接
            response = self.session.get(models_url, timeout=MODEL_DETECT_TIMEOUT)
            try:
                response.raise_for_status()
                data = response.json()
                
                if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                    # 使用第一个可用的模型
                    actual_model = data["data"][0].get("id", self.model)
                    _auto_log(f"[配置] 发现并使用 LM Studio 加载的模型: {actual_model}")
                    self.model = actual_model
                else:
                    _auto_log(f"[警告] 无法从 LM Studio 获取模型列表，使用默认模型: {self.model}", level=logging.WARNING)
            finally:
                # 【修复】模型检测响应统一关闭，异常路径也归还连接池
                try:
                    response.close()
                except Exception:
                    pass
        except Exception as e:
            _auto_log(f"[警告] 检测模型失败: {e}，使用默认模型: {self.model}", level=logging.WARNING)
        
    def _call_responses_api(self, input_content: str, temperature: float, timeout: int = None, _retry: bool = True) -> str:
        """
        新版 Responses API 调用（有状态，支持 KV cache 复用）
        
        Args:
            input_content: 用户输入内容
            temperature: 生成温度
            timeout: 超时时间（秒），默认使用 LLM_API_TIMEOUT
            
        Returns:
            LLM 生成的文本
        """
        payload = {
            "model": self.model,
            "input": input_content,
            "temperature": temperature,
            "stop": LLM_STOP_SEQUENCES  # 【修复】统一停止序列，移除省略号截断
        }
        
        # 始终发送 reasoning_effort="none"，确保关闭 Think Mode
        # 即使 REASONING_EFFORT 为空或未设置，也要显式关闭
        payload["reasoning_effort"] = REASONING_EFFORT if REASONING_EFFORT else "none"
        
        # 如果不是第一轮对话，添加上一轮的 response_id
        if self._previous_response_id:
            payload["previous_response_id"] = self._previous_response_id
        
        # 使用自定义超时或默认超时
        api_timeout = timeout if timeout is not None else LLM_API_TIMEOUT
        
        try:
            # 使用 session 对象，复用连接
            response = self.session.post(
                self.responses_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=api_timeout
            )
            
            try:
                response.raise_for_status()
                data = response.json()
                
                # 每次会话后，LMStudio都会更新会话ID用于下一轮对话
                new_id = data.get("id")
                if new_id:
                    self._previous_response_id = new_id
                else:
                    _auto_log("[警告] ⚠ 响应缺少 id 字段，Session 链可能断开", level=logging.WARNING)
                
                # 提取输出内容
                # LM Studio Responses API 返回格式：data["output"][0]["text"]
                if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                    output_item = data["output"][0]
                    
                    # 尝试多种可能的字段名
                    content = None
                    
                    # 优先尝试 text 字段（LM Studio 的实际格式）
                    if "text" in output_item:
                        content = output_item["text"]
                    # 兼容 content 字段（标准格式）
                    elif "content" in output_item:
                        content = output_item["content"]
                    # 其他情况
                    else:
                        content = str(output_item)
                    
                    # 确保返回的是字符串
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # 如果 content 本身是列表，尝试拼接
                        return "\n".join(str(item) for item in content)
                    else:
                        return str(content)
                else:
                    return f"{ERROR_PREFIX} Responses API 返回格式异常"
            finally:
                # 【修复】非流式响应统一关闭：raise_for_status 抛错（连接未归还）等路径也归还连接池
                try:
                    response.close()
                except Exception:
                    pass
                
        except requests.exceptions.ConnectionError:
            _auto_log(f"[调试] ConnectionError: 无法连接到 {self.responses_url}，准备重建连接重试")
            if _retry:
                # 【修复】重连仅重建连接、保留 _previous_response_id 会话链：
                # ConnectionError 发生在连接层，本次请求未成功，_previous_response_id
                # 仍是上一轮的会话链 ID，重试应继续携带以延续上下文；绝不可清空。
                self._rebuild_session_keep_state()
                return self._call_responses_api(input_content, temperature, timeout=timeout, _retry=False)
            return f"{ERROR_PREFIX} 无法连接到 LM Studio，请确保服务已启动"
        except requests.exceptions.Timeout:
            _auto_log(f"[调试] Timeout: Responses API 请求超过 {api_timeout} 秒仍未响应")
            return f"{ERROR_PREFIX} 请求超时（>{api_timeout}秒），请检查：1) LM Studio 是否运行 2) 模型是否已加载 3) 尝试清空历史记录"
        except Exception as e:
            _auto_log(f"[调试] 异常类型: {type(e).__name__}, 消息: {str(e)}")
            return f"{ERROR_PREFIX} API 调用失败: {str(e)}"
    
    def _call_responses_api_streaming(self, input_content: str, temperature: float, stage: str = "", timeout: int = None, _retry: bool = True) -> str:
        """
        智能流式调用 Responses API，带进度显示和超时检测
        
        Args:
            input_content: 用户输入内容
            temperature: 生成温度
            stage: 阶段标识（用于日志显示）
            timeout: 超时时间（秒），默认使用 LLM_API_TIMEOUT
            
        Returns:
            LLM 生成的文本
        """
        import time
        import json
        
        payload = {
            "model": self.model,
            "input": input_content,
            "temperature": temperature,
            "stream": True,  # 启用流式
            "stop": LLM_STOP_SEQUENCES  # 【修复】统一停止序列，移除省略号截断
        }
        
        # 始终发送 reasoning_effort="none"，确保关闭 Think Mode
        payload["reasoning_effort"] = REASONING_EFFORT if REASONING_EFFORT else "none"
        
        # 如果不是第一轮对话，添加上一轮的 response_id
        if self._previous_response_id:
            payload["previous_response_id"] = self._previous_response_id
        
        # 反思阶段使用传入超时，其它阶段使用默认超时
        api_timeout = timeout if timeout is not None else LLM_API_TIMEOUT
        no_token_timeout = timeout if timeout is not None else STREAM_NO_TOKEN_TIMEOUT

        try:
            start_time = time.time()
            last_token_time = start_time
            
            response = self.session.post(
                self.responses_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=api_timeout,
                stream=True  # 关键：启用流式接收
            )
            try:
                response.raise_for_status()
                
                # 【调试】记录响应头信息
                if stage:
                    _auto_log(f"[调试] 响应状态码: {response.status_code}")
                    _auto_log(f"[调试] 响应头 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                
                full_text = ""
                token_count = 0
                
                if stage:
                    _auto_log(f"[信息] 🔄 {stage} 开始生成...")
                
                new_id = None
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        
                        # 【修复】无 token 超时检查提前到"每收到一条 SSE 行"时执行（含 keepalive 注释行）：
                        # 原位置在 data: 分支内部，服务器只发 keepalive 不发 token 时永不执行，
                        # 导致 STREAM_NO_TOKEN_TIMEOUT 失效；完全静默仍由 requests 读超时（api_timeout）兜底
                        if time.time() - last_token_time > no_token_timeout:
                            _auto_log(f"[警告] ⚠ {stage} 超过 {no_token_timeout} 秒无新 token，可能已卡住", level=logging.WARNING)
                            break
                        
                        # 【调试】记录原始SSE数据
                        if stage and token_count == 0 and len(full_text) == 0:
                            _auto_log(f"[调试] 收到SSE行: {line_str[:200]}")
                        
                        # SSE 格式：data: {...}
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 去掉 "data: "
                            
                            if data_str == '[DONE]':
                                _auto_log("[调试] 收到[DONE]标记")
                                break
                            
                            try:
                                data = json.loads(data_str)

                                # 【修复】显式 SSE 错误事件（引擎级拒绝，如上下文超限）：
                                # 立即以错误契约返回，避免被当作"无文本"空结果吞掉
                                if data.get("type") in ("error", "response.failed"):
                                    err_obj = data.get("error")
                                    if not err_obj and isinstance(data.get("response"), dict):
                                        err_obj = data["response"].get("error")
                                    if isinstance(err_obj, dict):
                                        err_msg = err_obj.get("message", str(err_obj))
                                    else:
                                        err_msg = str(err_obj)
                                    _auto_log(f"[警告] ⚠ {stage} 引擎拒绝请求: {err_msg}", level=logging.WARNING)
                                    return f"{ERROR_PREFIX} 引擎拒绝请求: {err_msg}"

                                if new_id is None:
                                    # 1. 尝试从 response 对象中获取 (最常见)
                                    if 'response' in data and isinstance(data['response'], dict):
                                        if 'id' in data['response']:
                                            new_id = data['response']['id']
                                            _auto_log(f"[调试] 从 response 对象获取新 ID: {new_id[:15]}...")
                                    
                                    # 2. 尝试从顶层获取
                                    elif 'id' in data:
                                        new_id = data['id']
                                        _auto_log(f"[调试] 从顶层获取新 ID: {new_id[:15]}...")
                                
                                # 【调试】记录数据结构
                                if stage and token_count < 3:
                                    _auto_log(f"[调试] JSON数据结构: {list(data.keys())}")
                                    if 'delta' in data:
                                        _auto_log(f"[调试] 检测到delta字段: '{data['delta']}'")
                                    if 'text' in data:
                                        _auto_log(f"[调试] 检测到text字段: '{str(data['text'])[:50]}'")
                                    if 'output' in data:
                                        _auto_log(f"[调试] output类型: {type(data['output'])}, 长度: {len(data['output']) if isinstance(data['output'], list) else 'N/A'}")
                                        if isinstance(data['output'], list) and len(data['output']) > 0:
                                            _auto_log(f"[调试] output[0]的键: {list(data['output'][0].keys()) if isinstance(data['output'][0], dict) else type(data['output'][0])}")
                                
                                # 流式模式下，仅信任 delta 增量，避免 output 全量导致的重复
                                token = ""
                                
                                # 1. 优先提取 delta (增量数据，最可靠)
                                if 'delta' in data:
                                    delta_val = data['delta']
                                    if isinstance(delta_val, str) and delta_val:
                                        token = delta_val
                                
                                # 2. 兜底逻辑：仅在流式结束且之前没收到任何内容时，尝试提取全量
                                elif not full_text:
                                    if 'text' in data and data['text']:
                                        token = data['text']
                                    elif 'output' in data and isinstance(data['output'], list) and len(data['output']) > 0:
                                        output_item = data['output'][0]
                                        if isinstance(output_item, dict):
                                            if 'text' in output_item and output_item['text']:
                                                token = output_item['text']
                                            elif 'content' in output_item and output_item['content']:
                                                token = output_item['content']
                                        elif isinstance(output_item, str) and output_item:
                                            token = output_item
                                
                                if token:
                                    full_text += token
                                    token_count += 1
                                    last_token_time = time.time()
                                    
                                    # 每 50 个 token 显示一次进度
                                    if token_count % 50 == 0:
                                        elapsed = time.time() - start_time
                                        speed = token_count / elapsed if elapsed > 0 else 0
                                        if stage:
                                            _auto_log(f"[进度] {stage}: {token_count} tokens, {speed:.1f} t/s")
                                
                            except json.JSONDecodeError as e:
                                _auto_log(f"[调试] JSON解析失败: {e}, 原始数据: {data_str[:100]}")
                                continue
                
                # 【修复】流式响应处理完毕（no_token 超时 break / [DONE] break / 自然耗尽）后统计与收尾，
                # 连接释放统一由 finally 兜底：正常返回 / 引擎拒绝提前 return / 异常，均立即归还连接池
                elapsed = time.time() - start_time
                if stage:
                    _auto_log(f"[信息] ✓ {stage} 完成: {token_count} tokens, 耗时 {elapsed:.1f}s")
                
                if new_id:
                    self._previous_response_id = new_id
                else:
                    _auto_log(f"[警告] ⚠ {stage} 未能获取新 response_id，Session 链接可能断开", level=logging.WARNING)
                
                # 【修复】如果最终没有获取到文本，返回空字符串而不是错误
                if not full_text:
                    _auto_log(f"[警告] ⚠ {stage} 未生成任何文本内容", level=logging.WARNING)
                    return ""
                
                return full_text
            finally:
                # 【修复】统一关闭流式响应：正常返回 / 引擎拒绝提前 return / 异常三条路径均归还连接池，
                # 避免长会话中未关闭的流式连接累积占用连接池导致后续请求排队超时
                try:
                    response.close()
                except Exception:
                    pass
            
        except requests.exceptions.ConnectionError:
            _auto_log(f"[调试] ConnectionError: 无法连接到 {self.responses_url}，准备重建连接重试")
            if _retry:
                # 【修复】重连仅重建连接、保留 _previous_response_id 会话链（同上）
                self._rebuild_session_keep_state()
                return self._call_responses_api_streaming(
                    input_content, temperature, stage=stage, timeout=timeout, _retry=False)
            return f"{ERROR_PREFIX} 无法连接到 LM Studio，请确保服务已启动"
        except requests.exceptions.Timeout:
            _auto_log(f"[调试] Timeout: 流式请求超过 {api_timeout} 秒")
            return f"{ERROR_PREFIX} 请求超时（>{api_timeout}秒），请检查：1) LM Studio 是否运行 2) 模型是否已加载"
        except Exception as e:
            _auto_log(f"[调试] 异常类型: {type(e).__name__}, 消息: {str(e)}")
            return f"{ERROR_PREFIX} 流式调用失败: {str(e)}"
    
    def _parse_json_response(self, text: str) -> str:
        """
        解析可能包含 JSON 格式的响应
        
        某些情况下，LLM 可能返回类似 {'type': 'reasoning_text', 'text': '...'} 的 JSON 格式
        需要从中提取实际的文本内容
        
        【简化】Responses API 稳定后仅保留标准 json.loads 一层解析；
        早期的单引号修复 / ast.literal_eval / 正则提取等启发式兜底已移除
        （它们存在误伤合法 JSON 内容的可能）
        
        Args:
            text: LLM 原始响应文本
            
        Returns:
            提取后的纯文本内容
        """
        import json
        
        original_text = text
        text = text.strip()
        
        # 检查是否是 JSON/字典对象格式（以 { 开头）
        if text.startswith('{'):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # 非合法 JSON，按普通文本原样返回
                return original_text
            
            # 如果成功解析为字典，提取 'text' 或 'content' 字段
            if isinstance(parsed, dict):
                for key in ('text', 'content'):
                    if key in parsed:
                        extracted = parsed[key]
                        if isinstance(extracted, str):
                            return extracted
                        elif isinstance(extracted, (dict, list)):
                            return json.dumps(extracted, ensure_ascii=False)
                        else:
                            return str(extracted)
                _auto_log("[调试]  未找到 'text' 或 'content' 字段，返回原始文本")
        
        # 返回原始文本（未检测到 JSON 格式）
        return original_text
    
    def chat(self, messages: list[dict], temperature: float = None, timeout: int = None,
             stage: str = "", stream: bool = True) -> str:
        """
        统一 LLM 调用入口
        
        Args:
            messages: 消息列表
            temperature: 生成温度（None 时使用实例默认值）
            timeout: 超时时间（秒），默认使用 LLM_API_TIMEOUT
            stage: 阶段标识（流式调用时用于日志与调试记录）
            stream: True=流式调用（推荐，防超时）；False=非流式调用
            
        Returns:
            LLM 生成的文本（错误时返回 ERROR_PREFIX 前缀字符串，可用 is_llm_error() 判断）
        """
        current_temperature = temperature if temperature is not None else self.temperature
        
        # 【封装】消息序列化统一在此完成，调用方无需再手动拼接
        input_content = self._messages_to_input(messages)
        
        if stream:
            return self._call_responses_api_streaming(
                input_content, current_temperature, stage=stage, timeout=timeout
            )
        else:
            raw_response = self._call_responses_api(
                input_content, current_temperature, timeout=timeout
            )
            # 解析可能的 JSON 格式响应（非流式路径保留兜底）
            return self._parse_json_response(raw_response)
    
    def _messages_to_input(self, messages: list[dict]):
        """
        将 messages 列表转换为 Responses API 的 input
        
        优化策略：
        - 首次请求：智能选择关键上下文，避免过长（延迟加载）
        - 后续请求：构建完整的对话历史（包含 assistant 消息）
        
        【新增】返回 str（纯文本，原有行为不变）或 list（含图像时的结构化 input）
        """
        # 【新增】多模态检测：存在结构化 content（图像）时构建结构化 input
        if any(isinstance(msg.get("content"), list) for msg in messages):
            return self._messages_to_structured_input(messages)
        
        # 如果没有 previous_response_id，说明是首次请求或 session 已重置
        if not self._previous_response_id:
            # 提取 system prompt
            system_prompt = ""
            conversation_parts = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    # 自带 【EGO:】/【系统:】 角色标签的消息不再叠加 [用户] 前缀
                    conversation_parts.append(_apply_role_prefix("user", msg["content"]))
                elif msg["role"] == "assistant":
                    # 【修改】不过滤 <<EGO: >> 格式，原样传递
                    conversation_parts.append(_apply_role_prefix("assistant", msg["content"]))
        
            # 延迟加载策略：如果对话轮次过多，只保留最近的关键对话
            max_rounds = RESPONSES_API_MAX_INITIAL_ROUNDS
            max_parts = max_rounds * 3
            
            if len(conversation_parts) > max_parts:
                start_idx = len(conversation_parts) - max_parts
                recent_parts = conversation_parts[start_idx:]
                
                skipped_count = start_idx // 3
                summary_hint = f"[提示：之前有约 {skipped_count} 轮对话已被省略以优化性能，以下是最近的对话内容。]"
                
                parts = [system_prompt, summary_hint] + recent_parts
            else:
                parts = [system_prompt] + conversation_parts
            
            return "\n\n".join(parts)
        else:
            # 后续请求：构建完整的对话历史（包含 assistant 消息）
            input_parts = []
            
            for msg in messages:
                if msg["role"] == "user":
                    # 自带 【EGO:】/【系统:】 角色标签的消息不再叠加 [用户] 前缀
                    input_parts.append(_apply_role_prefix("user", msg["content"]))
                elif msg["role"] == "assistant":
                    # 【修改】不过滤 <<EGO: >> 格式，原样传递
                    input_parts.append(_apply_role_prefix("assistant", msg["content"]))
            
            # 返回所有消息的组合
            if input_parts:
                return "\n\n".join(input_parts)
            else:
                return ""
    
    # 【新增】构建多模态结构化 input（Responses API message items）
    def _messages_to_structured_input(self, messages: list[dict]) -> list:
        """
        将含图像的消息转换为 Responses API 结构化 input 列表。
        纯文本部分保持与原有逻辑相同的 "[用户]/[自己]" 前缀约定。
        """
        items = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                text = content if isinstance(content, str) else "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "input_text"
                )
                if text:
                    items.append({"role": "system", "content": text})
                continue
            
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_text":
                        parts.append({
                            "type": "input_text",
                            "text": _apply_role_prefix(role, part.get("text", "")),
                        })
                    else:
                        # input_image 等结构原样透传
                        parts.append(part)
                items.append({"role": role, "content": parts})
            else:
                items.append({"role": role, "content": _apply_role_prefix(role, str(content))})
        
        return items

    def close(self):
        """关闭 Session，释放连接资源"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def reset_and_reconnect(self):
        """
        完全重置连接：关闭旧 Session 并创建新 Session
        
        用于 /clear 等需要彻底清理资源的场景
        """
        # 先关闭旧连接
        self.close()
        
        # 创建新的 Session 和适配器
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 重置 previous_response_id
        self._previous_response_id = None
        
        _auto_log("[系统] ✓ 已重置 LLM 连接并重建 Session")
    
    def _rebuild_session_keep_state(self):
        """仅重建连接、保留 _previous_response_id 会话链：用于连接层失败后的安全重连。
    
        与 reset_and_reconnect() 的区别：后者用于 /clear 等需要彻底清空会话的场景，
        会置 _previous_response_id = None；而连接层失败（ConnectionError）时本次请求
        并未成功，_previous_response_id 仍是上一轮有效的会话链 ID，必须保留以便重试延续上下文。
        """
        try:
            self.close()
        except Exception as e:
            _auto_log(f"[调试] 关闭旧 Session 失败（不影响重连）: {e}")
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        _auto_log("[系统] ✓ 已重建 Session（保留会话链，用于连接重连）")
    
    def __del__(self):
        """析构函数，确保 Session 被正确关闭"""
        try:
            self.close()
        except Exception:
            pass
        
    def warmup_session(self, messages: list[dict]) -> bool:
        """
        Session 预热：快速重建 KV cache
        
        Args:
            messages: 用于预热的消息列表
            
        Returns:
            bool: 预热是否成功
        """

        _auto_log("[预热] 开始重建 KV cache...")
        
        try:
            # 添加一个虚拟的用户输入来触发响应
            warmup_messages = messages.copy()
            warmup_messages.append({
                "role": "user",
                "content": "[请确认你已理解上述对话历史。回复'已就绪'即可。]"
            })
            
            # 使用低温度，快速响应
            response = self.chat(warmup_messages, temperature=TEMPERATURE_WARMUP)
            
            if "已就绪" in response or "ready" in response.lower():
                _auto_log(f"[预热] ✓ KV cache 重建完成，session_id: {self._previous_response_id}")
                return True
            else:
                _auto_log(f"[预热] ⚠ 预热响应: {response[:50]}...", level=logging.WARNING)
                return True  # 即使响应不完美，也认为预热成功
        except Exception as e:
            _auto_log(f"[警告] ✗ 预热失败: {e}，将在首次请求时重建上下文", level=logging.WARNING)
            return False