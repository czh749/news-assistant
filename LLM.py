"""
大语言模型交互模块 (LLM Module)

本模块封装与智谱AI大语言模型的交互，提供对话功能。
使用LangChain框架统一接口，支持多轮对话。

核心功能：
1. 单轮对话：发送消息获取模型回复
2. 多轮对话：维护对话历史，支持上下文理解
3. 系统提示：支持设置系统级提示词定义模型角色
4. 历史管理：支持清空、获取、设置对话历史

技术说明：
- 使用智谱ChatGLM系列模型（默认glm-4.5-flash）
- temperature参数控制生成随机性（0-1，默认0.2）
- 支持SystemMessage定义模型行为
- 自动维护对话历史，实现多轮交互

使用场景：
- 新闻摘要生成：输入新闻内容生成摘要
- 简报生成：基于检索结果生成个性化简报
- 问答生成：基于搜索结果生成回答

使用示例：
    llm = ZhipuLLM()
    
    # 单轮对话
    reply = llm.chat("你好")
    
    # 带系统提示的多轮对话
    reply = llm.chat("量子计算最新进展", system_prompt="你是新闻助手", session_id="demo")
    reply = llm.chat("还有哪些？", session_id="demo")  # 基于上文回答
    
    # 清空历史
    llm.clear_history()
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from typing import Dict, List, Optional
import logging
import time
import random
import threading
from config import settings

logger = logging.getLogger(__name__)


class ZhipuLLM:
    def __init__(
        self,
        model_name: str = "glm-4.5-flash",
        temperature: float = 0.2,
        max_history_messages: int = 20,
    ):
        """
        初始化ZhipuLLM实例
        
        Args:
            model_name: 模型名称，默认为"glm-4.5-flash"
            temperature: 生成文本的随机性参数，范围0-1，默认为0.2
        """
        if not settings.ZHIPU_API_KEY:
            raise RuntimeError("缺少 ZHIPU_API_KEY，请在环境变量或 .env 中配置")

        self.llm = ChatZhipuAI(model=model_name, temperature=temperature)
        self.max_history_messages = max(2, max_history_messages)
        self._histories: Dict[str, List[BaseMessage]] = {}
        self._session_locks: Dict[str, threading.Lock] = {}
        self._history_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self.last_request_time = 0  # 上次请求时间戳
        self.min_request_interval = 2.0  # 最小请求间隔（秒）
    
    def _wait_for_rate_limit(self):
        """等待以满足速率限制要求"""
        with self._rate_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def chat(
        self,
        text: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """
        与模型进行对话（支持多轮，带重试机制）
        
        Args:
            text: 用户输入的文本
            system_prompt: 系统提示词（会话首次调用时生效）
            max_retries: 最大重试次数，默认为3
            session_id: 会话标识。为空时执行无状态调用，避免批处理任务互相污染
            
        Returns:
            str: 模型的回复内容
        """
        if max_retries < 1:
            raise ValueError("max_retries 必须大于 0")

        if session_id:
            session_lock = self._get_session_lock(session_id)
            with session_lock:
                messages = self.get_history(session_id)
                if not messages and system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=text))
                response_text = self._invoke_with_retries(messages, max_retries)
                messages.append(AIMessage(content=response_text))
                self.set_history(messages, session_id)
                return response_text

        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=text))
        return self._invoke_with_retries(messages, max_retries)

    def _invoke_with_retries(self, messages: List[BaseMessage], max_retries: int) -> str:
        """调用模型并对限流错误执行指数退避。"""
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                error_msg = str(e).lower()
                if '429' in str(e) or 'too many requests' in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning("API 请求过于频繁，等待 %.1f 秒后重试", wait_time)
                        time.sleep(wait_time)
                        continue
                raise

        raise Exception("API 调用失败，已达到最大重试次数")

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._history_lock:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def clear_history(self, session_id: Optional[str] = None) -> None:
        """清空指定会话；未提供会话标识时清空所有会话。"""
        with self._history_lock:
            if session_id is None:
                self._histories.clear()
            else:
                self._histories.pop(session_id, None)

    def get_history(self, session_id: Optional[str] = None) -> List[BaseMessage]:
        """
        获取当前对话历史
        
        Returns:
            List[BaseMessage]: 对话历史消息列表
        """
        if not session_id:
            return []
        with self._history_lock:
            return self._histories.get(session_id, []).copy()

    def set_history(self, history: List[BaseMessage], session_id: str) -> None:
        """
        设置对话历史
        
        Args:
            history: 新的对话历史消息列表
            session_id: 会话标识
        """
        trimmed = history[-self.max_history_messages:]
        if history and isinstance(history[0], SystemMessage) and history[0] not in trimmed:
            trimmed = [history[0], *trimmed[-(self.max_history_messages - 1):]]
        with self._history_lock:
            self._histories[session_id] = trimmed.copy()

# 使用示例（多轮对话）
if __name__ == "__main__":
    # 创建实例
    llm = ZhipuLLM()
    
    # 开始多轮对话
    print("=== 多轮对话示例 ===")
    print("输入 'exit' 退出对话，输入 'clear' 清空历史")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() == "exit":
            print("对话结束")
            break
        
        if user_input.lower() == "clear":
            llm.clear_history()
            print("对话历史已清空")
            continue
        
        # 获取模型回复
        try:
            response = llm.chat(user_input, session_id="cli")
            print(f"AI: {response}")
        except Exception as e:
            print(f"发生错误: {e}")
