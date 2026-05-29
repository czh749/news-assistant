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
    reply = llm.chat("量子计算最新进展", system_prompt="你是新闻助手")
    reply = llm.chat("还有哪些？")  # 基于上文回答
    
    # 清空历史
    llm.clear_history()
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from typing import List, Optional
import os
import time
import random
from config import settings

# 设置API密钥从配置读取
os.environ["ZHIPUAI_API_KEY"] = settings.ZHIPU_API_KEY


class ZhipuLLM:
    def __init__(self, model_name: str = "glm-4.5-flash", temperature: float = 0.2):
        """
        初始化ZhipuLLM实例
        
        Args:
            model_name: 模型名称，默认为"glm-4.5-flash"
            temperature: 生成文本的随机性参数，范围0-1，默认为0.2
        """
        self.llm = ChatZhipuAI(model=model_name, temperature=temperature)
        self.history_messages: List[BaseMessage] = []  # 存储对话历史
        self.last_request_time = 0  # 上次请求时间戳
        self.min_request_interval = 2.0  # 最小请求间隔（秒）
    
    def _wait_for_rate_limit(self):
        """等待以满足速率限制要求"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def chat(self, text: str, system_prompt: Optional[str] = None, max_retries: int = 3) -> str:
        """
        与模型进行对话（支持多轮，带重试机制）
        
        Args:
            text: 用户输入的文本
            system_prompt: 系统提示词（仅在首次调用或历史为空时生效）
            max_retries: 最大重试次数，默认为3
            
        Returns:
            str: 模型的回复内容
        """
        # 如果历史为空且提供了系统提示词，则添加到历史中
        if not self.history_messages and system_prompt:
            self.history_messages.append(SystemMessage(content=system_prompt))
        
        # 添加用户消息到历史
        self.history_messages.append(HumanMessage(content=text))
        
        # 带指数退避的重试机制
        for attempt in range(max_retries):
            try:
                # 等待速率限制
                self._wait_for_rate_limit()
                
                # 使用完整历史调用模型
                response = self.llm.invoke(self.history_messages)
                
                # 添加模型回复到历史
                self.history_messages.append(AIMessage(content=response.content))
                
                return response.content
                
            except Exception as e:
                error_msg = str(e).lower()
                # 判断是否是速率限制错误
                if '429' in str(e) or 'too many requests' in error_msg:
                    if attempt < max_retries - 1:
                        # 指数退避 + 随机抖动
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"API 请求过于频繁，等待 {wait_time:.1f} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                # 其他错误或重试次数用尽，抛出异常
                # 回滚已添加的用户消息，避免历史污染
                self.history_messages.pop()
                raise
        
        # 重试次数用尽
        self.history_messages.pop()
        raise Exception("API 调用失败，已达到最大重试次数")
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history_messages.clear()
    
    def get_history(self) -> List[BaseMessage]:
        """
        获取当前对话历史
        
        Returns:
            List[BaseMessage]: 对话历史消息列表
        """
        return self.history_messages.copy()  # 返回副本，避免外部直接修改
    
    def set_history(self, history: List[BaseMessage]) -> None:
        """
        设置对话历史
        
        Args:
            history: 新的对话历史消息列表
        """
        self.history_messages = history.copy()  # 使用副本，避免外部直接修改

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
            response = llm.chat(user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"发生错误: {e}")