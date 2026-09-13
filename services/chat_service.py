"""
对话服务模块 (Chat Service Module)

本模块是智能新闻助手的核心对话处理模块，负责整合多个子系统，
实现完整的用户对话流程。

核心功能：
1. 消息处理：接收用户消息，识别意图，调用相应处理逻辑
2. 搜索流程：执行向量检索 -> 获取新闻详情 -> LLM生成回答的完整流程
3. 兴趣管理：处理用户的兴趣设置和取消操作
4. 对话响应：封装统一的响应格式

架构设计：
- 采用分层处理模式：意图识别层 -> 业务处理层 -> 响应生成层
- 依赖注入：支持各组件的自定义配置和测试模拟
- 降级策略：各步骤失败时有对应的降级处理方案

数据流：
用户消息 -> 意图识别 -> 路由到处理函数 -> (搜索: 向量化->向量检索->MinIO获取->LLM生成)
                                      -> (兴趣: 调用用户服务)
                                      -> (帮助/问候: 返回固定回复)
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import settings
from LLM import ZhipuLLM
from embedding import ZhipuEmbedding
from storage.milvus_client import MilvusClient
from storage.minio_client import MinioClient
from services.intent_service import IntentService, IntentType, get_intent_service
from services.user_service import UserService, get_user_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    搜索结果数据结构
    
    封装从向量检索和相关存储中获取的完整新闻信息。
    用于在对话服务中传递搜索结果，最终生成用户回复。
    
    Attributes:
        news_id: 新闻唯一标识符
        title: 新闻标题
        summary: 新闻摘要或内容预览
        source: 新闻来源（如新浪新闻、腾讯新闻等）
        url: 新闻原文链接
        publish_time: 发布时间
        similarity: 与用户查询的相似度分数（向量检索结果）
    """
    news_id: str
    title: str
    summary: str
    source: str
    url: str
    publish_time: str
    similarity: float


@dataclass
class ChatResponse:
    """
    对话响应数据结构
    
    封装对话服务的完整响应信息，包含要回复给用户的文本、
    识别出的意图、搜索结果列表以及处理状态。
    
    Attributes:
        reply_text: 要发送给用户的回复文本
        intent: 识别出的意图类型字符串（如"search"、"interest"等）
        search_results: 搜索结果列表（如果是搜索意图）
        success: 处理是否成功
        error_message: 错误信息（处理失败时）
    """
    reply_text: str             # 回复给用户的消息文本
    intent: str                 # 识别的意图类型
    search_results: List[SearchResult]  # 相关新闻结果列表
    success: bool               # 处理是否成功
    error_message: Optional[str] = None  # 可选的错误描述


class ChatService:
    """
    对话服务类
    
    核心服务类，负责处理用户对话的完整流程。
    整合意图识别、向量检索、存储访问和LLM生成等多个组件。
    
    组件说明：
    - intent_service: 识别用户意图（搜索/兴趣/帮助/问候）
    - llm: 大语言模型，用于生成自然语言回答
    - embedding: 嵌入模型，将文本转换为向量用于检索
    - milvus: 向量数据库客户端，执行相似度搜索
    - minio: 对象存储客户端，获取完整新闻内容
    - user_service: 用户服务，管理兴趣偏好
    
    设计模式：
    - 依赖注入：各组件可通过构造函数传入，便于测试和自定义
    - 懒加载：未传入的组件会自动创建默认实例
    """
    
    def __init__(
        self,
        intent_service: Optional[IntentService] = None,
        llm: Optional[ZhipuLLM] = None,
        embedding: Optional[ZhipuEmbedding] = None,
        milvus_client: Optional[MilvusClient] = None,
        minio_client: Optional[MinioClient] = None,
        user_service: Optional[UserService] = None
    ):
        """
        初始化对话服务
        
        所有参数都是可选的，如果未提供则自动创建默认实例。
        这种设计便于单元测试时注入模拟对象。
        
        Args:
            intent_service: 意图识别服务，用于分析用户意图
            llm: LLM模型，用于生成自然语言回答
            embedding: 嵌入模型，用于查询向量化
            milvus_client: Milvus向量数据库客户端
            minio_client: MinIO对象存储客户端
            user_service: 用户服务，用于管理用户兴趣
        """
        # 使用传入的组件或创建默认实例（依赖注入模式）
        self.intent_service = intent_service or get_intent_service()
        self.llm = llm or ZhipuLLM(model_name=settings.ZHIPU_MODEL)
        self.embedding = embedding or ZhipuEmbedding()
        self.milvus = milvus_client or MilvusClient()
        self.minio = minio_client or MinioClient()
        self.user_service = user_service or get_user_service()
        
        logger.info("对话服务初始化完成")
    
    def process_message(self, user_message: str, user_id: Optional[str] = None) -> ChatResponse:
        """
        处理用户消息（核心入口方法）
        
        这是对话服务的主要入口，处理所有用户消息的完整流程：
        1. 意图识别：分析用户消息确定意图类型
        2. 意图路由：根据意图类型调用相应处理函数
        3. 异常处理：捕获并处理所有可能的异常
        
        支持的消息类型：
        - 搜索类消息："搜索AI新闻"、"量子计算最新进展"
        - 兴趣设置："关注科技新闻"、"取消关注财经"
        - 帮助请求："帮助"、"怎么用"
        - 问候语："你好"、"早上好"
        
        Args:
            user_message: 用户输入的原始消息文本
            user_id: 用户唯一标识（可选，用于个性化服务如兴趣管理）
            
        Returns:
            ChatResponse: 包含回复文本、意图类型、处理状态的完整响应
        """
        try:
            # ========== 步骤1: 意图识别 ==========
            # 调用意图识别服务分析用户消息
            intent_result = self.intent_service.recognize(user_message)
            logger.info(
                "收到用户消息，长度=%s，识别意图=%s",
                len(user_message),
                intent_result.intent.value,
            )
            
            # ========== 步骤2: 根据意图类型路由处理 ==========
            if intent_result.intent == IntentType.SEARCH:
                # 搜索意图：执行向量检索和LLM生成
                return self._handle_search_intent(intent_result, user_id)
            
            elif intent_result.intent == IntentType.SET_INTEREST:
                # 兴趣设置意图：添加或删除用户兴趣
                return self._handle_interest_intent(intent_result, user_id)
            
            elif intent_result.intent == IntentType.HELP:
                # 帮助意图：返回使用指南
                return ChatResponse(
                    reply_text=self.intent_service.get_help_message(),
                    intent=intent_result.intent.value,
                    search_results=[],
                    success=True
                )
            
            elif intent_result.intent == IntentType.GREETING:
                # 问候意图：返回友好问候
                return ChatResponse(
                    reply_text=self.intent_service.get_greeting_message(),
                    intent=intent_result.intent.value,
                    search_results=[],
                    success=True
                )
            
            else:
                # 未知意图：默认尝试作为搜索处理
                logger.warning("未知意图，按搜索请求处理")
                return self._handle_search_intent(intent_result, user_id)
                
        except Exception as e:
            # ========== 异常处理 ==========
            # 捕获所有异常，确保不会暴露内部错误给用户
            logger.error(f"处理消息失败: {e}", exc_info=True)
            return ChatResponse(
                reply_text="抱歉，处理您的请求时出现了错误，请稍后再试。",
                intent="error",
                search_results=[],
                success=False,
                error_message=str(e)
            )
    
    def _handle_search_intent(
        self,
        intent_result,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        处理搜索意图 - RAG完整流程
        
        执行检索增强生成(Retrieval-Augmented Generation)的完整流程：
        
        处理流程：
        1. 查询构建：从意图结果中提取查询词
        2. 向量化：使用Embedding模型将查询转换为向量
        3. 向量检索：在Milvus中搜索相似新闻（top_k=5）
        4. 数据获取：从MinIO获取新闻完整内容
        5. 回答生成：使用LLM基于检索结果生成自然语言回答
        
        降级策略：
        - 向量化失败：返回服务不可用提示
        - 检索失败：返回检索失败提示
        - 无结果：返回建议换词提示
        - MinIO获取失败：使用Milvus的基本信息
        - LLM生成失败：返回格式化的列表作为降级
        
        Args:
            intent_result: 意图识别结果，包含查询关键词和实体
            
        Returns:
            ChatResponse: 包含生成回答和相关新闻列表的响应
        """
        # 保存用户原始问题，用于LLM生成回答
        user_question = intent_result.raw_text
        
        # ========== 步骤1: 构建查询词 ==========
        # 优先使用从意图中提取的查询实体
        query = intent_result.entities.get("query", "")
        # 如果没有实体但有提取的关键词，使用关键词拼接
        if not query and intent_result.keywords:
            query = " ".join(intent_result.keywords)
        # 如果都没有，使用用户原始输入
        if not query:
            query = user_question
        
        logger.info(f"搜索查询: {query}")
        
        # ========== 步骤2: 向量化查询 ==========
        try:
            # 使用Embedding服务将查询文本转换为向量
            query_vector = self.embedding.embed_query(query)
        except Exception as e:
            logger.error(f"向量化查询失败: {e}")
            return ChatResponse(
                reply_text="抱歉，搜索服务暂时不可用，请稍后再试。",
                intent="search",
                search_results=[],
                success=False,
                error_message="embedding_failed"
            )
        
        # ========== 步骤3: Milvus向量检索 ==========
        try:
            # 在Milvus中搜索相似向量，返回最相似的5条新闻
            search_results = self.milvus.search(
                query_vector=query_vector,
                top_k=5,
                output_fields=["news_id", "title", "source", "category", "publish_time"]
            )
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return ChatResponse(
                reply_text="抱歉，新闻检索服务暂时不可用，请稍后再试。",
                intent="search",
                search_results=[],
                success=False,
                error_message="search_failed"
            )
        
        # 检查是否有检索结果
        if not search_results:
            return ChatResponse(
                reply_text=f"抱歉，没有找到与\"{query}\"相关的新闻。\n\n您可以尝试：\n• 换用其他关键词\n• 使用更通用的表述\n• 检查关键词拼写",
                intent="search",
                search_results=[],
                success=True
            )
        
        # ========== 步骤4: 从MinIO获取新闻详情 ==========
        news_list = []
        for result in search_results:
            news_id = result.get("news_id")
            try:
                # 尝试从MinIO下载新闻完整数据
                news_data = self._get_news_from_minio(news_id, result.get('source'))
                
                if news_data:
                    # MinIO中获取成功，使用完整数据
                    news_list.append(SearchResult(
                        news_id=news_id,
                        title=news_data.get("title", result.get("title", "无标题")),
                        summary=news_data.get("summary", news_data.get("content", "")[:200] + "..."),
                        source=news_data.get("source", result.get("source", "未知来源")),
                        url=news_data.get("url", ""),
                        publish_time=news_data.get("publish_time", ""),
                        similarity=result.get("distance", 0.0)
                    ))
                else:
                    # MinIO中未找到，使用Milvus返回的基本信息（降级）
                    news_list.append(SearchResult(
                        news_id=news_id,
                        title=result.get("title", "无标题"),
                        summary="",
                        source=result.get("source", "未知来源"),
                        url="",
                        publish_time=result.get("publish_time", ""),
                        similarity=result.get("distance", 0.0)
                    ))
            except Exception as e:
                logger.warning(f"获取新闻详情失败 {news_id}: {e}")
                continue
        
        # 检查是否成功获取到任何新闻详情
        if not news_list:
            return ChatResponse(
                reply_text="找到了相关新闻，但无法获取详情，请稍后重试。",
                intent="search",
                search_results=[],
                success=True
            )
        
        # ========== 步骤5: LLM生成回答 ==========
        reply_text = self._generate_search_reply(
            user_question,
            news_list,
            session_id=user_id,
        )
        
        return ChatResponse(
            reply_text=reply_text,
            intent="search",
            search_results=news_list,
            success=True
        )
    
    def _get_news_from_minio(self, news_id: str, source: str) -> Optional[Dict[str, Any]]:
        """
        从MinIO获取新闻数据
        
        由于新闻在MinIO中的存储路径不固定，采用遍历搜索的方式：
        1. 列出所有新闻文件
        2. 查找包含news_id的文件
        3. 下载并返回新闻数据
        
        注意：这种实现方式适合新闻数量不太大的场景。
        如果新闻数量很大，应该使用更高效的索引机制。
        
        Args:
            news_id: 新闻唯一标识符
            source: 新闻来源（当前未使用，保留用于未来优化）
            
        Returns:
            Optional[Dict]: 新闻数据字典，如果未找到则返回None
        """
        try:
            # 列出MinIO中所有新闻文件
            objects = self.minio.list_news(prefix="news/")
            
            # 遍历查找包含news_id的文件
            for obj_name in objects:
                if news_id in obj_name:
                    # 找到匹配文件，下载数据
                    return self.minio.download_news(obj_name)
            
            # 未找到匹配的新闻
            return None
        except Exception as e:
            logger.error(f"从MinIO获取新闻失败: {e}")
            return None
    
    def _generate_search_reply(
        self,
        user_question: str,
        news_list: List[SearchResult],
        session_id: Optional[str] = None,
    ) -> str:
        """
        使用LLM基于搜索结果生成完整回答（RAG核心）
        
        这是RAG架构的核心环节：
        1. 将检索到的新闻构建成上下文
        2. 构造Prompt，包含用户问题和新闻上下文
        3. 调用LLM生成针对性的自然语言回答
        4. 如果LLM失败，使用降级方案返回格式化列表
        
        Prompt设计要点：
        - 明确的角色设定（新闻助手）
        - 清晰的任务描述（基于新闻回答问题）
        - 结构化的新闻上下文（带编号）
        - 具体的输出要求（引用来源、字数限制、格式规范）
        
        Args:
            user_question: 用户原始问题文本
            news_list: 检索到的新闻列表（最多取前5条）
            
        Returns:
            str: LLM生成的回答或格式化的新闻列表（降级方案）
        """
        if len(news_list) == 0:
            return f"抱歉，我没有找到与\"{user_question}\"相关的新闻信息。"
        
        # 构建新闻上下文
        news_context = []
        for i, news in enumerate(news_list[:5], 1):
            context = f"[{i}] 标题: {news.title}\n"
            context += f"来源: {news.source}\n"
            if news.summary:
                context += f"内容: {news.summary}\n"
            if news.url:
                context += f"链接: {news.url}\n"
            news_context.append(context)
        
        all_context = "\n---\n".join(news_context)
        
        # 使用LLM生成完整回答
        try:
            prompt = f"""你是智能新闻助手。请根据以下检索到的新闻信息，回答用户的问题。

用户问题: {user_question}

检索到的新闻信息:
{all_context}

请按以下要求回答:
1. 直接回答用户的问题，不要偏离主题
2. 基于上述新闻内容进行归纳总结
3. 如果有多条相关新闻，请综合信息给出完整答案
4. 在回答中引用新闻来源（用[1]、[2]等标注）
5. 回答控制在200字以内，简洁明了
6. 最后列出相关新闻标题和链接供用户查看

回答格式:
[针对用户问题的回答内容]

相关新闻:
[1] 新闻标题 - 链接
[2] 新闻标题 - 链接
..."""
            
            answer = self.llm.chat(
                prompt,
                system_prompt="你是专业的新闻助手，擅长基于检索到的新闻内容回答用户问题。回答要准确、简洁、有依据。",
                session_id=session_id,
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM生成回答失败: {e}")
            
            # 降级方案：返回格式化列表
            reply = f"关于\"{user_question}\"，我找到以下相关新闻：\n\n"
            
            for i, news in enumerate(news_list[:5], 1):
                reply += f"{i}. {news.title}\n"
                reply += f"   来源: {news.source}\n"
                if news.url:
                    reply += f"   链接: {news.url}\n"
                if news.summary:
                    reply += f"   摘要: {news.summary[:80]}...\n"
                reply += "\n"
            
            return reply
    
    def _handle_interest_intent(self, intent_result, user_id: Optional[str]) -> ChatResponse:
        """
        处理兴趣设置意图
        
        处理用户的兴趣偏好管理请求，包括添加关注和取消关注。
        
        处理流程：
        1. 从意图结果中提取兴趣关键词和操作类型（add/remove）
        2. 验证输入完整性（兴趣关键词不能为空）
        3. 验证用户身份（需要user_id）
        4. 调用用户服务执行相应操作
        5. 根据操作结果生成反馈消息
        
        Args:
            intent_result: 意图识别结果，包含interest实体和action实体
            user_id: 用户唯一标识，用于关联用户和兴趣偏好
            
        Returns:
            ChatResponse: 包含操作结果的响应，告知用户操作成功或失败
        """
        # 提取意图中的关键信息
        interest = intent_result.entities.get("interest", "")
        action = intent_result.entities.get("action", "add")
        
        # 验证是否提取到兴趣关键词
        if not interest:
            return ChatResponse(
                reply_text='请告诉我要关注什么话题，例如："关注科技新闻"或"对财经感兴趣"',
                intent="interest",
                search_results=[],
                success=True
            )
        
        # 验证用户身份（飞书用户会有open_id作为user_id）
        if not user_id:
            return ChatResponse(
                reply_text='请先登录后再设置兴趣偏好。',
                intent="interest",
                search_results=[],
                success=True
            )
        
        # 根据操作类型执行相应操作
        if action == "remove":
            # 取消关注操作
            success = self.user_service.remove_interest(user_id, interest)
            if success:
                reply = f'[OK] 已取消关注 "{interest}" 相关新闻。\n\n您将不再收到该话题的推送。'
            else:
                reply = '取消关注失败，请稍后重试。'
        else:
            # 添加关注操作
            success = self.user_service.add_interest(user_id, interest, weight=1.0)
            if success:
                reply = f'[OK] 已成功关注 "{interest}"！\n\n我会为您推荐相关新闻，并在每日简报中包含该话题的最新资讯。'
            else:
                reply = '添加关注失败，请稍后重试。'
        
        return ChatResponse(
            reply_text=reply,
            intent="interest",
            search_results=[],
            success=success
        )
    
    def clear_conversation_history(self, user_id: Optional[str] = None):
        """
        清空对话历史
        
        清除LLM的对话历史记录，开始新的对话上下文。
        在多轮对话场景中使用，确保新的对话不受历史影响。
        """
        self.llm.clear_history(user_id)
        logger.info("对话历史已清空: %s", user_id or "all")


# =============================================================================
# 全局服务实例（单例模式）
# =============================================================================
# 使用单例模式确保整个应用中只有一个ChatService实例
# 避免重复初始化各个组件，提高性能和资源利用率
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """
    获取对话服务单例
    
    懒加载方式获取ChatService的全局唯一实例。
    首次调用时创建实例，后续调用返回已创建的实例。
    
    Returns:
        ChatService: 全局唯一的对话服务实例
    """
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


# =============================================================================
# 使用示例和测试代码
# =============================================================================
# 当直接运行此文件时执行的测试代码
# 用于验证对话服务的各项功能
if __name__ == "__main__":
    # 配置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 打印测试标题
    print("=" * 60)
    print("对话服务测试")
    print("=" * 60)
    
    # 创建服务实例
    chat_service = ChatService()
    
    # 定义测试用例，覆盖各种意图类型
    test_messages = [
        "你好",                      # 问候意图
        "搜索人工智能新闻",          # 搜索意图
        "查一下科技资讯",            # 搜索意图（变体表达）
        "关注财经新闻",              # 兴趣设置意图
        "帮助",                     # 帮助意图
    ]
    
    # 逐个测试并打印结果
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"用户: {msg}")
        print("-" * 60)
        
        try:
            response = chat_service.process_message(msg)
            print(f"意图: {response.intent}")
            print(f"成功: {response.success}")
            print(f"回复:\n{response.reply_text}")
            if response.search_results:
                print(f"搜索结果数: {len(response.search_results)}")
        except Exception as e:
            print(f"错误: {e}")
