"""
定时任务服务模块 (Scheduler Service Module)

本模块负责管理系统的定时任务，包括每日简报生成和新闻爬虫调度。
使用APScheduler库实现后台定时任务调度。

核心功能：
1. 每日简报生成：每天9:00为所有设置兴趣的用户生成个性化新闻简报
2. 新闻爬虫调度：每6小时自动执行新闻爬取任务，保持数据更新
3. 手动简报生成：支持为指定用户手动触发简报生成

定时任务说明：
- 每日简报：使用CronTrigger，每天9:00执行
- 爬虫任务：使用IntervalTrigger，每6小时执行一次

简报生成流程：
1. 获取所有设置兴趣的用户
2. 对每个用户的每个兴趣关键词进行向量检索
3. 合并、去重、排序检索结果
4. 使用LLM生成个性化简报内容
5. 保存推送记录并推送给用户

注意事项：
- 简报生成依赖Milvus向量检索，确保向量数据库可用
- 飞书推送功能需要配置飞书SDK
- 任务在后台线程执行，不会阻塞主程序
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from LLM import ZhipuLLM
from storage.milvus_client import MilvusClient
from storage.minio_client import MinioClient
from storage.mysql_client import MySQLClient, get_mysql_client
from services.user_service import UserService, get_user_service
from embedding import ZhipuEmbedding

logger = logging.getLogger(__name__)


@dataclass
class DailyBriefing:
    """
    每日简报数据类
    
    封装每日简报的核心信息。
    
    Attributes:
        user_id: 用户ID（飞书open_id）
        date: 简报日期（格式：YYYY-MM-DD）
        content: 简报的文本内容（LLM生成）
        news_count: 包含的新闻数量
        news_ids: 简报中引用的新闻ID列表
    """
    user_id: str
    date: str
    content: str
    news_count: int
    news_ids: List[str]


class SchedulerService:
    """
    定时任务服务类
    
    管理系统所有的定时任务，包括每日简报生成和新闻爬虫调度。
    
    核心功能：
    1. 调度器管理：创建、启动、停止APScheduler调度器
    2. 任务注册：注册每日简报和爬虫任务
    3. 简报生成：为所有用户生成个性化每日简报
    4. 爬虫调度：定期执行新闻爬虫，保持数据新鲜
    
    组件依赖：
    - user_service: 获取用户和兴趣信息
    - llm: 生成简报内容
    - embedding: 兴趣关键词向量化
    - milvus: 向量检索相关新闻
    - minio: 获取新闻详情
    - db: 保存推送记录
    
    使用示例：
        service = SchedulerService()
        service.start()  # 启动所有定时任务
        # 程序保持运行...
        service.stop()   # 停止定时任务
    """
    
    def __init__(
        self,
        user_service: Optional[UserService] = None,
        llm: Optional[ZhipuLLM] = None,
        embedding: Optional[ZhipuEmbedding] = None,
        milvus_client: Optional[MilvusClient] = None,
        minio_client: Optional[MinioClient] = None,
        mysql_client: Optional[MySQLClient] = None,
        message_sender: Optional[Callable[[str, str], bool]] = None,
    ):
        """
        初始化定时任务服务
        
        创建APScheduler调度器和各个依赖组件。
        注意：此时调度器尚未启动，需要调用start()方法启动。
        
        Args:
            user_service: 用户服务，用于获取用户兴趣
            llm: LLM模型，用于生成简报内容
            embedding: 嵌入模型，用于向量化
            milvus_client: Milvus向量数据库客户端
            minio_client: MinIO对象存储客户端
            mysql_client: MySQL数据库客户端
            message_sender: 飞书消息发送函数，参数为用户ID和消息正文
        """
        # 初始化各个依赖组件（支持依赖注入）
        self.user_service = user_service or get_user_service()
        self.llm = llm or ZhipuLLM(model_name=settings.ZHIPU_MODEL)
        self.embedding = embedding or ZhipuEmbedding()
        self.milvus = milvus_client or MilvusClient()
        self.minio = minio_client or MinioClient()
        self.db = mysql_client or get_mysql_client()
        self._message_sender = message_sender
        
        # 创建后台调度器（BackgroundScheduler在后台线程运行）
        self.scheduler = BackgroundScheduler()
        
        logger.info("定时任务服务初始化完成")

    @staticmethod
    def _build_date_filter(date: str) -> str:
        """构造仅匹配指定自然日的 Milvus 发布时间过滤表达式。"""
        normalized_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        return f'publish_time like "{normalized_date}%"'
    
    def start(self):
        """
        启动定时任务调度器
        
        注册所有定时任务并启动调度器。
        
        已注册任务：
        1. 每日简报生成：每天上午9:00执行
           - CronTrigger(hour=9, minute=0)
           - 为所有设置兴趣的用户生成个性化简报
        
        2. 新闻爬虫：每1小时执行一次
           - IntervalTrigger(hours=1)
           - 自动抓取新浪新闻，保持数据新鲜
        
        使用replace_existing=True确保重复启动时不会报错
        """
        # ========== 注册每日简报任务 ==========
        self.scheduler.add_job(
            self.generate_daily_briefings,  # 执行函数
            trigger=CronTrigger(hour=9, minute=0),  # 每天9:00触发
            id='daily_briefing',  # 任务ID
            name='每日简报生成',  # 任务名称
            replace_existing=True  # 如果已存在则替换
        )
        
        # ========== 注册爬虫任务 ==========
        from apscheduler.triggers.interval import IntervalTrigger
        self.scheduler.add_job(
            self.run_crawler_task,  # 执行函数
            trigger=IntervalTrigger(hours=1),  # 每1小时触发
            id='news_crawler',  # 任务ID
            name='新闻爬虫',  # 任务名称
            replace_existing=True  # 如果已存在则替换
        )
        
        # 启动调度器（在后台线程运行）
        self.scheduler.start()
        logger.info("定时任务调度器已启动")
        logger.info("已注册任务: 每日简报(9:00), 新闻爬虫(每小时)")
    
    def stop(self):
        """
        停止定时任务调度器
        
        优雅地关闭调度器，等待正在执行的任务完成。
        在程序退出前调用此方法。
        """
        self.scheduler.shutdown()
        logger.info("定时任务调度器已停止")
    
    def run_crawler_task(self):
        """
        运行新闻爬虫任务
        
        每1小时执行一次的定时任务，调用新浪新闻爬虫模块抓取最新新闻。
        
        执行方式：
        - 使用 scrapy crawl sina 在 sina_news 目录下执行
        - 超时设置5分钟，防止爬虫卡死影响调度器
        - 捕获输出并记录到日志
        
        异常处理：
        - 捕获并记录所有异常，确保不影响其他任务
        - 超时后会终止爬虫进程
        """
        logger.info("开始执行定时爬虫任务...")
        
        try:
            import subprocess
            import sys
            import os
            
            # 爬虫项目目录
            crawler_dir = os.path.join(settings.BASE_DIR, "sina_news")
            
            # 使用 scrapy crawl sina 命令执行爬虫
            result = subprocess.run(
                [sys.executable, "-m", "scrapy", "crawl", "sina"],
                capture_output=True,
                text=True,
                timeout=900,
                cwd=crawler_dir
            )
            
            # 检查执行结果
            if result.returncode == 0:
                logger.info("爬虫任务执行成功")
                if result.stdout:
                    logger.info(f"爬虫输出: {result.stdout[-500:]}")
            else:
                logger.error(f"爬虫任务失败 (code={result.returncode})")
                if result.stderr:
                    logger.error(f"错误信息: {result.stderr[-500:]}")
                
        except subprocess.TimeoutExpired:
            logger.error("爬虫任务超时（5分钟），已终止")
        except Exception as e:
            logger.error(f"爬虫任务异常: {e}", exc_info=True)
    
    def generate_daily_briefings(self):
        """
        生成所有用户的每日简报
        
        核心定时任务，每天9:00自动执行。
        
        处理流程：
        1. 获取所有设置了兴趣偏好的用户
        2. 计算目标日期（默认昨天）
        3. 为每个用户单独生成个性化简报
        4. 异常隔离：单个用户失败不影响其他用户
        
        性能考虑：
        - 串行处理用户，避免同时占用过多资源
        - 每个用户处理失败只记录日志，继续处理下一个
        """
        logger.info("开始生成每日简报...")
        
        try:
            # 获取所有有兴趣设置的用户
            users_with_interests = self.user_service.get_users_with_interests()
            
            if not users_with_interests:
                logger.info("没有用户设置兴趣，跳过简报生成")
                return
            
            logger.info(f"找到 {len(users_with_interests)} 个有兴趣的用户")
            
            # 获取昨日日期（简报默认汇总昨日新闻）
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 逐个为用户生成简报
            for user_info in users_with_interests:
                try:
                    self._generate_user_briefing(user_info, yesterday)
                except Exception as e:
                    # 单个用户失败不影响其他用户
                    logger.error(f"为用户 {user_info['user_id']} 生成简报失败: {e}")
                    continue
            
            logger.info("每日简报生成完成")
            
        except Exception as e:
            logger.error(f"生成每日简报失败: {e}", exc_info=True)
    
    def _generate_user_briefing(self, user_info: Dict[str, Any], date: str):
        """
        为单个用户生成简报
        
        针对单个用户的完整简报生成流程：
        
        1. 兴趣检索：对每个兴趣关键词进行向量检索（每个兴趣Top 3）
        2. 结果聚合：合并所有兴趣的检索结果
        3. 去重排序：按相似度*权重排序，去重后取Top 10
        4. 详情获取：从MinIO获取新闻完整内容
        5. 内容生成：使用LLM生成个性化简报
        6. 记录保存：保存推送记录到数据库
        7. 消息推送：调用飞书API推送给用户
        
        Args:
            user_info: 用户信息字典，包含user_id和interests列表
            date: 简报日期字符串（YYYY-MM-DD格式）
        """
        user_id = user_info['user_id']
        interests = user_info['interests']
        date_filter = self._build_date_filter(date)
        
        logger.info(f"为用户 {user_id} 生成简报，兴趣数: {len(interests)}")
        
        # ========== 步骤1: 根据兴趣检索相关新闻 ==========
        interest_news = []
        for interest in interests:
            keyword = interest['keyword']
            weight = interest['weight']
            
            try:
                # 向量化兴趣关键词
                query_vector = self.embedding.embed_query(keyword)
                
                # Milvus向量检索，每个兴趣取Top 3
                results = self.milvus.search(
                    query_vector=query_vector,
                    top_k=3,
                    output_fields=["news_id", "title", "source", "publish_time"],
                    filter_expr=date_filter,
                )
                
                # 将权重信息附加到结果中（用于后续排序）
                for result in results:
                    result['weight'] = weight
                    result['interest'] = keyword
                    interest_news.append(result)
                
            except Exception as e:
                logger.warning(f"检索兴趣 '{keyword}' 相关新闻失败: {e}")
                continue
        
        if not interest_news:
            logger.info(f"用户 {user_id} 没有匹配到相关新闻")
            return
        
        # ========== 步骤2: 去重并按相似度排序 ==========
        seen_ids = set()
        unique_news = []
        # 按相似度*权重排序，优先展示更相关的新闻
        for news in sorted(interest_news, key=lambda x: x.get('distance', 0) * x.get('weight', 1), reverse=True):
            news_id = news.get('news_id')
            if news_id not in seen_ids:
                seen_ids.add(news_id)
                unique_news.append(news)
        
        # 取Top 10条新闻生成简报
        selected_news = unique_news[:10]
        
        # ========== 步骤3: 从MinIO获取新闻详情 ==========
        news_details = []
        for news in selected_news:
            try:
                news_data = self._get_news_detail(news.get('news_id'))
                if news_data:
                    news_details.append(news_data)
            except Exception as e:
                logger.warning(f"获取新闻详情失败: {e}")
                continue
        
        if not news_details:
            logger.info(f"用户 {user_id} 无法获取新闻详情")
            return
        
        # ========== 步骤4: 使用LLM生成简报内容 ==========
        briefing_content = self._generate_briefing_content(user_info, news_details, date)
        
        # ========== 步骤5: 保存推送记录 ==========
        push_record_id = self._save_push_record(
            user_id=user_id,
            push_type='daily',
            content=briefing_content,
            news_ids=[n['news_id'] for n in selected_news],
            status='pending',
        )
        
        # ========== 步骤6: 推送给用户 ==========
        try:
            if not self._push_to_user(user_id, briefing_content):
                raise RuntimeError("飞书接口返回发送失败")
            self._update_push_record_status(push_record_id, 'sent')
        except Exception:
            self._update_push_record_status(push_record_id, 'failed')
            raise
        
        logger.info(f"用户 {user_id} 的简报生成完成，包含 {len(news_details)} 条新闻")
    
    def _get_news_detail(self, news_id: str) -> Optional[Dict[str, Any]]:
        """
        获取新闻详情
        
        Args:
            news_id: 新闻ID
            
        Returns:
            Optional[Dict]: 新闻详情
        """
        try:
            # 从MinIO搜索新闻文件
            objects = self.minio.list_news(prefix="news/")
            
            for obj_name in objects:
                if news_id in obj_name:
                    return self.minio.download_news(obj_name)
            
            return None
        except Exception as e:
            logger.error(f"获取新闻详情失败: {e}")
            return None
    
    def _generate_briefing_content(self, user_info: Dict, news_list: List[Dict], date: str) -> str:
        """
        使用LLM生成简报内容
        
        Args:
            user_info: 用户信息
            news_list: 新闻列表
            date: 日期
            
        Returns:
            str: 简报内容
        """
        # 构建新闻上下文
        news_context = []
        for i, news in enumerate(news_list[:8], 1):  # 最多8条
            context = f"[{i}] {news.get('title', '无标题')}\n"
            context += f"来源: {news.get('source', '未知')}\n"
            if news.get('summary'):
                context += f"摘要: {news.get('summary')[:100]}...\n"
            elif news.get('content'):
                context += f"内容: {news.get('content')[:100]}...\n"
            news_context.append(context)
        
        all_context = "\n".join(news_context)
        
        # 构建兴趣描述
        interests = [i['keyword'] for i in user_info.get('interests', [])]
        interest_str = "、".join(interests[:5])  # 最多显示5个兴趣
        
        prompt = f"""你是智能新闻助手。请为以下用户生成一份个性化的每日新闻简报。

用户信息:
- 关注领域: {interest_str}

昨日相关新闻:
{all_context}

请生成一份简洁友好的每日简报，要求:
1. 开头用一句问候语
2. 概括昨日该领域的主要动态（100字以内）
3. 列出3-5条重点新闻标题
4. 整体控制在300字以内
5. 语气友好专业

简报格式:
📰 每日新闻简报 - {date}

[问候语]

[昨日动态概括]

📌 重点新闻:
1. [新闻标题]
2. [新闻标题]
...

💡 提示: 发送"搜索+关键词"可查询更多新闻"""
        
        try:
            briefing = self.llm.chat(
                prompt,
                system_prompt="你是专业的新闻编辑，擅长生成简洁、有价值的每日新闻简报。"
            )
            return briefing
        except Exception as e:
            logger.error(f"LLM生成简报失败: {e}")
            
            # 降级方案：简单拼接
            fallback = f"📰 每日新闻简报 - {date}\n\n"
            fallback += f"根据您关注的{interest_str}领域，为您精选了以下新闻：\n\n"
            for i, news in enumerate(news_list[:5], 1):
                fallback += f"{i}. {news.get('title', '无标题')}\n"
            return fallback
    
    def _save_push_record(
        self,
        user_id: str,
        push_type: str,
        content: str,
        news_ids: List[str],
        status: str = 'pending',
    ) -> Optional[int]:
        """
        保存推送记录
        
        Args:
            user_id: 用户ID
            push_type: 推送类型
            content: 推送内容
            news_ids: 新闻ID列表
        """
        import json
        
        try:
            record_id = self.db.insert(
                "push_records",
                {
                    "user_id": user_id,
                    "push_type": push_type,
                    "content": content,
                    "news_ids": json.dumps(news_ids, ensure_ascii=False),
                    "status": status,
                },
            )
            logger.info(f"保存推送记录: {user_id}")
            return record_id
        except Exception as e:
            logger.error(f"保存推送记录失败: {e}")
            return None

    def _update_push_record_status(self, record_id: Optional[int], status: str) -> None:
        """更新推送结果；记录创建失败时不阻断实际消息发送。"""
        if record_id is None:
            return
        try:
            self.db.update(
                "push_records",
                {"status": status},
                "id = %s",
                (record_id,),
            )
        except Exception as e:
            logger.error("更新推送记录 %s 状态失败: %s", record_id, e)
    
    def _push_to_user(self, user_id: str, content: str) -> bool:
        """
        推送给用户（飞书）
        
        Args:
            user_id: 用户ID
            content: 推送内容
        """
        if self._message_sender is None:
            from bot.feishu_sdk_client import FeishuSDKClient

            self._message_sender = FeishuSDKClient().send_text_message

        sent = self._message_sender(user_id, content)
        if sent:
            logger.info("简报推送成功: user_id=%s", user_id)
        else:
            logger.error("简报推送失败: user_id=%s", user_id)
        return sent
    
    def generate_briefing_for_user(self, user_id: str) -> Optional[str]:
        """
        为指定用户生成简报（手动触发）
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[str]: 简报内容
        """
        user = self.user_service.get_user(user_id)
        if not user:
            logger.warning(f"用户 {user_id} 不存在")
            return None
        
        interests = self.user_service.get_user_interests(user_id)
        if not interests:
            logger.info(f"用户 {user_id} 没有设置兴趣")
            return None
        
        user_info = {
            'user_id': user_id,
            'name': user.name,
            'interests': [{'keyword': i.interest_keyword, 'weight': i.weight} for i in interests]
        }
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        date_filter = self._build_date_filter(yesterday)
        
        try:
            # 这里简化处理，实际应该调用 _generate_user_briefing
            # 但为了返回内容，我们单独处理
            interest_news = []
            for interest in user_info['interests']:
                try:
                    keyword = interest['keyword']
                    query_vector = self.embedding.embed_query(keyword)
                    results = self.milvus.search(
                        query_vector=query_vector,
                        top_k=3,
                        output_fields=["news_id", "title", "source", "publish_time"],
                        filter_expr=date_filter,
                    )
                    for r in results:
                        r['interest'] = keyword
                        interest_news.append(r)
                except Exception as e:
                    logger.warning(f"检索失败: {e}")
                    continue
            
            if not interest_news:
                return "暂无相关新闻"
            
            # 获取详情并生成简报
            news_details = []
            for news in interest_news[:5]:
                detail = self._get_news_detail(news.get('news_id'))
                if detail:
                    news_details.append(detail)
            
            if not news_details:
                return "暂无相关新闻详情"
            
            return self._generate_briefing_content(user_info, news_details, yesterday)
            
        except Exception as e:
            logger.error(f"生成简报失败: {e}")
            return None


# =============================================================================
# 全局服务实例（单例模式）
# =============================================================================
# 使用单例模式确保整个应用中只有一个SchedulerService实例
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """
    获取定时任务服务单例
    
    懒加载方式获取SchedulerService的全局唯一实例。
    
    Returns:
        SchedulerService: 全局唯一的定时任务服务实例
    """
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service


# =============================================================================
# 使用示例和测试代码
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("定时任务服务测试")
    print("=" * 60)
    
    service = SchedulerService()
    
    # 手动触发一次简报生成（用于测试）
    print("\n手动生成简报...")
    service.generate_daily_briefings()
    
    print("\n测试完成")
