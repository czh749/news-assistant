"""
向量化处理服务

监听已生成摘要的新闻，自动向量化并存储到 Milvus
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import settings
from storage import MinioClient, MilvusClient
from embedding import ZhipuEmbedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    向量化处理服务
    
    轮询监听已生成摘要的新闻，向量化后存入 Milvus
    """
    
    def __init__(
        self,
        poll_interval: int = 600,
        max_retries: int = 3,
        retry_delay: int = 5,
        max_text_length: int = 2000
    ):
        """
        初始化向量化服务
        
        Args:
            poll_interval: 轮询间隔（秒），默认 600 秒（10分钟）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            max_text_length: 文本最大长度（字符）
        """
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_text_length = max_text_length
        
        # 初始化客户端
        self.minio_client = MinioClient()
        self.milvus_client = MilvusClient()
        self.embedding = ZhipuEmbedding(model_name=settings.ZHIPU_EMBEDDING_MODEL)
        
        self.running = False
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        截断过长文本
        
        Args:
            text: 原始文本
            max_length: 最大长度
            
        Returns:
            str: 截断后的文本
        """
        max_len = max_length or self.max_text_length
        if len(text) > max_len:
            logger.debug(f"文本过长 ({len(text)} 字)，截断至 {max_len} 字")
            return text[:max_len]
        return text
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        生成文本向量
        
        Args:
            text: 待向量化的文本
            
        Returns:
            List[float]: 向量，失败返回 None
        """
        # 截断文本
        text = self.truncate_text(text)
        
        for attempt in range(self.max_retries):
            try:
                vector = self.embedding.embed_query(text)
                logger.info(f"向量生成成功，维度: {len(vector)}")
                return vector
                
            except Exception as e:
                logger.error(f"向量生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，放弃生成向量")
                    return None
    
    def process_news(self, object_name: str) -> bool:
        """
        处理单条新闻，生成向量并存储
        
        Args:
            object_name: MinIO 对象路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 1. 下载新闻数据
            news_data = self.minio_client.download_news(object_name)
            if not news_data:
                logger.error(f"无法下载新闻: {object_name}")
                return False
            
            # 2. 检查是否有摘要
            if not news_data.get('has_summary', False):
                logger.debug(f"新闻尚未生成摘要，跳过: {object_name}")
                self.stats['skipped'] += 1
                return True
            
            # 3. 检查是否已向量化
            if news_data.get('has_embedding', False):
                logger.debug(f"新闻已向量化，跳过: {object_name}")
                self.stats['skipped'] += 1
                return True
            
            # 4. 准备向量化文本（标题 + 摘要）
            title = news_data.get('title', '')
            summary = news_data.get('summary', '')
            
            if not title:
                logger.warning(f"新闻缺少标题: {object_name}")
                return False
            
            # 合并标题和摘要
            text_for_embedding = f"{title}\n{summary}" if summary else title
            
            # 5. 生成向量
            logger.info(f"正在向量化: {title[:50]}...")
            vector = self.generate_embedding(text_for_embedding)
            
            if not vector:
                return False
            
            # 6. 存储到 Milvus
            news_id = news_data.get('id', '')
            source = news_data.get('source', 'unknown')
            category = news_data.get('category', '')
            publish_time = news_data.get('publish_time', '')
            
            self.milvus_client.insert_news(
                news_id=news_id,
                title=title,
                vector=vector,
                source=source,
                category=category,
                publish_time=publish_time
            )
            
            # 7. 更新 MinIO 中的状态
            news_data['has_embedding'] = True
            news_data['embedding_generated_at'] = datetime.now().isoformat()
            
            self.minio_client.upload_news(
                news_data=news_data,
                news_id=news_id,
                source=source
            )
            
            logger.info(f"向量已存储到 Milvus: {news_id}")
            self.stats['success'] += 1
            return True
            
        except Exception as e:
            logger.error(f"处理新闻失败 {object_name}: {e}")
            return False
    
    def get_pending_news(self) -> List[str]:
        """
        获取待向量化的新闻列表
        
        Returns:
            list: 对象路径列表
        """
        try:
            # 获取今天的新闻
            date_str = datetime.now().strftime("%Y-%m-%d")
            all_news = self.minio_client.list_news_by_date(date_str)
            
            pending = []
            for obj_name in all_news:
                try:
                    news_data = self.minio_client.download_news(obj_name)
                    # 有摘要但未向量化的新闻
                    if (news_data and 
                        news_data.get('has_summary', False) and 
                        not news_data.get('has_embedding', False)):
                        pending.append(obj_name)
                except Exception as e:
                    logger.warning(f"检查新闻状态时出错 {obj_name}: {e}")
                    continue
            
            return pending
            
        except Exception as e:
            logger.error(f"获取待处理新闻列表失败: {e}")
            return []
    
    def run_once(self) -> int:
        """
        执行一次处理循环
        
        Returns:
            int: 处理的新闻数量
        """
        logger.info("开始扫描待向量化新闻...")
        
        pending_news = self.get_pending_news()
        if not pending_news:
            logger.info("没有待向量化的新闻")
            return 0
        
        logger.info(f"发现 {len(pending_news)} 条待向量化新闻")
        
        processed = 0
        for obj_name in pending_news:
            self.stats['processed'] += 1
            
            if self.process_news(obj_name):
                processed += 1
            else:
                self.stats['failed'] += 1
            
            # 避免请求过快
            time.sleep(0.5)
        
        logger.info(f"本次处理完成: {processed}/{len(pending_news)}")
        return processed
    
    def run(self):
        """启动服务（持续运行）"""
        logger.info("=" * 50)
        logger.info("向量化服务启动")
        logger.info(f"轮询间隔: {self.poll_interval} 秒")
        logger.info(f"最大重试: {self.max_retries} 次")
        logger.info("=" * 50)
        
        self.running = True
        
        try:
            while self.running:
                self.run_once()
                
                logger.info(f"统计 - 处理: {self.stats['processed']}, "
                          f"成功: {self.stats['success']}, "
                          f"失败: {self.stats['failed']}, "
                          f"跳过: {self.stats['skipped']}")
                
                logger.info(f"等待 {self.poll_interval} 秒后下次扫描...")
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("收到停止信号，服务正在关闭...")
            self.stop()
    
    def stop(self):
        """停止服务"""
        self.running = False
        logger.info("向量化服务已停止")
        logger.info(f"最终统计: {self.stats}")


def main():
    """主入口"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    service = EmbeddingService(
        poll_interval=60,
        max_retries=3
    )
    
    try:
        service.run()
    except Exception as e:
        logger.error(f"服务异常: {e}")
        raise


if __name__ == "__main__":
    main()
