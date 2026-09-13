"""
摘要生成服务

监听 MinIO 新文件，调用 GLM API 生成摘要，更新回 MinIO
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from storage import MinioClient
from LLM import ZhipuLLM

logger = logging.getLogger(__name__)


# 摘要生成 Prompt 模板
SUMMARY_PROMPT_TEMPLATE = """请为以下新闻生成一段简洁的摘要，要求：
1. 摘要长度控制在 100-200 字
2. 保留新闻的核心信息和关键数据
3. 语言简洁明了，突出重点
4. 不要添加个人观点或评价

新闻标题：{title}

新闻内容：
{content}

请直接输出摘要内容，不需要任何前缀说明："""


class SummaryService:
    """
    摘要生成服务
    
    轮询监听 MinIO 中的新新闻文件，自动生成摘要并更新
    """
    
    def __init__(
        self,
        poll_interval: int = 600,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        初始化摘要服务
        
        Args:
            poll_interval: 轮询间隔（秒），默认 600 秒（10分钟）
            max_retries: 最大重试次数，默认 3 次
            retry_delay: 重试间隔（秒），默认 5 秒
        """
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.minio_client = MinioClient()
        self.llm = ZhipuLLM(
            model_name=settings.ZHIPU_MODEL,
            temperature=0.3  # 低温度，输出更稳定
        )
        
        self.running = False
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def generate_summary(self, title: str, content: str) -> Optional[str]:
        """
        调用 GLM API 生成摘要
        
        Args:
            title: 新闻标题
            content: 新闻内容
            
        Returns:
            str: 生成的摘要，失败返回 None
        """
        # 截断过长的内容（避免超出模型上下文限制）
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            title=title,
            content=content
        )
        
        for attempt in range(self.max_retries):
            try:
                summary = self.llm.chat(prompt)
                
                # 清理摘要内容
                summary = summary.strip()
                if summary.startswith('摘要：') or summary.startswith('摘要:'):
                    summary = summary[3:].strip()
                
                logger.info(f"摘要生成成功，长度: {len(summary)} 字")
                return summary
                
            except Exception as e:
                logger.error(f"摘要生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，放弃生成摘要")
                    return None
    
    def process_news(self, object_name: str) -> bool:
        """
        处理单条新闻，生成摘要并更新
        
        支持重试机制：
        - 失败时记录重试次数
        - 使用指数退避计算下次重试时间
        - 超过最大重试次数后放弃
        
        Args:
            object_name: MinIO 中的对象路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 1. 下载新闻数据
            news_data = self.minio_client.download_news(object_name)
            if not news_data:
                logger.error(f"无法下载新闻: {object_name}")
                return False
            
            # 2. 检查是否已生成摘要
            if news_data.get('has_summary', False):
                logger.debug(f"新闻已有摘要，跳过: {object_name}")
                self.stats['skipped'] += 1
                return True
            
            # 3. 检查必要字段
            title = news_data.get('title', '')
            content = news_data.get('content', '')
            
            if not title or not content:
                logger.warning(f"新闻缺少标题或内容: {object_name}")
                return False
            
            # 4. 生成摘要
            logger.info(f"正在生成摘要: {title[:50]}...")
            summary = self.generate_summary(title, content)
            
            if not summary:
                # 生成失败，更新重试信息
                self._update_retry_info(news_data, object_name, success=False)
                return False
            
            # 5. 更新新闻数据（成功）
            news_data['summary'] = summary
            news_data['has_summary'] = True
            news_data['summary_generated_at'] = datetime.now().isoformat()
            news_data['summary_retry_count'] = 0  # 重置重试计数
            news_data['summary_next_retry'] = ''  # 清空重试时间
            
            # 6. 重新上传到 MinIO
            news_id = news_data.get('id', '')
            source = news_data.get('source', 'unknown')
            
            self.minio_client.upload_news(
                news_data=news_data,
                news_id=news_id,
                source=source
            )
            
            logger.info(f"摘要已更新到 MinIO: {object_name}")
            self.stats['success'] += 1
            return True
            
        except Exception as e:
            logger.error(f"处理新闻失败 {object_name}: {e}")
            # 尝试更新重试信息
            try:
                news_data = self.minio_client.download_news(object_name)
                if news_data:
                    self._update_retry_info(news_data, object_name, success=False)
            except Exception:
                pass
            return False
    
    def _update_retry_info(self, news_data: dict, object_name: str, success: bool = False):
        """
        更新新闻的重试信息
        
        Args:
            news_data: 新闻数据
            object_name: MinIO对象路径
            success: 是否成功
        """
        try:
            if success:
                # 成功，重置重试信息
                news_data['summary_retry_count'] = 0
                news_data['summary_next_retry'] = ''
            else:
                # 失败，增加重试计数并设置下次重试时间
                retry_count = news_data.get('summary_retry_count', 0) + 1
                news_data['summary_retry_count'] = retry_count
                
                # 指数退避：1分钟, 2分钟, 4分钟, 8分钟...
                retry_delay = min(2 ** (retry_count - 1), 60)  # 最大60分钟
                next_retry = datetime.now() + timedelta(minutes=retry_delay)
                news_data['summary_next_retry'] = next_retry.isoformat()
                
                logger.info(f"新闻 {object_name} 处理失败，第{retry_count}次重试，下次重试时间: {next_retry.strftime('%H:%M:%S')}")
            
            # 保存更新
            news_id = news_data.get('id', '')
            source = news_data.get('source', 'unknown')
            self.minio_client.upload_news(
                news_data=news_data,
                news_id=news_id,
                source=source
            )
        except Exception as e:
            logger.error(f"更新重试信息失败: {e}")
    
    def get_pending_news(self) -> list:
        """
        获取待处理的新闻列表（没有摘要的新闻）
        
        支持重试机制：
        - 首次处理失败的新闻会记录失败次数和下次重试时间
        - 使用指数退避策略，失败后等待时间逐渐增加
        - 超过最大重试次数的新闻会被跳过
        
        Returns:
            list: 对象路径列表
        """
        try:
            # 获取今天的新闻
            date_str = datetime.now().strftime("%Y-%m-%d")
            all_news = self.minio_client.list_news_by_date(date_str)
            
            pending = []
            now = datetime.now()
            
            for obj_name in all_news:
                try:
                    news_data = self.minio_client.download_news(obj_name)
                    if not news_data:
                        continue
                    
                    # 已有摘要，跳过
                    if news_data.get('has_summary', False):
                        continue
                    
                    # 检查重试次数和下次重试时间
                    retry_count = news_data.get('summary_retry_count', 0)
                    next_retry = news_data.get('summary_next_retry', '')
                    
                    # 超过最大重试次数，跳过
                    if retry_count >= self.max_retries:
                        logger.warning(f"新闻 {obj_name} 已超过最大重试次数({self.max_retries})，跳过")
                        continue
                    
                    # 如果设置了下次重试时间，检查是否到达
                    if next_retry:
                        try:
                            next_retry_time = datetime.fromisoformat(next_retry)
                            if now < next_retry_time:
                                # 还未到重试时间，跳过
                                continue
                        except ValueError:
                            # 时间格式错误，继续处理
                            pass
                    
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
        logger.info("开始扫描待处理新闻...")
        
        pending_news = self.get_pending_news()
        if not pending_news:
            logger.info("没有待处理的新闻")
            return 0
        
        logger.info(f"发现 {len(pending_news)} 条待处理新闻")
        
        processed = 0
        for i, obj_name in enumerate(pending_news):
            self.stats['processed'] += 1
            
            if self.process_news(obj_name):
                processed += 1
            else:
                self.stats['failed'] += 1
            
            # 避免请求过快，每处理一条新闻等待 3 秒
            # 每处理 5 条后额外等待，避免触发频率限制
            if i < len(pending_news) - 1:
                wait_time = 3
                if (i + 1) % 5 == 0:
                    wait_time = 10  # 每 5 条后额外等待
                    logger.info(f"已处理 {i + 1} 条新闻，额外等待 {wait_time} 秒...")
                time.sleep(wait_time)
        
        logger.info(f"本次处理完成: {processed}/{len(pending_news)}")
        return processed
    
    def run(self):
        """启动服务（持续运行）"""
        logger.info("=" * 50)
        logger.info("摘要生成服务启动")
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
        logger.info("摘要服务已停止")
        logger.info(f"最终统计: {self.stats}")


def main():
    """主入口"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    service = SummaryService(
        poll_interval=60,  # 每 60 秒扫描一次
        max_retries=3
    )
    
    try:
        service.run()
    except Exception as e:
        logger.error(f"服务异常: {e}")
        raise


if __name__ == "__main__":
    main()
