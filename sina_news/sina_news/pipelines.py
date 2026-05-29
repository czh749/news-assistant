# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import sys
import os
import scrapy
import scrapy.exceptions
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from itemadapter import ItemAdapter
from sina_news.items import SinaNewsItem


class MinioPipeline:
    """
    MinIO 存储 Pipeline
    
    将爬取的新闻数据存储到 MinIO 对象存储
    存储路径规范: news/{date}/{source}/{id}.json
    """
    
    def __init__(self):
        self.minio_client = None
        self.stats = {
            'uploaded': 0,
            'failed': 0,
            'skipped': 0
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        """从 crawler 创建 pipeline 实例"""
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signal=scrapy.signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signal=scrapy.signals.spider_closed)
        return pipeline
    
    def spider_opened(self, spider):
        """爬虫启动时初始化 MinIO 客户端"""
        try:
            from storage.minio_client import get_minio_client
            self.minio_client = get_minio_client()
            spider.logger.info("MinIO Pipeline 已启动")
        except Exception as e:
            spider.logger.error(f"MinIO 客户端初始化失败: {e}")
            self.minio_client = None
    
    def spider_closed(self, spider):
        """爬虫关闭时输出统计信息"""
        spider.logger.info(
            f"MinIO Pipeline 统计 - 成功: {self.stats['uploaded']}, "
            f"失败: {self.stats['failed']}, 跳过: {self.stats['skipped']}"
        )
    
    def process_item(self, item, spider):
        """处理 Item，存储到 MinIO"""
        if not isinstance(item, SinaNewsItem):
            spider.logger.warning(f"未知的 Item 类型: {type(item)}")
            return item
        
        if self.minio_client is None:
            spider.logger.error("MinIO 客户端未初始化，跳过存储")
            self.stats['skipped'] += 1
            return item
        
        try:
            # 转换为字典
            news_data = item.to_dict()
            
            # 获取新闻 ID 和来源
            news_id = item.get('news_id', '')
            source = item.get('source', 'sina')
            
            if not news_id:
                spider.logger.warning("新闻 ID 为空，跳过存储")
                self.stats['skipped'] += 1
                return item
            
            # 检查是否已存在（去重）
            date = datetime.now().strftime("%Y-%m-%d")
            object_name = f"news/{date}/{source}/{news_id}.json"
            
            if self.minio_client.check_exists(object_name):
                spider.logger.debug(f"新闻已存在，跳过: {object_name}")
                self.stats['skipped'] += 1
                return item
            
            # 上传到 MinIO
            uploaded_path = self.minio_client.upload_news(
                news_data=news_data,
                news_id=news_id,
                source=source
            )
            
            spider.logger.info(f"新闻已存储到 MinIO: {uploaded_path}")
            self.stats['uploaded'] += 1
            
            # 将存储路径添加到 item 中
            item['minio_path'] = uploaded_path
            
        except Exception as e:
            spider.logger.error(f"存储到 MinIO 失败: {e}")
            self.stats['failed'] += 1
        
        return item


class DuplicateFilterPipeline:
    """
    去重 Pipeline
    
    基于 URL 进行去重，防止重复抓取
    """
    
    def __init__(self):
        self.crawled_urls = set()
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls()
    
    def process_item(self, item, spider):
        """处理 Item，检查是否重复"""
        adapter = ItemAdapter(item)
        url = adapter.get('url', '')
        
        if url in self.crawled_urls:
            spider.logger.debug(f"重复的新闻，跳过: {url}")
            raise scrapy.exceptions.DropItem(f"重复的新闻: {url}")
        
        self.crawled_urls.add(url)
        return item
