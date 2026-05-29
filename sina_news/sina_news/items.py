"""
新浪新闻 Item 定义
"""

import scrapy


class SinaNewsItem(scrapy.Item):
    """新浪新闻 Item"""
    
    # 基本信息
    title = scrapy.Field()          # 新闻标题
    content = scrapy.Field()        # 新闻正文
    url = scrapy.Field()            # 原文链接
    news_id = scrapy.Field()        # 唯一ID（MD5生成）
    
    # 来源信息
    source = scrapy.Field()         # 来源（固定为sina）
    author = scrapy.Field()         # 作者
    publish_time = scrapy.Field()   # 发布时间
    
    # 分类信息
    category = scrapy.Field()       # 分类
    tags = scrapy.Field()           # 标签
    
    # 元数据
    crawl_time = scrapy.Field()     # 抓取时间
    summary = scrapy.Field()        # 摘要（后续生成）
    minio_path = scrapy.Field()     # MinIO存储路径
    
    def to_dict(self):
        """转换为字典"""
        return {
            'title': self.get('title', ''),
            'content': self.get('content', ''),
            'url': self.get('url', ''),
            'id': self.get('news_id', ''),
            'source': self.get('source', 'sina'),
            'author': self.get('author', ''),
            'publish_time': self.get('publish_time', ''),
            'category': self.get('category', ''),
            'tags': self.get('tags', []),
            'crawl_time': self.get('crawl_time', ''),
            'summary': self.get('summary', ''),
            'has_summary': False,
            'has_embedding': False,
            'status': 'active'
        }
