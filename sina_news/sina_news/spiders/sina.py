"""
新浪新闻爬虫模块 (Sina News Spider)

本模块实现基于Scrapy框架的新浪新闻爬虫，
用于自动抓取新浪新闻网站的新闻内容。

爬取范围：
- 新浪新闻首页 (news.sina.com.cn)
- 科技频道 (tech.sina.com.cn)
- 财经频道 (finance.sina.com.cn)
- 体育频道 (sports.sina.com.cn)

核心功能：
1. 列表页解析：从入口页面提取新闻链接
2. 详情页解析：提取新闻标题、正文、作者、时间等
3. 智能限流：控制爬取速度，避免被封
4. 数量限制：支持设置最大爬取数量
5. URL去重：避免重复爬取相同新闻

爬取策略：
- 多入口：从多个频道首页开始爬取，覆盖不同分类
- 链接过滤：使用正则匹配识别新闻详情页URL
- 内容清洗：清理标题中的站点名称等噪音
- 选择器回退：使用多种CSS选择器提高提取成功率

技术说明：
- 使用Scrapy框架实现异步高效爬取
- 支持Robots协议
- 自动处理相对URL
- 生成新闻唯一ID（基于URL的MD5）

使用方式：
    # 命令行运行
    cd sina_news
    scrapy crawl sina
    
    # 指定数量
    scrapy crawl sina -a max_news=100
    
    # Python模块方式
    from sina_news.spiders.sina import SinaSpider
    from scrapy.crawler import CrawlerProcess
    
    process = CrawlerProcess()
    process.crawl(SinaSpider, max_news=50)
    process.start()
"""

import re
import hashlib
from datetime import datetime
from urllib.parse import urljoin
import scrapy
from sina_news.items import SinaNewsItem


class SinaSpider(scrapy.Spider):
    """新浪新闻爬虫"""
    
    name = "sina"
    allowed_domains = ["news.sina.com.cn", "sina.com.cn", "tech.sina.com.cn", 
                       "finance.sina.com.cn", "sports.sina.com.cn"]
    
    # 多个入口页面，覆盖不同分类
    start_urls = [
        "https://news.sina.com.cn/",
        "https://tech.sina.com.cn/",
        "https://finance.sina.com.cn/",
        "https://sports.sina.com.cn/",
    ]
    
    # 爬取限制
    max_news = 50  # 最多爬取新闻数量
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crawled_urls = set()
        self.crawled_count = 0
        self.is_stopped = False  # 全局停止标志
        if 'max_news' in kwargs:
            self.max_news = int(kwargs['max_news'])
    
    def parse(self, response):
        """解析新闻列表页，提取新闻链接"""
        self.logger.info(f"解析列表页: {response.url}")
        
        # 检查是否已停止
        if self.is_stopped:
            return
        
        # 检查是否达到爬取上限
        if self.crawled_count >= self.max_news:
            if not self.is_stopped:
                self.logger.info(f"达到最大新闻数量限制 {self.max_news}，停止爬取")
                self.is_stopped = True
            return
        
        # 提取所有可能的新闻链接
        # 新浪新闻URL通常包含 /c/ 日期格式 或 doc- 格式
        all_links = response.css('a::attr(href)').getall()
        
        news_links = []
        for link in all_links:
            if link and self._is_news_url(link):
                full_url = urljoin(response.url, link)
                # 去重
                if full_url not in self.crawled_urls:
                    news_links.append(full_url)
        
        self.logger.info(f"发现 {len(news_links)} 条新闻链接，当前已爬取: {self.crawled_count}/{self.max_news}")
        
        # 请求详情页
        for link in news_links:
            # 双重检查：如果已停止或达到上限，直接返回
            if self.is_stopped or self.crawled_count >= self.max_news:
                if not self.is_stopped:
                    self.logger.info(f"达到最大数量，停止发送新请求")
                    self.is_stopped = True
                return
            
            self.crawled_urls.add(link)
            yield scrapy.Request(
                url=link, 
                callback=self.parse_detail,
                errback=self.parse_error,
                meta={'source_url': response.url}
            )
    
    def _is_news_url(self, url: str) -> bool:
        """判断是否是新闻详情页URL"""
        if not url:
            return False
        # 过滤掉非http链接
        if not url.startswith('http'):
            return False
        # 新浪新闻URL模式
        patterns = [
            r'/c/\d{4}-\d{2}-\d{2}/',  # 如 /c/2024-01-15/
            r'/doc-[a-z0-9]+\.shtml',   # 如 /doc-iiznezxt2247504.shtml
        ]
        return any(re.search(p, url) for p in patterns)
    
    def _generate_id(self, url: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()[:16]
    
    def parse_error(self, failure):
        """处理请求错误"""
        self.logger.error(f"请求失败: {failure.request.url}")
    
    def parse_detail(self, response):
        """解析新闻详情页"""
        # 如果已停止，不再处理新响应
        if self.is_stopped:
            return
        
        # 再次检查是否达到上限（可能已经满了）
        if self.crawled_count >= self.max_news:
            return
        
        if response.status != 200:
            self.logger.warning(f"页面状态异常: {response.url}, status={response.status}")
            return
        
        # 检查内容类型
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', errors='ignore')
        if 'text/html' not in content_type:
            self.logger.warning(f"非HTML页面: {response.url}, type={content_type}")
            return
        
        item = SinaNewsItem()
        
        # 提取标题 - 尝试多种选择器
        title = self._extract_title(response)
        if not title:
            self.logger.warning(f"无法提取标题: {response.url}")
            return
        
        item['title'] = title
        
        # 提取正文
        content = self._extract_content(response)
        if not content or len(content) < 50:
            self.logger.warning(f"内容太短或无法提取: {response.url}")
            return
        
        item['content'] = content
        item['url'] = response.url
        item['news_id'] = self._generate_id(response.url)
        item['source'] = 'sina'
        
        # 提取作者
        item['author'] = self._extract_author(response)
        
        # 提取发布时间
        item['publish_time'] = self._extract_publish_time(response)
        
        # 提取分类
        item['category'] = self._extract_category(response)
        
        # 提取标签
        item['tags'] = self._extract_tags(response)
        
        item['crawl_time'] = datetime.now().isoformat()
        
        self.crawled_count += 1
        self.logger.info(f"成功提取新闻 [{self.crawled_count}/{self.max_news}]: {title[:30]}...")
        
        yield item
    
    def _extract_title(self, response) -> str:
        """提取标题"""
        selectors = [
            'h1.main-title::text',
            'h1#artibodyTitle::text',
            'h1.article-title::text',
            'div.page-header h1::text',
            'h1::text',
            'title::text',
        ]
        
        for selector in selectors:
            title = response.css(selector).get('')
            if title:
                title = title.strip()
                # 清理标题中的站点名称
                title = re.sub(r'[_\-]?\s*新浪.*?$', '', title, flags=re.IGNORECASE)
                title = re.sub(r'[_\-]?\s*sina.*?$', '', title, flags=re.IGNORECASE)
                if len(title) > 5:  # 标题至少5个字符
                    return title
        return ''
    
    def _extract_content(self, response) -> str:
        """提取正文内容"""
        # 尝试多种选择器组合
        content_selectors = [
            'div#artibody p',           # 主要正文容器
            'div.article-content p',
            'div#article_content p',
            'div.content p',
            'div.main-content p',
            'div.text p',
            'div#artibody div',
        ]
        
        for selector in content_selectors:
            paragraphs = response.css(selector)
            if paragraphs:
                texts = []
                for p in paragraphs:
                    text = p.css('::text').get('').strip()
                    if text and len(text) > 10:  # 过滤短文本
                        texts.append(text)
                
                if texts:
                    content = '\n'.join(texts)
                    if len(content) > 100:
                        return content
        
        return ''
    
    def _extract_author(self, response) -> str:
        """提取作者"""
        selectors = [
            'p.show_author::text',
            'span.source::text',
            'span.author::text',
            'a.source::text',
            'span.source_ent::text',
            'span#source::text',
            'div.source::text',
        ]
        
        for selector in selectors:
            author = response.css(selector).get('')
            if author:
                author = author.strip()
                # 清理 "责任编辑：" 等前缀
                author = re.sub(r'^[\s\u3000]*(责任编辑[：:]?|作者[：:]?|来源[：:]?)\s*', '', author)
                if author and len(author) < 50:
                    return author
        return ''
    
    def _extract_publish_time(self, response) -> str:
        """提取发布时间"""
        selectors = [
            'span.date::text',
            'span.time-source::text',
            'span.titer::text',
            'span#pub_date::text',
            'div.date::text',
            'time::text',
            'span.pub-time::text',
        ]
        
        for selector in selectors:
            time_str = response.css(selector).get('')
            if time_str:
                time_str = time_str.strip()
                # 尝试规范化时间格式
                time_str = re.sub(r'[\s\u3000]+', ' ', time_str)
                if len(time_str) >= 10:  # 至少包含日期
                    return time_str
        return ''
    
    def _extract_category(self, response) -> str:
        """提取分类"""
        # 从面包屑导航提取
        breadcrumbs = response.css('div.bread a::text').getall()
        if breadcrumbs:
            # 通常最后一个或倒数第二个是分类
            for cat in reversed(breadcrumbs):
                cat = cat.strip()
                if cat and cat not in ['新浪首页', '首页', '新闻中心']:
                    return cat
        
        # 从URL推断分类
        url = response.url
        if 'tech' in url:
            return '科技'
        elif 'finance' in url:
            return '财经'
        elif 'sports' in url:
            return '体育'
        elif 'ent' in url:
            return '娱乐'
        
        return ''
    
    def _extract_tags(self, response) -> list:
        """提取标签"""
        tags = response.css('div.keywords a::text').getall()
        if not tags:
            tags = response.css('a.tag::text').getall()
        if not tags:
            tags = response.css('div.tags a::text').getall()
        
        return [tag.strip() for tag in tags if tag.strip()]
