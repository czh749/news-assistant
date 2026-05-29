"""
新闻数据模型模块 (News Data Model Module)

本模块定义新闻数据的结构、验证规则和序列化方法。
是整个系统中新闻数据的标准表示。

核心组件：
1. News数据类：新闻的核心数据结构，支持序列化和反序列化
2. NewsSchema类：JSON Schema定义，用于数据验证和API文档

设计要点：
- 使用Python dataclass实现，简洁高效
- 自动生成ID：基于URL的MD5哈希生成16位唯一ID
- 状态标记：has_summary和has_embedding记录处理状态
- 灵活的扩展字段：extra字段支持存储任意额外信息
- 多重序列化支持：支持dict、JSON、MinIO路径等多种格式

使用示例：
    # 创建新闻对象
    news = News(
        title="新闻标题",
        content="新闻内容...",
        url="https://example.com/news/123"
    )
    
    # 序列化为JSON
    json_str = news.to_json()
    
    # 从JSON恢复
    news = News.from_json(json_str)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import hashlib


@dataclass
class News:
    """
    新闻数据模型类
    
    Attributes:
        id: 新闻唯一ID（自动生成）
        title: 新闻标题
        content: 新闻正文内容
        summary: 新闻摘要（由AI生成）
        source: 新闻来源（如sina, 163等）
        url: 原文链接
        publish_time: 发布时间
        crawl_time: 抓取时间
        author: 作者
        category: 分类
        tags: 标签列表
        has_summary: 是否已生成摘要
        has_embedding: 是否已向量化
        status: 状态（active/deleted）
    """
    
    # 必需字段
    title: str
    content: str
    url: str
    
    # 可选字段
    id: Optional[str] = None
    summary: Optional[str] = None
    source: str = "unknown"
    publish_time: Optional[str] = None
    crawl_time: str = field(default_factory=lambda: datetime.now().isoformat())
    author: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    has_summary: bool = False
    has_embedding: bool = False
    status: str = "active"
    
    # 扩展字段（存储额外信息）
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，自动生成ID"""
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """基于URL生成唯一ID"""
        hash_obj = hashlib.md5(self.url.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "News":
        """从字典创建News对象"""
        # 过滤掉类中不存在的字段
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    @classmethod
    def from_json(cls, json_str: str) -> "News":
        """从JSON字符串创建News对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_text_for_embedding(self) -> str:
        """获取用于向量化的文本（标题+摘要）"""
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        return " ".join(parts)
    
    def get_full_text(self) -> str:
        """获取完整文本（标题+内容）"""
        return f"{self.title}\n\n{self.content}"
    
    def set_summary(self, summary: str):
        """设置摘要"""
        self.summary = summary
        self.has_summary = True
    
    def set_embedding_done(self):
        """标记已完成向量化"""
        self.has_embedding = True
    
    def to_minio_path(self) -> str:
        """生成MinIO存储路径"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"news/{date_str}/{self.source}/{self.id}.json"


class NewsSchema:
    """
    新闻数据JSON Schema定义
    
    用于验证和文档说明
    """
    
    SCHEMA = {
        "type": "object",
        "required": ["title", "content", "url"],
        "properties": {
            "id": {
                "type": "string",
                "description": "新闻唯一ID（16位MD5）"
            },
            "title": {
                "type": "string",
                "description": "新闻标题",
                "maxLength": 500
            },
            "content": {
                "type": "string",
                "description": "新闻正文内容"
            },
            "summary": {
                "type": ["string", "null"],
                "description": "新闻摘要（AI生成）",
                "maxLength": 2000
            },
            "source": {
                "type": "string",
                "description": "新闻来源",
                "enum": ["sina", "163", "qq", "sohu", "unknown"]
            },
            "url": {
                "type": "string",
                "description": "原文链接",
                "format": "uri"
            },
            "publish_time": {
                "type": ["string", "null"],
                "description": "发布时间（ISO 8601格式）"
            },
            "crawl_time": {
                "type": "string",
                "description": "抓取时间（ISO 8601格式）"
            },
            "author": {
                "type": ["string", "null"],
                "description": "作者"
            },
            "category": {
                "type": ["string", "null"],
                "description": "分类"
            },
            "tags": {
                "type": "array",
                "description": "标签列表",
                "items": {"type": "string"}
            },
            "has_summary": {
                "type": "boolean",
                "description": "是否已生成摘要"
            },
            "has_embedding": {
                "type": "boolean",
                "description": "是否已向量化"
            },
            "status": {
                "type": "string",
                "description": "状态",
                "enum": ["active", "deleted", "archived"]
            },
            "extra": {
                "type": "object",
                "description": "扩展字段"
            }
        }
    }
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """获取JSON Schema"""
        return cls.SCHEMA
    
    @classmethod
    def get_example(cls) -> Dict[str, Any]:
        """获取示例数据"""
        return {
            "id": "a1b2c3d4e5f67890",
            "title": "我国自主研发量子计算机 '悟空' 算力再突破",
            "content": "我国自主研发的量子计算机...",
            "summary": "我国量子计算机取得重大突破，实现百倍提速...",
            "source": "sina",
            "url": "https://news.sina.com.cn/2024/01/01/abc123.html",
            "publish_time": "2024-01-01T10:00:00",
            "crawl_time": "2024-01-01T12:00:00",
            "author": "张三",
            "category": "科技",
            "tags": ["量子计算", "科技突破", "国产"],
            "has_summary": True,
            "has_embedding": True,
            "status": "active",
            "extra": {
                "view_count": 10000,
                "comment_count": 500
            }
        }


if __name__ == "__main__":
    # 测试代码
    print("=== 测试 News 模型 ===")
    
    # 创建新闻对象
    news = News(
        title="测试新闻标题",
        content="这是测试新闻的详细内容...",
        url="https://example.com/news/123",
        source="sina",
        author="测试作者",
        tags=["测试", "新闻"]
    )
    
    print(f"新闻ID: {news.id}")
    print(f"MinIO路径: {news.to_minio_path()}")
    
    # 转换为JSON
    json_str = news.to_json(indent=2)
    print(f"\nJSON格式:\n{json_str}")
    
    # 从JSON恢复
    restored = News.from_json(json_str)
    print(f"\n恢复后的标题: {restored.title}")
    
    # 获取示例数据
    print("\n=== JSON Schema 示例 ===")
    example = NewsSchema.get_example()
    print(json.dumps(example, ensure_ascii=False, indent=2))
