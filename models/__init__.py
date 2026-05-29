"""
数据模型模块

提供新闻、用户等数据模型
"""

from .news import News, NewsSchema

__all__ = ["News", "NewsSchema"]
