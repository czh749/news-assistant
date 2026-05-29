"""
MinIO 对象存储客户端封装模块 (MinIO Client Module)

本模块提供MinIO对象存储的封装，用于存储新闻内容的JSON文件。
MinIO是一个兼容Amazon S3 API的开源对象存储系统。

核心功能：
1. 新闻上传：将新闻数据以JSON格式上传到MinIO
2. 新闻下载：根据对象路径下载新闻数据
3. 文件列表：列出指定前缀的对象
4. 文件删除：删除指定的新闻文件
5. 预签名URL：生成临时访问链接

存储路径设计：
    news/{YYYY-MM-DD}/{source}/{news_id}.json
    示例: news/2024-01-15/sina/a1b2c3d4e5f67890.json

设计特点：
- 自动创建存储桶：初始化时自动检查并创建bucket
- 分层存储：按日期和来源组织文件，便于管理
- 流式上传：使用BytesIO实现流式上传，支持大文件
- 异常处理：捕获S3Error并转换为友好的错误信息

使用示例：
    client = MinioClient()
    
    # 上传新闻
    path = client.upload_news(news_dict, news_id="123", source="sina")
    
    # 下载新闻
    data = client.download_news("news/2024-01-15/sina/123.json")
    
    # 列出今天的新闻
    files = client.list_news_by_date("2024-01-15")
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, BinaryIO
from minio import Minio
from minio.error import S3Error
from config import settings

logger = logging.getLogger(__name__)


class MinioClient:
    """MinIO 客户端封装类"""
    
    def __init__(self):
        """初始化MinIO客户端"""
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.bucket_name = settings.MINIO_BUCKET_NEWS
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """确保存储桶存在，不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"创建存储桶: {self.bucket_name}")
            else:
                logger.info(f"存储桶已存在: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"存储桶操作失败: {e}")
            raise
    
    def upload_news(
        self,
        news_data: Dict[str, Any],
        news_id: str,
        source: str = "sina",
        date: Optional[str] = None
    ) -> str:
        """
        上传新闻JSON数据到MinIO
        
        Args:
            news_data: 新闻数据字典
            news_id: 新闻唯一ID
            source: 新闻来源，默认sina
            date: 日期字符串，格式YYYY-MM-DD，默认今天
            
        Returns:
            str: 存储的对象路径
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 构建存储路径: news/{date}/{source}/{id}.json
        object_name = f"news/{date}/{source}/{news_id}.json"
        
        # 将数据转为JSON字节
        json_data = json.dumps(news_data, ensure_ascii=False, indent=2).encode('utf-8')
        
        try:
            # 使用put_object上传
            from io import BytesIO
            data_stream = BytesIO(json_data)
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(json_data),
                content_type='application/json'
            )
            
            logger.info(f"上传成功: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"上传失败: {e}")
            raise
    
    def download_news(self, object_name: str) -> Optional[Dict[str, Any]]:
        """
        下载新闻JSON数据
        
        Args:
            object_name: 对象路径，如 news/2024-01-01/sina/123.json
            
        Returns:
            Dict: 新闻数据字典，失败返回None
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            return data
            
        except S3Error as e:
            logger.error(f"下载失败: {e}")
            return None
    
    def list_news(
        self,
        prefix: str = "news/",
        recursive: bool = True
    ) -> List[str]:
        """
        列出新闻文件
        
        Args:
            prefix: 路径前缀
            recursive: 是否递归列出
            
        Returns:
            List[str]: 对象名称列表
        """
        try:
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            return [obj.object_name for obj in objects]
            
        except S3Error as e:
            logger.error(f"列出对象失败: {e}")
            return []
    
    def list_news_by_date(
        self,
        date: str,
        source: Optional[str] = None
    ) -> List[str]:
        """
        按日期列出新闻文件
        
        Args:
            date: 日期，格式YYYY-MM-DD
            source: 新闻来源，如sina
            
        Returns:
            List[str]: 对象名称列表
        """
        if source:
            prefix = f"news/{date}/{source}/"
        else:
            prefix = f"news/{date}/"
        
        return self.list_news(prefix=prefix)
    
    def delete_news(self, object_name: str) -> bool:
        """
        删除新闻文件
        
        Args:
            object_name: 对象路径
            
        Returns:
            bool: 是否成功
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"删除成功: {object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"删除失败: {e}")
            return False
    
    def check_exists(self, object_name: str) -> bool:
        """
        检查对象是否存在
        
        Args:
            object_name: 对象路径
            
        Returns:
            bool: 是否存在
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_presigned_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """
        获取临时访问URL
        
        Args:
            object_name: 对象路径
            expires: URL有效期（秒），默认1小时
            
        Returns:
            str: 临时URL
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            logger.error(f"生成URL失败: {e}")
            return None


# 全局客户端实例（单例模式）
_minio_client: Optional[MinioClient] = None


def get_minio_client() -> MinioClient:
    """获取MinIO客户端单例"""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinioClient()
    return _minio_client


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    client = MinioClient()
    
    # 测试上传
    test_news = {
        "id": "test001",
        "title": "测试新闻标题",
        "content": "这是测试新闻内容...",
        "source": "sina",
        "publish_time": datetime.now().isoformat(),
        "url": "https://example.com/news/1"
    }
    
    object_path = client.upload_news(
        news_data=test_news,
        news_id="test001",
        source="sina"
    )
    print(f"上传路径: {object_path}")
    
    # 测试下载
    downloaded = client.download_news(object_path)
    print(f"下载数据: {downloaded}")
    
    # 测试列出
    files = client.list_news_by_date(datetime.now().strftime("%Y-%m-%d"))
    print(f"今日文件: {files}")
