"""
智能新闻助手 - 统一配置模块

集中管理所有配置项，支持从环境变量读取，提供默认值。
使用方法:
    from config.settings import settings
    api_key = settings.ZHIPU_API_KEY
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, encoding="utf-8")


class Settings:
    """统一配置类"""
    
    # ======================================
    # 智谱AI配置
    # ======================================
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_MODEL: str = os.getenv("ZHIPU_MODEL", "glm-4.5-flash")
    ZHIPU_EMBEDDING_MODEL: str = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2")
    
    # ======================================
    # MySQL数据库配置
    # ======================================
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "123456")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "news_db")
    
    @property
    def MYSQL_URL(self) -> str:
        """生成MySQL连接URL"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    # ======================================
    # MinIO对象存储配置
    # ======================================
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET_NEWS: str = os.getenv("MINIO_BUCKET_NEWS", "news-bucket")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    # ======================================
    # Milvus向量数据库配置
    # ======================================
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "news_collection")
    
    # ======================================
    # 飞书机器人配置
    # ======================================
    FEISHU_APP_ID: str = os.getenv("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET: str = os.getenv("FEISHU_APP_SECRET", "")
    FEISHU_ENCRYPT_KEY: str = os.getenv("FEISHU_ENCRYPT_KEY", "")
    FEISHU_VERIFICATION_TOKEN: str = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
    
    # ======================================
    # 爬虫配置
    # ======================================
    CRAWL_INTERVAL: int = int(os.getenv("CRAWL_INTERVAL", "3600"))
    CRAWL_USER_AGENT: str = os.getenv(
        "CRAWL_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    CRAWL_DELAY: float = float(os.getenv("CRAWL_DELAY", "1"))
    
    # ======================================
    # 应用配置
    # ======================================
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 项目根目录
    BASE_DIR: Path = Path(__file__).parent.parent
    
    def check_required_config(self) -> list:
        """
        检查必需的配置项是否已设置
        
        Returns:
            list: 未设置的必需配置项列表
        """
        required = [
            ("ZHIPU_API_KEY", self.ZHIPU_API_KEY),
        ]
        missing = [name for name, value in required if not value]
        return missing
    
    def print_config(self, hide_secret: bool = True):
        """
        打印当前配置（用于调试）
        
        Args:
            hide_secret: 是否隐藏敏感信息
        """
        print("=" * 50)
        print("智能新闻助手 - 当前配置")
        print("=" * 50)
        
        configs = [
            ("智谱API Key", self._mask_secret(self.ZHIPU_API_KEY) if hide_secret else self.ZHIPU_API_KEY),
            ("智谱模型", self.ZHIPU_MODEL),
            ("Embedding模型", self.ZHIPU_EMBEDDING_MODEL),
            ("", ""),
            ("MySQL主机", self.MYSQL_HOST),
            ("MySQL端口", self.MYSQL_PORT),
            ("MySQL数据库", self.MYSQL_DATABASE),
            ("", ""),
            ("MinIO端点", self.MINIO_ENDPOINT),
            ("MinIO存储桶", self.MINIO_BUCKET_NEWS),
            ("", ""),
            ("Milvus主机", self.MILVUS_HOST),
            ("Milvus端口", self.MILVUS_PORT),
            ("Milvus集合", self.MILVUS_COLLECTION),
            ("", ""),
            ("飞书App ID", self.FEISHU_APP_ID or "未配置"),
            ("", ""),
            ("调试模式", self.DEBUG),
            ("日志级别", self.LOG_LEVEL),
        ]
        
        for name, value in configs:
            if name:
                print(f"  {name}: {value}")
            else:
                print()
        
        print("=" * 50)
    
    @staticmethod
    def _mask_secret(value: str, show_chars: int = 8) -> str:
        """隐藏敏感信息，只显示前后部分"""
        if not value or len(value) <= show_chars * 2:
            return "***"
        return f"{value[:show_chars]}...{value[-show_chars:]}"


# 全局配置实例
settings = Settings()


if __name__ == "__main__":
    # 测试配置加载
    settings.print_config()
    
    # 检查必需配置
    missing = settings.check_required_config()
    if missing:
        print(f"\n[警告] 以下必需配置项未设置: {', '.join(missing)}")
    else:
        print("\n[OK] 所有必需配置项已设置")
