"""
Milvus 向量数据库客户端封装模块 (Milvus Client Module)

本模块提供Milvus向量数据库的封装，用于存储和检索新闻的向量表示。
Milvus是一个开源的向量数据库，专为处理大规模向量数据而设计。

核心功能：
1. 集合管理：创建、获取、删除向量集合
2. 向量插入：批量或单条插入新闻向量
3. 相似度搜索：基于余弦相似度的向量检索
4. 索引管理：自动创建IVF_FLAT索引加速搜索
5. 统计信息：获取集合的实体数量等统计信息

向量字段设计：
    - id: 主键（字符串，新闻ID的MD5值）
    - news_id: 新闻唯一ID
    - title: 新闻标题
    - source: 新闻来源
    - category: 分类
    - publish_time: 发布时间
    - embedding: 向量嵌入（1024维浮点数，来自智谱Embedding模型）

索引配置：
    - 类型: IVF_FLAT（适合中小规模数据）
    - 度量: COSINE（余弦相似度）
    - nlist: 128（聚类中心数）

使用示例：
    client = MilvusClient()
    
    # 创建集合
    client.create_collection()
    
    # 插入向量
    client.insert_news(news_id="123", title="标题", vector=[...])
    
    # 搜索相似向量
    results = client.search(query_vector=[...], top_k=5)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    MilvusException
)
from config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus 向量数据库客户端封装"""
    
    # 向量维度（智谱 embedding-2 模型为 1024 维）
    DIM = 1024
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        初始化 Milvus 客户端
        
        Args:
            collection_name: 集合名称，默认使用配置文件中的名称
        """
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self._connection_alias = "default"
        self._collection = None
        
        self._connect()
    
    def _connect(self):
        """建立 Milvus 连接"""
        try:
            connections.connect(
                alias=self._connection_alias,
                host=self.host,
                port=self.port
            )
            logger.info(f"Milvus 连接成功: {self.host}:{self.port}")
        except MilvusException as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise
    
    def disconnect(self):
        """断开 Milvus 连接"""
        connections.disconnect(self._connection_alias)
        logger.info("Milvus 连接已断开")
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None
    ) -> Collection:
        """
        创建新闻向量集合
        
        Args:
            collection_name: 集合名称，默认使用初始化时的名称
            dim: 向量维度，默认 1024
            
        Returns:
            Collection: Milvus 集合对象
        """
        name = collection_name or self.collection_name
        vector_dim = dim or self.DIM
        
        # 如果集合已存在，直接返回
        if utility.has_collection(name, using=self._connection_alias):
            logger.info(f"集合已存在: {name}")
            self._collection = Collection(name)
            return self._collection
        
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name="news_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="新闻唯一ID"
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="新闻标题"
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="新闻来源"
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="分类"
            ),
            FieldSchema(
                name="publish_time",
                dtype=DataType.VARCHAR,
                max_length=30,
                description="发布时间"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=vector_dim,
                description="向量嵌入"
            ),
        ]
        
        # 创建集合 Schema
        schema = CollectionSchema(
            fields=fields,
            description="新闻向量集合",
            enable_dynamic_field=True
        )
        
        # 创建集合
        collection = Collection(
            name=name,
            schema=schema,
            using=self._connection_alias
        )
        
        # 创建索引（IVF_FLAT 算法，适合小规模数据）
        index_params = {
            "metric_type": "COSINE",  # 余弦相似度
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"集合创建成功: {name}, 维度: {vector_dim}")
        self._collection = collection
        return collection
    
    def get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        获取集合对象
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Collection: 集合对象
        """
        name = collection_name or self.collection_name
        
        if not utility.has_collection(name, using=self._connection_alias):
            raise ValueError(f"集合不存在: {name}")
        
        self._collection = Collection(name)
        return self._collection
    
    def insert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        news_ids: List[str],
        titles: List[str],
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        publish_times: Optional[List[str]] = None
    ) -> List[int]:
        """
        插入向量数据
        
        Args:
            vectors: 向量列表，每个向量是 1024 维的浮点数列表
            ids: 主键ID列表
            news_ids: 新闻ID列表
            titles: 标题列表
            sources: 来源列表
            categories: 分类列表
            publish_times: 发布时间列表
            
        Returns:
            List[int]: 插入的实体ID列表
        """
        if self._collection is None:
            self.get_collection()
        
        count = len(vectors)
        
        # 准备数据
        entities = [
            ids,
            news_ids,
            titles,
            sources or ["unknown"] * count,
            categories or [""] * count,  # 使用空字符串替代 None
            publish_times or [""] * count,  # 使用空字符串替代 None
            vectors
        ]
        
        try:
            # 插入数据
            insert_result = self._collection.insert(entities)
            self._collection.flush()
            
            logger.info(f"插入 {count} 条向量数据成功")
            return insert_result.primary_keys
            
        except MilvusException as e:
            logger.error(f"插入向量失败: {e}")
            raise
    
    def insert_news(
        self,
        news_id: str,
        title: str,
        vector: List[float],
        source: str = "unknown",
        category: Optional[str] = None,
        publish_time: Optional[str] = None
    ) -> List[int]:
        """
        插入单条新闻向量
        
        Args:
            news_id: 新闻ID
            title: 新闻标题
            vector: 向量
            source: 来源
            category: 分类
            publish_time: 发布时间
            
        Returns:
            List[int]: 插入的实体ID列表
        """
        return self.insert_vectors(
            vectors=[vector],
            ids=[news_id],
            news_ids=[news_id],
            titles=[title],
            sources=[source],
            categories=[category] if category else [""],  # 传递空字符串而不是 None
            publish_times=[publish_time] if publish_time else [""]  # 传递空字符串而不是 None
        )
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            collection_name: 集合名称
            output_fields: 返回的字段列表
            
        Returns:
            List[Dict]: 搜索结果列表，包含 id, distance, entity 等信息
        """
        if self._collection is None:
            self.get_collection(collection_name)
        
        # 加载集合到内存
        self._collection.load()
        
        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 执行搜索
        results = self._collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields or ["news_id", "title", "source"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "news_id": hit.entity.get("news_id"),
                    "title": hit.entity.get("title"),
                    "source": hit.entity.get("source"),
                })
        
        return formatted_results
    
    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        根据ID删除向量
        
        Args:
            ids: 要删除的ID列表
            
        Returns:
            bool: 是否成功
        """
        if self._collection is None:
            self.get_collection()
        
        try:
            id_list = ", ".join([f'"{id}"' for id in ids])
            expr = f"id in [{id_list}]"
            self._collection.delete(expr)
            logger.info(f"删除 {len(ids)} 条向量数据")
            return True
        except MilvusException as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def drop_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除集合（危险操作）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            bool: 是否成功
        """
        name = collection_name or self.collection_name
        
        try:
            utility.drop_collection(name, using=self._connection_alias)
            logger.warning(f"集合已删除: {name}")
            return True
        except MilvusException as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            Dict: 统计信息
        """
        if self._collection is None:
            self.get_collection()
        
        self._collection.load()
        stats = {
            "collection_name": self._collection.name,
            "num_entities": self._collection.num_entities,
            "is_empty": self._collection.is_empty
        }
        return stats
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            connections.connect(
                host=self.host,
                port=self.port
            )
            version = utility.get_server_version()
            logger.info(f"Milvus 版本: {version}")
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False


# 全局客户端实例（单例模式）
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 1. 测试连接
    client = MilvusClient()
    if not client.test_connection():
        print("[失败] Milvus 连接失败")
        exit(1)
    print("[OK] Milvus 连接成功")
    
    # 2. 创建集合
    collection = client.create_collection()
    print(f"[OK] 集合创建/获取成功: {collection.name}")
    
    # 3. 测试插入
    import random
    test_vector = [random.random() for _ in range(1024)]
    client.insert_news(
        news_id="test001",
        title="测试新闻标题",
        vector=test_vector,
        source="sina",
        category="科技"
    )
    print("[OK] 向量插入成功")
    
    # 4. 测试搜索
    results = client.search(test_vector, top_k=3)
    print(f"[OK] 搜索返回 {len(results)} 条结果")
    for r in results:
        print(f"  - {r['title']} (相似度: {r['distance']:.4f})")
    
    # 5. 获取统计
    stats = client.get_stats()
    print(f"[OK] 集合统计: {stats}")
    
    print("\n所有测试通过！")
