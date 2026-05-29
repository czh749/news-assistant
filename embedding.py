"""
向量嵌入模块 (Embedding Module)

本模块提供文本向量化的封装，使用智谱AI的Embedding API
将文本转换为高维向量表示，用于向量检索和相似度计算。

核心功能：
1. 单条向量化：将单个文本转换为1024维向量
2. 批量向量化：支持批量处理，带限流保护
3. 元数据附加：向量化同时附加业务元数据
4. 相似度搜索：封装Milvus检索的简化接口

技术说明：
- 使用智谱 embedding-2 模型，输出1024维向量
- 向量表示文本的语义信息，语义相近的文本向量距离近
- 批量处理时自动限流，避免API调用频率限制

应用场景：
- 新闻向量化存储到Milvus
- 用户查询向量化用于检索
- 兴趣关键词向量化用于推荐

使用示例：
    embedding = ZhipuEmbedding()
    
    # 单条向量化
    vector = embedding.embed_query("人工智能新闻")
    
    # 批量向量化
    vectors = embedding.embed_documents(["新闻1", "新闻2", "新闻3"])
    
    # 带元数据
    result = embedding.embed_with_metadata("新闻", metadata={"source": "sina"})
"""

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from typing import List, Optional, Dict, Any
import os
import logging
from config import settings

# 设置API密钥从配置读取
os.environ["ZHIPUAI_API_KEY"] = settings.ZHIPU_API_KEY

logger = logging.getLogger(__name__)


class ZhipuEmbedding:
    """
    智谱 Embedding 封装类
    
    提供文本向量化和 Milvus 存储对接功能
    """
    
    # 向量维度（智谱 embedding-2 为 1024 维）
    DIMENSION = 1024
    
    def __init__(self, model_name: str = None):
        """
        初始化智谱 Embedding 模型
        
        Args:
            model_name: Embedding 模型名称，默认从配置读取
        """
        self.model_name = model_name or settings.ZHIPU_EMBEDDING_MODEL
        self.embeddings = ZhipuAIEmbeddings(model=self.model_name)
        logger.info(f"Embedding 模型初始化: {self.model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        为单个文本生成嵌入向量
        
        Args:
            text: 待向量化的文本
            
        Returns:
            List[float]: 嵌入向量（1024 维）
        """
        try:
            vector = self.embeddings.embed_query(text)
            logger.debug(f"生成向量成功，维度: {len(vector)}")
            return vector
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            raise
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        批量为文档生成嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批次大小（避免 API 限流）
            
        Returns:
            List[List[float]]: 向量列表
        """
        try:
            all_vectors = []
            
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}，数量: {len(batch)}")
                
                vectors = self.embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
                
                # 避免请求过快
                if i + batch_size < len(texts):
                    import time
                    time.sleep(0.5)
            
            logger.info(f"批量向量化完成，总数: {len(all_vectors)}")
            return all_vectors
            
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            raise
    
    def embed_with_metadata(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成向量并附加元数据
        
        Args:
            text: 待向量化的文本
            metadata: 元数据字典
            
        Returns:
            Dict: 包含向量和元数据的字典
        """
        vector = self.embed_query(text)
        
        result = {
            'vector': vector,
            'dimension': len(vector),
            'model': self.model_name,
            'text_length': len(text)
        }
        
        if metadata:
            result['metadata'] = metadata
        
        return result
    
    def similarity_search(
        self,
        query_text: str,
        collection,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索（简化接口）
        
        Args:
            query_text: 查询文本
            collection: Milvus 集合对象
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        from storage import MilvusClient
        
        # 生成查询向量
        query_vector = self.embed_query(query_text)
        
        # 创建 Milvus 客户端并搜索
        milvus_client = MilvusClient()
        results = milvus_client.search(
            query_vector=query_vector,
            top_k=top_k
        )
        
        return results


# 全局实例
_embedding_instance: Optional[ZhipuEmbedding] = None


def get_embedding() -> ZhipuEmbedding:
    """获取 Embedding 实例（单例）"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = ZhipuEmbedding()
    return _embedding_instance


# 使用示例
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建实例
    embedding = ZhipuEmbedding()
    
    # 单个文本嵌入
    text = "我国自主研发量子计算机 '悟空' 算力再突破 实现特定问题百倍提速"
    query_embedding = embedding.embed_query(text)
    print(f"单个文本嵌入向量维度: {len(query_embedding)}")
    print(f"向量前5个值: {query_embedding[:5]}")
    
    # 多个文本嵌入
    texts = [
        "世界人工智能大会今日开幕 多款国产大模型首发亮相",
        "全国统一电子病历查询系统上线 跨省市就医告别纸质证明",
        "我国自主研发量子计算机 '悟空' 算力再突破"
    ]
    document_embeddings = embedding.embed_documents(texts)
    print(f"\n多个文本嵌入数量: {len(document_embeddings)}")
    print(f"第一个向量维度: {len(document_embeddings[0])}")
    
    # 带元数据的向量
    result = embedding.embed_with_metadata(
        text=text,
        metadata={'source': 'sina', 'category': '科技'}
    )
    print(f"\n带元数据的向量: {result['dimension']} 维")
    print(f"元数据: {result.get('metadata')}")
