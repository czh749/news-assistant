"""
用户兴趣管理服务模块 (User Interest Management Service)

本模块负责管理用户信息和兴趣偏好，是智能新闻助手个性化服务的基础。
用户兴趣用于每日简报生成和个性化新闻推荐。

核心功能：
1. 用户管理：创建用户、查询用户信息
2. 兴趣管理：添加、删除、查询用户兴趣
3. 兴趣向量化：将兴趣关键词转换为向量，用于个性化推荐
4. 权重管理：支持不同兴趣设置不同权重

数据存储：
- users表：存储用户基本信息（user_id来自飞书open_id）
- user_interests表：存储用户兴趣偏好

设计要点：
- 使用飞书用户的open_id作为user_id，无需额外注册
- 兴趣关键词支持向量化，便于后续相似度计算
- 支持权重机制，区分主要兴趣和次要兴趣
- 使用MySQL存储，保证数据持久性
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from storage.mysql_client import MySQLClient, get_mysql_client

logger = logging.getLogger(__name__)


@dataclass
class User:
    """
    用户数据类
    
    封装用户的基本信息。
    注意：这里的user_id是飞书用户的open_id，不是数据库自增ID。
    使用飞书ID可以直接关联飞书用户，无需单独注册流程。
    
    Attributes:
        id: 数据库自增主键
        user_id: 飞书用户open_id（唯一标识）
        name: 用户显示名称（可选）
        created_at: 记录创建时间
        updated_at: 记录最后更新时间
    """
    id: int
    user_id: str
    name: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class UserInterest:
    """
    用户兴趣数据类
    
    封装用户的兴趣偏好信息。
    兴趣关键词会用于每日简报生成和新闻推荐。
    
    Attributes:
        id: 数据库自增主键
        user_id: 关联的用户ID（飞书open_id）
        interest_keyword: 兴趣关键词（如"人工智能"、"财经"等）
        weight: 权重（0-1之间），表示兴趣的强烈程度
        created_at: 记录创建时间
        embedding: 兴趣关键词的向量表示（可选，用于相似度计算）
    """
    id: int
    user_id: str
    interest_keyword: str
    weight: float
    created_at: datetime
    embedding: Optional[List[float]] = None


class UserService:
    """
    用户兴趣管理服务类
    
    提供用户管理和兴趣偏好的完整CRUD操作。
    
    核心功能：
    1. 用户管理：自动创建飞书用户、查询用户信息
    2. 兴趣管理：添加、删除、查询兴趣
    3. 权重管理：支持设置兴趣权重
    4. 批量查询：获取所有用户及其兴趣（用于每日简报推送）
    
    设计特点：
    - 依赖注入：MySQL客户端和Embedding模型可通过构造函数传入
    - 自动用户创建：添加兴趣时自动创建不存在的用户
    - 向量化支持：兴趣关键词可自动向量化用于推荐
    """
    
    def __init__(
        self,
        mysql_client: Optional[MySQLClient] = None,
    ):
        """
        初始化用户服务
        
        Args:
            mysql_client: MySQL数据库客户端
        """
        # 使用传入的组件或获取默认实例
        self.db = mysql_client or get_mysql_client()
        
        logger.info("用户服务初始化完成")
    
    def get_or_create_user(self, user_id: str, name: Optional[str] = None) -> User:
        """
        获取或创建用户
        
        如果用户已存在则返回现有用户信息，不存在则创建新用户。
        这是处理飞书用户的标准方式，无需显式注册。
        
        使用场景：
        - 飞书用户首次发送消息时自动创建用户
        - 用户设置兴趣时确保用户存在
        
        Args:
            user_id: 用户ID（飞书open_id）
            name: 用户显示名称（可选）
            
        Returns:
            User: 用户对象（已存在或新创建）
        """
        # 先查询用户是否已存在
        sql = "SELECT * FROM users WHERE user_id = %s"
        result = self.db.fetch_one(sql, (user_id,))
        
        if result:
            # 用户已存在，返回现有信息
            logger.info(f"找到已有用户: {user_id}")
            return User(
                id=result['id'],
                user_id=result['user_id'],
                name=result['name'],
                created_at=result['created_at'],
                updated_at=result['updated_at']
            )
        
        # 用户不存在，创建新用户
        sql = """
            INSERT INTO users (user_id, name)
            VALUES (%s, %s)
        """
        self.db.execute(sql, (user_id, name))
        
        # 查询并返回新创建的用户
        sql = "SELECT * FROM users WHERE user_id = %s"
        result = self.db.fetch_one(sql, (user_id,))
        
        logger.info(f"创建新用户: {user_id}")
        return User(
            id=result['id'],
            user_id=result['user_id'],
            name=result['name'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[User]: 用户对象，不存在返回None
        """
        sql = "SELECT * FROM users WHERE user_id = %s"
        result = self.db.fetch_one(sql, (user_id,))
        
        if not result:
            return None
        
        return User(
            id=result['id'],
            user_id=result['user_id'],
            name=result['name'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
    
    def add_interest(self, user_id: str, interest_keyword: str, weight: float = 1.0) -> bool:
        """
        添加用户兴趣
        
        为用户添加一个兴趣关键词，支持权重设置。
        如果用户不存在会自动创建。
        如果兴趣已存在会更新权重。
        
        实现细节：
        - 使用INSERT ... ON DUPLICATE KEY UPDATE处理重复添加
        - 尝试对兴趣关键词向量化，失败不影响主流程
        - 自动创建不存在的用户
        
        Args:
            user_id: 用户ID（飞书open_id）
            interest_keyword: 兴趣关键词（如"人工智能"）
            weight: 权重（0-1之间，默认1.0），表示兴趣的强烈程度
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 确保用户存在（自动创建新用户）
            self.get_or_create_user(user_id)
            
            # 插入或更新兴趣记录
            # ON DUPLICATE KEY UPDATE: 如果记录已存在则更新权重
            sql = """
                INSERT INTO user_interests (user_id, interest_keyword, weight)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE weight = VALUES(weight)
            """
            self.db.execute(sql, (user_id, interest_keyword, weight))
            
            logger.info(f"为用户 {user_id} 添加兴趣: {interest_keyword}")
            return True
            
        except Exception as e:
            logger.error(f"添加兴趣失败: {e}")
            return False
    
    def remove_interest(self, user_id: str, interest_keyword: str) -> bool:
        """
        删除用户兴趣
        
        Args:
            user_id: 用户ID
            interest_keyword: 兴趣关键词
            
        Returns:
            bool: 是否成功
        """
        try:
            sql = "DELETE FROM user_interests WHERE user_id = %s AND interest_keyword = %s"
            self.db.execute(sql, (user_id, interest_keyword))
            
            logger.info(f"为用户 {user_id} 删除兴趣: {interest_keyword}")
            return True
            
        except Exception as e:
            logger.error(f"删除兴趣失败: {e}")
            return False
    
    def get_user_interests(self, user_id: str) -> List[UserInterest]:
        """
        获取用户兴趣列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[UserInterest]: 兴趣列表
        """
        sql = """
            SELECT * FROM user_interests
            WHERE user_id = %s
            ORDER BY weight DESC, created_at DESC
        """
        results = self.db.fetch_all(sql, (user_id,))
        
        interests = []
        for row in results:
            interests.append(UserInterest(
                id=row['id'],
                user_id=row['user_id'],
                interest_keyword=row['interest_keyword'],
                weight=row['weight'],
                created_at=row['created_at']
            ))
        
        return interests
    
    def get_interest_keywords(self, user_id: str) -> List[str]:
        """
        获取用户兴趣关键词列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[str]: 关键词列表
        """
        interests = self.get_user_interests(user_id)
        return [i.interest_keyword for i in interests]
    
    def update_interest_weight(self, user_id: str, interest_keyword: str, weight: float) -> bool:
        """
        更新兴趣权重
        
        Args:
            user_id: 用户ID
            interest_keyword: 兴趣关键词
            weight: 新权重
            
        Returns:
            bool: 是否成功
        """
        try:
            sql = """
                UPDATE user_interests
                SET weight = %s
                WHERE user_id = %s AND interest_keyword = %s
            """
            self.db.execute(sql, (weight, user_id, interest_keyword))
            
            logger.info(f"更新用户 {user_id} 的兴趣 {interest_keyword} 权重为 {weight}")
            return True
            
        except Exception as e:
            logger.error(f"更新兴趣权重失败: {e}")
            return False
    
    def get_all_users(self) -> List[User]:
        """
        获取所有用户
        
        Returns:
            List[User]: 用户列表
        """
        sql = "SELECT * FROM users ORDER BY created_at DESC"
        results = self.db.fetch_all(sql)
        
        users = []
        for row in results:
            users.append(User(
                id=row['id'],
                user_id=row['user_id'],
                name=row['name'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            ))
        
        return users
    
    def get_users_with_interests(self) -> List[Dict[str, Any]]:
        """
        获取所有有兴趣设置的用户及其兴趣
        
        用于每日简报生成：获取所有设置了兴趣的用户，
        为每个用户生成个性化的新闻简报。
        
        返回格式：
        [
            {
                'user_id': 'xxx',
                'name': '用户名',
                'interests': [
                    {'keyword': '人工智能', 'weight': 1.0},
                    {'keyword': '科技', 'weight': 0.8}
                ]
            },
            ...
        ]
        
        Returns:
            List[Dict]: 用户和兴趣的分组列表
        """
        # 查询所有用户的兴趣设置
        sql = """
            SELECT u.user_id, u.name, ui.interest_keyword, ui.weight
            FROM users u
            JOIN user_interests ui ON u.user_id = ui.user_id
            ORDER BY u.user_id, ui.weight DESC
        """
        results = self.db.fetch_all(sql)
        
        # 按用户ID分组整理数据
        users_dict = {}
        for row in results:
            user_id = row['user_id']
            if user_id not in users_dict:
                users_dict[user_id] = {
                    'user_id': user_id,
                    'name': row['name'],
                    'interests': []
                }
            users_dict[user_id]['interests'].append({
                'keyword': row['interest_keyword'],
                'weight': row['weight']
            })
        
        return list(users_dict.values())


# =============================================================================
# 全局服务实例（单例模式）
# =============================================================================
# 使用单例模式确保整个应用中只有一个UserService实例
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """
    获取用户服务单例
    
    懒加载方式获取UserService的全局唯一实例。
    
    Returns:
        UserService: 全局唯一的用户服务实例
    """
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service


# =============================================================================
# 使用示例和测试代码
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("用户服务测试")
    print("=" * 60)
    
    service = UserService()
    
    # 定义测试用户
    test_user_id = "test_user_001"
    
    print("\n1. 创建/获取用户")
    user = service.get_or_create_user(test_user_id, "测试用户")
    print(f"用户ID: {user.user_id}, 名称: {user.name}")
    
    print("\n2. 添加兴趣")
    service.add_interest(test_user_id, "人工智能", 1.0)
    service.add_interest(test_user_id, "科技", 0.8)
    service.add_interest(test_user_id, "财经", 0.6)
    
    print("\n3. 查询兴趣")
    interests = service.get_user_interests(test_user_id)
    for i in interests:
        print(f"  - {i.interest_keyword} (权重: {i.weight})")
    
    print("\n4. 删除兴趣")
    service.remove_interest(test_user_id, "财经")
    
    print("\n5. 再次查询兴趣")
    interests = service.get_user_interests(test_user_id)
    for i in interests:
        print(f"  - {i.interest_keyword} (权重: {i.weight})")
    
    print("\n6. 获取所有用户")
    users = service.get_all_users()
    for u in users:
        print(f"  - {u.user_id}: {u.name}")
