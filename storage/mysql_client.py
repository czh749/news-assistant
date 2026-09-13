"""
MySQL 数据库客户端封装模块 (MySQL Client Module)

本模块提供MySQL数据库的连接管理和基础CRUD操作封装。
使用PyMySQL库实现，支持字典类型的游标返回。

核心功能：
1. 连接管理：自动创建和关闭数据库连接
2. 事务支持：使用上下文管理器自动处理commit/rollback
3. 基础CRUD：封装常用的增删改查操作
4. 参数化查询：防止SQL注入攻击
5. 批量操作：支持批量插入提高效率

设计特点：
- 上下文管理器：使用with语句自动管理连接和事务
- 字典游标：查询结果直接返回字典格式，便于使用
- 异常处理：自动回滚事务并记录错误日志
- 单例模式：全局共享同一个客户端实例

使用示例：
    client = MySQLClient()
    
    # 查询单条
    result = client.fetchone("SELECT * FROM users WHERE id = %s", (1,))
    
    # 插入数据
    client.insert("users", {"name": "张三", "age": 25})
    
    # 使用上下文管理器执行复杂操作
    with client.get_cursor() as cursor:
        cursor.execute("INSERT INTO logs (msg) VALUES (%s)", ("操作日志",))
"""

import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
import pymysql
from pymysql.cursors import DictCursor
from pymysql.connections import Connection
from config import settings

logger = logging.getLogger(__name__)


class MySQLClient:
    """MySQL 客户端封装类"""
    
    def __init__(self):
        """初始化MySQL连接配置"""
        self.host = settings.MYSQL_HOST
        self.port = settings.MYSQL_PORT
        self.user = settings.MYSQL_USER
        self.password = settings.MYSQL_PASSWORD
        self.database = settings.MYSQL_DATABASE
        self._connection: Optional[Connection] = None
    
    def _get_connection(self) -> Connection:
        """获取数据库连接"""
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=False
            )
            return conn
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """
        获取数据库游标的上下文管理器
        
        使用示例:
            with client.get_cursor() as cursor:
                cursor.execute("SELECT * FROM news")
                results = cursor.fetchall()
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def execute(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> int:
        """
        执行SQL语句（INSERT/UPDATE/DELETE）
        
        Args:
            sql: SQL语句
            params: SQL参数
            
        Returns:
            int: 影响的行数
        """
        with self.get_cursor() as cursor:
            affected = cursor.execute(sql, params)
            logger.debug(f"执行SQL: {sql}, 影响行数: {affected}")
            return affected
    
    def fetch_one(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        查询单条记录
        
        Args:
            sql: SQL语句
            params: SQL参数
            
        Returns:
            Dict: 单条记录字典，无结果返回None
        """
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()
    
    def fetch_all(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        查询多条记录
        
        Args:
            sql: SQL语句
            params: SQL参数
            
        Returns:
            List[Dict]: 记录字典列表
        """
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    # Backwards-compatible aliases for callers using PyMySQL-style names.
    def fetchone(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        return self.fetch_one(sql, params)

    def fetchall(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        return self.fetch_all(sql, params)
    
    def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> int:
        """
        插入单条记录
        
        Args:
            table: 表名
            data: 数据字典
            
        Returns:
            int: 新记录ID
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with self.get_cursor() as cursor:
            cursor.execute(sql, tuple(data.values()))
            return cursor.lastrowid
    
    def insert_many(
        self,
        table: str,
        data_list: List[Dict[str, Any]]
    ) -> int:
        """
        批量插入记录
        
        Args:
            table: 表名
            data_list: 数据字典列表
            
        Returns:
            int: 影响的行数
        """
        if not data_list:
            return 0
        
        columns = ', '.join(data_list[0].keys())
        placeholders = ', '.join(['%s'] * len(data_list[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        values = [tuple(d.values()) for d in data_list]
        
        with self.get_cursor() as cursor:
            affected = cursor.executemany(sql, values)
            return affected
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        where_params: Tuple
    ) -> int:
        """
        更新记录
        
        Args:
            table: 表名
            data: 要更新的数据
            where: WHERE条件
            where_params: WHERE参数
            
        Returns:
            int: 影响的行数
        """
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        
        params = tuple(data.values()) + where_params
        
        return self.execute(sql, params)
    
    def delete(
        self,
        table: str,
        where: str,
        where_params: Tuple
    ) -> int:
        """
        删除记录
        
        Args:
            table: 表名
            where: WHERE条件
            where_params: WHERE参数
            
        Returns:
            int: 影响的行数
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.execute(sql, where_params)
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False


# 全局客户端实例（单例模式）
_mysql_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    """获取MySQL客户端单例"""
    global _mysql_client
    if _mysql_client is None:
        _mysql_client = MySQLClient()
    return _mysql_client


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    client = MySQLClient()
    
    # 测试连接
    if client.test_connection():
        print("数据库连接成功!")
    else:
        print("数据库连接失败!")
