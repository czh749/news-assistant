"""
飞书机器人模块

提供消息接收、加解密、发送等功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import time
import hmac
import hashlib
import base64
import logging
from typing import Dict, Any, Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import requests

from config import settings

logger = logging.getLogger(__name__)


class FeishuCrypto:
    """飞书消息加解密工具"""
    
    def __init__(self, encrypt_key: str):
        """
        初始化加密工具
        
        Args:
            encrypt_key: 飞书应用的 Encrypt Key
        """
        # 飞书要求密钥为 32 字节，需要对 key 进行 base64 解码
        self.key = base64.b64decode(encrypt_key + "==")  # 补齐 base64 padding
        
    def decrypt(self, encrypted_data: str) -> str:
        """
        解密飞书消息
        
        Args:
            encrypted_data: base64 编码的加密数据
            
        Returns:
            str: 解密后的 JSON 字符串
        """
        try:
            # base64 解码
            encrypted_bytes = base64.b64decode(encrypted_data)
            
            # 提取 IV（前 16 字节）
            iv = encrypted_bytes[:16]
            encrypted_content = encrypted_bytes[16:]
            
            # AES 解密
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(encrypted_content)
            
            # 去除 padding
            decrypted = unpad(decrypted, AES.block_size)
            
            # 解码为字符串
            return decrypted.decode('utf-8')
            
        except Exception as e:
            logger.error(f"消息解密失败: {e}")
            raise
    
    def encrypt(self, message: str) -> str:
        """
        加密消息（用于返回加密消息）
        
        Args:
            message: 明文消息
            
        Returns:
            str: base64 编码的加密数据
        """
        try:
            # 生成随机 IV
            import os
            iv = os.urandom(16)
            
            # AES 加密
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            encrypted = cipher.encrypt(pad(message.encode('utf-8'), AES.block_size))
            
            # 拼接 IV + 密文
            result = iv + encrypted
            
            # base64 编码
            return base64.b64encode(result).decode('utf-8')
            
        except Exception as e:
            logger.error(f"消息加密失败: {e}")
            raise


class FeishuBot:
    """飞书机器人"""
    
    def __init__(self):
        """初始化飞书机器人"""
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET
        self.encrypt_key = settings.FEISHU_ENCRYPT_KEY
        self.verification_token = settings.FEISHU_VERIFICATION_TOKEN
        
        self.crypto = FeishuCrypto(self.encrypt_key) if self.encrypt_key else None
        
        # 飞书 API 地址
        self.base_url = "https://open.feishu.cn/open-apis"
        
        # 访问令牌缓存
        self._access_token = None
        self._token_expire_time = 0
        
    def verify_signature(self, timestamp: str, nonce: str, signature: str, body: str) -> bool:
        """
        验证请求签名（防止伪造请求）
        
        Args:
            timestamp: 时间戳
            nonce: 随机字符串
            signature: 签名
            body: 请求体
            
        Returns:
            bool: 是否验证通过
        """
        if not self.encrypt_key:
            logger.warning("未配置 Encrypt Key，跳过签名验证")
            return True
        
        try:
            # 拼接签名字符串
            sign_str = timestamp + nonce + self.encrypt_key + body
            
            # 计算签名
            calculated_signature = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
            
            # 验证
            return calculated_signature == signature
            
        except Exception as e:
            logger.error(f"签名验证失败: {e}")
            return False
    
    def get_access_token(self) -> str:
        """
        获取飞书访问令牌
        
        Returns:
            str: access_token
        """
        # 如果令牌未过期，直接返回
        if self._access_token and time.time() < self._token_expire_time:
            return self._access_token
        
        try:
            url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
            
            data = {
                "app_id": self.app_id,
                "app_secret": self.app_secret
            }
            
            response = requests.post(url, json=data)
            result = response.json()
            
            if result.get("code") != 0:
                raise Exception(f"获取 access_token 失败: {result.get('msg')}")
            
            self._access_token = result["tenant_access_token"]
            self._token_expire_time = time.time() + result["expire"] - 300  # 提前5分钟过期
            
            logger.info("获取 access_token 成功")
            return self._access_token
            
        except Exception as e:
            logger.error(f"获取 access_token 失败: {e}")
            raise
    
    def send_text_message(self, receive_id: str, receive_id_type: str, text: str) -> bool:
        """
        发送文本消息
        
        Args:
            receive_id: 接收者ID
            receive_id_type: 接收者类型（open_id, user_id, union_id, email, chat_id）
            text: 文本内容
            
        Returns:
            bool: 是否发送成功
        """
        try:
            access_token = self.get_access_token()
            
            url = f"{self.base_url}/im/v1/messages"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "receive_id_type": receive_id_type
            }
            
            data = {
                "receive_id": receive_id,
                "msg_type": "text",
                "content": json.dumps({"text": text})
            }
            
            response = requests.post(url, headers=headers, params=params, json=data)
            result = response.json()
            
            if result.get("code") != 0:
                logger.error(f"发送消息失败: {result.get('msg')}")
                return False
            
            logger.info(f"消息发送成功: {receive_id}")
            return True
            
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            return False
    
    def send_card_message(self, receive_id: str, receive_id_type: str, card: Dict[str, Any]) -> bool:
        """
        发送卡片消息
        
        Args:
            receive_id: 接收者ID
            receive_id_type: 接收者类型
            card: 卡片内容（字典格式）
            
        Returns:
            bool: 是否发送成功
        """
        try:
            access_token = self.get_access_token()
            
            url = f"{self.base_url}/im/v1/messages"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "receive_id_type": receive_id_type
            }
            
            data = {
                "receive_id": receive_id,
                "msg_type": "interactive",
                "content": json.dumps(card)
            }
            
            response = requests.post(url, headers=headers, params=params, json=data)
            result = response.json()
            
            if result.get("code") != 0:
                logger.error(f"发送卡片消息失败: {result.get('msg')}")
                return False
            
            logger.info(f"卡片消息发送成功: {receive_id}")
            return True
            
        except Exception as e:
            logger.error(f"发送卡片消息异常: {e}")
            return False
    
    def handle_message(self, event: Dict[str, Any]) -> Optional[str]:
        """
        处理接收到的消息
        
        Args:
            event: 飞书事件数据
            
        Returns:
            str: 回复消息内容（可选）
        """
        try:
            # 提取消息内容
            message = event.get("message", {})
            message_type = message.get("message_type")
            content = message.get("content")
            
            if message_type != "text":
                return "目前只支持文本消息"
            
            # 解析消息内容
            content_dict = json.loads(content)
            text = content_dict.get("text", "").strip()
            
            # 提取发送者信息
            sender = event.get("sender", {})
            sender_id = sender.get("sender_id", {}).get("user_id")
            
            logger.info(f"收到消息 - 用户: {sender_id}, 内容: {text}")
            
            # 返回消息内容（由上层处理）
            return text
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return "消息处理失败，请稍后重试"
    
    def create_news_card(self, title: str, summary: str, source: str, url: str) -> Dict[str, Any]:
        """
        创建新闻卡片
        
        Args:
            title: 新闻标题
            summary: 新闻摘要
            source: 新闻来源
            url: 新闻链接
            
        Returns:
            Dict: 卡片配置
        """
        card = {
            "type": "template",
            "data": {
                "template_id": "AAqk8gJ4",  # 默认卡片模板（可自定义）
                "template_variable": {
                    "title": title,
                    "summary": summary,
                    "source": source,
                    "url": url
                }
            }
        }
        
        # 如果没有自定义模板，使用标准卡片格式
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "content": f"**{title}**\n\n{summary}",
                        "tag": "lark_md"
                    }
                },
                {
                    "tag": "div",
                    "fields": [
                        {
                            "is_short": True,
                            "text": {
                                "content": f"**来源**: {source}",
                                "tag": "lark_md"
                            }
                        }
                    ]
                },
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {
                                "content": "查看原文",
                                "tag": "plain_text"
                            },
                            "type": "primary",
                            "url": url
                        }
                    ]
                }
            ]
        }
        
        return card


# 使用示例
if __name__ == "__main__":
    bot = FeishuBot()
    
    # 测试获取 access_token
    try:
        token = bot.get_access_token()
        print(f"Access Token: {token[:20]}...")
    except Exception as e:
        print(f"错误: {e}")
