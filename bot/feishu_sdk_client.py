"""
飞书机器人 - 官方 SDK 长连接模式

使用飞书官方 SDK 的 WebSocket 长连接，无需内网穿透
参考文档：https://open.feishu.cn/document/developer-sdk/bot-development/bot-overview
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lark_oapi as lark

from config import settings

logger = logging.getLogger(__name__)


class FeishuSDKClient:
    """飞书官方 SDK 客户端"""

    def __init__(self):
        """初始化客户端"""
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET

        # 消息处理器
        self._message_handler: Optional[Callable] = None
        
        # 消息去重：记录已处理的消息ID，防止重复处理
        self._processed_messages: set = set()
        self._max_message_history = 100  # 最多保留100条消息记录

    def set_message_handler(self, handler: Callable):
        """
        设置消息处理器

        Args:
            handler: 处理函数，接收消息字典参数
        """
        self._message_handler = handler

    def start(self):
        """启动事件监听"""
        logger.info("=" * 60)
        logger.info("飞书机器人 - 官方 SDK 长连接模式")
        logger.info("=" * 60)
        logger.info("优势：")
        logger.info("  ✅ 无需内网穿透")
        logger.info("  ✅ 无需公网地址")
        logger.info("  ✅ 使用官方 SDK")
        logger.info("  ✅ 实时接收消息")
        logger.info("=" * 60)
        logger.info("正在初始化...")

        # 创建事件处理器
        def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
            """处理接收到的消息"""
            try:
                self._handle_message(data)
            except Exception as e:
                logger.error(f"处理消息失败: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # 构建事件分发器
        event_handler = lark.EventDispatcherHandler.builder(
            settings.FEISHU_ENCRYPT_KEY or "",
            settings.FEISHU_VERIFICATION_TOKEN or ""
        ).register_p2_im_message_receive_v1(do_p2_im_message_receive_v1) \
         .build()

        # 创建 WebSocket 客户端
        cli = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )

        logger.info("开始监听消息事件...")
        logger.info("按 Ctrl+C 可停止服务")
        logger.info("")

        # 启动客户端（阻塞运行）
        try:
            cli.start()
        except KeyboardInterrupt:
            logger.info("\n服务已停止")
        except Exception as e:
            logger.error(f"服务异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _handle_message(self, data: lark.im.v1.P2ImMessageReceiveV1):
        """
        处理消息事件

        Args:
            data: 飞书消息事件数据
        """
        try:
            # 提取消息信息
            event = data.event
            message = event.message
            sender = event.sender

            # 提取发送者信息
            open_id = sender.sender_id.open_id

            # 提取消息内容
            message_type = message.message_type
            content = message.content
            
            # 获取消息ID用于去重
            message_id = message.message_id
            
            # 检查消息是否已处理过（防止重复处理）
            if message_id in self._processed_messages:
                logger.debug(f"消息 {message_id} 已处理过，跳过")
                return
            
            # 记录消息ID
            self._processed_messages.add(message_id)
            # 限制历史记录大小，防止内存无限增长
            if len(self._processed_messages) > self._max_message_history:
                # 移除最旧的消息ID（集合是无序的，这里简单清空一半）
                self._processed_messages = set(list(self._processed_messages)[self._max_message_history//2:])

            logger.info(f"收到消息 - 类型: {message_type}, 发送者: {open_id}, 消息ID: {message_id}")

            if message_type != "text":
                logger.info("目前只支持文本消息")
                return

            # 解析文本内容
            content_dict = json.loads(content)
            text = content_dict.get("text", "").strip()

            logger.info(f"消息内容: {text}")

            # 调用消息处理器
            if self._message_handler:
                self._message_handler({
                    "open_id": open_id,
                    "text": text,
                    "message": message,
                    "sender": sender,
                    "event": data,
                    "message_id": message_id
                })

        except Exception as e:
            logger.error(f"处理消息事件失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def send_text_message(self, open_id: str, text: str) -> bool:
        """
        发送文本消息

        Args:
            open_id: 用户的 open_id
            text: 文本内容

        Returns:
            bool: 是否发送成功
        """
        try:
            # 创建客户端
            cli = lark.Client.builder() \
                .app_id(self.app_id) \
                .app_secret(self.app_secret) \
                .build()

            # 构造请求
            request = (lark.im.v1.CreateMessageRequest.builder()
                      .receive_id_type("open_id")
                      .request_body(lark.im.v1.CreateMessageRequestBody.builder()
                                   .receive_id(open_id)
                                   .msg_type("text")
                                   .content(json.dumps({"text": text}))
                                   .build())
                      .build())

            # 发送消息
            response = cli.im.v1.message.create(request)

            if response.code == 0:
                logger.info(f"消息发送成功: {open_id}")
                return True
            else:
                logger.error(f"消息发送失败: {response.msg}")
                return False

        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建客户端
    client = FeishuSDKClient()

    # 设置消息处理器
    def handle_message(msg_data):
        print(f"\n收到消息: {msg_data['text']}")
        # 发送回复
        client.send_text_message(
            msg_data['open_id'],
            f"收到你的消息：{msg_data['text']}"
        )

    client.set_message_handler(handle_message)

    # 启动监听
    client.start()
