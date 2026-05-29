"""
飞书机器人 - 官方 SDK 模式启动脚本

使用飞书官方 SDK 长连接，无需内网穿透
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bot.feishu_sdk_client import FeishuSDKClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("飞书机器人 - 官方 SDK 长连接模式")
    logger.info("=" * 60)
    logger.info("优势：")
    logger.info("  ✅ 无需内网穿透")
    logger.info("  ✅ 无需公网地址")
    logger.info("  ✅ 使用官方 SDK，稳定可靠")
    logger.info("=" * 60)

    # 创建客户端
    client = FeishuSDKClient()

    # 设置消息处理器
    def handle_message(msg_data):
        """处理收到的消息"""
        try:
            open_id = msg_data["open_id"]
            text = msg_data["text"]

            logger.info(f"收到用户[{open_id}]消息: {text}")

            # 处理消息（集成对话服务）
            reply = process_message(text, open_id)

            # 发送回复
            client.send_text_message(open_id, reply)

            logger.info(f"回复已发送")

        except Exception as e:
            logger.error(f"处理消息失败: {e}")

    client.set_message_handler(handle_message)

    # 启动监听
    try:
        client.start()
    except KeyboardInterrupt:
        logger.info("\n服务已停止")
    except Exception as e:
        logger.error(f"服务异常: {e}")
        import traceback
        logger.error(traceback.format_exc())


def process_message(text: str, open_id: str) -> str:
    """
    处理用户消息 - 集成对话服务

    Args:
        text: 用户消息文本
        open_id: 用户open_id

    Returns:
        str: 回复内容
    """
    from services.chat_service import get_chat_service

    try:
        # 获取对话服务
        chat_service = get_chat_service()

        # 处理消息
        response = chat_service.process_message(text, user_id=open_id)

        # 返回回复内容
        return response.reply_text

    except Exception as e:
        logger.error(f"处理消息失败: {e}")
        return "抱歉，处理您的消息时出现了错误，请稍后再试。"


if __name__ == "__main__":
    main()
