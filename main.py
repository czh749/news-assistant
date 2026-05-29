"""
智能新闻助手 - 主入口

整合所有服务，提供统一启动入口

使用方法:
    python main.py start      # 启动所有服务（飞书机器人 + 定时任务 + 摘要 + 向量化）
    python main.py crawler    # 手动运行爬虫一次
    python main.py summary    # 手动运行摘要生成一次
    python main.py embedding  # 手动运行向量化一次
    python main.py test       # 运行端到端测试

服务架构:
    - 飞书机器人：处理用户对话，接收消息并回复
    - 定时任务：每日简报推送 + 每小时爬虫
    - 摘要服务：轮询生成新闻摘要（每60秒）
    - 向量化服务：轮询生成向量存入Milvus（每60秒）
    - 存储服务：MySQL + MinIO + Milvus（需单独启动）

自动处理流程:
    爬虫抓取新闻 → MinIO存储 → 摘要服务生成摘要 → 向量化服务生成向量 → Milvus存储
"""

import sys
import signal
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsAssistant:
    """
    智能新闻助手主类
    
    负责管理所有服务的生命周期：
    - 定时任务服务（SchedulerService）：定时爬虫 + 每日简报
    - 飞书机器人（FeishuSDKClient）：用户交互入口
    - 对话服务（ChatService）：处理用户消息逻辑
    - 摘要服务（SummaryService）：轮询生成新闻摘要
    - 向量化服务（EmbeddingService）：轮询生成向量存入Milvus
    
    Attributes:
        running: 服务运行状态标志
        scheduler: 定时任务服务实例
        feishu_client: 飞书机器人客户端实例
        chat_service: 对话服务实例
        summary_service: 摘要服务实例
        embedding_service: 向量化服务实例
        summary_thread: 摘要服务线程
        embedding_thread: 向量化服务线程
    """
    
    def __init__(self):
        """初始化新闻助手，注册信号处理器"""
        self.running = False
        self.scheduler = None
        self.feishu_client = None
        self.chat_service = None
        self.summary_service = None
        self.embedding_service = None
        self.summary_thread = None
        self.embedding_thread = None
        
        # 注册信号处理（支持Ctrl+C优雅退出）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        处理系统退出信号
        
        Args:
            signum: 信号编号
            frame: 当前栈帧
        """
        logger.info(f"收到信号 {signum}，正在关闭服务...")
        self.stop()
        sys.exit(0)
    
    def start_scheduler(self):
        """
        启动定时任务服务
        
        定时任务包括：
        - 每日9:00生成并推送简报
        - 每小时爬取一次新闻
        """
        from services.scheduler_service import get_scheduler_service
        
        logger.info("启动定时任务服务...")
        self.scheduler = get_scheduler_service()
        self.scheduler.start()
        logger.info("定时任务服务已启动")
    
    def start_summary_service(self):
        """
        启动摘要生成服务（后台线程）
        
        轮询监听MinIO中的新新闻，自动生成摘要
        默认每60秒扫描一次
        """
        import threading
        from services.summary_service import SummaryService
        
        logger.info("启动摘要生成服务...")
        
        self.summary_service = SummaryService(poll_interval=600)
        
        def run_summary():
            """在新线程中运行摘要服务"""
            try:
                self.summary_service.run()
            except Exception as e:
                logger.error(f"摘要服务异常: {e}")
        
        # 在后台线程启动
        self.summary_thread = threading.Thread(target=run_summary, daemon=True)
        self.summary_thread.start()
        logger.info("摘要生成服务已启动（后台线程，每10分钟轮询）")
    
    def start_embedding_service(self):
        """
        启动向量化服务（后台线程）
        
        轮询监听已生成摘要的新闻，生成向量存入Milvus
        默认每60秒扫描一次
        """
        import threading
        from services.embedding_service import EmbeddingService
        
        logger.info("启动向量化服务...")
        
        self.embedding_service = EmbeddingService(poll_interval=600)
        
        def run_embedding():
            """在新线程中运行向量化服务"""
            try:
                self.embedding_service.run()
            except Exception as e:
                logger.error(f"向量化服务异常: {e}")
        
        # 在后台线程启动
        self.embedding_thread = threading.Thread(target=run_embedding, daemon=True)
        self.embedding_thread.start()
        logger.info("向量化服务已启动（后台线程，每10分钟轮询）")
    
    def start_feishu_bot(self):
        """
        启动飞书机器人
        
        使用官方SDK长连接模式，无需内网穿透
        支持消息接收和回复
        """
        from bot.feishu_sdk_client import FeishuSDKClient
        from services.chat_service import get_chat_service
        
        logger.info("启动飞书机器人...")
        
        # 创建飞书SDK客户端
        self.feishu_client = FeishuSDKClient()
        
        # 获取对话服务实例
        self.chat_service = get_chat_service()
        
        # 设置消息处理器（回调函数）
        def handle_message(msg_data: dict):
            """
            处理收到的飞书消息
            
            Args:
                msg_data: 消息数据，包含open_id和text
            """
            try:
                open_id = msg_data["open_id"]  # 用户唯一标识
                text = msg_data["text"]         # 消息内容
                
                logger.info(f"收到用户[{open_id}]消息: {text}")
                
                # 调用对话服务处理消息
                response = self.chat_service.process_message(text, user_id=open_id)
                
                # 发送回复给用户
                self.feishu_client.send_text_message(open_id, response.reply_text)
                
                logger.info(f"回复已发送")
                
            except Exception as e:
                logger.error(f"处理消息失败: {e}", exc_info=True)
        
        # 注册消息处理器
        self.feishu_client.set_message_handler(handle_message)
        
        # 启动长连接监听（阻塞方法）
        self.feishu_client.start()
    
    def start_all(self):
        """
        启动所有服务
        
        启动顺序：
        1. 定时任务服务（后台线程）
        2. 摘要生成服务（后台线程）
        3. 向量化服务（后台线程）
        4. 飞书机器人（主线程阻塞）
        """
        logger.info("=" * 60)
        logger.info("智能新闻助手 - 启动所有服务")
        logger.info("=" * 60)
        
        self.running = True
        
        # 启动定时任务（后台运行）
        self.start_scheduler()
        
        # 启动摘要生成服务（后台线程）
        self.start_summary_service()
        
        # 启动向量化服务（后台线程）
        self.start_embedding_service()
        
        logger.info("=" * 60)
        logger.info("所有后台服务已启动")
        logger.info("=" * 60)
        
        # 启动飞书机器人（阻塞主线程）
        self.start_feishu_bot()
    
    def stop(self):
        """停止所有服务，释放资源"""
        logger.info("正在停止所有服务...")
        self.running = False
        
        # 停止定时任务
        if self.scheduler:
            self.scheduler.stop()
        
        # 停止摘要服务
        if self.summary_service:
            self.summary_service.stop()
        
        # 停止向量化服务
        if self.embedding_service:
            self.embedding_service.stop()
        
        logger.info("所有服务已停止")
    
    def run_crawler(self):
        """手动运行爬虫一次"""
        logger.info("运行爬虫...")
        try:
            from services.scheduler_service import get_scheduler_service
            scheduler = get_scheduler_service()
            scheduler.run_crawler_task()
            logger.info("爬虫运行完成")
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
    
    def run_summary(self):
        """手动运行摘要生成一次"""
        logger.info("运行摘要生成...")
        try:
            from services.summary_service import SummaryService
            service = SummaryService()
            count = service.run_once()
            logger.info(f"摘要生成完成，处理了 {count} 条新闻")
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
    
    def run_embedding(self):
        """手动运行向量化一次"""
        logger.info("运行向量化...")
        try:
            from services.embedding_service import EmbeddingService
            service = EmbeddingService()
            count = service.run_once()
            logger.info(f"向量化完成，处理了 {count} 条新闻")
        except Exception as e:
            logger.error(f"向量化失败: {e}")


def main():
    """
    主函数
    
    解析命令行参数并执行相应命令
    """
    parser = argparse.ArgumentParser(
        description='智能新闻助手 - 统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py start      # 启动完整服务
  python main.py crawler    # 手动运行爬虫
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'crawler', 'summary', 'embedding', 'test'],
        help='要执行的命令'
    )
    
    args = parser.parse_args()
    
    assistant = NewsAssistant()
    
    if args.command == 'start':
        # 启动所有服务（飞书机器人 + 定时任务）
        assistant.start_all()
    
    elif args.command == 'crawler':
        # 手动运行爬虫
        assistant.run_crawler()
    
    elif args.command == 'summary':
        # 手动运行摘要生成
        assistant.run_summary()
    
    elif args.command == 'embedding':
        # 手动运行向量化
        assistant.run_embedding()
    
    elif args.command == 'test':
        # 运行健康检查
        logger.info("运行服务健康检查...")
        from config import settings
        settings.print_config()
        logger.info("健康检查完成")


if __name__ == "__main__":
    main()
