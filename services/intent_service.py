"""
意图识别服务模块 (Intent Recognition Service Module)

本模块负责识别和理解用户的自然语言输入，将其归类为特定的意图类型。
是整个对话系统的入口，决定了后续如何处理用户的请求。

核心功能：
1. 意图分类：将用户输入分类为搜索、设置兴趣、帮助、问候等意图
2. 关键词提取：从用户输入中提取关键信息用于搜索
3. 实体识别：识别特定类型的实体（如查询词、兴趣领域等）
4. 置信度计算：评估识别结果的可靠程度

设计思路：
- 使用基于规则的正则表达式匹配，简单高效
- 支持多模式匹配，提高识别准确率
- 模块化设计，便于后续扩展为机器学习模型
"""

import re
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """
    意图类型枚举
    
    定义系统支持的所有用户意图类型。
    每种意图对应不同的处理逻辑：
    - SEARCH: 触发新闻搜索流程，调用向量检索和LLM生成回答
    - SET_INTEREST: 触发用户兴趣管理，添加或删除关注的领域
    - HELP: 返回系统的使用帮助信息
    - GREETING: 返回友好的问候语
    - UNKNOWN: 无法识别的意图，返回默认提示
    """
    SEARCH = "search"           # 搜索新闻 - 用户想查找特定主题的新闻
    SET_INTEREST = "interest"   # 设置兴趣 - 用户想管理个人兴趣偏好
    HELP = "help"               # 帮助 - 用户需要了解如何使用系统
    GREETING = "greeting"       # 问候 - 用户打招呼
    UNKNOWN = "unknown"         # 未知 - 无法识别的意图


@dataclass
class IntentResult:
    """
    意图识别结果数据类
    
    封装意图识别的完整结果，包含识别的意图类型、置信度、
    提取的关键词和实体信息。
    
    Attributes:
        intent: 识别出的意图类型（IntentType枚举值）
        confidence: 置信度分数（0-1之间），表示识别的可靠程度
        keywords: 从用户输入中提取的关键词列表，用于搜索
        raw_text: 用户的原始输入文本，用于日志记录和调试
        entities: 提取的实体字典，包含意图特定的结构化数据
    """
    intent: IntentType         # 识别出的意图类型
    confidence: float          # 置信度 0-1，越高表示越确定
    keywords: List[str]        # 提取的关键词，用于新闻检索
    raw_text: str              # 原始文本，保留原始输入
    entities: Dict[str, Any]   # 提取的实体，如查询词、兴趣领域等


class IntentService:
    """
    意图识别服务类
    
    核心服务类，负责分析用户输入并识别其意图。
    
    实现原理：
    1. 基于正则表达式的模式匹配
    2. 每个意图类型配置多个正则模式，覆盖不同的表达方式
    3. 按优先级顺序匹配，第一个匹配到的意图即为识别结果
    4. 无法匹配时默认尝试作为搜索意图处理
    
    使用示例：
        service = IntentService()
        result = service.recognize("搜索人工智能新闻")
        print(result.intent)  # IntentType.SEARCH
        print(result.keywords)  # ['人工智能']
    """
    
    # 意图关键词模式 - 定义每个意图类型对应的正则表达式列表
    # 每个意图可以有多个模式，覆盖不同的表达方式
    INTENT_PATTERNS = {
        # 搜索意图模式 - 匹配用户想要搜索新闻的各种表达方式
        IntentType.SEARCH: [
            r"搜索(.+?)(新闻|资讯|信息)?",           # "搜索人工智能新闻"
            r"查[找询]?(.+?)(新闻|资讯|信息)?",     # "查找/查询/查科技资讯"
            r"(?:有没有|有啥)(.+?)(新闻|资讯|信息)?", # "有没有科技新闻"
            r"(.+?)(?:的)?(?:最新)?(?:新闻|资讯|信息)", # "人工智能的新闻"
            r"[告诉给]我(.+?)(?:的)?(?:新闻|资讯|信息)?", # "告诉我科技新闻"
            r"想了解(.+?)",                          # "想了解量子计算"
            r"(.+?)最近怎么样",                      # "比特币最近怎么样"
            r"最近(.+?)有什么新闻",                  # "最近科技有什么新闻"
        ],
        # 兴趣设置意图模式 - 匹配用户管理兴趣偏好的表达
        IntentType.SET_INTEREST: [
            r"(?:关注|订阅)(.+?)(?:新闻|资讯|信息)+", # "关注科技新闻"（+ 要求至少匹配一个后缀）
            r"(?:关注|订阅)(.+)",                      # "关注科技"（无后缀，贪婪匹配剩余内容）
            r"对(.+?)感兴趣",                        # "对财经感兴趣"
            r"(?:喜欢|想[要看])(.+?)(?:的)?(?:新闻|资讯|信息)+", # "喜欢看体育新闻"
            r"[添加]?(?:兴趣|爱好)(.+?)",            # "添加兴趣科技"
            r"取消关注(.+?)",                        # "取消关注娱乐"
            r"不再关注(.+?)",                        # "不再关注八卦"
        ],
        # 帮助意图模式 - 匹配用户需要帮助的场景
        IntentType.HELP: [
            r"帮助|help|怎么用|使用说明|指令",       # "帮助" "help"
            r"[你能]做什么",                         # "你能做什么"
            r"有什么功能",                          # "有什么功能"
            r"[如怎]么用",                          # "怎么用"
            r"[请指]教",                            # "请教"
        ],
        # 问候意图模式 - 匹配用户打招呼
        IntentType.GREETING: [
            r"你好|您好|哈喽|嗨|hi|hello",           # 各种问候语
            r"早上好|下午好|晚上好",                 # 时间段问候
            r"[在吗？]?",                            # "在吗"
        ],
    }
    
    # 停用词集合 - 在关键词提取时需要过滤的常见词
    # 这些词出现频率高但信息含量低，不应作为搜索关键词
    STOP_WORDS = {
        # 常见虚词
        "的", "了", "是", "在", "和", "与", "或", "有", "个",
        # 人称代词
        "我", "你", "他", "她", "它", "们",
        # 指示代词
        "这", "那", "这些", "那些",
        # 常见功能词（在新闻搜索语境下属于停用词）
        "新闻", "资讯", "信息", "一下", "最近", "最新", "相关"
    }
    
    def __init__(self):
        """
        初始化意图识别服务
        
        完成以下初始化工作：
        1. 编译所有正则表达式模式，提高匹配效率
        2. 初始化日志记录
        """
        self._compile_patterns()
        logger.info("意图识别服务初始化完成")
    
    def _compile_patterns(self):
        """
        编译正则表达式模式
        
        将所有字符串形式的正则表达式预编译为Pattern对象，
        这样在匹配时可以大幅提高性能，避免每次重新编译。
        
        编译时使用re.IGNORECASE标志，实现大小写不敏感匹配。
        """
        self.compiled_patterns = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            # 对每个意图类型的所有模式进行编译
            self.compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def recognize(self, text: str) -> IntentResult:
        """
        识别用户意图
        
        核心方法，分析用户输入文本并返回识别结果。
        按照配置的顺序依次尝试匹配各个意图类型的正则模式，
        第一个匹配成功的意图即为识别结果。
        
        匹配优先级顺序（由高到低）：
        1. SEARCH（搜索意图）
        2. SET_INTEREST（兴趣设置意图）
        3. HELP（帮助意图）
        4. GREETING（问候意图）
        
        如果没有任何模式匹配成功，会尝试将输入作为通用搜索处理，
        只有当完全无法提取关键词时才返回UNKNOWN。
        
        Args:
            text: 用户输入文本，可以是任意自然语言
            
        Returns:
            IntentResult: 包含意图类型、置信度、关键词和实体的完整结果
        """
        # 去除首尾空白字符
        text = text.strip()
        
        # 处理空输入情况
        if not text:
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                keywords=[],
                raw_text=text,
                entities={}
            )
        
        # 逐个意图类型进行匹配
        # 按照INTENT_PATTERNS字典中的顺序依次尝试
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                # 使用search()而非match()，允许模式在文本任意位置匹配
                match = pattern.search(text)
                if match:
                    # 匹配成功，提取相关信息
                    keywords = self._extract_keywords(match, text)  # 提取关键词
                    confidence = self._calculate_confidence(match, text, intent)  # 计算置信度
                    entities = self._extract_entities(match, intent)  # 提取实体
                    
                    logger.info(f"识别意图: {intent.value}, 置信度: {confidence:.2f}")
                    
                    # 返回识别结果
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        keywords=keywords,
                        raw_text=text,
                        entities=entities
                    )
        
        # 没有匹配到任何预定义意图，尝试作为通用搜索处理
        # 这是为了处理用户直接输入关键词的情况，如"人工智能"
        keywords = self._extract_general_keywords(text)
        if keywords:
            return IntentResult(
                intent=IntentType.SEARCH,
                confidence=0.5,  # 通用搜索的置信度较低
                keywords=keywords,
                raw_text=text,
                entities={"query": text}  # 将整个文本作为查询词
            )
        
        # 完全无法识别，返回UNKNOWN
        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            keywords=[],
            raw_text=text,
            entities={}
        )
    
    def _extract_keywords(self, match: re.Match, text: str) -> List[str]:
        """
        从匹配中提取关键词
        
        从正则匹配结果的分组中提取有意义的关键词。
        分组是正则表达式中用括号()捕获的内容，
        通常包含用户关心的核心信息，如搜索主题、兴趣领域等。
        
        提取规则：
        1. 遍历所有捕获分组
        2. 过滤掉空值和仅包含空白字符的分组
        3. 过滤停用词（如"的"、"新闻"等）
        4. 过滤长度小于2的词（避免单字词）
        
        Args:
            match: 正则匹配结果对象，包含所有捕获的分组
            text: 原始文本（备用，当前未使用）
            
        Returns:
            List[str]: 经过过滤的关键词列表，可用于新闻检索
        """
        keywords = []
        
        # 遍历正则表达式中定义的所有捕获分组
        # groups()返回所有分组匹配的字符串元组
        for group in match.groups():
            if group and len(group.strip()) > 0:
                keyword = group.strip()
                # 应用过滤规则
                if keyword not in self.STOP_WORDS and len(keyword) >= 2:
                    keywords.append(keyword)
        
        return keywords
    
    def _extract_general_keywords(self, text: str) -> List[str]:
        """
        通用关键词提取（用于默认搜索意图）
        
        当用户的输入没有匹配任何预定义意图模式时使用。
        采用简单的规则进行分词：按标点符号和空白字符切分。
        
        适用场景：
        - 用户直接输入关键词，如"人工智能"
        - 用户输入简短的查询词，如"科技新闻"
        - 用户输入无法被正则模式匹配的自然语言
        
        Args:
            text: 用户输入文本
            
        Returns:
            List[str]: 提取的关键词列表，过滤后返回
        """
        # 使用正则表达式按常见标点符号切分文本
        # 包括：中文标点（，。？！）和英文标点（,.?!）以及空白字符
        words = re.split(r'[，。？！,.?!\s]+', text)
        
        # 对每个词进行清洗和过滤
        keywords = [
            w.strip() for w in words 
            if w.strip()  # 过滤空字符串
            and w.strip() not in self.STOP_WORDS  # 过滤停用词
            and len(w.strip()) >= 2  # 过滤单字词
        ]
        return keywords
    
    def _calculate_confidence(self, match: re.Match, text: str, intent: IntentType) -> float:
        """
        计算意图识别置信度
        
        根据匹配质量计算一个0-1之间的置信度分数，
        用于评估识别结果的可靠程度。
        
        计算因素：
        1. 基础置信度：根据意图类型设定不同的基础值
           - 问候意图(GREETING): 0.9（匹配通常很准确）
           - 帮助意图(HELP): 0.85
           - 其他意图: 0.7
        2. 覆盖率：匹配文本长度与原文本长度的比例
           匹配越多原文本内容，置信度越高
        
        公式：confidence = base * (0.5 + 0.5 * coverage)
        这样即使完全匹配(coverage=1)，置信度也只是base，不会过高
        
        Args:
            match: 正则匹配结果对象
            text: 原始用户输入文本
            intent: 识别出的意图类型
            
        Returns:
            float: 置信度分数（0-1之间）
        """
        # 设置基础置信度，根据意图类型有所不同
        base_confidence = 0.7
        
        # 问候和帮助意图的匹配通常更精确，给予更高基础置信度
        if intent == IntentType.GREETING:
            base_confidence = 0.9
        elif intent == IntentType.HELP:
            base_confidence = 0.85
        
        # 计算匹配覆盖率 = 匹配文本长度 / 原文本长度
        # 覆盖率越高，说明正则模式捕获的内容越多，置信度越高
        matched_text = match.group(0)
        coverage = len(matched_text) / len(text) if text else 0
        
        # 计算最终置信度
        # 0.5是基础权重，0.5*coverage是覆盖率权重
        confidence = base_confidence * (0.5 + 0.5 * coverage)
        
        # 确保置信度不超过1.0
        return min(confidence, 1.0)
    
    def _extract_entities(self, match: re.Match, intent: IntentType) -> Dict[str, Any]:
        """
        提取实体信息
        
        从正则匹配结果中提取意图相关的结构化数据。
        不同类型的意图会提取不同的实体字段。
        
        实体提取规则：
        - SEARCH: 提取查询词(query)
        - SET_INTEREST: 提取兴趣领域(interest)和操作类型(action: add/remove)
        - HELP/GREETING: 无需提取特定实体
        
        Args:
            match: 正则匹配结果对象
            intent: 识别出的意图类型
            
        Returns:
            Dict[str, Any]: 包含提取实体的字典，不同意图返回不同字段
        """
        entities = {}
        
        if intent == IntentType.SEARCH:
            # 提取搜索查询词
            # 如果有捕获分组，使用第一个分组作为查询词
            if match.groups() and match.group(1):
                entities["query"] = match.group(1).strip()
            else:
                # 如果没有分组，使用整个匹配文本作为查询词
                entities["query"] = match.group(0).strip()
                
        elif intent == IntentType.SET_INTEREST:
            # 提取兴趣关键词
            if match.groups() and match.group(1):
                entities["interest"] = match.group(1).strip()
            
            # 判断操作类型：添加关注还是取消关注
            # 通过检查关键词来判断用户意图
            raw_text = match.string
            if any(word in raw_text for word in ["取消", "不再", "删除", "移除"]):
                entities["action"] = "remove"  # 删除兴趣
            else:
                entities["action"] = "add"     # 添加兴趣
        
        return entities
    
    def is_search_intent(self, text: str) -> bool:
        """
        快速判断是否为搜索意图
        
        便捷方法，快速判断用户输入是否属于搜索类请求。
        适用于需要快速过滤的场景，如决定是否启用搜索流程。
        
        Args:
            text: 用户输入文本
            
        Returns:
            bool: True表示是搜索意图，False表示其他意图
        """
        result = self.recognize(text)
        return result.intent == IntentType.SEARCH
    
    def get_help_message(self) -> str:
        """
        获取帮助信息
        
        返回系统的使用说明，包含所有支持的功能和示例。
        当用户触发帮助意图时，将此文本作为回复发送给用户。
        
        Returns:
            str: 格式化的帮助文本，包含使用指南和示例
        """
        return """🤖 智能新闻助手使用指南

📰 **搜索新闻**
• "搜索人工智能新闻"
• "查一下最近的经济资讯"
• "告诉我科技领域有什么新闻"
• "量子计算最近怎么样"

⭐ **管理兴趣**
• "关注科技新闻" - 添加兴趣
• "对财经感兴趣" - 设置关注领域
• "取消关注体育" - 移除兴趣

❓ **获取帮助**
• 发送"帮助"查看本指南

💡 **小贴士**
• 可以直接发送关键词，我会自动搜索
• 支持多种自然语言表达方式
• 搜索结果会包含来源链接，可点击查看详情
"""

    def get_greeting_message(self) -> str:
        """
        获取问候回复
        
        随机选择一条友好的问候语返回给用户。
        当用户触发问候意图时使用，增加交互的亲切感。
        
        每条问候语都包含：
        - 友好的问候
        - 身份介绍（智能新闻助手）
        - 功能提示（搜索新闻、管理兴趣）
        - 行动号召（告诉用户下一步可以做什么）
        
        Returns:
            str: 随机选择的问候语文本
        """
        import random
        # 预定义的问候语列表，每次随机选择一条
        greetings = [
            "你好！我是智能新闻助手，可以帮你搜索新闻、管理兴趣偏好。发送'帮助'了解更多。",
            "您好！有什么新闻想了解的吗？我可以帮你搜索最新资讯。",
            "哈喽！我是你的新闻助手，告诉我你想了解什么话题吧。",
            "你好！想搜索什么新闻呢？直接告诉我关键词就可以。",
        ]
        return random.choice(greetings)


# =============================================================================
# 全局服务实例（单例模式）
# =============================================================================
# 使用单例模式确保整个应用中只有一个IntentService实例
# 这样可以避免重复编译正则表达式，提高性能
_intent_service: Optional[IntentService] = None


def get_intent_service() -> IntentService:
    """
    获取意图识别服务单例
    
    使用单例模式管理IntentService实例。
    首次调用时创建实例，后续调用返回已创建的实例。
    
    为什么使用单例：
    1. 性能优化：避免重复编译正则表达式模式
    2. 内存优化：避免创建多个相同的对象
    3. 一致性：确保整个应用使用相同的配置和状态
    
    Returns:
        IntentService: 全局唯一的意图识别服务实例
    """
    global _intent_service
    # 懒加载：首次调用时才创建实例
    if _intent_service is None:
        _intent_service = IntentService()
    return _intent_service


# =============================================================================
# 使用示例和测试代码
# =============================================================================
# 当直接运行此文件时执行的测试代码
# 用于验证意图识别服务的正确性
if __name__ == "__main__":
    # 配置日志级别为INFO，显示运行日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建服务实例
    service = IntentService()
    
    # 定义测试用例，覆盖各种意图类型
    test_cases = [
        "搜索人工智能新闻",          # 搜索意图
        "查一下最近的科技资讯",      # 搜索意图（变体表达）
        "关注财经新闻",              # 兴趣设置意图（添加）
        "对体育赛事感兴趣",          # 兴趣设置意图（添加）
        "帮助",                     # 帮助意图
        "怎么用",                   # 帮助意图（变体表达）
        "你好",                     # 问候意图
        "量子计算最新进展",          # 搜索意图（无显式关键词）
        "告诉我关于新能源的消息",    # 搜索意图
    ]
    
    # 打印测试标题
    print("=" * 60)
    print("意图识别测试")
    print("=" * 60)
    
    # 逐个测试并打印结果
    for text in test_cases:
        result = service.recognize(text)
        print(f"\n输入: {text}")
        print(f"  意图: {result.intent.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  关键词: {result.keywords}")
        print(f"  实体: {result.entities}")
