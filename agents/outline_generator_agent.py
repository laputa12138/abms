import logging
import json
from typing import List, Dict, Optional, Union

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

class OutlineGeneratorAgentError(Exception):
    """Custom exception for OutlineGeneratorAgent errors."""
    pass

class OutlineGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating a report outline based on a given topic
    and associated keywords.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一个报告大纲撰写助手。请根据以下主题和关键词，生成一份详细的中文报告大纲。
大纲应包含主要章节和子章节（如果适用）。请确保大纲结构清晰、逻辑连贯，并覆盖主题的核心方面。

主题：
{topic_cn} (英文参考: {topic_en})

关键词：
中文: {keywords_cn}
英文: {keywords_en}

请以Markdown列表格式返回大纲。例如：
- 章节一：介绍
  - 1.1 背景
  - 1.2 研究意义
- 章节二：主要发现
  - 2.1 发现点A
  - 2.2 发现点B
- 章节三：结论

输出的大纲内容：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        """
        Initializes the OutlineGeneratorAgent.

        Args:
            llm_service (LLMService): An instance of the LLMService.
            prompt_template (Optional[str]): A custom prompt template for the LLM.
                                             If None, DEFAULT_PROMPT_TEMPLATE is used.
        """
        super().__init__(agent_name="OutlineGeneratorAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise OutlineGeneratorAgentError("LLMService is required for OutlineGeneratorAgent.")

    def run(self, analyzed_topic: Dict[str, any]) -> str:
        """
        Generates a report outline using the LLM.

        Args:
            analyzed_topic (Dict[str, any]): A dictionary from TopicAnalyzerAgent, containing:
                - 'generalized_topic_cn': Generalized topic in Chinese.
                - 'generalized_topic_en': Generalized topic in English.
                - 'keywords_cn': A list of Chinese keywords.
                - 'keywords_en': A list of English keywords.

        Returns:
            str: The generated report outline in Markdown format.

        Raises:
            OutlineGeneratorAgentError: If the LLM call fails or input is invalid.
        """
        self._log_input(analyzed_topic=analyzed_topic)

        required_keys = ["generalized_topic_cn", "generalized_topic_en", "keywords_cn", "keywords_en"]
        if not all(key in analyzed_topic for key in required_keys):
            msg = f"Invalid input: analyzed_topic dictionary missing one or more required keys: {required_keys}"
            logger.error(msg)
            raise OutlineGeneratorAgentError(msg)

        topic_cn = analyzed_topic["generalized_topic_cn"]
        topic_en = analyzed_topic["generalized_topic_en"]
        # Ensure keywords are strings for the prompt
        keywords_cn_str = ", ".join(analyzed_topic["keywords_cn"]) if isinstance(analyzed_topic["keywords_cn"], list) else str(analyzed_topic["keywords_cn"])
        keywords_en_str = ", ".join(analyzed_topic["keywords_en"]) if isinstance(analyzed_topic["keywords_en"], list) else str(analyzed_topic["keywords_en"])

        prompt = self.prompt_template.format(
            topic_cn=topic_cn,
            topic_en=topic_en,
            keywords_cn=keywords_cn_str,
            keywords_en=keywords_en_str
        )

        try:
            logger.info(f"Sending request to LLM for outline generation. Topic: '{topic_cn}'")
            # System prompt is more about the role, main instructions are in the user prompt.
            outline_markdown = self.llm_service.chat(query=prompt, system_prompt="你是一个专业的报告大纲规划师。")

            logger.debug(f"Raw LLM response for outline generation: {outline_markdown}")

            if not outline_markdown or not outline_markdown.strip():
                logger.warning("LLM returned an empty or whitespace-only outline.")
                # Return a default basic outline or raise error, depending on desired behavior
                # For now, let's raise an error if it's completely empty.
                raise OutlineGeneratorAgentError("LLM returned an empty outline.")

            self._log_output(outline_markdown)
            return outline_markdown.strip()

        except LLMServiceError as e:
            logger.error(f"LLM service error during outline generation: {e}")
            raise OutlineGeneratorAgentError(f"LLM service failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in OutlineGeneratorAgent: {e}")
            raise OutlineGeneratorAgentError(f"Unexpected error in outline generation: {e}")

if __name__ == '__main__':
    print("OutlineGeneratorAgent Example (requires running Xinference for LLMService)")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            logger.info(f"MockLLMService received query (first 100 chars): {query[:100]}")
            if "ABMS系统" in query:
                return """
- 章节一：ABMS系统概述
  - 1.1 ABMS定义与目标
  - 1.2 发展背景与重要性
- 章节二：ABMS核心技术与架构
  - 2.1 数据链与网络技术
  - 2.2 人工智能与机器学习应用
  - 2.3 传感器与数据融合
  - 2.4 JADC2框架下的角色
- 章节三：ABMS面临的挑战与未来展望
  - 3.1 技术挑战
  - 3.2 安全挑战
  - 3.3 未来发展方向
- 章节四：结论
                """
            elif "气候变化" in query:
                return """
- 1. 引言
  - 1.1 气候变化的定义与科学共识
  - 1.2 研究背景与意义
- 2. 气候变化的主要表现
  - 2.1 全球气温上升
  - 2.2 极端天气事件频发
  - 2.3 海平面上升
  - 2.4 生物多样性减少
- 3. 气候变化对自然生态系统的影响
  - 3.1 对水资源的影响
  - 3.2 对农业和粮食安全的影响
  - 3.3 对海洋生态的影响
- 4. 气候变化对人类社会经济的影响
  - 4.1 对人类健康的影响
  - 4.2 对经济发展的影响
  - 4.3 对社会公平与可持续发展的影响
- 5. 应对气候变化的策略与行动
  - 5.1 国际合作与政策框架
  - 5.2 减排措施
  - 5.3 适应措施
- 6. 结论与展望
                """
            return "- 默认章节：请检查输入"


    try:
        # llm_service_instance = LLMService()
        print("Using MockLLMService for OutlineGeneratorAgent example.")
        llm_service_instance = MockLLMService()

        outline_agent = OutlineGeneratorAgent(llm_service=llm_service_instance)

        # Test case 1 (simulating output from TopicAnalyzerAgent)
        analyzed_topic1 = {
            "generalized_topic_cn": "先进战斗管理系统（ABMS）",
            "generalized_topic_en": "Advanced Battle Management System (ABMS)",
            "keywords_cn": ["ABMS", "先进战斗管理", "美军", "JADC2", "多域作战", "指挥控制"],
            "keywords_en": ["ABMS", "Advanced Battle Management", "US Military", "JADC2", "Multi-Domain Operations", "Command and Control"]
        }
        print(f"\nGenerating outline for topic: '{analyzed_topic1['generalized_topic_cn']}'")
        outline1 = outline_agent.run(analyzed_topic=analyzed_topic1)
        print("Generated Outline 1:")
        print(outline1)

        # Test case 2
        analyzed_topic2 = {
            "generalized_topic_cn": "气候变化及其全球影响",
            "generalized_topic_en": "Climate Change and its Global Impacts",
            "keywords_cn": ["气候变化", "全球变暖", "极端天气", "环境影响", "碳排放"],
            "keywords_en": ["Climate Change", "Global Warming", "Extreme Weather", "Environmental Impact", "Carbon Emissions"]
        }
        print(f"\nGenerating outline for topic: '{analyzed_topic2['generalized_topic_cn']}'")
        outline2 = outline_agent.run(analyzed_topic=analyzed_topic2)
        print("Generated Outline 2:")
        print(outline2)

    except (LLMServiceError, OutlineGeneratorAgentError) as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nOutlineGeneratorAgent example finished.")
