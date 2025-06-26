import logging
from typing import List, Dict, Optional, Union

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

class ChapterWriterAgentError(Exception):
    """Custom exception for ChapterWriterAgent errors."""
    pass

class ChapterWriterAgent(BaseAgent):
    """
    Agent responsible for writing a chapter of a report based on a given
    chapter title and relevant retrieved content.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位专业的报告撰写员。请根据以下报告章节标题和相关的参考资料，撰写详细、流畅、专业、连贯的中文章节内容。

章节标题：
{chapter_title}

参考资料（请基于这些资料进行撰写，不要杜撰）：
{retrieved_content_formatted}

请确保内容与章节标题紧密相关，并充分、合理地利用提供的参考资料。避免直接复制粘贴参考资料，而是要理解、整合信息，并用自己的话语有条理地表达出来。
输出的章节内容应具有良好的可读性和专业性。

撰写的章节内容（纯文本，不需要包含章节标题本身）：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        """
        Initializes the ChapterWriterAgent.

        Args:
            llm_service (LLMService): An instance of the LLMService.
            prompt_template (Optional[str]): A custom prompt template for the LLM.
                                             If None, DEFAULT_PROMPT_TEMPLATE is used.
        """
        super().__init__(agent_name="ChapterWriterAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise ChapterWriterAgentError("LLMService is required for ChapterWriterAgent.")

    def _format_retrieved_content(self, retrieved_content: List[Dict[str, any]]) -> str:
        """
        Formats the list of retrieved content dictionaries into a string for the prompt.
        Each piece of content is numbered.
        """
        if not retrieved_content:
            return "无参考资料提供。"

        formatted_str = ""
        for i, item in enumerate(retrieved_content):
            doc_text = item.get('document', '无效的参考资料片段')
            score = item.get('score', 'N/A') # Could be distance or relevance
            source = item.get('source', 'unknown') # vector_search or reranker
            formatted_str += f"参考资料 {i+1} (来源: {source}, 得分/距离: {score:.4f}):\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        return formatted_str.strip()

    def run(self, chapter_title: str, retrieved_content: List[Dict[str, any]]) -> str:
        """
        Writes a chapter using the LLM based on the title and retrieved content.

        Args:
            chapter_title (str): The title of the chapter to be written.
            retrieved_content (List[Dict[str, any]]): A list of dictionaries,
                where each dictionary contains 'document' (str) and 'score' (float),
                representing relevant content snippets.

        Returns:
            str: The written chapter content (text).

        Raises:
            ChapterWriterAgentError: If the LLM call fails or input is invalid.
        """
        self._log_input(chapter_title=chapter_title, retrieved_content_count=len(retrieved_content))

        if not chapter_title:
            msg = "Chapter title cannot be empty."
            logger.error(msg)
            raise ChapterWriterAgentError(msg)

        formatted_content_str = self._format_retrieved_content(retrieved_content)

        prompt = self.prompt_template.format(
            chapter_title=chapter_title,
            retrieved_content_formatted=formatted_content_str
        )

        try:
            logger.info(f"Sending request to LLM for chapter writing. Chapter: '{chapter_title}'")
            # System prompt guides the LLM's persona.
            chapter_text = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一位精通特定领域知识的专业中文报告撰写者，擅长整合信息并清晰表达。"
            )

            logger.debug(f"Raw LLM response for chapter writing (first 200 chars): {chapter_text[:200]}")

            if not chapter_text or not chapter_text.strip():
                logger.warning(f"LLM returned empty content for chapter: '{chapter_title}'.")
                # Depending on requirements, could return a placeholder or raise error.
                # For now, we'll return empty string, pipeline can decide how to handle.
                return ""

            self._log_output(chapter_text)
            return chapter_text.strip()

        except LLMServiceError as e:
            logger.error(f"LLM service error during chapter writing for '{chapter_title}': {e}")
            raise ChapterWriterAgentError(f"LLM service failed for chapter '{chapter_title}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in ChapterWriterAgent for '{chapter_title}': {e}")
            raise ChapterWriterAgentError(f"Unexpected error writing chapter '{chapter_title}': {e}")

if __name__ == '__main__':
    print("ChapterWriterAgent Example (requires running Xinference for LLMService)")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            logger.info(f"MockLLMService received query (first 150 chars): {query[:150]}")
            if "ABMS系统概述" in query:
                return """
根据提供的资料，ABMS（先进战斗管理系统）是美军为了应对未来复杂战场环境而提出的一项关键发展项目。
其核心目标在于构建一个能够连接所有传感器、平台和作战人员的统一网络，实现信息的快速共享和协同决策，从而在多域作战中取得优势。
资料1指出ABMS的设计理念是开放架构和敏捷开发，这与传统武器系统的封闭模式形成对比。
资料2则强调了JADC2（联合全域指挥控制）是ABMS实现其目标的重要框架，ABMS可以被视为JADC2概念下的具体技术和系统实现之一。
总的来说，ABMS旨在通过技术创新，提升美军的战场感知、指挥控制和作战效能。
                """
            elif "气候变化的主要表现" in query:
                return """
气候变化已成为全球共同关注的焦点议题，其表现形式多样且影响深远。
根据参考资料1，全球平均气温的持续上升是气候变化最显著的特征之一，这导致了冰川融化和海平面上升。
同时，如资料2所述，极端天气事件，例如热浪、干旱、洪水和强风暴的频率和强度也在增加，对全球各地的生态系统和人类社会造成了严重威胁。
此外，资料3提到生物多样性的减少也是气候变化的一个重要表现，许多物种因无法适应快速变化的环境而面临生存危机。
这些现象共同构成了当前气候变化的主要表现，警示我们需要采取紧急行动。
                """
            return "根据提供的内容，本章节主要讨论了[自动生成内容]..."

    try:
        # llm_service_instance = LLMService()
        print("Using MockLLMService for ChapterWriterAgent example.")
        llm_service_instance = MockLLMService()

        writer_agent = ChapterWriterAgent(llm_service=llm_service_instance)

        # Test case 1
        title1 = "ABMS系统概述"
        content1 = [
            {"document": "ABMS (Advanced Battle Management System) is designed with an open architecture approach.", "score": 0.95, "source": "reranker"},
            {"document": "JADC2 is the overarching framework for connecting sensors and shooters across all domains. ABMS is a key component of JADC2.", "score": 0.92, "source": "reranker"},
            {"document": "The primary goal of ABMS is to enable rapid decision-making in multi-domain operations.", "score": 0.88, "source": "reranker"}
        ]
        print(f"\nWriting chapter: '{title1}'")
        chapter_text1 = writer_agent.run(chapter_title=title1, retrieved_content=content1)
        print(f"Generated Chapter Text for '{title1}':\n{chapter_text1}")

        # Test case 2
        title2 = "气候变化的主要表现"
        content2 = [
            {"document": "Global warming leads to rising sea levels and melting glaciers.", "score": 0.98, "source": "reranker"},
            {"document": "Extreme weather events like heatwaves and floods are becoming more frequent and intense due to climate change.", "score": 0.95, "source": "reranker"},
            {"document": "Climate change is a major driver of biodiversity loss.", "score": 0.90, "source": "reranker"}
        ]
        print(f"\nWriting chapter: '{title2}'")
        chapter_text2 = writer_agent.run(chapter_title=title2, retrieved_content=content2)
        print(f"Generated Chapter Text for '{title2}':\n{chapter_text2}")

        # Test case 3: No content
        title3 = "一个没有内容的章节"
        content3 = []
        print(f"\nWriting chapter: '{title3}' (with no reference content)")
        chapter_text3 = writer_agent.run(chapter_title=title3, retrieved_content=content3)
        print(f"Generated Chapter Text for '{title3}':\n'{chapter_text3}'")


    except (LLMServiceError, ChapterWriterAgentError) as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nChapterWriterAgent example finished.")
