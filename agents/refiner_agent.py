import logging
import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

class RefinerAgentError(Exception):
    """Custom exception for RefinerAgent errors."""
    pass

class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining (improving) content based on
    evaluation feedback.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位报告修改专家。请根据以下原始内容和评审反馈，对内容进行修改和完善，输出修改后的中文版本。
你的目标是解决反馈中指出的问题，并提升内容的整体质量，包括相关性、流畅性、完整性和准确性。

原始内容：
---
{original_content}
---

评审反馈：
---
{evaluation_feedback}
---

请仔细阅读评审反馈，理解需要改进的关键点。
在修改时，请尽量保留原始内容的合理部分，重点针对反馈中提出的不足之处进行优化。
如果反馈中包含具体的修改建议，请优先考虑采纳。

修改后的内容（纯文本）：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        """
        Initializes the RefinerAgent.

        Args:
            llm_service (LLMService): An instance of the LLMService.
            prompt_template (Optional[str]): A custom prompt template for the LLM.
                                             If None, DEFAULT_PROMPT_TEMPLATE is used.
        """
        super().__init__(agent_name="RefinerAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise RefinerAgentError("LLMService is required for RefinerAgent.")

    def _format_feedback(self, feedback_data: Dict[str, any]) -> str:
        """Formats the structured feedback into a string for the prompt."""
        score = feedback_data.get('score', 'N/A')
        feedback_text = feedback_data.get('feedback_cn', '无具体反馈文本。')
        criteria = feedback_data.get('evaluation_criteria_met', {})

        criteria_str = "\n具体评估标准反馈：\n"
        for k, v in criteria.items():
            criteria_str += f"- {k}: {v}\n"

        return f"总体评分: {score}\n\n评审意见:\n{feedback_text}\n{criteria_str if criteria else ''}".strip()


    def run(self, original_content: str, evaluation_feedback: Dict[str, any]) -> str:
        """
        Refines the content based on evaluation feedback using the LLM.

        Args:
            original_content (str): The original content to be refined.
            evaluation_feedback (Dict[str, any]): A dictionary containing evaluation
                                                  results (score, feedback_cn, etc.)
                                                  from the EvaluatorAgent.

        Returns:
            str: The refined content.

        Raises:
            RefinerAgentError: If the LLM call fails or input is invalid.
        """
        self._log_input(original_content_length=len(original_content), evaluation_feedback=evaluation_feedback)

        if not original_content: # Allow empty original content if feedback suggests writing from scratch
            logger.warning("RefinerAgent received empty original_content. Refinement will be based solely on feedback if it implies generation.")

        if not evaluation_feedback or not evaluation_feedback.get("feedback_cn"):
            logger.warning("RefinerAgent received insufficient evaluation feedback. Returning original content.")
            return original_content # Or raise error if feedback is strictly required

        formatted_feedback_str = self._format_feedback(evaluation_feedback)

        prompt = self.prompt_template.format(
            original_content=original_content if original_content else "无原始内容提供，请根据反馈生成。",
            evaluation_feedback=formatted_feedback_str
        )

        try:
            logger.info(f"Sending request to LLM for content refinement. Original content length: {len(original_content)}")
            refined_text = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一位经验丰富的编辑和内容优化师，擅长根据反馈精确改进文稿。"
            )
            logger.debug(f"Raw LLM response for refinement (first 200 chars): {refined_text[:200]}")

            if not refined_text and original_content: # If LLM returns empty but there was original content
                logger.warning("LLM returned empty refined content. This might be an error or intentional. Returning original content as fallback.")
                # Fallback to original content if LLM fails to produce anything meaningful
                # This behavior might need adjustment based on how critical refinement is.
                # return original_content
                # For now, let's assume empty response means "no changes needed" if original was good, or "cannot refine"
                # A more robust system might check the score from feedback. If score was high, no change is ok.
                # If score was low and LLM gives empty, that's an issue.
                # Given the prompt, LLM should always return *something*.
                # If it's empty, it's more likely an LLM error or a very poor attempt.
                # Let's assume for now that an empty response means the LLM failed to refine meaningfully.
                raise RefinerAgentError("LLM returned empty string during refinement process.")


            self._log_output(refined_text)
            return refined_text.strip() if refined_text else original_content # Prefer refined, fallback to original if refined is empty string

        except LLMServiceError as e:
            logger.error(f"LLM service error during content refinement: {e}")
            raise RefinerAgentError(f"LLM service failed during refinement: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in RefinerAgent: {e}")
            raise RefinerAgentError(f"Unexpected error in content refinement: {e}")

if __name__ == '__main__':
    print("RefinerAgent Example (requires running Xinference for LLMService)")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            logger.info(f"MockLLMService received query for refinement (first 150 chars): {query[:150]}")
            # Simulate refinement based on feedback
            if "补充ABMS在实际演习中的应用案例" in query: # Check for specific feedback
                return query.split("原始内容：\n---\n")[1].split("\n---")[0] + \
                       "\n\n经过修订，补充内容如下：例如，在“勇敢之盾”等军事演习中，ABMS的部分技术得到了验证和应用，展示了其在连接不同军种资产、加速数据共享方面的潜力。这些演习也暴露了一些需要进一步完善的问题，如网络弹性和互操作性等。"
            elif "可以更深入地探讨各项表现之间的关联性" in query:
                 original_content_part = query.split("原始内容：\n---\n")[1].split("\n---")[0]
                 return original_content_part + \
                        "\n\n（修订后）这些气候变化的主要表现并非孤立存在，而是相互关联、相互加剧的。例如，全球气温上升直接导致冰川融化和海平面上升，进而影响沿海生态系统和人类居住区。同时，气温升高也会改变大气环流模式，增加极端天气事件发生的概率和强度，而这些极端天气又可能进一步破坏生态系统，影响生物多样性。"
            return "（模拟的修订内容）基于反馈，对原文进行了调整和优化。"

    try:
        # llm_service_instance = LLMService()
        print("Using MockLLMService for RefinerAgent example.")
        llm_service_instance = MockLLMService()

        refiner_agent = RefinerAgent(llm_service=llm_service_instance)

        # Test case 1
        original_text1 = """根据提供的资料，ABMS（先进战斗管理系统）是美军为了应对未来复杂战场环境而提出的一项关键发展项目。其核心目标在于构建一个能够连接所有传感器、平台和作战人员的统一网络，实现信息的快速共享和协同决策，从而在多域作战中取得优势。ABMS旨在通过技术创新，提升美军的战场感知、指挥控制和作战效能。"""
        feedback1 = {
            "score": 85,
            "feedback_cn": "内容与主题高度相关，对ABMS的定义和目标阐述清晰。语言流畅，逻辑性强。可以进一步补充ABMS在实际演习中的应用案例以增强完整性。准确性方面，基于常识判断，核心概念描述正确。",
            "evaluation_criteria_met": {
                "relevance": "高", "fluency": "优秀",
                "completeness": "良好，但可补充实例。", "accuracy": "高"
            }
        }
        print(f"\nRefining content 1 (length: {len(original_text1)}):")
        refined_text1 = refiner_agent.run(original_content=original_text1, evaluation_feedback=feedback1)
        print("Refined Text 1:")
        print(refined_text1)

        # Test case 2
        original_text2 = """气候变化已成为全球共同关注的焦点议题，其表现形式多样且影响深远。全球平均气温的持续上升是气候变化最显著的特征之一。极端天气事件的频率和强度也在增加。生物多样性的减少也是气候变化的一个重要表现。"""
        feedback2 = {
            "score": 78,
            "feedback_cn": "报告章节对气候变化的主要表现有较好的概述。但在完整性方面，可以更深入地探讨各项表现之间的关联性。",
            "evaluation_criteria_met": {
                "relevance": "高", "fluency": "良好",
                "completeness": "一般，可以更深入分析各项表现的内在联系。", "accuracy": "较高"
            }
        }
        print(f"\nRefining content 2 (length: {len(original_text2)}):")
        refined_text2 = refiner_agent.run(original_content=original_text2, evaluation_feedback=feedback2)
        print("Refined Text 2:")
        print(refined_text2)

        # Test case 3: Empty original content, feedback suggests generation
        original_text3 = ""
        feedback3 = {
            "score": 20,
            "feedback_cn": "原始内容缺失。请根据主题“AI伦理的重要性”撰写一段引言。",
            "evaluation_criteria_met": {"completeness": "严重缺失"}
        }
        # For this mock, the LLM won't actually generate from scratch based on this feedback,
        # but in a real scenario, the prompt is designed to handle this.
        # The current mock will return the "（模拟的修订内容）..." string.
        print(f"\nRefining content 3 (empty original, feedback implies generation):")
        refined_text3 = refiner_agent.run(original_content=original_text3, evaluation_feedback=feedback3)
        print("Refined Text 3 (Mocked - would be generated from scratch by real LLM):")
        print(refined_text3)


    except (LLMServiceError, RefinerAgentError) as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nRefinerAgent example finished.")
