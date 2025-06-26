import logging
import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

class EvaluatorAgentError(Exception):
    """Custom exception for EvaluatorAgent errors."""
    pass

class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating the generated content (e.g., a chapter or
    the full report) based on predefined criteria and providing a score and feedback.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位资深的报告评审员。请根据以下标准评估提供的报告内容：
1.  **相关性**：内容是否紧扣主题和章节要求？信息是否与讨论的核心问题直接相关？
2.  **流畅性**：语句是否通顺自然？段落之间过渡是否平滑？逻辑是否清晰？
3.  **完整性**：信息是否全面？论点是否得到了充分的论证和支持？是否涵盖了应有的关键点？
4.  **准确性**：所陈述的事实、数据和信息是否准确无误？（请基于常识或普遍接受的知识进行判断，除非提供了特定领域的参考标准）

待评估内容：
---
{content_to_evaluate}
---

请对以上内容进行综合评估，并严格按照以下JSON格式返回你的评分和反馈意见。不要添加任何额外的解释或说明文字。
总评分范围为0-100分。反馈意见应具体指出优点和需要改进的地方。

JSON输出格式：
{
  "score": <总评分，整数>,
  "feedback_cn": "具体的中文反馈意见，包括优点和改进建议。",
  "evaluation_criteria_met": {
    "relevance": "<关于相关性的简短评价，例如：高/中/低，具体说明>",
    "fluency": "<关于流畅性的简短评价，例如：优秀/良好/一般/较差，具体说明>",
    "completeness": "<关于完整性的简短评价，例如：非常全面/基本全面/部分缺失/严重缺失，具体说明>",
    "accuracy": "<关于准确性的简短评价（基于常识），例如：高/待核实/部分存疑/低，具体说明>"
  }
}
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        """
        Initializes the EvaluatorAgent.

        Args:
            llm_service (LLMService): An instance of the LLMService.
            prompt_template (Optional[str]): A custom prompt template for the LLM.
                                             If None, DEFAULT_PROMPT_TEMPLATE is used.
        """
        super().__init__(agent_name="EvaluatorAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise EvaluatorAgentError("LLMService is required for EvaluatorAgent.")

    def run(self, content_to_evaluate: str) -> Dict[str, any]:
        """
        Evaluates the given content using the LLM.

        Args:
            content_to_evaluate (str): The text content to be evaluated.

        Returns:
            Dict[str, any]: A dictionary containing:
                - 'score': An overall score (0-100).
                - 'feedback_cn': Detailed feedback in Chinese.
                - 'evaluation_criteria_met': A dict with sub-scores/comments for each criterion.

        Raises:
            EvaluatorAgentError: If the LLM call fails or the response is not as expected.
        """
        self._log_input(content_to_evaluate_length=len(content_to_evaluate))

        if not content_to_evaluate or not content_to_evaluate.strip():
            logger.warning("EvaluatorAgent received empty content to evaluate.")
            # Return a default low score or raise error, depending on desired behavior.
            return {
                "score": 0,
                "feedback_cn": "无法评估空内容。",
                "evaluation_criteria_met": {
                    "relevance": "无法评估", "fluency": "无法评估",
                    "completeness": "无法评估", "accuracy": "无法评估"
                }
            }

        prompt = self.prompt_template.format(content_to_evaluate=content_to_evaluate)

        try:
            logger.info(f"Sending request to LLM for content evaluation. Content length: {len(content_to_evaluate)}")
            raw_response = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一个严格且公正的AI内容评审专家。"
            )
            logger.debug(f"Raw LLM response for evaluation: {raw_response}")

            try:
                json_start_index = raw_response.find('{')
                json_end_index = raw_response.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_string = raw_response[json_start_index:json_end_index]
                    parsed_response = json.loads(json_string)
                else:
                    logger.error(f"Could not find valid JSON object in LLM evaluation response: {raw_response}")
                    raise EvaluatorAgentError("LLM evaluation response does not contain a valid JSON object.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from LLM for evaluation: {raw_response}. Error: {e}")
                raise EvaluatorAgentError(f"LLM evaluation response was not valid JSON: {e}")

            required_keys = ["score", "feedback_cn", "evaluation_criteria_met"]
            if not all(key in parsed_response for key in required_keys):
                logger.error(f"LLM evaluation response missing required keys. Response: {parsed_response}")
                raise EvaluatorAgentError("LLM evaluation response is missing one or more required keys.")

            if not isinstance(parsed_response.get("score"), int) or \
               not isinstance(parsed_response.get("feedback_cn"), str) or \
               not isinstance(parsed_response.get("evaluation_criteria_met"), dict):
                logger.error(f"LLM evaluation response has malformed data types. Response: {parsed_response}")
                raise EvaluatorAgentError("LLM evaluation response has malformed data types for required keys.")

            self._log_output(parsed_response)
            return parsed_response

        except LLMServiceError as e:
            logger.error(f"LLM service error during content evaluation: {e}")
            raise EvaluatorAgentError(f"LLM service failed during evaluation: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in EvaluatorAgent: {e}")
            raise EvaluatorAgentError(f"Unexpected error in content evaluation: {e}")

if __name__ == '__main__':
    print("EvaluatorAgent Example (requires running Xinference for LLMService)")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            logger.info(f"MockLLMService received query for evaluation (first 100 chars): {query[:100]}")
            if "ABMS系统是美军" in query: # Assuming this is part of the content
                return """
                {
                  "score": 85,
                  "feedback_cn": "内容与主题高度相关，对ABMS的定义和目标阐述清晰。语言流畅，逻辑性强。可以进一步补充ABMS在实际演习中的应用案例以增强完整性。准确性方面，基于常识判断，核心概念描述正确。",
                  "evaluation_criteria_met": {
                    "relevance": "高，内容紧扣ABMS系统概述。",
                    "fluency": "优秀，语句通顺，表达清晰。",
                    "completeness": "良好，核心概念完整，但可补充实例。",
                    "accuracy": "高（基于常识判断）。"
                  }
                }
                """
            elif "气候变化已成为全球共同关注" in query:
                return """
                {
                  "score": 78,
                  "feedback_cn": "报告章节对气候变化的主要表现有较好的概述，相关性强，语言也比较流畅。但在完整性方面，可以更深入地探讨各项表现之间的关联性。准确性较高，但部分数据如能引用来源会更好。",
                  "evaluation_criteria_met": {
                    "relevance": "高，准确描述了气候变化表现。",
                    "fluency": "良好，段落过渡自然。",
                    "completeness": "一般，可以更深入分析各项表现的内在联系和具体数据支撑。",
                    "accuracy": "较高，但建议关键数据注明来源。"
                  }
                }
                """
            return """
                {
                  "score": 50,
                  "feedback_cn": "内容笼统，缺乏具体细节和深入分析。",
                  "evaluation_criteria_met": {
                    "relevance": "中", "fluency": "一般", "completeness": "不足", "accuracy": "待核实"
                  }
                }
            """

    try:
        # llm_service_instance = LLMService()
        print("Using MockLLMService for EvaluatorAgent example.")
        llm_service_instance = MockLLMService()

        evaluator_agent = EvaluatorAgent(llm_service=llm_service_instance)

        # Test case 1: Good content
        content1 = """根据提供的资料，ABMS（先进战斗管理系统）是美军为了应对未来复杂战场环境而提出的一项关键发展项目。其核心目标在于构建一个能够连接所有传感器、平台和作战人员的统一网络，实现信息的快速共享和协同决策，从而在多域作战中取得优势。资料1指出ABMS的设计理念是开放架构和敏捷开发，这与传统武器系统的封闭模式形成对比。资料2则强调了JADC2（联合全域指挥控制）是ABMS实现其目标的重要框架，ABMS可以被视为JADC2概念下的具体技术和系统实现之一。总的来说，ABMS旨在通过技术创新，提升美军的战场感知、指挥控制和作战效能。"""
        print(f"\nEvaluating content 1 (length: {len(content1)}):")
        evaluation1 = evaluator_agent.run(content_to_evaluate=content1)
        print("Evaluation Result 1:")
        print(json.dumps(evaluation1, indent=2, ensure_ascii=False))

        # Test case 2: Average content
        content2 = """气候变化已成为全球共同关注的焦点议题，其表现形式多样且影响深远。根据参考资料1，全球平均气温的持续上升是气候变化最显著的特征之一，这导致了冰川融化和海平面上升。同时，如资料2所述，极端天气事件，例如热浪、干旱、洪水和强风暴的频率和强度也在增加，对全球各地的生态系统和人类社会造成了严重威胁。此外，资料3提到生物多样性的减少也是气候变化的一个重要表现，许多物种因无法适应快速变化的环境而面临生存危机。这些现象共同构成了当前气候变化的主要表现，警示我们需要采取紧急行动。"""
        print(f"\nEvaluating content 2 (length: {len(content2)}):")
        evaluation2 = evaluator_agent.run(content_to_evaluate=content2)
        print("Evaluation Result 2:")
        print(json.dumps(evaluation2, indent=2, ensure_ascii=False))

        # Test case 3: Empty content
        content3 = ""
        print(f"\nEvaluating content 3 (empty):")
        evaluation3 = evaluator_agent.run(content_to_evaluate=content3)
        print("Evaluation Result 3:")
        print(json.dumps(evaluation3, indent=2, ensure_ascii=False))


    except (LLMServiceError, EvaluatorAgentError) as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nEvaluatorAgent example finished.")
