import logging
import json
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

class TopicAnalyzerAgentError(Exception):
    """Custom exception for TopicAnalyzerAgent errors."""
    pass

class TopicAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing the user's topic, generalizing it,
    and extracting relevant keywords in both Chinese and English.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一个主题分析专家。请分析以下用户提供的主题，对其进行理解、扩展和泛化，并生成相关的中文和英文关键词/主题概念，以便于后续的文档检索。请确保关键词的全面性。

用户主题：'{user_topic}'

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或说明文字：
{
  "generalized_topic_cn": "泛化后的中文主题",
  "generalized_topic_en": "Generalized English Topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"]
}
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        """
        Initializes the TopicAnalyzerAgent.

        Args:
            llm_service (LLMService): An instance of the LLMService.
            prompt_template (Optional[str]): A custom prompt template for the LLM.
                                             If None, DEFAULT_PROMPT_TEMPLATE is used.
        """
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")

    def run(self, user_topic: str) -> Dict[str, any]:
        """
        Analyzes the user topic using the LLM.

        Args:
            user_topic (str): The topic provided by the user.

        Returns:
            Dict[str, any]: A dictionary containing:
                - 'generalized_topic_cn': Generalized topic in Chinese.
                - 'generalized_topic_en': Generalized topic in English.
                - 'keywords_cn': A list of Chinese keywords.
                - 'keywords_en': A list of English keywords.

        Raises:
            TopicAnalyzerAgentError: If the LLM call fails or the response is not as expected.
        """
        self._log_input(user_topic=user_topic)

        prompt = self.prompt_template.format(user_topic=user_topic)

        try:
            logger.info(f"Sending request to LLM for topic analysis. User topic: '{user_topic}'")
            raw_response = self.llm_service.chat(query=prompt, system_prompt="你是一个高效的主题分析助手。") # System prompt can be minimal if main instructions are in user prompt

            logger.debug(f"Raw LLM response for topic analysis: {raw_response}")

            # Attempt to parse the JSON response
            # The LLM might sometimes add extra text around the JSON.
            # We'll try to extract the JSON part.
            try:
                # Find the start and end of the JSON object
                json_start_index = raw_response.find('{')
                json_end_index = raw_response.rfind('}') + 1

                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_string = raw_response[json_start_index:json_end_index]
                    parsed_response = json.loads(json_string)
                else:
                    logger.error(f"Could not find valid JSON object in LLM response: {raw_response}")
                    raise TopicAnalyzerAgentError("LLM response does not contain a valid JSON object.")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from LLM: {raw_response}. Error: {e}")
                raise TopicAnalyzerAgentError(f"LLM response was not valid JSON: {e}")

            # Validate the structure of the parsed response
            required_keys = ["generalized_topic_cn", "generalized_topic_en", "keywords_cn", "keywords_en"]
            if not all(key in parsed_response for key in required_keys):
                logger.error(f"LLM response missing required keys. Response: {parsed_response}")
                raise TopicAnalyzerAgentError("LLM response is missing one or more required keys.")

            if not isinstance(parsed_response["keywords_cn"], list) or \
               not isinstance(parsed_response["keywords_en"], list):
                logger.error(f"Keywords in LLM response are not lists. Response: {parsed_response}")
                raise TopicAnalyzerAgentError("Keywords in LLM response must be lists.")

            self._log_output(parsed_response)
            return parsed_response

        except LLMServiceError as e:
            logger.error(f"LLM service error during topic analysis: {e}")
            raise TopicAnalyzerAgentError(f"LLM service failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in TopicAnalyzerAgent: {e}")
            raise TopicAnalyzerAgentError(f"Unexpected error in topic analysis: {e}")

if __name__ == '__main__':
    # This is an example of how to use the TopicAnalyzerAgent.
    # Requires a running Xinference server for LLMService.
    print("TopicAnalyzerAgent Example (requires running Xinference for LLMService)")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock LLMService for example if Xinference is not available
    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            logger.info(f"MockLLMService received query (first 100 chars): {query[:100]}")
            # Simulate a valid JSON response based on the prompt
            if "ABMS系统" in query:
                return """
                {
                  "generalized_topic_cn": "先进战斗管理系统（ABMS）",
                  "generalized_topic_en": "Advanced Battle Management System (ABMS)",
                  "keywords_cn": ["ABMS", "先进战斗管理", "美军", "JADC2", "多域作战", "指挥控制"],
                  "keywords_en": ["ABMS", "Advanced Battle Management", "US Military", "JADC2", "Multi-Domain Operations", "Command and Control"]
                }
                """
            elif "气候变化的影响" in query:
                 return """
                {
                  "generalized_topic_cn": "气候变化及其全球影响",
                  "generalized_topic_en": "Climate Change and its Global Impacts",
                  "keywords_cn": ["气候变化", "全球变暖", "极端天气", "环境影响", "碳排放"],
                  "keywords_en": ["Climate Change", "Global Warming", "Extreme Weather", "Environmental Impact", "Carbon Emissions"]
                }
                """
            return "{}" # Default empty JSON

    try:
        # llm_service_instance = LLMService() # Uses defaults from settings
        # print("Attempting to use actual LLMService.")
        # Forcing mock for example run
        print("Using MockLLMService for TopicAnalyzerAgent example.")
        llm_service_instance = MockLLMService()

        analyzer_agent = TopicAnalyzerAgent(llm_service=llm_service_instance)

        # Test case 1
        topic1 = "介绍美国的ABMS系统"
        print(f"\nAnalyzing topic: '{topic1}'")
        analysis_result1 = analyzer_agent.run(user_topic=topic1)
        print("Analysis Result 1:")
        print(json.dumps(analysis_result1, indent=2, ensure_ascii=False))

        # Test case 2
        topic2 = "气候变化的影响"
        print(f"\nAnalyzing topic: '{topic2}'")
        analysis_result2 = analyzer_agent.run(user_topic=topic2)
        print("Analysis Result 2:")
        print(json.dumps(analysis_result2, indent=2, ensure_ascii=False))

    except (LLMServiceError, TopicAnalyzerAgentError) as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nTopicAnalyzerAgent example finished.")
