import logging
import json # Keep for other JSON operations if any, or remove if only repair is used for loading
import json_repair # Added
from typing import Dict, Optional, List, Any # Added List, Any

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_GENERATE_OUTLINE # Import constants

logger = logging.getLogger(__name__)

class TopicAnalyzerAgentError(Exception):
    """Custom exception for TopicAnalyzerAgent errors."""
    pass

class TopicAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing the user's topic, generalizing it,
    extracting relevant keywords, generating expanded search queries,
    and elaborating on a global theme. Updates the WorkflowState.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个高效的主题分析助手。"
        "你的任务是准确理解用户主题，然后执行以下操作："
        "1. 生成一个泛化的中文主题表述。"
        "2. 生成一个泛化的英文主题表述。"
        "3. 提取3-5个核心中文关键词。"
        "4. 提取3-5个核心英文关键词。"
        "5. 针对用户主题，生成3-7条相关的、多样化的搜索查询建议，用于后续的资料检索。这些查询应该包括同义词、相关概念、上位词、下位词、以及可能的子主题。查询可以是中文或英文。"
        "6. 针对用户主题进行详细阐述（至少50字），作为整个报告的全局背景和核心思想，为后续章节撰写提供明确方向。"
        "请严格按照指定的JSON格式返回结果。"
    )
    DEFAULT_PROMPT_TEMPLATE_WITH_EXPANDED_QUERIES = """请分析以下用户提供的主题：'{user_topic}'

并严格按照以下JSON格式返回结果：
{{
  "generalized_topic_cn": "对用户主题进行泛化后的中文主题表述",
  "generalized_topic_en": "Generalized English Topic based on user_topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"],
  "expanded_queries": [
    "与用户主题相关的第一个扩展检索查询",
    "第二个不同的检索查询，可能是子主题或相关概念",
    "第三个检索查询，尝试使用同义词或不同角度",
    "更多查询..."
  ],
  "report_global_theme_elaboration": "针对'{user_topic}'的详细阐述，作为整个报告的全局背景和核心思想。这段阐述应至少50字，力求全面、深刻，为后续章节的撰写提供明确的方向。"
}}
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        from config import settings as app_settings
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE_WITH_EXPANDED_QUERIES
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")
        if not self.prompt_template:
            raise TopicAnalyzerAgentError("Prompt template is required for TopicAnalyzerAgent.")


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Analyzes the user topic using the LLM and updates the WorkflowState.
        Then, adds a task to generate the outline.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload for this task, expects 'user_topic'.

        Raises:
            TopicAnalyzerAgentError: If the LLM call fails or the response is not as expected.
        """
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for user_topic: {task_payload.get('user_topic')}")

        user_topic = task_payload.get('user_topic')
        if not user_topic:
            err_msg = "User topic not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)

        self._log_input(user_topic=user_topic) # BaseAgent helper

        # print
        print('--' * 20)
        print(f"TopicAnalyzerAgent executing for user topic: '{user_topic}'")
        print('--' * 20)
        prompt = self.prompt_template.format(user_topic=user_topic)

        try:
            logger.info(f"Sending request to LLM for topic analysis. User topic: '{user_topic}' using system prompt: '{self.system_prompt}'")
            raw_response = self.llm_service.chat(query=prompt, system_prompt=self.system_prompt)
            logger.debug(f"Raw LLM response for topic analysis: {raw_response}")

            try:
                json_start_index = raw_response.find('{')
                json_end_index = raw_response.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_string = raw_response[json_start_index:json_end_index]
                    parsed_response = json_repair.loads(json_string)
                else:
                    logger.warning(f"No clear JSON object found in LLM response for topic analysis, attempting to repair entire response: {raw_response}")
                    parsed_response = json_repair.loads(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                raise TopicAnalyzerAgentError(f"LLM response was not valid or repairable JSON: {raw_response}. Error: {e}")

            # Define all required keys including the new 'expanded_queries'
            required_keys = [
                "generalized_topic_cn", "generalized_topic_en",
                "keywords_cn", "keywords_en",
                "expanded_queries", # New key
                "report_global_theme_elaboration"
            ]
            missing_keys = [key for key in required_keys if key not in parsed_response]
            if missing_keys:
                raise TopicAnalyzerAgentError(f"LLM response missing required keys: {', '.join(missing_keys)}. Response: {parsed_response}")

            # Validate types of critical fields
            if not isinstance(parsed_response.get("keywords_cn"), list) or \
               not isinstance(parsed_response.get("keywords_en"), list):
                raise TopicAnalyzerAgentError(f"Keywords in LLM response are not lists. Response: {parsed_response}")

            expanded_queries = parsed_response.get("expanded_queries")
            if not isinstance(expanded_queries, list) or \
               not all(isinstance(q, str) for q in expanded_queries) or \
               not expanded_queries: # Ensure it's not an empty list
                raise TopicAnalyzerAgentError(
                    f"'expanded_queries' must be a non-empty list of strings. Found: {expanded_queries}. Response: {parsed_response}"
                )

            # Validate report_global_theme_elaboration
            theme_elaboration = parsed_response.get("report_global_theme_elaboration")
            if not isinstance(theme_elaboration, str) or not theme_elaboration.strip():
                logger.warning(
                    f"LLM response provided an empty or non-string report_global_theme_elaboration. "
                    f"Key present: {'report_global_theme_elaboration' in parsed_response}, Value: {theme_elaboration}"
                )
                # Retain the potentially empty/None value for now; fallback logic is downstream.


            # Update WorkflowState with all analysis results, including expanded_queries
            workflow_state.update_topic_analysis(parsed_response)

            # Specifically update the report_global_theme in workflow_state (handles fallback if needed)
            theme_elaboration = parsed_response.get("report_global_theme_elaboration")
            if theme_elaboration and isinstance(theme_elaboration, str) and theme_elaboration.strip():
                workflow_state.update_report_global_theme(theme_elaboration)
            else:
                # Fallback or use a default if theme is crucial and missing
                fallback_theme = f"基于用户主题“{user_topic}”的综合分析报告。"
                workflow_state.update_report_global_theme(fallback_theme)
                logger.warning(f"Using fallback global theme: {fallback_theme} due to missing/empty elaboration from LLM.")


            # Add next task: Generate Outline
            workflow_state.add_task(
                task_type=TASK_TYPE_GENERATE_OUTLINE,
                payload={'topic_details': parsed_response}, # Pass analysis results to next task
                priority=2 # Assuming topic analysis is priority 1
            )

            self._log_output(parsed_response) # BaseAgent helper
            success_msg = f"Topic analysis successful for '{user_topic}'. Next task (Generate Outline) added."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except LLMServiceError as e:
            err_msg = f"LLM service failed for topic '{user_topic}': {e}"
            workflow_state.log_event(f"LLM service error during topic analysis for '{user_topic}'", {"error": str(e)}, level="ERROR") # Keep
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg) # Re-raise
        except TopicAnalyzerAgentError as e:
            err_msg = f"Topic analysis failed for '{user_topic}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR") # Keep
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise # Re-raise
        except Exception as e:
            err_msg = f"Unexpected error in topic analysis for '{user_topic}': {e}"
            workflow_state.log_event(f"Unexpected error in TopicAnalyzerAgent for '{user_topic}'", {"error": str(e)}, level="CRITICAL") # Keep
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg) # Re-raise

if __name__ == '__main__':
    # Updated example for WorkflowState interaction
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            # Updated mock response to include expanded_queries
            if "ABMS系统" in query:
                return json.dumps({
                    "generalized_topic_cn": "先进战斗管理系统（ABMS）",
                    "generalized_topic_en": "Advanced Battle Management System (ABMS)",
                    "keywords_cn": ["ABMS", "JADC2", "军事"],
                    "keywords_en": ["ABMS", "JADC2", "military"],
                    "expanded_queries": [
                        "ABMS系统架构",
                        "JADC2概念与ABMS关系",
                        "ABMS关键技术",
                        "ABMS发展现状与挑战",
                        "美军ABMS系统介绍"
                    ],
                    "report_global_theme_elaboration": "ABMS是美国空军开发的一套先进的战场管理和指挥控制系统，旨在连接各种传感器、武器和数据源，实现跨域作战的无缝协同。它对于提升态势感知能力、加速决策循环以及实现JADC2概念至关重要。"
                })
            return json.dumps({
                "generalized_topic_cn": "模拟主题",
                "generalized_topic_en": "Mock Topic",
                "keywords_cn": ["关键词1"],
                "keywords_en": ["Keyword1"],
                "expanded_queries": ["模拟主题的定义", "模拟主题的应用", "模拟主题的未来发展"],
                "report_global_theme_elaboration": "这是一个关于模拟主题的详细阐述，旨在为报告提供全局背景和核心思想，确保内容连贯一致。"
            })

    # Mock WorkflowState for testing the agent
    class MockWorkflowState(WorkflowState):
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            # self.updated_analysis = None # Not needed, topic_analysis_results is directly updated by parent
            self.added_tasks_topic_analyzer = [] # Use a unique name to avoid conflicts if other tests use 'added_tasks'

        def update_topic_analysis(self, results: Dict[str, Any]):
            # self.updated_analysis = results # Store if direct inspection is needed pre-super call
            super().update_topic_analysis(results) # Call parent for logging etc.
            logger.debug(f"MockWorkflowState: Topic analysis updated with: {results}")


        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_topic_analyzer.append({'type': task_type, 'payload': payload, 'priority': priority})
            logger.debug(f"MockWorkflowState (TopicAnalyzer): Task added - Type: {task_type}, Payload: {payload}")


    llm_service_instance = MockLLMService()
    analyzer_agent = TopicAnalyzerAgent(llm_service=llm_service_instance)

    test_topic = "介绍美国的ABMS系统"
    mock_state = MockWorkflowState(user_topic=test_topic)
    mock_state.current_processing_task_id = "taa_task_test_001" # Simulate orchestrator setting this

    task_payload_for_agent = {'user_topic': test_topic, 'priority':1} # Simulate payload from orchestrator

    print(f"\nExecuting TopicAnalyzerAgent for topic: '{test_topic}' with MockWorkflowState (incl. Expanded Queries)")
    try:
        analyzer_agent.execute_task(mock_state, task_payload_for_agent)

        print("\nWorkflowState after TopicAnalyzerAgent execution:")
        print(f"  Topic Analysis Results: {json.dumps(mock_state.topic_analysis_results, indent=2, ensure_ascii=False)}")
        print(f"  Report Global Theme: {mock_state.get_report_global_theme()}")
        print(f"  Tasks added by agent: {json.dumps(mock_state.added_tasks_topic_analyzer, indent=2, ensure_ascii=False)}")

        assert mock_state.topic_analysis_results is not None
        assert mock_state.topic_analysis_results['generalized_topic_cn'] == "先进战斗管理系统（ABMS）"
        assert "ABMS是美国空军开发的一套先进的战场管理和指挥控制系统" in mock_state.get_report_global_theme()

        assert "expanded_queries" in mock_state.topic_analysis_results
        assert isinstance(mock_state.topic_analysis_results["expanded_queries"], list)
        assert len(mock_state.topic_analysis_results["expanded_queries"]) > 0
        assert "ABMS系统架构" in mock_state.topic_analysis_results["expanded_queries"]

        assert len(mock_state.added_tasks_topic_analyzer) == 1
        added_task = mock_state.added_tasks_topic_analyzer[0]
        assert added_task['type'] == TASK_TYPE_GENERATE_OUTLINE
        assert added_task['payload']['topic_details'] == mock_state.topic_analysis_results
        print("\nTopicAnalyzerAgent test successful with MockWorkflowState (including expanded_queries).")

    except Exception as e:
        print(f"Error during TopicAnalyzerAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nTopicAnalyzerAgent example finished.")
