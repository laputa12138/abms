import logging
import json # Keep for other JSON operations if any, or remove if only repair is used for loading
import json_repair # Added
from typing import Dict, Optional

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
    and extracting relevant keywords. Updates the WorkflowState.
    """

    # DEFAULT_PROMPT_TEMPLATE will be removed and sourced from settings
    DEFAULT_SYSTEM_PROMPT = "你是一个高效的主题分析助手，能够准确理解用户主题，生成泛化主题、关键词，并对用户主题进行详细的阐述作为报告的全局背景。"
    DEFAULT_PROMPT_TEMPLATE_WITH_THEME_GENERATION = """请分析以下用户提供的主题，并严格按照以下JSON格式返回结果。
用户主题：'{user_topic}'

JSON输出格式：
{{
  "generalized_topic_cn": "对用户主题进行泛化后的中文主题表述",
  "generalized_topic_en": "Generalized English Topic based on user_topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"],
  "report_global_theme_elaboration": "针对'{user_topic}'的详细阐述，作为整个报告的全局背景和核心思想。这段阐述应至少50字，力求全面、深刻，为后续章节的撰写提供明确的方向。"
}}
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        # Import settings here or ensure it's available if prompt_template is None
        from config import settings as app_settings # Use a different alias to avoid conflict if 'settings' is a var
        # Check if the old DEFAULT_TOPIC_ANALYZER_PROMPT is still in app_settings for backward compatibility or specific use cases
        # For this change, we assume we want to use the new prompt that includes theme elaboration.
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE_WITH_THEME_GENERATION
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")
        if not self.prompt_template: # Should always be true now with internal default
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
                    # Attempt to repair and load the JSON string
                    parsed_response = json_repair.loads(json_string)
                else:
                    # If no clear JSON structure is found, try to repair the whole raw_response
                    logger.warning(f"No clear JSON object found in LLM response for topic analysis, attempting to repair entire response: {raw_response}")
                    parsed_response = json_repair.loads(raw_response) # Try repairing the whole thing
            except (json.JSONDecodeError, ValueError) as e: # json_repair can also raise ValueError
                raise TopicAnalyzerAgentError(f"LLM response was not valid or repairable JSON: {raw_response}. Error: {e}")

            # Updated required keys to include the new theme elaboration
            required_keys = ["generalized_topic_cn", "generalized_topic_en", "keywords_cn", "keywords_en", "report_global_theme_elaboration"]
            missing_keys = [key for key in required_keys if key not in parsed_response]
            if missing_keys:
                raise TopicAnalyzerAgentError(f"LLM response missing required keys: {', '.join(missing_keys)}. Response: {parsed_response}")

            if not isinstance(parsed_response.get("keywords_cn"), list) or \
               not isinstance(parsed_response.get("keywords_en"), list):
                raise TopicAnalyzerAgentError(f"Keywords in LLM response are not lists. Response: {parsed_response}")

            if not isinstance(parsed_response.get("report_global_theme_elaboration"), str) or \
               not parsed_response.get("report_global_theme_elaboration").strip():
                logger.warning(f"LLM response provided an empty or non-string report_global_theme_elaboration. Key present: {'report_global_theme_elaboration' in parsed_response}, Value: {parsed_response.get('report_global_theme_elaboration')}")
                # Allow to proceed but log a warning. Downstream might use a default.
                # Or, make this a hard fail:
                # raise TopicAnalyzerAgentError(f"report_global_theme_elaboration is missing, empty or not a string. Response: {parsed_response}")


            # Update WorkflowState with all analysis results
            workflow_state.update_topic_analysis(parsed_response)

            # Specifically update the report_global_theme in workflow_state
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
            # Updated mock response to include report_global_theme_elaboration
            if "ABMS系统" in query:
                return json.dumps({
                    "generalized_topic_cn": "先进战斗管理系统（ABMS）",
                    "generalized_topic_en": "Advanced Battle Management System (ABMS)",
                    "keywords_cn": ["ABMS", "JADC2", "军事"],
                    "keywords_en": ["ABMS", "JADC2", "military"],
                    "report_global_theme_elaboration": "ABMS是美国空军开发的一套先进的战场管理和指挥控制系统，旨在连接各种传感器、武器和数据源，实现跨域作战的无缝协同。它对于提升态势感知能力、加速决策循环以及实现JADC2概念至关重要。"
                })
            return json.dumps({
                "generalized_topic_cn": "模拟主题",
                "generalized_topic_en": "Mock Topic",
                "keywords_cn": ["关键词1"],
                "keywords_en": ["Keyword1"],
                "report_global_theme_elaboration": "这是一个关于模拟主题的详细阐述，旨在为报告提供全局背景和核心思想，确保内容连贯一致。"
            })

    # Mock WorkflowState for testing the agent
    class MockWorkflowState(WorkflowState):
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.updated_analysis = None
            self.added_tasks = []

        def update_topic_analysis(self, results: Dict[str, Any]):
            self.updated_analysis = results
            super().update_topic_analysis(results) # Call parent for logging etc.

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks.append({'type': task_type, 'payload': payload, 'priority': priority})
            # Don't call super().add_task here if we just want to inspect, or do if full behavior is needed.
            # For this test, just capturing is enough.
            logger.debug(f"MockWorkflowState: Task added - Type: {task_type}, Payload: {payload}")


    llm_service_instance = MockLLMService()
    analyzer_agent = TopicAnalyzerAgent(llm_service=llm_service_instance)

    test_topic = "介绍美国的ABMS系统"
    mock_state = MockWorkflowState(user_topic=test_topic)

    task_payload_for_agent = {'user_topic': test_topic}

    print(f"\nExecuting TopicAnalyzerAgent for topic: '{test_topic}' with MockWorkflowState")
    try:
        analyzer_agent.execute_task(mock_state, task_payload_for_agent)

        print("\nWorkflowState after TopicAnalyzerAgent execution:")
        print(f"  Topic Analysis Results: {json.dumps(mock_state.topic_analysis_results, indent=2, ensure_ascii=False)}")
        print(f"  Report Global Theme: {mock_state.get_report_global_theme()}") # Check the new theme
        print(f"  Tasks added by agent: {json.dumps(mock_state.added_tasks, indent=2, ensure_ascii=False)}")

        assert mock_state.topic_analysis_results is not None
        assert mock_state.topic_analysis_results['generalized_topic_cn'] == "先进战斗管理系统（ABMS）"
        assert "ABMS是美国空军开发的一套先进的战场管理和指挥控制系统" in mock_state.get_report_global_theme() # Verify theme content
        assert len(mock_state.added_tasks) == 1
        assert mock_state.added_tasks[0]['type'] == TASK_TYPE_GENERATE_OUTLINE
        assert mock_state.added_tasks[0]['payload']['topic_details'] == mock_state.topic_analysis_results
        print("\nTopicAnalyzerAgent test successful with MockWorkflowState (including theme generation).")

    except Exception as e:
        print(f"Error during TopicAnalyzerAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nTopicAnalyzerAgent example finished.")
