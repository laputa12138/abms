import logging
import json
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

    DEFAULT_PROMPT_TEMPLATE = """你是一个主题分析专家。请分析以下用户提供的主题，对其进行理解、扩展和泛化，并生成相关的中文和英文关键词/主题概念，以便于后续的文档检索。请确保关键词的全面性。

用户主题：'{user_topic}'

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或说明文字：
{{
  "generalized_topic_cn": "泛化后的中文主题",
  "generalized_topic_en": "Generalized English Topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"]
}}
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")

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
            logger.info(f"Sending request to LLM for topic analysis. User topic: '{user_topic}'")
            raw_response = self.llm_service.chat(query=prompt, system_prompt="你是一个高效的主题分析助手。")
            logger.debug(f"Raw LLM response for topic analysis: {raw_response}")

            try:
                json_start_index = raw_response.find('{')
                json_end_index = raw_response.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_string = raw_response[json_start_index:json_end_index]
                    parsed_response = json.loads(json_string)
                else:
                    raise TopicAnalyzerAgentError(f"LLM response does not contain a valid JSON object: {raw_response}")
            except json.JSONDecodeError as e:
                raise TopicAnalyzerAgentError(f"LLM response was not valid JSON: {raw_response}. Error: {e}")

            required_keys = ["generalized_topic_cn", "generalized_topic_en", "keywords_cn", "keywords_en"]
            if not all(key in parsed_response for key in required_keys):
                raise TopicAnalyzerAgentError(f"LLM response missing required keys. Response: {parsed_response}")
            if not isinstance(parsed_response.get("keywords_cn"), list) or \
               not isinstance(parsed_response.get("keywords_en"), list):
                raise TopicAnalyzerAgentError(f"Keywords in LLM response are not lists. Response: {parsed_response}")

            # Update WorkflowState
            workflow_state.update_topic_analysis(parsed_response)

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

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "ABMS系统" in query: return json.dumps({"generalized_topic_cn": "先进战斗管理系统（ABMS）", "generalized_topic_en": "Advanced Battle Management System (ABMS)", "keywords_cn": ["ABMS", "JADC2"], "keywords_en": ["ABMS", "JADC2"]})
            return json.dumps({"generalized_topic_cn": "模拟主题", "generalized_topic_en": "Mock Topic", "keywords_cn": ["关键词1"], "keywords_en": ["Keyword1"]})

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
        print(f"  Tasks added by agent: {json.dumps(mock_state.added_tasks, indent=2, ensure_ascii=False)}")

        assert mock_state.topic_analysis_results is not None
        assert mock_state.topic_analysis_results['generalized_topic_cn'] == "先进战斗管理系统（ABMS）"
        assert len(mock_state.added_tasks) == 1
        assert mock_state.added_tasks[0]['type'] == TASK_TYPE_GENERATE_OUTLINE
        assert mock_state.added_tasks[0]['payload']['topic_details'] == mock_state.topic_analysis_results
        print("\nTopicAnalyzerAgent test successful with MockWorkflowState.")

    except Exception as e:
        print(f"Error during TopicAnalyzerAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nTopicAnalyzerAgent example finished.")
