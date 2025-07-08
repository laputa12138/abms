import logging
import json # Keep for other JSON operations if any, or remove if only repair is used for loading
import json_repair # Added
from typing import Dict, Optional, List, Any # Added List, Any

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_GENERATE_OUTLINE # Import constants
from config import settings as app_settings # Import settings

logger = logging.getLogger(__name__)

class TopicAnalyzerAgentError(Exception):
    """Custom exception for TopicAnalyzerAgent errors."""
    pass

class TopicAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing the user's topic, generalizing it,
    extracting relevant keywords, identifying core research questions,
    potential methodologies, generating expanded search queries,
    and elaborating on a global theme. Updates the WorkflowState.
    """

    # DEFAULT_SYSTEM_PROMPT and DEFAULT_PROMPT_TEMPLATE_WITH_EXPANDED_QUERIES
    # are removed as the prompt now comes from app_settings.DEFAULT_TOPIC_ANALYZER_PROMPT

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        # The main prompt template is now sourced from settings
        self.prompt_template = prompt_template or app_settings.DEFAULT_TOPIC_ANALYZER_PROMPT
        # System prompt can be simpler or also configurable if needed
        self.system_prompt = system_prompt or "You are a helpful AI assistant specialized in topic analysis and research planning."


        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")
        if not self.prompt_template:
            raise TopicAnalyzerAgentError("Prompt template (from settings.DEFAULT_TOPIC_ANALYZER_PROMPT) is required for TopicAnalyzerAgent.")


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

        # The DEFAULT_TOPIC_ANALYZER_PROMPT from settings now includes {max_expanded_queries}
        max_queries = app_settings.DEFAULT_MAX_EXPANDED_QUERIES_TOPIC
        prompt_formatted = self.prompt_template.format(
            user_topic=user_topic,
            max_expanded_queries=max_queries
        )

        try:
            logger.info(f"Sending request to LLM for topic analysis. User topic: '{user_topic}'. Using system prompt: '{self.system_prompt}'")
            # The detailed instructions are now part of prompt_formatted (from settings)
            raw_response = self.llm_service.chat(query=prompt_formatted, system_prompt=self.system_prompt)
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

            # Define all required keys based on the new prompt template in settings.py
            required_keys = [
                "generalized_topic_cn", "generalized_topic_en",
                "keywords_cn", "keywords_en",
                "core_research_questions_cn",   # New key
                "potential_methodologies_cn", # New key
                "expanded_queries",
                # "report_global_theme_elaboration" # This was in the old agent-specific prompt,
                                                  # the new settings prompt doesn't explicitly ask for it as a separate key.
                                                  # The general idea of a theme should be covered by the other elements.
                                                  # If it's still needed, the prompt in settings.py must be updated.
                                                  # For now, let's assume it's not a separate required key from the new prompt.
            ]
            # Let's check if the new prompt *implies* a theme elaboration or if we need to add it back.
            # The new prompt asks for:泛化主题, 关键词, 核心研究问题, 潜在研究方法, 扩展查询.
            # It doesn't explicitly ask for "report_global_theme_elaboration".
            # We will rely on `workflow_state.update_report_global_theme` being called with a generated theme later if needed,
            # or the sum of these parts will constitute the "theme".
            # For now, removing "report_global_theme_elaboration" from required_keys.
            # If a global theme string is still desired, it should be explicitly asked for in the settings prompt and added here.

            missing_keys = [key for key in required_keys if key not in parsed_response]
            if missing_keys:
                raise TopicAnalyzerAgentError(f"LLM response missing required keys: {', '.join(missing_keys)}. Response: {parsed_response}")

            # Validate types of critical fields
            if not isinstance(parsed_response.get("keywords_cn"), list) or \
               not isinstance(parsed_response.get("keywords_en"), list):
                raise TopicAnalyzerAgentError(f"Keywords in LLM response are not lists. Response: {parsed_response}")

            # Validate new fields (core_research_questions_cn, potential_methodologies_cn)
            if not isinstance(parsed_response.get("core_research_questions_cn"), list) or \
               not all(isinstance(q, str) for q in parsed_response.get("core_research_questions_cn", [])):
                raise TopicAnalyzerAgentError(f"'core_research_questions_cn' must be a list of strings. Found: {parsed_response.get('core_research_questions_cn')}. Response: {parsed_response}")

            if not isinstance(parsed_response.get("potential_methodologies_cn"), list) or \
               not all(isinstance(m, str) for m in parsed_response.get("potential_methodologies_cn", [])):
                raise TopicAnalyzerAgentError(f"'potential_methodologies_cn' must be a list of strings. Found: {parsed_response.get('potential_methodologies_cn')}. Response: {parsed_response}")


            expanded_queries = parsed_response.get("expanded_queries")
            if not isinstance(expanded_queries, list) or \
               not all(isinstance(q, str) for q in expanded_queries) or \
               not expanded_queries: # Ensure it's not an empty list
                raise TopicAnalyzerAgentError(
                    f"'expanded_queries' must be a non-empty list of strings. Found: {expanded_queries}. Response: {parsed_response}"
                )

            # Handling report_global_theme:
            # Since the new prompt doesn't explicitly ask for "report_global_theme_elaboration",
            # we can construct a theme from other parts or leave it for a later stage.
            # For now, let's update topic_analysis_results and let downstream agents decide on theme.
            # ChapterWriterAgent will need a global theme. We can synthesize one here or have another agent do it.
            # Let's synthesize a simple one for now if the key 'report_global_theme_elaboration' is missing.
            theme_elaboration = parsed_response.get("report_global_theme_elaboration") # Check if LLM still provided it
            if not theme_elaboration or not isinstance(theme_elaboration, str) or not theme_elaboration.strip():
                 # Synthesize a theme if not provided or empty
                theme_elaboration = (
                    f"本报告围绕主题“{parsed_response.get('generalized_topic_cn', user_topic)}”展开，"
                    f"重点探讨其核心研究问题，如“{'、'.join(parsed_response.get('core_research_questions_cn', ['未指定核心问题']))}”，"
                    f"并可能采用“{'、'.join(parsed_response.get('potential_methodologies_cn', ['未指定方法']))}”等视角进行分析。"
                )
                logger.info(f"Synthesized report_global_theme_elaboration as it was missing or empty in LLM response. Theme: {theme_elaboration}")
                parsed_response["report_global_theme_elaboration"] = theme_elaboration # Add to parsed_response for consistency


            # Update WorkflowState with all analysis results
            workflow_state.update_topic_analysis(parsed_response)

            # Specifically update the report_global_theme in workflow_state
            # This uses the (potentially synthesized) theme_elaboration from parsed_response
            workflow_state.update_report_global_theme(parsed_response["report_global_theme_elaboration"])


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
            workflow_state.log_event(f"LLM service error during topic analysis for '{user_topic}'", {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)
        except TopicAnalyzerAgentError as e:
            err_msg = f"Topic analysis failed for '{user_topic}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise
        except Exception as e:
            err_msg = f"Unexpected error in topic analysis for '{user_topic}': {e}"
            workflow_state.log_event(f"Unexpected error in TopicAnalyzerAgent for '{user_topic}'", {"error": str(e)}, level="CRITICAL")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)

if __name__ == '__main__':
    # Updated example for WorkflowState interaction
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            # Mock response for the new prompt structure
            if "ABMS系统" in query: # Check for user_topic in the formatted query
                return json.dumps({
                    "generalized_topic_cn": "先进战斗管理系统（ABMS）",
                    "generalized_topic_en": "Advanced Battle Management System (ABMS)",
                    "keywords_cn": ["ABMS", "JADC2", "军事指挥与控制", "网络中心战"],
                    "keywords_en": ["ABMS", "JADC2", "Military C2", "Network Centric Warfare"],
                    "core_research_questions_cn": [
                        "ABMS如何提升美军的作战效能和决策速度？",
                        "ABMS在实现JADC2目标中面临哪些关键技术挑战和伦理困境？",
                        "未来ABMS技术将如何演进，对国际军事格局可能产生哪些影响？"
                    ],
                    "potential_methodologies_cn": [
                        "案例研究（分析ABMS项目进展和演习）",
                        "技术评估（分析关键技术如AI、云计算在ABMS中的应用）",
                        "文献综述（总结现有研究和报告）"
                    ],
                    "expanded_queries": [
                        "ABMS系统架构和组成部分",
                        "JADC2概念与ABMS的关系",
                        "ABMS在多域作战中的应用案例",
                        "ABMS项目的最新进展和挑战",
                        "人工智能在ABMS中的作用",
                        "ABMS数据链和通信技术",
                        "对比分析各国类似的军事指挥控制系统"
                    ],
                    # "report_global_theme_elaboration" is not explicitly in the new prompt's JSON structure.
                    # The agent will synthesize it or it can be added back to the prompt if a specific LLM-generated one is desired.
                })
            # Fallback generic response
            return json.dumps({
                "generalized_topic_cn": "通用模拟主题",
                "generalized_topic_en": "General Mock Topic",
                "keywords_cn": ["模拟关键词1", "模拟关键词2"],
                "keywords_en": ["Mock Keyword1", "Mock Keyword2"],
                "core_research_questions_cn": ["模拟核心问题1？", "模拟核心问题2？"],
                "potential_methodologies_cn": ["模拟方法1", "模拟方法2"],
                "expanded_queries": ["模拟查询1", "模拟查询2", "模拟查询3"]
            })

    # Mock WorkflowState for testing the agent
    class MockWorkflowState(WorkflowState):
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.added_tasks_topic_analyzer = []

        def update_topic_analysis(self, results: Dict[str, Any]):
            super().update_topic_analysis(results)
            logger.debug(f"MockWorkflowState: Topic analysis updated with: {results}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_topic_analyzer.append({'type': task_type, 'payload': payload, 'priority': priority})
            logger.debug(f"MockWorkflowState (TopicAnalyzer): Task added - Type: {task_type}, Payload: {payload}")

    llm_service_instance = MockLLMService()
    # The agent will now use DEFAULT_TOPIC_ANALYZER_PROMPT from settings.py
    analyzer_agent = TopicAnalyzerAgent(llm_service=llm_service_instance)

    test_topic = "介绍美国的ABMS系统"
    mock_state = MockWorkflowState(user_topic=test_topic)
    mock_state.current_processing_task_id = "taa_task_test_002"

    task_payload_for_agent = {'user_topic': test_topic, 'priority':1}

    print(f"\nExecuting TopicAnalyzerAgent for topic: '{test_topic}' with new prompt structure")
    try:
        analyzer_agent.execute_task(mock_state, task_payload_for_agent)

        print("\nWorkflowState after TopicAnalyzerAgent execution:")
        print(f"  Topic Analysis Results: {json.dumps(mock_state.topic_analysis_results, indent=2, ensure_ascii=False)}")
        print(f"  Report Global Theme (from state): {mock_state.get_report_global_theme()}")
        print(f"  Tasks added by agent: {json.dumps(mock_state.added_tasks_topic_analyzer, indent=2, ensure_ascii=False)}")

        assert mock_state.topic_analysis_results is not None
        assert mock_state.topic_analysis_results['generalized_topic_cn'] == "先进战斗管理系统（ABMS）"
        assert "core_research_questions_cn" in mock_state.topic_analysis_results
        assert len(mock_state.topic_analysis_results["core_research_questions_cn"]) == 3
        assert "ABMS如何提升美军的作战效能和决策速度？" in mock_state.topic_analysis_results["core_research_questions_cn"]

        assert "potential_methodologies_cn" in mock_state.topic_analysis_results
        assert "案例研究（分析ABMS项目进展和演习）" in mock_state.topic_analysis_results["potential_methodologies_cn"]

        assert "expanded_queries" in mock_state.topic_analysis_results
        assert "ABMS系统架构和组成部分" in mock_state.topic_analysis_results["expanded_queries"]

        # Check synthesized theme if "report_global_theme_elaboration" was not in mock LLM output for this key
        # The mock LLM for ABMS *doesn't* return "report_global_theme_elaboration" key, so agent should synthesize it.
        synthesized_theme_part = "本报告围绕主题“先进战斗管理系统（ABMS）”展开"
        assert synthesized_theme_part in mock_state.get_report_global_theme()
        assert "未指定核心问题" not in mock_state.get_report_global_theme() # Should use actual questions


        assert len(mock_state.added_tasks_topic_analyzer) == 1
        added_task = mock_state.added_tasks_topic_analyzer[0]
        assert added_task['type'] == TASK_TYPE_GENERATE_OUTLINE
        assert added_task['payload']['topic_details'] == mock_state.topic_analysis_results
        print("\nTopicAnalyzerAgent test successful with new prompt structure.")

    except Exception as e:
        print(f"Error during TopicAnalyzerAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nTopicAnalyzerAgent example finished.")
