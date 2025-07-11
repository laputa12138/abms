import logging
import json
import json_repair # For robust JSON parsing
from typing import Dict, Optional, List, Any
import uuid # For test section unique IDs

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_GENERATE_OUTLINE
from config import settings as app_settings

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

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or app_settings.DEFAULT_TOPIC_ANALYZER_PROMPT
        self.system_prompt = system_prompt or "You are a helpful AI assistant specialized in topic analysis and research planning."
        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")
        if not self.prompt_template:
            raise TopicAnalyzerAgentError("Prompt template (from settings.DEFAULT_TOPIC_ANALYZER_PROMPT) is required for TopicAnalyzerAgent.")

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for user_topic: {task_payload.get('user_topic')}")

        user_topic = task_payload.get('user_topic')
        if not user_topic:
            err_msg = "User topic not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)

        self._log_input(user_topic=user_topic)
        print('--' * 20)
        print(f"TopicAnalyzerAgent executing for user topic: '{user_topic}'")
        print('--' * 20)

        max_queries = app_settings.DEFAULT_MAX_EXPANDED_QUERIES_TOPIC
        prompt_formatted = self.prompt_template.format(user_topic=user_topic, max_expanded_queries=max_queries)

        try:
            logger.info(f"Sending request to LLM for topic analysis. User topic: '{user_topic}'.")
            raw_response = self.llm_service.chat(query=prompt_formatted, system_prompt=self.system_prompt)
            logger.debug(f"Raw LLM response for topic analysis: {raw_response}")

            parsed_response = {}
            try:
                json_start_index = raw_response.find('{')
                json_end_index = raw_response.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_string = raw_response[json_start_index:json_end_index]
                    parsed_response = json_repair.loads(json_string)
                else:
                    logger.warning(f"No clear JSON object found in LLM response, attempting to repair entire response: {raw_response}")
                    # Try to repair the whole thing if no clear start/end, but this is risky
                    parsed_response = json_repair.loads(raw_response)
            except Exception as e_parse:
                raise TopicAnalyzerAgentError(f"LLM response was not valid or repairable JSON: {raw_response}. Error: {e_parse}")

            # --- Start of Robust Key Handling and Validation Logic ---

            # 1. Define strictly required keys and check for their presence
            strictly_required_keys = [
                "generalized_topic_cn", "generalized_topic_en",
                "keywords_cn", "keywords_en", "core_research_questions_cn",
            ]
            missing_strict_keys = [key for key in strictly_required_keys if key not in parsed_response]
            if missing_strict_keys:
                raise TopicAnalyzerAgentError(f"LLM response missing strictly required fields: {', '.join(missing_strict_keys)}. Parsed Response: {parsed_response}")

            # 2. Define recoverable keys (these might be missing or malformed by LLM)
            recoverable_keys = ["potential_methodologies_cn", "expanded_queries"]

            # Attempt to recover these keys if they are nested within a 'class' field by mistake
            # This is a specific heuristic based on observed LLM error patterns.
            if 'class' in parsed_response and isinstance(parsed_response['class'], str):
                class_field_value = parsed_response['class']
                logger.warning(f"Found 'class' field in LLM response. Attempting to recover recoverable_keys from its content: {class_field_value[:200]}...")
                try:
                    # Check if the content of 'class' field looks like it might contain the missing JSON parts
                    if ('"potential_methodologies_cn":' in class_field_value or \
                        '"expanded_queries":' in class_field_value):

                        # Try to make the content of 'class' a valid JSON object string by wrapping with {}
                        potential_json_from_class_str = f"{{{class_field_value}}}"
                        recovered_data_from_class = json_repair.loads(potential_json_from_class_str)

                        for r_key in recoverable_keys:
                            # Only assign if the key is missing in parsed_response AND found in the recovered_data
                            if r_key not in parsed_response and r_key in recovered_data_from_class:
                                parsed_response[r_key] = recovered_data_from_class[r_key]
                                logger.info(f"Successfully recovered '{r_key}' from 'class' field's content.")
                    else:
                        logger.info("Content of 'class' field did not match expected pattern for recovery of 'potential_methodologies_cn' or 'expanded_queries'.")
                except Exception as e_recover_class:
                    # Log error but don't let it stop the process; defaults will be applied later.
                    logger.error(f"Failed to parse or recover from 'class' field. Error: {e_recover_class}. Content: {class_field_value[:500]}", exc_info=False)

            # 3. Ensure all recoverable keys exist in parsed_response, defaulting to empty lists if necessary.
            #    Also, ensure their values are lists and all items within those lists are strings.
            for key in recoverable_keys:
                if key not in parsed_response:
                    logger.warning(f"Key '{key}' is missing after all recovery attempts. Defaulting to an empty list.")
                    parsed_response[key] = []
                elif not isinstance(parsed_response[key], list):
                    logger.warning(f"Key '{key}' was found but its value is not a list (type: {type(parsed_response[key])}). Correcting to an empty list. Original value: {str(parsed_response[key])[:100]}")
                    parsed_response[key] = []
                else:
                    # If it's a list, ensure all items are strings. Non-string items are removed.
                    original_list = parsed_response[key]
                    # Convert basic types to str, filter out others (like None if json_repair puts it)
                    corrected_list = [str(item) for item in original_list if isinstance(item, (str, int, float, bool))]
                    if len(corrected_list) != len(original_list):
                         logger.warning(f"Corrected/filtered non-string items in list for key '{key}'. Original: {original_list}, Corrected: {corrected_list}")
                    parsed_response[key] = corrected_list

            # --- Type Validations for strictly_required_keys (already checked for presence) ---
            # These ensure the core fields have the correct list-of-string structure.
            for key_to_validate in ["keywords_cn", "keywords_en", "core_research_questions_cn"]:
                if not (isinstance(parsed_response[key_to_validate], list) and all(isinstance(item, str) for item in parsed_response[key_to_validate])):
                    # This error means that even a strictly required key, though present, is not a list of strings.
                    raise TopicAnalyzerAgentError(f"Field '{key_to_validate}' must be a list of strings. Found: {parsed_response[key_to_validate]}. Full Response: {parsed_response}")

            # Final type check for recoverable_keys (now they must be lists of strings)
            for key_to_validate in recoverable_keys:
                 if not (isinstance(parsed_response[key_to_validate], list) and all(isinstance(item, str) for item in parsed_response[key_to_validate])):
                    logger.error(f"Internal logic error or unrecoverable type for '{key_to_validate}'. Expected list of strings, got {type(parsed_response[key_to_validate])}. Value: {str(parsed_response[key_to_validate])[:200]}. Forcing to empty list.")
                    # Force to empty list of strings to prevent downstream errors if somehow still not a list of strings.
                    parsed_response[key_to_validate] = [str(item) for item in parsed_response.get(key_to_validate, []) if isinstance(item, str)]


            if not parsed_response.get("expanded_queries"):
                 logger.warning("'expanded_queries' list is empty after all processing.")

            # Synthesize global theme
            theme_elaboration = parsed_response.get("report_global_theme_elaboration")
            if not theme_elaboration or not isinstance(theme_elaboration, str) or not theme_elaboration.strip():
                core_questions_str = '、'.join(parsed_response.get('core_research_questions_cn', ['未指定核心问题']))
                methodologies_str = '、'.join(parsed_response.get('potential_methodologies_cn', ['通用分析方法'])) # Now this should be a list
                theme_elaboration = (
                    f"本报告围绕主题“{parsed_response.get('generalized_topic_cn', user_topic)}”展开，"
                    f"核心研究问题包括：“{core_questions_str}”。"
                    f"分析中可能采用的方法有：“{methodologies_str}”。"
                )
                logger.info(f"Synthesized report_global_theme_elaboration. Theme: {theme_elaboration}")
                parsed_response["report_global_theme_elaboration"] = theme_elaboration
            # --- End of Key Handling and Validation Logic ---

            workflow_state.update_topic_analysis(parsed_response)
            workflow_state.update_report_global_theme(parsed_response["report_global_theme_elaboration"])

            workflow_state.add_task(
                task_type=TASK_TYPE_GENERATE_OUTLINE,
                payload={'topic_details': parsed_response},
                priority=2
            )

            self._log_output(parsed_response)
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
    import uuid
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            # Test case 1: Valid response
            if "ABMS系统" in query:
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
                        "人工智能在ABMS中的作用"
                    ]
                })
            # Test case 2: Malformed response with 'class' field
            if "MalformedTest" in query:
                return json.dumps({
                    "generalized_topic_cn": "测试主题（错误格式）",
                    "generalized_topic_en": "Test Topic (Malformed)",
                    "keywords_cn": ["测试", "错误"],
                    "keywords_en": ["test", "error"],
                    "core_research_questions_cn": ["这是一个核心问题吗？"],
                    "class": '"potential_methodologies_cn": ["方法A", null, "方法C"], "expanded_queries": ["查询X", 123, "查询Z"]'
                })
            # Test case 3: Recoverable keys entirely missing, no 'class' field
            if "MissingRecoverable" in query:
                return json.dumps({
                    "generalized_topic_cn": "测试主题（缺失可恢复键）",
                    "generalized_topic_en": "Test Topic (Missing Recoverable)",
                    "keywords_cn": ["测试", "缺失"],
                    "keywords_en": ["test", "missing"],
                    "core_research_questions_cn": ["核心问题存在吗？"]
                    # potential_methodologies_cn and expanded_queries are missing
                })
            # Test case 4: Strictly required key missing
            if "MissingStrict" in query:
                 return json.dumps({
                    "generalized_topic_en": "Test Topic (Missing Strict)",
                    "keywords_cn": ["测试", "缺失严格"],
                    "keywords_en": ["test", "missing strict"],
                    "core_research_questions_cn": ["这个能通过吗？"]
                    # generalized_topic_cn is missing
                })

            # Default fallback mock response
            return json.dumps({
                "generalized_topic_cn": "通用模拟主题",
                "generalized_topic_en": "General Mock Topic",
                "keywords_cn": ["模拟关键词1"],
                "keywords_en": ["Mock Keyword1"],
                "core_research_questions_cn": ["模拟核心问题1？"],
                "potential_methodologies_cn": ["模拟方法1"],
                "expanded_queries": ["模拟查询1"]
            })

    class MockWorkflowState(WorkflowState):
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.added_tasks_topic_analyzer = []

        def update_topic_analysis(self, results: Dict[str, Any]):
            super().update_topic_analysis(results)
            logger.debug(f"MockWorkflowState: Topic analysis updated with: {json.dumps(results, ensure_ascii=False, indent=2)}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_topic_analyzer.append({'type': task_type, 'payload': payload, 'priority': priority})
            logger.debug(f"MockWorkflowState (TopicAnalyzer): Task added - Type: {task_type}, Payload: {payload}")

    llm_service_instance = MockLLMService()
    analyzer_agent = TopicAnalyzerAgent(llm_service=llm_service_instance)

    def run_test(test_name: str, topic: str, should_fail: bool = False):
        mock_state = MockWorkflowState(user_topic=topic)
        mock_state.current_processing_task_id = f"{test_name}_{str(uuid.uuid4())[:4]}"
        task_payload_for_agent = {'user_topic': topic, 'priority':1}

        print(f"\n--- Executing TopicAnalyzerAgent Test: '{test_name}' for topic: '{topic}' ---")

        try:
            analyzer_agent.execute_task(mock_state, task_payload_for_agent)
            if should_fail:
                print(f"--- Test '{test_name}' FAILED: Expected an exception but none was raised. ---")
                return False # Indicate test failure

            print(f"  Topic Analysis Results: {json.dumps(mock_state.topic_analysis_results, indent=2, ensure_ascii=False)}")

            # Assertions to ensure keys exist and are of correct type (list of strings)
            for key_to_check in ["potential_methodologies_cn", "expanded_queries"]:
                assert key_to_check in mock_state.topic_analysis_results, f"Key '{key_to_check}' missing in results."
                assert isinstance(mock_state.topic_analysis_results[key_to_check], list), f"Key '{key_to_check}' is not a list."
                assert all(isinstance(i, str) for i in mock_state.topic_analysis_results[key_to_check]), f"Key '{key_to_check}' does not contain all strings."

            # Specific checks for MalformedTest
            if topic == "MalformedTest":
                assert "方法A" in mock_state.topic_analysis_results["potential_methodologies_cn"]
                assert "方法C" in mock_state.topic_analysis_results["potential_methodologies_cn"]
                assert len(mock_state.topic_analysis_results["potential_methodologies_cn"]) == 2 # null was removed
                assert "查询X" in mock_state.topic_analysis_results["expanded_queries"]
                assert "查询Z" in mock_state.topic_analysis_results["expanded_queries"]
                assert len(mock_state.topic_analysis_results["expanded_queries"]) == 2 # 123 was removed

            # Specific checks for MissingRecoverable
            if topic == "MissingRecoverable":
                assert mock_state.topic_analysis_results["potential_methodologies_cn"] == []
                assert mock_state.topic_analysis_results["expanded_queries"] == []

            print(f"--- Test '{test_name}' PASSED ---")
            return True
        except TopicAnalyzerAgentError as e:
            if should_fail:
                print(f"--- Test '{test_name}' PASSED as expected with error: {e} ---")
                return True
            else:
                print(f"--- Test '{test_name}' FAILED with TopicAnalyzerAgentError: {e} ---")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"--- Test '{test_name}' FAILED with unexpected error: {e} ---")
            import traceback
            traceback.print_exc()
            return False

    # Run all tests
    test_results = []
    test_results.append(run_test("ValidABMS", "介绍美国的ABMS系统"))
    test_results.append(run_test("MalformedClassField", "MalformedTest"))
    test_results.append(run_test("MissingRecoverableKeys", "MissingRecoverable"))
    test_results.append(run_test("MissingStrictKey", "MissingStrict", should_fail=True))
    test_results.append(run_test("GenericTopic", "另一个通用主题"))

    if all(test_results):
        print("\nAll TopicAnalyzerAgent tests passed successfully with robust key handling.")
    else:
        print("\nSome TopicAnalyzerAgent tests FAILED.")

    print("\nTopicAnalyzerAgent example finished.")

