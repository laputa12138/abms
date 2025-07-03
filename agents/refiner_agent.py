import logging
import json
import json_repair
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED # Import constants

logger = logging.getLogger(__name__)

class RefinerAgentError(Exception):
    """Custom exception for RefinerAgent errors."""
    pass

class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining (improving) chapter content based on
    evaluation feedback. Updates WorkflowState and queues re-evaluation.
    """

    # DEFAULT_PROMPT_TEMPLATE will be removed and sourced from settings

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="RefinerAgent", llm_service=llm_service)
        from config import settings as app_settings
        self.prompt_template = prompt_template or app_settings.DEFAULT_REFINER_PROMPT
        if not self.llm_service:
            raise RefinerAgentError("LLMService is required for RefinerAgent.")

    def _format_feedback(self, feedback_data: Dict[str, any]) -> str:
        """Formats the structured feedback from EvaluatorAgent into a string for the prompt."""
        score = feedback_data.get('score', 'N/A')
        feedback_text = feedback_data.get('feedback_cn', '无具体反馈文本。')
        criteria_met = feedback_data.get('evaluation_criteria_met', {})

        criteria_str = "\n具体评估标准反馈：\n"
        if isinstance(criteria_met, dict):
            for k, v in criteria_met.items():
                criteria_str += f"- {k}: {v}\n"
        else: # Handle case where it might not be a dict (e.g. if parsing failed or format changed)
            criteria_str += str(criteria_met) + "\n"

        return f"总体评分: {score}\n\n评审意见:\n{feedback_text}\n{criteria_str if criteria_met else ''}".strip()

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Refines chapter content based on evaluation feedback stored in WorkflowState.
        Updates WorkflowState with refined content and queues for re-evaluation.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload, expects 'chapter_key' and 'chapter_title'.
        """
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for chapter_key: {task_payload.get('chapter_key')}, title: {task_payload.get('chapter_title')}")

        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title')

        if not chapter_key or not chapter_title:
            err_msg = "Chapter key or title not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise RefinerAgentError(err_msg)

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data or not chapter_data.get('content'):
            err_msg = f"No original content to refine for chapter '{chapter_title}'. Skipping refinement."
            workflow_state.log_event(err_msg, level="WARNING") # Keep this
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Back to eval
            logger.warning(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='success') # Task done, but chapter reverted
            return

        original_content = chapter_data['content']
        evaluations = chapter_data.get('evaluations', [])
        if not evaluations:
            err_msg = f"No evaluation feedback found for chapter '{chapter_title}' to refine upon. Skipping refinement."
            workflow_state.log_event(err_msg, level="WARNING") # Keep this
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Re-evaluate
            logger.warning(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='success') # Task done, chapter reverted
            return

        latest_evaluation_feedback = evaluations[-1]
        num_prior_evals = len(evaluations) # This is also refinement attempt number

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Refining chapter '{chapter_title}' (Refinement attempt {num_prior_evals}). "
                    f"Original content length: {len(original_content)}, Last score: {latest_evaluation_feedback.get('score')}")
        self._log_input(chapter_key=chapter_key, original_content_length=len(original_content),
                        evaluation_feedback_score=latest_evaluation_feedback.get('score')) # Agent's own structured log

        formatted_feedback_str = self._format_feedback(latest_evaluation_feedback)

        prompt = self.prompt_template.format(
            chapter_title=chapter_title, # Add chapter_title to prompt for context
            original_content=original_content,
            evaluation_feedback=formatted_feedback_str
        )

        try:
            logger.info(f"Sending request to LLM for content refinement of chapter '{chapter_title}'.")
            refined_text = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一位经验丰富的编辑和内容优化师，擅长根据反馈精确改进文稿。"
            )
            logger.debug(f"Raw LLM response for refinement (first 200 chars): {refined_text[:200]}")

            refined_content_from_llm = None
            modification_notes_from_llm = None

            if refined_text and refined_text.strip():
                try:
                    # Attempt to parse the JSON response
                    parsed_response = json_repair.loads(refined_text)
                    refined_content_from_llm = parsed_response.get("refined_content")
                    modification_notes_from_llm = parsed_response.get("modification_notes")

                    if refined_content_from_llm:
                        logger.info(f"Successfully parsed refined content for chapter '{chapter_title}'.")
                        if modification_notes_from_llm:
                            logger.info(f"Modification notes for '{chapter_title}': {modification_notes_from_llm}")
                        else:
                            logger.warning(f"No modification notes found in LLM response for '{chapter_title}'.")
                    else:
                        logger.warning(f"LLM response for '{chapter_title}' parsed, but 'refined_content' key is missing or empty. Raw response: {refined_text}")
                        # Fallback to original content if 'refined_content' is missing
                        refined_content_from_llm = None

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response from LLM for chapter '{chapter_title}'. Error: {e}. Raw response: {refined_text}")
                    # Fallback to original content if JSON parsing fails
                    refined_content_from_llm = None
                except Exception as e: # Catch any other unexpected errors during parsing
                    logger.error(f"Unexpected error parsing LLM JSON response for chapter '{chapter_title}'. Error: {e}. Raw response: {refined_text}")
                    refined_content_from_llm = None
            else:
                logger.warning(f"LLM returned empty or whitespace-only response for '{chapter_title}'.")

            # Determine what content to store
            if refined_content_from_llm and refined_content_from_llm.strip():
                refined_text_to_store = refined_content_from_llm.strip()
            else:
                logger.warning(f"Using original content for chapter '{chapter_title}' due to issues with LLM response or parsing.")
                refined_text_to_store = original_content

            # Update WorkflowState with the refined content (as a new version)
            workflow_state.update_chapter_content(chapter_key, refined_text_to_store, is_new_version=True)
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Always re-evaluate after refinement

            # Add next task: Re-evaluate Chapter
            workflow_state.add_task(
                task_type=TASK_TYPE_EVALUATE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title},
                priority=task_payload.get('priority', 8) + 1
            )

            self._log_output({"chapter_key": chapter_key, "refined_content_length": len(refined_text_to_store)})
            success_msg = f"Refinement successful for '{chapter_title}'. Next task (Evaluate Chapter) added."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except LLMServiceError as e:
            err_msg = f"LLM service failed for chapter '{chapter_title}': {e}"
            workflow_state.log_event(f"LLM service error during refinement of chapter '{chapter_title}'", {"error": str(e)}, level="ERROR") # Keep
            workflow_state.add_chapter_error(chapter_key, f"LLM service error during refinement: {e}")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise RefinerAgentError(err_msg) # Re-raise
        except Exception as e:
            err_msg = f"Unexpected error refining chapter '{chapter_title}': {e}"
            workflow_state.log_event(f"Unexpected error in RefinerAgent for '{chapter_title}'", {"error": str(e)}, level="CRITICAL") # Keep
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error during refinement: e")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise RefinerAgentError(err_msg) # Re-raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def __init__(self, response_type="valid"):
            self.response_type = response_type

        def chat(self, query: str, system_prompt: str) -> str:
            original_content_part = "默认原始内容"
            if "原始内容：\n---\n" in query:
                try:
                    original_content_part = query.split("原始内容：\n---\n")[1].split("\n---")[0].strip()
                except IndexError:
                    pass # Keep default if parsing fails

            if self.response_type == "valid":
                return json.dumps({
                    "refined_content": f"{original_content_part}\n\n[LLM Mock Refinement: Valid JSON response. 针对反馈进行了修改。]",
                    "modification_notes": "这是有效的修改说明。"
                })
            elif self.response_type == "missing_content":
                return json.dumps({
                    "modification_notes": "内容字段丢失的修改说明。"
                })
            elif self.response_type == "missing_notes":
                return json.dumps({
                    "refined_content": f"{original_content_part}\n\n[LLM Mock Refinement: Notes missing. 针对反馈进行了修改。]"
                })
            elif self.response_type == "invalid_json":
                return "{'refined_content': 'bad json, not really', modification_notes: 'notes'}" # Malformed
            elif self.response_type == "empty_string":
                return ""
            elif self.response_type == "empty_json": # Valid JSON but empty content
                return json.dumps({
                    "refined_content": " ",
                    "modification_notes": "内容为空格的修改说明。"
                })
            return "（模拟的LLM无法处理此refinement请求 - 未知响应类型）"

    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_REFINEMENT_NEEDED, List, Any # Added List, Any

    class MockWorkflowStateRA(WorkflowState): # RA for RefinerAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, original_content: str, latest_eval: Dict):
            super().__init__(user_topic)
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_REFINEMENT_NEEDED,
                'content': original_content, 'retrieved_docs': [],
                'evaluations': [latest_eval], # Ensure there's an evaluation to refine upon
                'versions': [], 'errors': []
            }
            self.added_tasks_ra = []

        def update_chapter_content(self, chapter_key: str, content: str,
                                   retrieved_docs: Optional[List[Dict[str, Any]]] = None, # Added List type hint
                                   is_new_version: bool = True):
            super().update_chapter_content(chapter_key, content, retrieved_docs, is_new_version)
            logger.debug(f"MockWorkflowStateRA: Chapter '{chapter_key}' content updated by Refiner.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateRA: Chapter '{chapter_key}' status updated to {status} by Refiner.")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_ra.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)


    original_c = "This is the first draft. It has some issues."
    evaluation_f = {"score": 60, "feedback_cn": "Needs more examples and better flow.",
                    "evaluation_criteria_met": {"completeness": "不足"}}
    test_chap_key_refine_base = "chap_to_refine"
    test_chap_title_refine = "Chapter Being Refined"

    test_cases = [
        ("valid", f"{original_c}\n\n[LLM Mock Refinement: Valid JSON response. 针对反馈进行了修改。]", True, "Valid JSON test successful."),
        ("missing_content", original_c, False, "Missing 'refined_content' key test successful (fallback to original)."),
        ("missing_notes", f"{original_c}\n\n[LLM Mock Refinement: Notes missing. 针对反馈进行了修改。]", True, "Missing 'modification_notes' key test successful."),
        ("invalid_json", original_c, False, "Invalid JSON test successful (fallback to original)."),
        ("empty_string", original_c, False, "Empty string response test successful (fallback to original)."),
        ("empty_json", original_c, False, "Empty 'refined_content' in JSON test successful (fallback to original).")
    ]

    for i, (response_type, expected_content_part, expect_refined, success_message) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: response_type = '{response_type}' ---")

        current_chap_key = f"{test_chap_key_refine_base}_{response_type}"
        llm_service_instance = MockLLMService(response_type=response_type)
        refiner_agent = RefinerAgent(llm_service=llm_service_instance)

        mock_state_ra = MockWorkflowStateRA(
            f"Refinement Test - {response_type}",
            current_chap_key,
            test_chap_title_refine,
            original_c,
            evaluation_f
        )
        task_payload_for_agent_ra = {'chapter_key': current_chap_key, 'chapter_title': test_chap_title_refine}

        print(f"Executing RefinerAgent for chapter: '{test_chap_title_refine}' (key: {current_chap_key})")
        try:
            refiner_agent.execute_task(mock_state_ra, task_payload_for_agent_ra)

            chapter_info = mock_state_ra.get_chapter_data(current_chap_key)
            assert chapter_info is not None, "Chapter info should exist"

            print(f"  Content after refinement: '{chapter_info.get('content', '')[:150]}...'")
            print(f"  Status: {chapter_info.get('status')}")
            print(f"  Versions: {len(chapter_info.get('versions', []))}")

            assert chapter_info.get('status') == STATUS_EVALUATION_NEEDED, "Status should be STATUS_EVALUATION_NEEDED"

            actual_content = chapter_info.get('content', '')
            if expect_refined:
                assert expected_content_part in actual_content, f"Expected refined content part '{expected_content_part}' not in '{actual_content}'"
                assert actual_content != original_c, "Content should have been refined"
            else:
                assert actual_content == original_c, f"Expected original content, but got '{actual_content}'"

            assert len(mock_state_ra.added_tasks_ra) == 1, "One task should be added for re-evaluation"
            assert mock_state_ra.added_tasks_ra[0]['type'] == TASK_TYPE_EVALUATE_CHAPTER, "Task type should be TASK_TYPE_EVALUATE_CHAPTER"

            print(f"  {success_message}")

        except Exception as e:
            print(f"Error during RefinerAgent test case '{response_type}': {e}")
            import traceback
            traceback.print_exc()
            raise # Re-raise to fail the test run if any assertion fails

    print("\nAll RefinerAgent test cases passed successfully.")
    print("\nRefinerAgent example finished.")
