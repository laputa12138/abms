import logging
import json # Keep for other JSON operations if any, or remove if only repair is used for loading
import json_repair # Added
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_REFINE_CHAPTER, STATUS_REFINEMENT_NEEDED, STATUS_COMPLETED, STATUS_ERROR # Import constants
from config import settings # For DEFAULT_MAX_REFINEMENT_ITERATIONS

logger = logging.getLogger(__name__)

class EvaluatorAgentError(Exception):
    """Custom exception for EvaluatorAgent errors."""
    pass

class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating generated chapter content based on predefined criteria.
    Updates WorkflowState with the evaluation and decides if refinement is needed,
    then queues the next task (refinement or marks chapter as complete).
    """

    # DEFAULT_PROMPT_TEMPLATE will be removed and sourced from settings
    # REFINEMENT_SCORE_THRESHOLD will now be taken from settings

    def __init__(self,
                 llm_service: LLMService,
                 prompt_template: Optional[str] = None):
        super().__init__(agent_name="EvaluatorAgent", llm_service=llm_service)
        from config import settings as app_settings
        self.prompt_template = prompt_template or app_settings.DEFAULT_EVALUATOR_PROMPT
        # refinement_threshold is read from settings within execute_task
        if not self.llm_service:
            raise EvaluatorAgentError("LLMService is required for EvaluatorAgent.")

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Evaluates chapter content from WorkflowState based on task_payload.
        Updates WorkflowState with evaluation and queues next task (refine or complete).

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
            raise EvaluatorAgentError(err_msg)

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data or not chapter_data.get('content'):
            err_msg = f"No content to evaluate for chapter '{chapter_title}' (Key: {chapter_key}). Marking as error."
            workflow_state.log_event(err_msg, level="ERROR") # Keep this
            workflow_state.add_chapter_error(chapter_key, "No content available for evaluation.")
            workflow_state.update_chapter_status(chapter_key, STATUS_ERROR)
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed') # Complete task as failed
            return

        content_to_evaluate = chapter_data['content']
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Evaluating content for chapter '{chapter_title}', length: {len(content_to_evaluate)}")
        self._log_input(chapter_key=chapter_key, content_length=len(content_to_evaluate)) # Agent's own structured log

        if not content_to_evaluate.strip():
            logger.warning(f"[{self.agent_name}] Task ID: {task_id} - Received empty content for chapter '{chapter_title}'. Assigning score 0.")
            evaluation_result = {"score": 0, "feedback_cn": "无法评估空内容。",
                                 "evaluation_criteria_met": {k: "无法评估" for k in ["relevance", "fluency", "completeness", "accuracy"]}}
        else:
            report_topic = workflow_state.user_topic or "未提供报告主题"

            # Prepare report_outline_summary
            outline_summary_parts = []
            if workflow_state.parsed_outline:
                for item in workflow_state.parsed_outline:
                    indent = "  " * (item.get('level', 1) - 1)
                    outline_summary_parts.append(f"{indent}- {item.get('title', '未命名章节')}")
            report_outline_summary = "\n".join(outline_summary_parts) if outline_summary_parts else "报告大纲不可用或为空。"

            # Prepare retrieved_references_summary
            # Using 'retrieved_docs' from the chapter_data, which should contain docs used for writing.
            # These are the most relevant for evaluating how well references were used for this specific chapter.
            chapter_retrieved_docs = chapter_data.get('retrieved_docs', [])
            references_summary_parts = []
            if chapter_retrieved_docs:
                for i, doc in enumerate(chapter_retrieved_docs[:5]): # Show summary of first 5 docs for brevity
                    doc_text_snippet = doc.get('document', '摘要不可用')[:100] # First 100 chars as snippet
                    source_name = doc.get('source_document_name', '未知来源')
                    references_summary_parts.append(f"- [{i+1}] 来自 '{source_name}': \"{doc_text_snippet}...\"")
            retrieved_references_summary = "\n".join(references_summary_parts) if references_summary_parts else "本章节无特定检索参考资料或摘要不可用。"

            prompt = self.prompt_template.format(
                report_topic=report_topic,
                report_outline_summary=report_outline_summary,
                chapter_title=chapter_title,
                retrieved_references_summary=retrieved_references_summary,
                content_to_evaluate=content_to_evaluate
            )
            try:
                logger.info(f"[{self.agent_name}] Task ID: {task_id} - Sending request to LLM for evaluation of chapter '{chapter_title}'. Prompt includes report_topic, outline_summary, references_summary.")
                raw_response = self.llm_service.chat(query=prompt, system_prompt="你是一个严格且公正的AI内容评审专家。")
                logger.debug(f"[{self.agent_name}] Task ID: {task_id} - Raw LLM response for evaluation: {raw_response}")

                try:
                    json_start_index = raw_response.find('{')
                    json_end_index = raw_response.rfind('}') + 1
                    if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                        json_string = raw_response[json_start_index:json_end_index]
                        evaluation_result = json_repair.loads(json_string)
                    else:
                        # If no clear JSON structure is found, try to repair the whole raw_response
                        logger.warning(f"No clear JSON object found in LLM eval response, attempting to repair entire response: {raw_response}")
                        evaluation_result = json_repair.loads(raw_response)
                except (json.JSONDecodeError, ValueError) as e: # json_repair can also raise ValueError
                    raise EvaluatorAgentError(f"LLM eval response not valid or repairable JSON: {raw_response}. Error: {e}")

                required_keys = ["score", "feedback_cn", "evaluation_criteria_met"]
                if not all(key in evaluation_result for key in required_keys):
                    raise EvaluatorAgentError(f"LLM eval response missing keys: {evaluation_result}")
                if not isinstance(evaluation_result.get("score"), int) or \
                   not isinstance(evaluation_result.get("feedback_cn"), str) or \
                   not isinstance(evaluation_result.get("evaluation_criteria_met"), dict):
                    raise EvaluatorAgentError(f"LLM eval response malformed types: {evaluation_result}")

            except LLMServiceError as e:
                err_msg = f"LLM service failed for chapter '{chapter_title}': {e}"
                workflow_state.log_event(f"LLM service error evaluating chapter '{chapter_title}'", {"error": str(e)}, level="ERROR") # Keep
                workflow_state.add_chapter_error(chapter_key, f"LLM service error during evaluation: {e}")
                logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
                if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
                raise EvaluatorAgentError(err_msg) # Re-raise
            except Exception as e:
                err_msg = f"Error processing LLM evaluation response for '{chapter_title}': {e}"
                workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR") # Keep
                workflow_state.add_chapter_error(chapter_key, f"Processing LLM evaluation response error: {e}")
                logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
                if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
                raise EvaluatorAgentError(err_msg) # Re-raise

        workflow_state.add_chapter_evaluation(chapter_key, evaluation_result)
        current_score = evaluation_result.get('score', 0)
        max_ref_iters = workflow_state.get_flag('max_refinement_iterations', settings.DEFAULT_MAX_REFINEMENT_ITERATIONS)
        # Count existing evaluations for this chapter to determine refinement attempts.
        # The current evaluation (just added) means this is evaluation attempt number N.
        # Refinement is needed if this is attempt N and N <= max_refinement_iterations.
        num_evaluation_attempts = len(workflow_state.get_chapter_data(chapter_key).get('evaluations', []))

        # Use threshold from settings
        current_refinement_threshold = settings.DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD

        decision_log_payload = {
            "chapter_key": chapter_key, "title": chapter_title, "score": current_score,
            "threshold": current_refinement_threshold, # Using value from settings
            "evaluation_attempts": num_evaluation_attempts,
            "max_refinement_iterations": max_ref_iters
        }

        if current_score < current_refinement_threshold and num_evaluation_attempts <= max_ref_iters:
            workflow_state.update_chapter_status(chapter_key, STATUS_REFINEMENT_NEEDED)
            workflow_state.add_task(
                task_type=TASK_TYPE_REFINE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title},
                priority=task_payload.get('priority', 7) + 1
            )
            log_msg = (f"Evaluation score {current_score} < {current_refinement_threshold}. "
                       f"Refinement attempt {num_evaluation_attempts}/{max_ref_iters}. Queuing REFINE task.")
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {log_msg} {decision_log_payload}")
            if task_id: workflow_state.complete_task(task_id, log_msg, status='success')
        else:
            final_status_reason = ""
            if current_score >= current_refinement_threshold:
                final_status_reason = f"Score {current_score} >= threshold {current_refinement_threshold}."
            else:
                final_status_reason = (f"Score {current_score} < threshold {current_refinement_threshold}, "
                                       f"but max refinement attempts ({num_evaluation_attempts}/{max_ref_iters}) reached.")

            workflow_state.update_chapter_status(chapter_key, STATUS_COMPLETED)
            log_msg = f"Chapter '{chapter_title}' processing complete. {final_status_reason} Marking chapter COMPLETED."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {log_msg} {decision_log_payload}")
            if task_id: workflow_state.complete_task(task_id, log_msg, status='success')

        self._log_output(evaluation_result) # Agent's own structured log

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "Chapter Title: Good Chapter" in query:
                return json.dumps({"score": 90, "feedback_cn": "内容优秀，无需修改。", "evaluation_criteria_met": {"relevance": "高", "fluency": "优秀", "completeness": "全面", "accuracy": "高"}})
            elif "Chapter Title: Needs Work Chapter" in query:
                return json.dumps({"score": 65, "feedback_cn": "内容有一些问题，需要改进。", "evaluation_criteria_met": {"relevance": "中", "fluency": "一般", "completeness": "部分", "accuracy": "待核实"}})
            return json.dumps({"score": 50, "feedback_cn": "默认评估。", "evaluation_criteria_met": {}})

    from core.workflow_state import WorkflowState, TASK_TYPE_REFINE_CHAPTER, STATUS_COMPLETED, STATUS_REFINEMENT_NEEDED, STATUS_EVALUATION_NEEDED
    from config import settings # For default max iterations

    class MockWorkflowStateEA(WorkflowState): # EA for EvaluatorAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, content: str):
            super().__init__(user_topic)
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_EVALUATION_NEEDED,
                'content': content, 'retrieved_docs': [], 'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_ea = []
            self.max_ref_iter_config = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS # Store this for test

        def add_task(self, task_type: str, payload: Optional[Dict[str, any]] = None, priority: int = 0):
            self.added_tasks_ea.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)

        def get_flag(self, flag_name: str, default: Optional[any] = None) -> any:
            if flag_name == 'max_refinement_iterations': return self.max_ref_iter_config
            return super().get_flag(flag_name, default)


    llm_service_instance = MockLLMService()
    evaluator_agent = EvaluatorAgent(llm_service=llm_service_instance, refinement_threshold=80) # Using 80 as threshold

    # Test case 1: Good content, should complete
    chap_key_good = "chap_good"
    chap_title_good = "Good Chapter"
    content_good = "This is some excellent content for the good chapter."
    mock_state_ea_good = MockWorkflowStateEA("Test Topic", chap_key_good, chap_title_good, content_good)
    task_payload_good = {'chapter_key': chap_key_good, 'chapter_title': chap_title_good}

    print(f"\nExecuting EvaluatorAgent for '{chap_title_good}' (expected to complete)")
    try:
        evaluator_agent.execute_task(mock_state_ea_good, task_payload_good)
        chapter_info = mock_state_ea_good.get_chapter_data(chap_key_good)
        print(f"  Chapter '{chap_key_good}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations: {json.dumps(chapter_info.get('evaluations'), indent=2, ensure_ascii=False)}")
        print(f"  Tasks added: {json.dumps(mock_state_ea_good.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_COMPLETED
        assert not mock_state_ea_good.added_tasks_ea # No refinement task for good content
    except Exception as e: print(f"Error: {e}")

    # Test case 2: Needs work, should queue refinement
    chap_key_needs_work = "chap_needs_work"
    chap_title_needs_work = "Needs Work Chapter"
    content_needs_work = "This content needs some improvement."
    mock_state_ea_needs_work = MockWorkflowStateEA("Test Topic", chap_key_needs_work, chap_title_needs_work, content_needs_work)
    task_payload_needs_work = {'chapter_key': chap_key_needs_work, 'chapter_title': chap_title_needs_work}

    print(f"\nExecuting EvaluatorAgent for '{chap_title_needs_work}' (expected to queue refinement)")
    try:
        evaluator_agent.execute_task(mock_state_ea_needs_work, task_payload_needs_work)
        chapter_info = mock_state_ea_needs_work.get_chapter_data(chap_key_needs_work)
        print(f"  Chapter '{chap_key_needs_work}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations: {json.dumps(chapter_info.get('evaluations'), indent=2, ensure_ascii=False)}")
        print(f"  Tasks added: {json.dumps(mock_state_ea_needs_work.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_REFINEMENT_NEEDED
        assert len(mock_state_ea_needs_work.added_tasks_ea) == 1
        assert mock_state_ea_needs_work.added_tasks_ea[0]['type'] == TASK_TYPE_REFINE_CHAPTER
    except Exception as e: print(f"Error: {e}")

    # Test case 3: Needs work, but max refinements reached (simulate one prior eval)
    mock_state_ea_max_ref = MockWorkflowStateEA("Test Topic", chap_key_needs_work, chap_title_needs_work, content_needs_work)
    mock_state_ea_max_ref.max_ref_iter_config = 1 # Set max iterations to 1 for this test state
    # Simulate a previous evaluation already happened
    mock_state_ea_max_ref.chapter_data[chap_key_needs_work]['evaluations'].append({"score": 60, "feedback_cn": "First attempt was not good."})

    print(f"\nExecuting EvaluatorAgent for '{chap_title_needs_work}' (max refinements reached)")
    try:
        evaluator_agent.execute_task(mock_state_ea_max_ref, task_payload_needs_work)
        chapter_info = mock_state_ea_max_ref.get_chapter_data(chap_key_needs_work)
        print(f"  Chapter '{chap_key_needs_work}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations (count: {len(chapter_info.get('evaluations'))}): {json.dumps(chapter_info.get('evaluations')[-1], indent=2, ensure_ascii=False)}") # Show last eval
        print(f"  Tasks added: {json.dumps(mock_state_ea_max_ref.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_COMPLETED # Should be completed despite low score due to max_iterations
        assert not mock_state_ea_max_ref.added_tasks_ea # No new refinement task
    except Exception as e: print(f"Error: {e}")


    print("\nEvaluatorAgent example finished.")
