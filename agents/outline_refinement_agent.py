import logging
import json
from typing import Dict, Optional, List

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_APPLY_OUTLINE_REFINEMENT
from core.json_utils import clean_and_parse_json # Import the new helper

logger = logging.getLogger(__name__)

class OutlineRefinementAgentError(Exception):
    """Custom exception for OutlineRefinementAgent errors."""
    pass

class OutlineRefinementAgent(BaseAgent):
    """
    Agent responsible for suggesting refinements to an existing report outline.
    It uses an LLM to analyze the current outline and propose changes.
    """

    # DEFAULT_PROMPT_TEMPLATE is now moved to config.settings.py as DEFAULT_OUTLINE_REFINEMENT_PROMPT_CN

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="OutlineRefinementAgent", llm_service=llm_service)
        # Import the prompt from settings
        from config.settings import DEFAULT_OUTLINE_REFINEMENT_PROMPT_CN
        self.prompt_template = prompt_template or DEFAULT_OUTLINE_REFINEMENT_PROMPT_CN
        if not self.llm_service:
            raise OutlineRefinementAgentError("LLMService is required for OutlineRefinementAgent.")

    def _format_global_retrieved_docs_for_prompt(
        self,
        global_docs_map: Optional[Dict[str, List[Dict[str, any]]]],
        max_total_snippet_len: int = 4000  # 设置一个总的上下文长度限制
    ) -> str:
        """
        将全局检索到的所有文档整合成一个单一的、全面的上下文摘要，用于LLM prompt。
        这避免了将特定章节与特定文档片段直接关联，降低了LLM误解的风险。
        """
        if not global_docs_map:
            return "没有提供或找到全局检索信息。"

        all_docs = []
        # 从map中收集所有文档，并去重
        for docs in global_docs_map.values():
            for doc in docs:
                # 使用 'parent_id' 或 'child_id' 作为唯一标识符来去重
                doc_id = doc.get('parent_id') or doc.get('child_id')
                if doc_id and not any(d.get('parent_id') == doc_id or d.get('child_id') == doc_id for d in all_docs):
                    all_docs.append(doc)

        if not all_docs:
            return "没有可用的全局检索信息。"

        # 将所有文档内容拼接成一个大的上下文
        context_parts = [doc.get('document', '') for doc in all_docs]
        full_context = "\n\n---\n\n".join(filter(None, context_parts))

        # 截断到最大长度限制
        if len(full_context) > max_total_snippet_len:
            full_context = full_context[:max_total_snippet_len] + "..."
            logger.debug(f"Global retrieved context was truncated to {max_total_snippet_len} characters.")

        return full_context

    def _validate_suggestions(self, suggestions: List[Dict], parsed_outline: List[Dict]) -> List[Dict]:
        """Basic validation of suggestions from LLM."""
        if not isinstance(suggestions, list):
            logger.warning(f"LLM suggestions are not a list: {suggestions}")
            raise OutlineRefinementAgentError("LLM suggestions are not in the expected list format.")

        valid_suggestions = []
        existing_ids = {item['id'] for item in parsed_outline}

        for op in suggestions:
            if not isinstance(op, dict) or "action" not in op:
                logger.warning(f"Invalid operation format: {op}")
                continue

            action = op["action"]
            if action in ["delete", "modify_title", "modify_level", "move", "split"]:
                if "id" not in op or op["id"] not in existing_ids:
                    logger.warning(f"Operation '{action}' has missing or invalid id: {op.get('id')}")
                    continue
            if action == "add":
                if "title" not in op or "level" not in op:
                    logger.warning(f"Add operation missing title or level: {op}")
                    continue
                if op.get("after_id") and op["after_id"] not in existing_ids:
                    logger.warning(f"Add operation has invalid after_id: {op.get('after_id')}")
                    continue
            if action == "move":
                if op.get("after_id") and op["after_id"] not in existing_ids:
                     logger.warning(f"Move operation has invalid after_id: {op.get('after_id')}")
                     continue
            if action == "merge":
                if not ("primary_id" in op and op["primary_id"] in existing_ids and \
                        "secondary_id" in op and op["secondary_id"] in existing_ids):
                    logger.warning(f"Merge operation has missing or invalid primary_id or secondary_id: {op}")
                    continue
            if action == "split":
                if "new_chapters" not in op or not isinstance(op["new_chapters"], list) or not op["new_chapters"]:
                    logger.warning(f"Split operation has invalid new_chapters: {op}")
                    continue
                for new_chap in op["new_chapters"]:
                    if "title" not in new_chap or "level" not in new_chap:
                        logger.warning(f"Split operation's new_chapter missing title or level: {new_chap}")
                        # Mark the whole split op as problematic rather than trying to fix sub-parts
                        op = None # Invalidate this op
                        break
                if op is None: continue


            valid_suggestions.append(op)
        return valid_suggestions

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution.")

        current_outline_md = task_payload.get('current_outline_md')
        parsed_outline = task_payload.get('parsed_outline')
        topic_analysis_results = task_payload.get('topic_analysis_results', {})
        max_chapters = task_payload.get('max_chapters', 10)
        min_chapters = task_payload.get('min_chapters', 3)

        if not current_outline_md or not parsed_outline:
            err_msg = "Current outline (MD or parsed) not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineRefinementAgentError(err_msg)

        self._log_input(payload_keys=list(task_payload.keys()))

        topic_description = topic_analysis_results.get('generalized_topic_cn', workflow_state.user_topic)
        parsed_outline_json_str = json.dumps(parsed_outline, ensure_ascii=False, indent=2)

        # Fetch and format globally retrieved documents
        global_retrieved_docs = workflow_state.get_global_retrieved_docs_map()
        global_info_summary_str = self._format_global_retrieved_docs_for_prompt(
            global_retrieved_docs
        )
        logger.debug(f"Formatted global retrieved info summary for prompt:\n{global_info_summary_str[:500]}...")


        prompt = self.prompt_template.format(
            topic_description=topic_description,
            current_outline_md=current_outline_md,
            parsed_outline_json=parsed_outline_json_str,
            global_retrieved_info_summary=global_info_summary_str, # New field for prompt
            max_chapters=max_chapters,
            min_chapters=min_chapters
        )

        try:
            logger.info(f"Sending request to LLM for outline refinement suggestions. Topic: '{topic_description}'")
            llm_response_str = self.llm_service.chat(query=prompt, system_prompt="You are an AI assistant specializing in structuring and refining report outlines based on JSON instructions.")
            logger.debug(f"Raw LLM response for outline refinement: {llm_response_str}")

            if not llm_response_str or not llm_response_str.strip():
                logger.warning(f"[{self.agent_name}] LLM returned empty suggestions. Assuming no changes needed.")
                suggested_refinements = []
            else:
                try:
                    # Use the new robust JSON parsing helper
                    parsed_data = clean_and_parse_json(llm_response_str, context=f"outline_refinement_agent_task_{task_id}")

                    if parsed_data is not None:
                        suggested_refinements = self._validate_suggestions(parsed_data, parsed_outline)
                    else:
                        # clean_and_parse_json already logs the error with context
                        err_msg = f"Failed to decode or validate LLM JSON response for outline refinements after cleaning. Raw response: {llm_response_str[:500]}..."
                        logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
                        suggested_refinements = [] # Fallback to no refinements on parse error
                        workflow_state.log_event(f"LLM response for outline refinement was not valid JSON or failed validation.", {"llm_response": llm_response_str}, level="WARNING")

                except Exception as e_val: # Catch errors from _validate_suggestions or other unexpected issues
                    err_msg = f"Error processing LLM response after parsing for outline refinements: {e_val}. Raw response: {llm_response_str[:500]}..."
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
                    suggested_refinements = [] # Fallback to no refinements
                    workflow_state.log_event(f"Error processing LLM response for outline refinement.", {"error": str(e_val), "llm_response": llm_response_str}, level="ERROR")

            # Add task to apply these refinements
            apply_payload = {
                "original_outline_md": current_outline_md,
                "original_parsed_outline": parsed_outline,
                "suggested_refinements": suggested_refinements
            }
            workflow_state.add_task(
                task_type=TASK_TYPE_APPLY_OUTLINE_REFINEMENT,
                payload=apply_payload,
                priority=task_payload.get('priority', 2) # Apply should happen soon after suggestion
            )

            self._log_output({"num_suggestions": len(suggested_refinements)})
            success_msg = f"Outline refinement suggestions generated ({len(suggested_refinements)} operations). Task to apply refinements added."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except LLMServiceError as e:
            err_msg = f"LLM service failed for outline refinement: {e}"
            workflow_state.log_event(f"LLM service error during outline refinement suggestion", {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineRefinementAgentError(err_msg)
        except Exception as e:
            err_msg = f"Unexpected error in outline refinement suggestions: {e}"
            workflow_state.log_event(f"Unexpected error in OutlineRefinementAgent", {"error": str(e)}, level="CRITICAL")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineRefinementAgentError(err_msg)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def chat(self, query: str, system_prompt: str) -> str:
            # Simulate LLM returning a list of refinement operations as JSON string
            mock_suggestions = [
                {"action": "modify_title", "id": "ch_mock1", "new_title": "Revised Introduction"},
                {"action": "add", "title": "New Conclusion", "level": 1, "after_id": "ch_mock2"},
                {"action": "delete", "id": "ch_obsolete"}
            ]
            if "parsed_outline_json" in query: # Check if it's the refinement prompt
                 # Extract parsed_outline from the prompt to make suggestions more realistic
                try:
                    prompt_json_part = query.split("Current Outline (Parsed Structure with IDs):\n---\n")[1].split("\n---")[0]
                    parsed_outline_for_mock = json.loads(prompt_json_part)

                    # Make mock suggestions slightly dependent on input
                    suggestions = []
                    if any(item['id'] == 'ch_mock1' for item in parsed_outline_for_mock):
                         suggestions.append({"action": "modify_title", "id": "ch_mock1", "new_title": "Revised Mock Chapter 1 Title"})
                    if any(item['id'] == 'ch_mock2' for item in parsed_outline_for_mock):
                        suggestions.append({"action": "add", "title": "New Section after Mock 2", "level": 2, "after_id": "ch_mock2"})
                    if not suggestions: # Default if no specific IDs match
                         suggestions.append({"action": "add", "title": "Generic New Section", "level": 1, "after_id": None})
                    return json.dumps(suggestions)
                except Exception as e:
                    logger.error(f"MockLLMService error parsing prompt for suggestions: {e}")
                    return json.dumps([{"action": "add", "title": "Fallback New Section", "level": 1, "after_id": None}])
            return "[]" # Default empty list

    # Mock WorkflowState
    from core.workflow_state import WorkflowState, TASK_TYPE_APPLY_OUTLINE_REFINEMENT

    class MockWorkflowStateORA(WorkflowState): # ORA for OutlineRefinementAgent
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.added_tasks_ora = []

        def add_task(self, task_type: str, payload: Optional[Dict[str, any]] = None, priority: int = 0):
            self.added_tasks_ora.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority) # For logging or other side effects

    llm_service_instance = MockLLMService()
    outline_refinement_agent = OutlineRefinementAgent(llm_service=llm_service_instance)

    mock_user_topic = "Test Topic for Refinement"
    mock_state_ora = MockWorkflowStateORA(user_topic=mock_user_topic)

    initial_md_outline = "- Chapter 1 (ID: ch_mock1)\n- Chapter 2 (ID: ch_mock2)\n- Obsolete Chapter (ID: ch_obsolete)"
    initial_parsed_outline = [
        {"id": "ch_mock1", "title": "Chapter 1", "level": 1},
        {"id": "ch_mock2", "title": "Chapter 2", "level": 1},
        {"id": "ch_obsolete", "title": "Obsolete Chapter", "level": 1}
    ]
    mock_topic_analysis = {"generalized_topic_cn": "测试主题", "keywords_cn": ["测试"]}

    task_payload_for_agent_ora = {
        'current_outline_md': initial_md_outline,
        'parsed_outline': initial_parsed_outline,
        'topic_analysis_results': mock_topic_analysis,
        'max_chapters': 5,
        'min_chapters': 2,
        'priority': 2 # Example priority for the suggestion task itself
    }

    # Simulate current_processing_task_id being set by orchestrator
    mock_state_ora.current_processing_task_id = "mock_suggestion_task_id_123"

    # Simulate global retrieved docs being present in workflow state
    mock_global_docs = {
        "ch_mock1": [{"title": "Global Doc for Mock1", "text": "Some initial context for chapter 1..."}],
        "ch_mock2": [{"title": "Global Doc for Mock2", "text": "Preliminary findings for chapter 2..."}]
    }
    mock_state_ora.set_global_retrieved_docs_map(mock_global_docs)


    print(f"\nExecuting OutlineRefinementAgent with MockWorkflowStateORA (with global retrieved docs)")
    try:
        outline_refinement_agent.execute_task(mock_state_ora, task_payload_for_agent_ora)

        print("\nWorkflowState after OutlineRefinementAgent execution:")
        print(f"  Tasks added by agent: {json.dumps(mock_state_ora.added_tasks_ora, indent=2, ensure_ascii=False)}")

        assert len(mock_state_ora.added_tasks_ora) == 1
        added_apply_task = mock_state_ora.added_tasks_ora[0]
        assert added_apply_task['type'] == TASK_TYPE_APPLY_OUTLINE_REFINEMENT
        assert 'suggested_refinements' in added_apply_task['payload']
        assert added_apply_task['payload']['original_outline_md'] == initial_md_outline

        # Check if mock LLM produced expected suggestions based on input
        suggestions_in_payload = added_apply_task['payload']['suggested_refinements']
        assert any(s['action'] == 'modify_title' and s['id'] == 'ch_mock1' for s in suggestions_in_payload)
        assert any(s['action'] == 'add' and s['after_id'] == 'ch_mock2' for s in suggestions_in_payload)

        print("\nOutlineRefinementAgent test successful with MockWorkflowStateORA.")

    except Exception as e:
        print(f"Error during OutlineRefinementAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nOutlineRefinementAgent example finished.")
