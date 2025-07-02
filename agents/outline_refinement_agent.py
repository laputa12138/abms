import logging
import json
from typing import Dict, Optional, List

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_APPLY_OUTLINE_REFINEMENT

logger = logging.getLogger(__name__)

class OutlineRefinementAgentError(Exception):
    """Custom exception for OutlineRefinementAgent errors."""
    pass

class OutlineRefinementAgent(BaseAgent):
    """
    Agent responsible for suggesting refinements to an existing report outline.
    It uses an LLM to analyze the current outline and propose changes.
    """

    DEFAULT_PROMPT_TEMPLATE = """\
You are an expert outline reviewer and refiner. Your task is to review the provided report outline and suggest specific improvements.
The report is about: {topic_description}

Current Outline (Markdown format):
---
{current_outline_md}
---

Current Outline (Parsed Structure with IDs):
---
{parsed_outline_json}
---

Globally Retrieved Information (Preliminary context for each chapter based on titles):
---
{global_retrieved_info_summary}
---

Based on the current outline AND the globally retrieved information, please suggest refinements to make the outline more logical, comprehensive, coherent, and well-structured.
Pay attention to:
- Whether the retrieved information suggests missing subtopics or new relevant chapters.
- Whether any chapters seem to have very little supporting information, perhaps indicating they are too niche or could be merged.
- Whether information for different chapters seems highly overlapping, suggesting potential merges or restructuring.

Consider the following types of changes:
- Add new chapters or sub-chapters where important information seems missing (especially if suggested by retrieved content).
- Delete chapters or sub-chapters that are redundant, irrelevant, or too granular.
- Modify titles of chapters or sub-chapters to be clearer, more concise, or more impactful.
- Reorder chapters or sub-chapters for better flow and logical progression.
- Merge chapters that are too similar or cover overlapping content.
- Split chapters that are too broad or cover multiple distinct topics.
- Adjust levels (indentation) of chapters for proper hierarchy.

Constraints (if any):
- Maximum chapters: {max_chapters}
- Minimum chapters: {min_chapters}

Provide your suggestions as a JSON list of operations. Each operation should be an object with an "action" key and other necessary keys.
Supported actions and their formats:
1.  `{{ "action": "add", "title": "New Chapter Title", "level": <level_num>, "after_id": "<id_of_chapter_before_it_or_null>" }}` (if after_id is null, appends to the end of that level or overall outline)
2.  `{{ "action": "delete", "id": "<chapter_id_to_delete>" }}`
3.  `{{ "action": "modify_title", "id": "<chapter_id_to_modify>", "new_title": "Revised Title" }}`
4.  `{{ "action": "modify_level", "id": "<chapter_id_to_modify>", "new_level": <level_num> }}`
5.  `{{ "action": "move", "id": "<chapter_id_to_move>", "after_id": "<id_of_chapter_to_move_it_after_or_null>" }}` (if after_id is null, move to beginning of its level or overall outline)
6.  `{{ "action": "merge", "primary_id": "<chapter_id_to_merge_into>", "secondary_id": "<chapter_id_to_be_merged_and_deleted>", "new_title_for_primary": "Optional new title" }}`
7.  `{{ "action": "split", "id": "<chapter_id_to_split>", "new_chapters": [{{ "title": "Part 1", "level": <level_num> }}, {{ "title": "Part 2", "level": <level_num> }}] }}` (original chapter with 'id' will be deleted, new chapters get new IDs)

If no refinements are needed, return an empty JSON list: `[]`.

JSON Output of Suggested Refinements:
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="OutlineRefinementAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise OutlineRefinementAgentError("LLMService is required for OutlineRefinementAgent.")

    def _format_global_retrieved_docs_for_prompt(
        self,
        parsed_outline: List[Dict[str, any]],
        global_docs_map: Optional[Dict[str, List[Dict[str, any]]]],
        max_docs_per_chapter_summary: int = 2,
        max_text_snippet_len: int = 100 # Max length of text snippet from each doc
    ) -> str:
        """
        Formats the globally retrieved documents into a string summary for the LLM prompt.
        """
        if not global_docs_map:
            return "No globally retrieved information was provided or found."

        summary_lines = []
        for chapter_item in parsed_outline:
            chapter_id = chapter_item.get('id')
            chapter_title = chapter_item.get('title', 'Untitled Chapter')

            summary_lines.append(f"\nFor Chapter: \"{chapter_title}\" (ID: {chapter_id})")

            docs_for_chapter = global_docs_map.get(chapter_id)
            if docs_for_chapter:
                for i, doc in enumerate(docs_for_chapter[:max_docs_per_chapter_summary]):
                    doc_title = doc.get('title', f"Document {i+1}")
                    doc_text_snippet = doc.get('text', '')[:max_text_snippet_len]
                    if doc.get('text') and len(doc.get('text')) > max_text_snippet_len:
                        doc_text_snippet += "..."
                    summary_lines.append(f"  - Retrieved Doc: \"{doc_title}\"\n    Snippet: \"{doc_text_snippet}\"")
            else:
                summary_lines.append("  - No specific documents retrieved for this chapter title in the global pass.")

        return "\n".join(summary_lines).strip()

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
            parsed_outline,
            global_retrieved_docs
        )
        logger.debug(f"Formatted global retrieved info summary for prompt:\n{global_info_summary_str}")


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
                    # The LLM is asked to return JSON directly.
                    # Strip potential markdown code fences if LLM wraps JSON in them
                    if llm_response_str.strip().startswith("```json"):
                        llm_response_str = llm_response_str.strip()[7:]
                        if llm_response_str.endswith("```"):
                             llm_response_str = llm_response_str[:-3]
                    llm_response_str = llm_response_str.strip()

                    suggested_refinements = json.loads(llm_response_str)
                    suggested_refinements = self._validate_suggestions(suggested_refinements, parsed_outline)
                except json.JSONDecodeError as e:
                    err_msg = f"Failed to decode LLM JSON response for outline refinements: {e}. Response: {llm_response_str}"
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
                    # Potentially try to parse with more resilient methods or return empty list
                    suggested_refinements = [] # Fallback to no refinements on parse error
                    workflow_state.log_event(f"LLM response for outline refinement was not valid JSON.", {"error": str(e), "llm_response": llm_response_str}, level="WARNING")


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
