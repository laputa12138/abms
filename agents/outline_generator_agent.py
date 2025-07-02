import logging
import json
from typing import Dict, Optional, List # Added List
import uuid # For generating chapter IDs

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_PROCESS_CHAPTER # Import constants
# It might be useful to have a temporary ReportCompiler instance for its parsing logic,
# or replicate a simplified parser here if ReportCompilerAgent isn't available yet or for decoupling.
from agents.report_compiler_agent import ReportCompilerAgent # Assuming it's available for parsing

logger = logging.getLogger(__name__)

class OutlineGeneratorAgentError(Exception):
    """Custom exception for OutlineGeneratorAgent errors."""
    pass

class OutlineGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating a report outline based on topic analysis results.
    Updates the WorkflowState with the generated outline (both MD and parsed structure)
    and adds tasks to process each chapter.
    """

    # DEFAULT_PROMPT_TEMPLATE will be removed and sourced from settings

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="OutlineGeneratorAgent", llm_service=llm_service)
        from config import settings as app_settings
        self.prompt_template = prompt_template or app_settings.DEFAULT_OUTLINE_GENERATOR_PROMPT
        if not self.llm_service:
            raise OutlineGeneratorAgentError("LLMService is required for OutlineGeneratorAgent.")
        # For parsing the generated Markdown outline into a structured list with IDs
        # We can use the parser from ReportCompilerAgent or a similar one here.
        self._outline_parser = ReportCompilerAgent() # Temporary instance for parsing

    def _parse_markdown_outline_with_ids(self, markdown_outline: str) -> List[Dict[str, any]]:
        """
        Parses a Markdown list outline and adds unique IDs to each item.
        Leverages ReportCompilerAgent's parsing logic or a similar internal parser.
        """
        # This uses the ReportCompilerAgent's internal parser which should assign IDs.
        # If that parser doesn't assign 'id', we might need to add them here.
        # The ReportCompilerAgent._parse_markdown_outline was updated to add 'id' using title as key
        # but for WorkflowState, we need truly unique IDs, especially if titles are not unique or change.

        # Let's use a simplified version of the parser that adds UUIDs for chapter keys.
        parsed_items = []
        lines = markdown_outline.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue

            level = 0
            title = line

            # Basic level detection (can be enhanced like in ReportCompilerAgent)
            if line.startswith("- ") or line.startswith("* ") or line.startswith("+ "):
                level = 1
                # Count leading spaces for sub-levels, assuming 2 spaces per indent
                temp_line_for_level = line
                while temp_line_for_level.startswith("  "):
                    level +=1
                    temp_line_for_level = temp_line_for_level[2:]
                title = temp_line_for_level.lstrip("-*+ ").strip()

            elif line.startswith("#"):
                level = line.count("#")
                title = line.lstrip("# ").strip()
            else:
                continue # Skip lines not recognized as outline items

            if title:
                parsed_items.append({
                    "id": f"ch_{str(uuid.uuid4())[:8]}", # Unique ID for this chapter/section
                    "title": title,
                    "level": level
                })
        return parsed_items


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Generates a report outline using the LLM based on topic_details from payload.
        Updates WorkflowState and adds tasks for chapter processing.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload for this task, expects 'topic_details'.
        """
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution.")

        analyzed_topic = task_payload.get('topic_details')
        if not analyzed_topic:
            err_msg = "Topic details not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineGeneratorAgentError(err_msg)

        self._log_input(analyzed_topic=analyzed_topic)

        required_keys = ["generalized_topic_cn", "generalized_topic_en", "keywords_cn", "keywords_en"]
        if not all(key in analyzed_topic for key in required_keys):
            err_msg = f"Invalid input: analyzed_topic missing keys: {required_keys}. Payload: {analyzed_topic}"
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineGeneratorAgentError(err_msg)

        topic_cn = analyzed_topic["generalized_topic_cn"]
        topic_en = analyzed_topic["generalized_topic_en"]
        keywords_cn_str = ", ".join(analyzed_topic["keywords_cn"])
        keywords_en_str = ", ".join(analyzed_topic["keywords_en"])

        prompt = self.prompt_template.format(
            topic_cn=topic_cn, topic_en=topic_en,
            keywords_cn=keywords_cn_str, keywords_en=keywords_en_str
        )

        try:
            logger.info(f"Sending request to LLM for outline generation. Topic: '{topic_cn}'")
            outline_markdown = self.llm_service.chat(query=prompt, system_prompt="你是一个专业的报告大纲规划师。")
            logger.debug(f"Raw LLM response for outline generation: {outline_markdown}")

            if not outline_markdown or not outline_markdown.strip():
                raise OutlineGeneratorAgentError("LLM returned an empty outline.")

            outline_markdown = outline_markdown.strip()

            # Parse the generated MD outline into a structured list with unique IDs
            # This parsed structure will be stored in workflow_state.parsed_outline
            # And chapter_data keys will be based on these IDs.
            parsed_outline_with_ids = self._parse_markdown_outline_with_ids(outline_markdown)
            if not parsed_outline_with_ids:
                 raise OutlineGeneratorAgentError("Failed to parse the generated Markdown outline into a structured format.")

            # Update WorkflowState
            workflow_state.update_outline(outline_markdown, parsed_outline_with_ids)
            # Ensure outline_finalized is False as it will go to refinement next
            workflow_state.set_flag('outline_finalized', False)

            # Add a task to suggest outline refinements
            refinement_payload = {
                "current_outline_md": outline_markdown,
                "parsed_outline": parsed_outline_with_ids, # Use the freshly parsed one
                "topic_analysis_results": task_payload.get('topic_details'), # Pass along topic details
                # Constraints can be sourced from config or passed in task_payload if dynamic
                "max_chapters": workflow_state.get_flag('max_chapters_constraint', 10), # Example
                "min_chapters": workflow_state.get_flag('min_chapters_constraint', 3)  # Example
            }
            # Import the constant if not already imported at the top
            from core.workflow_state import TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT
            workflow_state.add_task(
                task_type=TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT,
                payload=refinement_payload,
                priority=task_payload.get('priority', 2) + 1 # Slightly lower priority than generation
            )

            self._log_output({"markdown_outline": outline_markdown, "parsed_items_count": len(parsed_outline_with_ids)})
            success_msg = f"Initial outline generation successful for '{topic_cn}'. {len(parsed_outline_with_ids)} chapters. Task for outline refinement suggestion added."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except LLMServiceError as e:
            err_msg = f"LLM service failed for outline generation on topic '{topic_cn}': {e}"
            workflow_state.log_event(f"LLM service error during outline generation for '{topic_cn}'", {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineGeneratorAgentError(err_msg) # Re-raise
        except OutlineGeneratorAgentError as e:
            err_msg = f"Outline generation failed for '{topic_cn}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise # Re-raise
        except Exception as e:
            err_msg = f"Unexpected error in outline generation for '{topic_cn}': {e}"
            workflow_state.log_event(f"Unexpected error in OutlineGeneratorAgent for '{topic_cn}'", {"error": str(e)}, level="CRITICAL")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise OutlineGeneratorAgentError(err_msg) # Re-raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "ABMS系统" in query: return "- 章节一：ABMS概述\n  - 1.1 定义\n- 章节二：核心技术"
            return "- 默认章节1\n  - 默认子章节1.1"

    # Mock WorkflowState for testing
    from core.workflow_state import WorkflowState, TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT # Updated import

    class MockWorkflowStateOGA(WorkflowState): # OGA for OutlineGeneratorAgent
        def __init__(self, user_topic: str, topic_analysis_results: Dict):
            super().__init__(user_topic)
            self.topic_analysis_results = topic_analysis_results # Pre-populate for the agent
            self.updated_outline_md = None
            self.updated_parsed_outline = None
            self.added_tasks_oga = [] # Specific list for this agent's added tasks

        def update_outline(self, outline_md: str, parsed_outline: List[Dict[str, Any]]):
            self.updated_outline_md = outline_md
            self.updated_parsed_outline = parsed_outline
            super().update_outline(outline_md, parsed_outline) # Call parent for full update

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_oga.append({'type': task_type, 'payload': payload, 'priority': priority})
            logger.debug(f"MockWorkflowStateOGA: Task added - Type: {task_type}, Payload: {payload}")


    llm_service_instance = MockLLMService()
    outline_agent = OutlineGeneratorAgent(llm_service=llm_service_instance)

    mock_topic_analysis = {
        "generalized_topic_cn": "ABMS系统", "generalized_topic_en": "ABMS",
        "keywords_cn": ["ABMS", "JADC2"], "keywords_en": ["ABMS", "JADC2"]
    }
    # Simulate current_processing_task_id being set by orchestrator
    mock_state_oga = MockWorkflowStateOGA(user_topic="ABMS系统", topic_analysis_results=mock_topic_analysis)
    mock_state_oga.current_processing_task_id = "mock_outline_gen_task_id_123"


    task_payload_for_agent_oga = {'topic_details': mock_topic_analysis, 'priority': 2}


    print(f"\nExecuting OutlineGeneratorAgent with MockWorkflowStateOGA (New flow with Refinement Suggestion)")
    try:
        outline_agent.execute_task(mock_state_oga, task_payload_for_agent_oga)

        print("\nWorkflowState after OutlineGeneratorAgent execution:")
        print(f"  Outline Markdown: \n{mock_state_oga.current_outline_md}")
        print(f"  Parsed Outline (from state): {json.dumps(mock_state_oga.parsed_outline, indent=2, ensure_ascii=False)}")
        print(f"  Tasks added by agent: {json.dumps(mock_state_oga.added_tasks_oga, indent=2, ensure_ascii=False)}")
        print(f"  Outline Finalized Flag: {mock_state_oga.get_flag('outline_finalized')}")


        assert mock_state_oga.current_outline_md is not None
        assert mock_state_oga.parsed_outline is not None and len(mock_state_oga.parsed_outline) > 0
        assert mock_state_oga.get_flag('outline_finalized') is False # Should be false, refinement comes next

        assert len(mock_state_oga.added_tasks_oga) == 1 # Only one task should be added (suggest refinement)
        added_task_details = mock_state_oga.added_tasks_oga[0]
        assert added_task_details['type'] == TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT
        assert 'current_outline_md' in added_task_details['payload']
        assert 'parsed_outline' in added_task_details['payload']
        assert 'topic_analysis_results' in added_task_details['payload']
        assert added_task_details['payload']['parsed_outline'] == mock_state_oga.parsed_outline

        print("\nOutlineGeneratorAgent test successful with MockWorkflowStateOGA (New flow).")

    except Exception as e:
        print(f"Error during OutlineGeneratorAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nOutlineGeneratorAgent example finished.")
