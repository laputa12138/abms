import logging
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService
from core.workflow_state import WorkflowState, TASK_TYPE_COMPILE_REPORT
from config import settings as app_settings

logger = logging.getLogger(__name__)

class MissingContentResolutionAgentError(Exception):
    pass

class MissingContentResolutionAgent(BaseAgent):
    def __init__(self, llm_service: Optional[LLMService] = None,
                 # Dependencies for actual retry logic (to be expanded in future iterations)
                 # content_retriever_agent: Optional[ContentRetrieverAgent] = None,
                 # chapter_writer_agent: Optional[ChapterWriterAgent] = None
                 ):
        super().__init__(agent_name="MissingContentResolutionAgent", llm_service=llm_service)
        # self.content_retriever_agent = content_retriever_agent # For future retry logic
        # self.chapter_writer_agent = chapter_writer_agent # For future retry logic
        self.max_retry_attempts = getattr(app_settings, 'MISSING_CONTENT_RESOLUTION_MAX_RETRY_ATTEMPTS', 1)

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution to resolve missing chapter content.")
        self._log_input(payload_keys=list(task_payload.keys()))

        # Placeholder that ChapterWriterAgent might have used
        old_placeholder_text = "(本章节未能生成内容，因为没有相关的文本片段可供处理，或者所有片段处理均失败。)"
        # Current placeholder used by ChapterWriterAgent (should be the primary one to check)
        current_missing_content_placeholder = getattr(app_settings,
                                                       'DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER',
                                                       "[本章节未能生成有效内容，我们正在努力改进。]")

        # Final placeholder if resolution (including retries) fails
        final_placeholder_if_retry_fails = getattr(app_settings,
                                                   'DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER_RETRY_FAILED',
                                                   "[尽管我们已尝试补充，本章节当前仍未能生成足够内容。我们对此表示歉意并会持续改进。]")

        parsed_outline = workflow_state.parsed_outline
        if not parsed_outline:
            logger.warning(f"[{self.agent_name}] No parsed outline found in workflow state. Cannot resolve missing content.")
            if task_id: workflow_state.complete_task(task_id, "No outline to process for missing content resolution.", status='success')
            # Add task to compile report as there's nothing to resolve
            # Ensure this priority is correct relative to other final tasks
            workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=task_payload.get('priority', 100) + 10) # Slightly higher than default compile
            workflow_state.set_flag('missing_content_resolution_completed', True) # Mark as completed
            return

        chapters_found_with_placeholder = 0
        chapters_resolved_count = 0

        for chapter_item in parsed_outline:
            chapter_key = chapter_item['id']
            chapter_title = chapter_item['title']
            chapter_data = workflow_state.get_chapter_data(chapter_key)

            current_content = chapter_data.get('content', '').strip() if chapter_data else ""

            if chapter_data and (current_content == old_placeholder_text or current_content == current_missing_content_placeholder):
                chapters_found_with_placeholder += 1
                logger.info(f"Found chapter '{chapter_title}' (key: {chapter_key}) with placeholder content: '{current_content}'. Attempting resolution.")

                retry_successful = False
                # --- Placeholder for Retry Logic ---
                # In a future iteration, this section would involve:
                # 1. Calling a (potentially modified or more lenient) ContentRetrieverAgent.
                # 2. If new documents are found, calling a (potentially modified) ChapterWriterAgent.
                # 3. This loop would run up to self.max_retry_attempts.
                #
                # For now, this simulation assumes the retry mechanism is not yet fully implemented,
                # so `retry_successful` remains False, leading to the final placeholder.
                #
                # Example conceptual flow for a single retry attempt:
                # if self.content_retriever_agent and self.chapter_writer_agent: # Check if agents are injected
                #     try:
                #         logger.info(f"Attempting to regenerate chapter '{chapter_title}' (Attempt 1/{self.max_retry_attempts}).")
                #         # Placeholder: retrieve_payload = {'chapter_title': chapter_title, 'broad_query': True}
                #         # retrieved_info = self.content_retriever_agent.execute_task_for_retry(workflow_state, retrieve_payload)
                #         # if retrieved_info and retrieved_info.get('retrieved_docs'):
                #         #     write_payload = {'chapter_key': chapter_key, 'chapter_title': chapter_title, 'retrieved_docs': retrieved_info['retrieved_docs'], 'is_retry': True}
                #         #     new_content = self.chapter_writer_agent.execute_task_for_retry(workflow_state, write_payload)
                #         #     if new_content and new_content not in [old_placeholder_text, current_missing_content_placeholder, final_placeholder_if_retry_fails]:
                #         #         workflow_state.update_chapter_content(chapter_key, new_content, is_new_version=True, retrieved_docs=retrieved_info['retrieved_docs'])
                #         #         logger.info(f"Successfully regenerated content for chapter '{chapter_title}' on retry.")
                #         #         retry_successful = True
                #         #         chapters_resolved_count += 1
                #         # else:
                #         #     logger.info(f"No new documents retrieved on retry for chapter '{chapter_title}'.")
                #     except Exception as e_retry:
                #         logger.error(f"Error during retry attempt for chapter '{chapter_title}': {e_retry}", exc_info=True)
                # else:
                #    logger.warning("ContentRetrieverAgent or ChapterWriterAgent not available for retry logic in MissingContentResolutionAgent.")


                if not retry_successful:
                    logger.warning(f"Content resolution retry (currently simulated as failed) for chapter '{chapter_title}'. Replacing with final placeholder: '{final_placeholder_if_retry_fails}'")
                    workflow_state.update_chapter_content(chapter_key, final_placeholder_if_retry_fails, is_new_version=True)
                    workflow_state.add_chapter_error(chapter_key, f"Content resolution failed after {self.max_retry_attempts} attempt(s); replaced with final placeholder.")
            elif chapter_data and (current_content == final_placeholder_if_retry_fails):
                 logger.info(f"Chapter '{chapter_title}' (key: {chapter_key}) already has the final 'retry failed' placeholder. Skipping.")


        log_msg = (f"Missing content resolution scan complete. "
                   f"Found {chapters_found_with_placeholder} chapter(s) with initial placeholders. "
                   f"Successfully resolved by retry: {chapters_resolved_count} chapter(s). "
                   f"Chapters that might now have the 'retry_failed' placeholder: {chapters_found_with_placeholder - chapters_resolved_count}.")

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - {log_msg}")
        self._log_output({
            "chapters_scanned": len(parsed_outline),
            "chapters_found_with_initial_placeholder": chapters_found_with_placeholder,
            "chapters_content_resolved_by_retry": chapters_resolved_count
        })

        workflow_state.set_flag('missing_content_resolution_completed', True) # Crucial: signal completion

        # Always add compile report task after this agent runs.
        # Orchestrator will check 'missing_content_resolution_completed' flag before adding COMPILE_REPORT,
        # but this agent should also ensure it's queued if it's the last step before compilation.
        # However, the orchestrator logic is now set up to handle this transition.
        # So, just completing this task is enough. Orchestrator will pick up from there.

        if task_id: workflow_state.complete_task(task_id, log_msg, status='success')
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Execution finished. Orchestrator will now proceed to ReportCompilerAgent if conditions met.")

