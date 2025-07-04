import logging
import json # Added for logging incomplete chapters
from typing import Dict, Any, Optional

from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT, STATUS_COMPLETED, STATUS_ERROR, STATUS_PENDING
# Import all agent classes
from agents.topic_analyzer_agent import TopicAnalyzerAgent
from agents.outline_generator_agent import OutlineGeneratorAgent
from agents.content_retriever_agent import ContentRetrieverAgent
from agents.chapter_writer_agent import ChapterWriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.refiner_agent import RefinerAgent
from agents.report_compiler_agent import ReportCompilerAgent
from agents.outline_refinement_agent import OutlineRefinementAgent
from agents.global_content_retriever_agent import GlobalContentRetrieverAgent
from agents.missing_content_resolution_agent import MissingContentResolutionAgent # New Agent
from core.workflow_state import TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT, \
    TASK_TYPE_APPLY_OUTLINE_REFINEMENT, TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE, \
    TASK_TYPE_RESOLVE_MISSING_CONTENT # New Task Type

logger = logging.getLogger(__name__)

class OrchestratorError(Exception):
    """Custom exception for Orchestrator errors."""
    pass

class Orchestrator:
    """
    Drives the report generation workflow by managing tasks from WorkflowState
    and dispatching them to appropriate agents.
    """
    def __init__(self,
                 workflow_state: WorkflowState,
                 topic_analyzer: TopicAnalyzerAgent,
                 outline_generator: OutlineGeneratorAgent,
                 global_content_retriever: GlobalContentRetrieverAgent,
                 outline_refiner: OutlineRefinementAgent,
                 content_retriever: ContentRetrieverAgent,
                 chapter_writer: ChapterWriterAgent,
                 evaluator: EvaluatorAgent,
                 refiner: RefinerAgent,
                 report_compiler: ReportCompilerAgent,
                 missing_content_resolver: MissingContentResolutionAgent, # New
                 max_workflow_iterations: int = 50,
                ):
        self.workflow_state = workflow_state
        self.agents = {
            TASK_TYPE_ANALYZE_TOPIC: topic_analyzer,
            TASK_TYPE_GENERATE_OUTLINE: outline_generator,
            TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE: global_content_retriever,
            TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT: outline_refiner,
            # TASK_TYPE_APPLY_OUTLINE_REFINEMENT is handled by Orchestrator directly
            # TASK_TYPE_PROCESS_CHAPTER is a meta-task, handled by adding retrieve then write
            TASK_TYPE_RETRIEVE_FOR_CHAPTER: content_retriever,
            TASK_TYPE_WRITE_CHAPTER: chapter_writer,
            TASK_TYPE_EVALUATE_CHAPTER: evaluator,
            TASK_TYPE_REFINE_CHAPTER: refiner,
            TASK_TYPE_RESOLVE_MISSING_CONTENT: missing_content_resolver, # New mapping
            TASK_TYPE_COMPILE_REPORT: report_compiler,
        }
        self.max_workflow_iterations = max_workflow_iterations
        logger.info("Orchestrator initialized with all agents (including MissingContentResolutionAgent) and workflow state.")

    def _execute_task_type(self, task: Dict[str, Any]):
        """Executes a specific task by calling the appropriate agent."""
        task_type = task['type']
        task_id = task['id']
        payload = task.get('payload', {})

        agent = self.agents.get(task_type)

        if agent:
            try:
                self.workflow_state.log_event(f"Orchestrator dispatching task '{task_type}' to agent '{agent.agent_name}'.",
                                             {"task_id": task_id, "payload": payload})
                agent.execute_task(self.workflow_state, payload)
                # Agents are responsible for calling workflow_state.complete_task for their own tasks.
            except Exception as e:
                logger.error(f"Error executing task {task_type} ({task_id}) with agent {getattr(agent, 'agent_name', 'UnknownAgent')}: {e}", exc_info=True)
                self.workflow_state.log_event(f"Agent execution error for task {task_type} ({task_id})",
                                             {"error": str(e), "agent": getattr(agent, 'agent_name', 'UnknownAgent')})
                self.workflow_state.complete_task(task_id, f"Agent failed: {e}", status='failed')
                if 'chapter_key' in payload:
                    self.workflow_state.add_chapter_error(payload['chapter_key'], f"Agent {task_type} failed: {e}")

        elif task_type == TASK_TYPE_PROCESS_CHAPTER:
            chapter_key = payload['chapter_key']
            self.workflow_state.update_chapter_status(chapter_key, STATUS_PENDING)
            self.workflow_state.add_task(TASK_TYPE_RETRIEVE_FOR_CHAPTER,
                                         payload=payload,
                                         priority=task.get('priority', 3))
            self.workflow_state.complete_task(task_id, f"PROCESS_CHAPTER task for '{payload.get('chapter_title')}' initiated retrieval.")

        elif task_type == TASK_TYPE_APPLY_OUTLINE_REFINEMENT:
            self._handle_apply_outline_refinement(task_id, payload)

        else:
            logger.warning(f"No agent or direct handler registered for task type: {task_type} (task_id: {task_id}).")
            self.workflow_state.log_event(f"Unknown task type: {task_type}", {"task_id": task_id})
            self.workflow_state.complete_task(task_id, f"Unknown task type {task_type}", status='failed')

    def _handle_apply_outline_refinement(self, task_id: str, payload: Dict[str, Any]):
        """
        Handles the application of suggested outline refinements.
        Updates the workflow state's outline and manages chapter tasks accordingly.
        """
        logger.info(f"Orchestrator handling TASK_TYPE_APPLY_OUTLINE_REFINEMENT (Task ID: {task_id}).")
        original_parsed_outline = payload.get('original_parsed_outline')
        suggested_refinements = payload.get('suggested_refinements')

        if not original_parsed_outline or suggested_refinements is None:
            err_msg = "Missing original_parsed_outline or suggested_refinements in payload for APPLY_OUTLINE_REFINEMENT."
            logger.error(err_msg)
            self.workflow_state.log_event(err_msg, {"task_id": task_id})
            self.workflow_state.complete_task(task_id, err_msg, status='failed')
            return

        original_ids = {item['id'] for item in original_parsed_outline}
        new_parsed_outline = self.workflow_state.apply_refinements_to_parsed_outline(
            original_parsed_outline,
            suggested_refinements
        )
        new_outline_md = self.workflow_state.generate_markdown_from_parsed_outline(new_parsed_outline)
        self.workflow_state.update_outline(new_outline_md, new_parsed_outline)
        logger.info(f"Outline updated after applying {len(suggested_refinements)} refinements. New chapter count: {len(new_parsed_outline)}")
        self.workflow_state.log_event("Outline updated via APPLY_OUTLINE_REFINEMENT.",
                                     {"num_refinements": len(suggested_refinements),
                                      "new_chapter_count": len(new_parsed_outline)})

        new_ids = {item['id'] for item in new_parsed_outline}
        deleted_ids = original_ids - new_ids
        if deleted_ids:
            logger.info(f"Chapters to remove tasks for (deleted): {deleted_ids}")
            tasks_to_remove = []
            for task_in_queue in self.workflow_state.pending_tasks:
                chap_key = task_in_queue.get('payload', {}).get('chapter_key')
                if chap_key and chap_key in deleted_ids:
                    tasks_to_remove.append(task_in_queue)

            for task_to_remove in tasks_to_remove:
                self.workflow_state.pending_tasks.remove(task_to_remove) # Direct removal, ensure thread-safety if applicable
                logger.info(f"Removed pending task {task_to_remove['id']} ({task_to_remove['type']}) for deleted chapter {task_to_remove.get('payload', {}).get('chapter_key')}")
                self.workflow_state.log_event(f"Task removed for deleted chapter.", {"removed_task_id": task_to_remove['id'], "chapter_id": task_to_remove.get('payload', {}).get('chapter_key')})

        self.workflow_state.pending_tasks = [
            pt for pt in self.workflow_state.pending_tasks
            if pt['type'] not in [TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER]
        ]
        logger.info("Cleared existing chapter processing tasks before adding new ones for the refined outline.")
        self.workflow_state.log_event("Cleared pending chapter-specific tasks before re-adding for refined outline.")

        for item in new_parsed_outline:
            chapter_key = item['id']
            chapter_title = item['title']
            self.workflow_state.add_task(
                task_type=TASK_TYPE_PROCESS_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title, 'level': item['level']},
                priority=3
            )
        logger.info(f"Added/Re-added PROCESS_CHAPTER tasks for all {len(new_parsed_outline)} chapters in the refined outline.")

        self.workflow_state.set_flag('outline_finalized', True)
        logger.info("Outline marked as finalized after applying refinements.")
        self.workflow_state.log_event("Outline finalized after APPLY_OUTLINE_REFINEMENT.",
                                     {"final_chapter_count": len(new_parsed_outline)})

        self.workflow_state.complete_task(task_id, f"Outline refinements applied. {len(new_parsed_outline)} chapters in new outline. Process tasks added.", status='success')


    def coordinate_workflow(self) -> None:
        """
        Main loop to coordinate the workflow by processing tasks from WorkflowState.
        """
        self.workflow_state.log_event("Orchestrator starting workflow coordination.")
        iteration_count = 0
        stall_patience_counter = 0
        STALL_PATIENCE_THRESHOLD = 5

        while not self.workflow_state.get_flag('report_generation_complete', False):
            if iteration_count >= self.max_workflow_iterations:
                self.workflow_state.log_event("Max workflow iterations reached by Orchestrator. Halting.", {"level": "ERROR"})
                self.workflow_state.set_flag('report_generation_complete', True)
                break

            task = self.workflow_state.get_next_task()

            if not task:
                stall_patience_counter += 1
                self.workflow_state.log_event(f"Task queue empty. Stall patience: {stall_patience_counter}/{STALL_PATIENCE_THRESHOLD}.")

                all_chapters_processed_flag = self.workflow_state.are_all_chapter_tasks_processed_or_terminal()
                outline_finalized_flag = self.workflow_state.get_flag('outline_finalized', False)

                # Check if it's time to run MissingContentResolutionAgent
                if all_chapters_processed_flag and \
                   outline_finalized_flag and \
                   not self.workflow_state.get_flag('missing_content_resolution_requested', False) and \
                   not self.workflow_state.get_flag('report_compilation_requested', False): # Ensure it runs before compilation

                    self.workflow_state.add_task(TASK_TYPE_RESOLVE_MISSING_CONTENT, payload={}, priority=90) # High priority, before compile
                    self.workflow_state.set_flag('missing_content_resolution_requested', True)
                    self.workflow_state.log_event("All chapter processing seems complete. Missing Content Resolution task added by Orchestrator.")
                    stall_patience_counter = 0

                # Check if it's time to compile the report (after missing content resolution)
                elif all_chapters_processed_flag and \
                     outline_finalized_flag and \
                     self.workflow_state.get_flag('missing_content_resolution_completed', False) and \
                     not self.workflow_state.get_flag('report_compilation_requested', False):

                    self.workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100) # Highest priority
                    self.workflow_state.set_flag('report_compilation_requested', True)
                    self.workflow_state.log_event("All chapters processed and missing content resolution done. Compilation task added by Orchestrator.")
                    stall_patience_counter = 0

                elif self.workflow_state.get_flag('report_generation_complete'):
                    break
                elif stall_patience_counter >= STALL_PATIENCE_THRESHOLD:
                    logger.warning(f"Orchestrator: Potential stall detected after {stall_patience_counter} iterations with empty queue. Halting.")
                    self.workflow_state.log_event(
                        "Stall detected: Task queue empty for too long and report not complete.",
                        {
                            "all_chapters_processed": all_chapters_processed_flag,
                            "outline_finalized": outline_finalized_flag,
                            "missing_content_resolution_requested": self.workflow_state.get_flag('missing_content_resolution_requested'),
                            "missing_content_resolution_completed": self.workflow_state.get_flag('missing_content_resolution_completed'),
                            "report_compilation_requested": self.workflow_state.get_flag('report_compilation_requested')
                        }
                    )
                    if not self.workflow_state.are_all_chapters_completed(): # This uses the old check, might need update too
                        incomplete_chapters = []
                        if self.workflow_state.parsed_outline: # Check if outline exists
                            for item in self.workflow_state.parsed_outline:
                                ch_key = item['id']
                                ch_data = self.workflow_state.get_chapter_data(ch_key)
                                ch_status = ch_data.get('status', 'NO_DATA') if ch_data else 'NO_DATA_FOR_KEY'
                                if ch_status != STATUS_COMPLETED:
                                    incomplete_chapters.append({
                                        'key': ch_key,
                                        'title': item.get('title', 'N/A'),
                                        'status': ch_status
                                    })
                            if incomplete_chapters:
                                logger.error(f"Orchestrator: Stall detected. Incomplete chapters: {json.dumps(incomplete_chapters, ensure_ascii=False, indent=2)}")
                                self.workflow_state.log_event("Stall details: Incomplete chapters logged.", {"incomplete_chapters_summary": incomplete_chapters})
                        else:
                            logger.error("Orchestrator: Stall detected. Parsed outline is missing, cannot list incomplete chapters.")
                            self.workflow_state.log_event("Stall details: Parsed outline missing.", {"level": "ERROR"})


                    self.workflow_state.set_flag('report_generation_complete', True)
                    break
            else:
                stall_patience_counter = 0
                self._execute_task_type(task)

            iteration_count += 1

            # Log chapter completion progress
            num_total_chapters = len(self.workflow_state.parsed_outline) if self.workflow_state.parsed_outline else 0
            num_completed_chapters = self.workflow_state.count_completed_chapters()
            progress_log_msg = (f"Orchestrator: Workflow iteration {iteration_count} complete. "
                                f"Chapters: {num_completed_chapters}/{num_total_chapters} completed.")
            logger.debug(progress_log_msg)
            if iteration_count % 5 == 0 or \
               (num_total_chapters > 0 and num_total_chapters == num_completed_chapters) or \
               not task : # Log to workflow state if queue was empty this iter
                self.workflow_state.log_event(progress_log_msg)

        final_log_details = {
            "total_iterations": iteration_count,
            "final_status_complete": self.workflow_state.get_flag('report_generation_complete')
        }
        if self.workflow_state.parsed_outline: # Avoid error if outline never generated
            final_log_details["chapters_completed_at_end"] = f"{self.workflow_state.count_completed_chapters()}/{len(self.workflow_state.parsed_outline)}"
        else:
            final_log_details["chapters_completed_at_end"] = "Outline not generated"

        self.workflow_state.log_event("Orchestrator finished workflow coordination.", final_log_details)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("Orchestrator Example Start")

    # --- Mock Agents and WorkflowState for testing Orchestrator ---
    # (These mocks would be more detailed in a full test suite)
    class MockAgent(BaseAgent):
        def __init__(self, agent_name, task_type_it_handles, next_task_type_to_add=None):
            super().__init__(agent_name=agent_name)
            self.task_type_it_handles = task_type_it_handles
            self.next_task_type_to_add = next_task_type_to_add
            self.called_with_payload = None

        def execute_task(self, workflow_state: WorkflowState, task_payload: Dict):
            logger.info(f"MockAgent '{self.agent_name}' executing task: {self.task_type_it_handles} with payload: {task_payload}")
            self.called_with_payload = task_payload

            # Simulate work & updating state (simplified)
            if self.task_type_it_handles == TASK_TYPE_ANALYZE_TOPIC:
                workflow_state.update_topic_analysis({"mock_analysis": "done"})
            elif self.task_type_it_handles == TASK_TYPE_GENERATE_OUTLINE:
                workflow_state.update_outline("- Mock Chapter 1\n- Mock Chapter 2",
                                             [{'id': 'c1', 'title': 'Mock Chapter 1', 'level': 1},
                                              {'id': 'c2', 'title': 'Mock Chapter 2', 'level': 1}])
            elif self.task_type_it_handles == TASK_TYPE_RETRIEVE_FOR_CHAPTER:
                chapter_key = task_payload.get('chapter_key')
                entry = workflow_state._get_chapter_entry(chapter_key, True)
                entry['retrieved_docs'] = [{"document": "mock retrieved doc for " + chapter_key}]
                workflow_state.update_chapter_status(chapter_key, "writing_needed_after_mock_retrieve") # Custom status for test
            # ... other agent simulations ...
            elif self.task_type_it_handles == TASK_TYPE_COMPILE_REPORT:
                 workflow_state.set_flag('final_report_md', "## Mock Final Report\nCompiled by Orchestrator test.")


            # Agent completes its own task in WorkflowState
            current_task_id = workflow_state.current_processing_task_id # Assuming Orchestrator set this
            if current_task_id: # Should always be set if task came from get_next_task
                 workflow_state.complete_task(current_task_id, f"{self.agent_name} finished task.")
            else: # Fallback if current_processing_task_id wasn't set as expected
                logger.error("MockAgent execute_task: current_processing_task_id not set in workflow_state!")


            # Agent adds next task
            if self.next_task_type_to_add:
                # Special handling for PROCESS_CHAPTER based on outline items
                if self.task_type_it_handles == TASK_TYPE_GENERATE_OUTLINE:
                    for item in workflow_state.parsed_outline:
                        workflow_state.add_task(self.next_task_type_to_add,
                                                payload={'chapter_key': item['id'], 'chapter_title': item['title']})
                else: # Standard next task
                    new_payload = {"source": self.agent_name}
                    if 'chapter_key' in task_payload: new_payload['chapter_key'] = task_payload['chapter_key']
                    if 'chapter_title' in task_payload: new_payload['chapter_title'] = task_payload['chapter_title']
                    workflow_state.add_task(self.next_task_type_to_add, payload=new_payload)

            # If this was the last agent in a chapter sequence (e.g. Refiner, or Evaluator if no refine)
            # it should mark chapter as completed.
            if self.task_type_it_handles == TASK_TYPE_EVALUATE_CHAPTER and 'chapter_key' in task_payload: # Simplified: assume eval completes
                workflow_state.update_chapter_status(task_payload['chapter_key'], STATUS_COMPLETED)


    mock_wf_state = WorkflowState(user_topic="Orchestrator Test")

    # Create mock agents
    mock_topic_analyzer = MockAgent("TopicAnalyzerMock", TASK_TYPE_ANALYZE_TOPIC, TASK_TYPE_GENERATE_OUTLINE)
    mock_outline_generator = MockAgent("OutlineGenMock", TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE) # Next is global retrieve
    mock_global_retriever = MockAgent("GlobalRetrieverMock", TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE, TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT) # Next is suggest refinement
    mock_outline_refiner = MockAgent("OutlineRefinerMock", TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT, TASK_TYPE_APPLY_OUTLINE_REFINEMENT)
    # APPLY_OUTLINE_REFINEMENT is handled by orchestrator, then adds PROCESS_CHAPTER tasks
    mock_retriever = MockAgent("RetrieverMock", TASK_TYPE_RETRIEVE_FOR_CHAPTER, TASK_TYPE_WRITE_CHAPTER)
    mock_writer = MockAgent("WriterMock", TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER)
    mock_evaluator = MockAgent("EvaluatorMock", TASK_TYPE_EVALUATE_CHAPTER) # No next task by default, unless refinement needed
    mock_refiner = MockAgent("RefinerMock", TASK_TYPE_REFINE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER) # Refine then re-evaluate
    mock_compiler = MockAgent("CompilerMock", TASK_TYPE_COMPILE_REPORT)


    orchestrator = Orchestrator(
        workflow_state=mock_wf_state,
        topic_analyzer=mock_topic_analyzer,
        outline_generator=mock_outline_generator,
        global_content_retriever=mock_global_retriever, # New
        outline_refiner=mock_outline_refiner,
        content_retriever=mock_retriever,
        chapter_writer=mock_writer,
        evaluator=mock_evaluator,
        refiner=mock_refiner,
        report_compiler=mock_compiler,
        max_workflow_iterations=25 # Limit for test, increased slightly for new steps
    )

    # Add initial task to workflow state
    mock_wf_state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={"user_topic": "Orchestrator Test"})
    mock_wf_state.set_flag('outline_finalized', True) # Assume for this test outline is final once generated

    logger.info("\n--- Starting Orchestrator.coordinate_workflow() ---")
    try:
        orchestrator.coordinate_workflow()

        logger.info("\n--- Orchestrator.coordinate_workflow() finished ---")
        print(f"Workflow Complete Flag: {mock_wf_state.get_flag('report_generation_complete')}")
        print(f"Final Report MD (from flag): {mock_wf_state.get_flag('final_report_md')}")
        print(f"Total errors: {mock_wf_state.error_count}")

        assert mock_wf_state.get_flag('report_generation_complete') is True
        assert "Mock Final Report" in (mock_wf_state.get_flag('final_report_md') or "")

        print("\nOrchestrator example test successful.")

    except Exception as e:
        print(f"Error during Orchestrator test: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nOrchestrator Example End")
