import logging
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
                 content_retriever: ContentRetrieverAgent, # This is the agent, not the service
                 chapter_writer: ChapterWriterAgent,
                 evaluator: EvaluatorAgent,
                 refiner: RefinerAgent,
                 report_compiler: ReportCompilerAgent,
                 max_workflow_iterations: int = 50,
                 # Optional: Pass RetrievalService if some tasks need direct access
                 # retrieval_service: Optional[RetrievalService] = None
                ):
        self.workflow_state = workflow_state
        self.agents = {
            TASK_TYPE_ANALYZE_TOPIC: topic_analyzer,
            TASK_TYPE_GENERATE_OUTLINE: outline_generator,
            # TASK_TYPE_PROCESS_CHAPTER is a meta-task, handled by adding retrieve then write
            TASK_TYPE_RETRIEVE_FOR_CHAPTER: content_retriever, # Agent uses RetrievalService internally
            TASK_TYPE_WRITE_CHAPTER: chapter_writer,
            TASK_TYPE_EVALUATE_CHAPTER: evaluator,
            TASK_TYPE_REFINE_CHAPTER: refiner,
            TASK_TYPE_COMPILE_REPORT: report_compiler,
        }
        # self.retrieval_service = retrieval_service # If needed directly
        self.max_workflow_iterations = max_workflow_iterations
        logger.info("Orchestrator initialized with all agents and workflow state.")

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
                # Assuming all agents now have an 'execute_task' method
                agent.execute_task(self.workflow_state, payload)
                # Agent's execute_task is responsible for adding next tasks and completing current one via workflow_state
                # So, we don't call workflow_state.complete_task here in orchestrator for agent-handled tasks.
                # Agent should call it. If agent raises error, it's caught below.
                # However, for tasks handled *directly* by orchestrator (like PROCESS_CHAPTER), we need to complete it.

                # If an agent's execute_task doesn't call complete_task itself, then orchestrator should.
                # For now, let's assume agents will call workflow_state.complete_task(task_id, ...)
                # This needs to be consistently implemented in all agents.
                # Let's refine: if agent.execute_task doesn't raise, we assume it handled completion.
                # This is a bit implicit. A better way: agent returns a status or next actions.
                # For now, let's assume agent calls complete_task.

            except Exception as e: # Catch errors from agent execution
                logger.error(f"Error executing task {task_type} ({task_id}) with agent {getattr(agent, 'agent_name', 'UnknownAgent')}: {e}", exc_info=True)
                self.workflow_state.log_event(f"Agent execution error for task {task_type} ({task_id})",
                                             {"error": str(e), "agent": getattr(agent, 'agent_name', 'UnknownAgent'), "level": "CRITICAL"})
                self.workflow_state.complete_task(task_id, f"Agent failed: {e}", status='failed')
                if 'chapter_key' in payload:
                    self.workflow_state.add_chapter_error(payload['chapter_key'], f"Agent {task_type} failed: {e}")

        elif task_type == TASK_TYPE_PROCESS_CHAPTER: # Meta-task handled by orchestrator
            # This task initiates the sequence for a chapter: retrieve -> write (-> eval -> refine)*
            chapter_key = payload['chapter_key']
            self.workflow_state.update_chapter_status(chapter_key, STATUS_PENDING) # Reset/confirm status
            # Add retrieval task as the first concrete step for this chapter
            self.workflow_state.add_task(TASK_TYPE_RETRIEVE_FOR_CHAPTER,
                                         payload=payload, # Pass on chapter_key, chapter_title
                                         priority=task.get('priority', 3))
            self.workflow_state.complete_task(task_id, f"PROCESS_CHAPTER task for '{payload.get('chapter_title')}' initiated retrieval.")

        # TODO: Handle TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT, TASK_TYPE_APPLY_OUTLINE_REFINEMENT here
        # These might involve more complex logic, potentially calling OutlineGeneratorAgent again or LLM.

        else:
            logger.warning(f"No agent or direct handler registered for task type: {task_type} (task_id: {task_id}).")
            self.workflow_state.log_event(f"Unknown task type: {task_type}", {"task_id": task_id, "level": "ERROR"})
            self.workflow_state.complete_task(task_id, f"Unknown task type {task_type}", status='failed')


    def coordinate_workflow(self) -> None:
        """
        Main loop to coordinate the workflow by processing tasks from WorkflowState.
        """
        self.workflow_state.log_event("Orchestrator starting workflow coordination.")
        iteration_count = 0

        while not self.workflow_state.get_flag('report_generation_complete', False):
            if iteration_count >= self.max_workflow_iterations:
                self.workflow_state.log_event("Max workflow iterations reached by Orchestrator. Halting.", {"level": "ERROR"})
                self.workflow_state.set_flag('report_generation_complete', True) # Force stop
                break

            task = self.workflow_state.get_next_task() # Pops task and marks 'in_progress'

            if not task:
                # No more tasks in the queue.
                # Check conditions for triggering report compilation.
                if self.workflow_state.are_all_chapters_completed() and \
                   self.workflow_state.get_flag('outline_finalized', False) and \
                   not self.workflow_state.get_flag('report_compilation_requested', False):

                    self.workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100) # Low priority
                    self.workflow_state.set_flag('report_compilation_requested', True)
                    self.workflow_state.log_event("All chapters completed, outline final. Compilation task added by Orchestrator.")
                    # Loop will continue and pick up this new task
                elif self.workflow_state.get_flag('report_generation_complete'): # If flag was set by compile task
                    break
                else:
                    # No tasks, not all chapters done, or outline not final, or compilation already requested but not done
                    # This might indicate a stall if no new tasks are being generated.
                    self.workflow_state.log_event("Task queue empty, but report not complete and compilation not triggerable yet. Checking for stall.",
                                                 {"all_chapters_done": self.workflow_state.are_all_chapters_completed(),
                                                  "outline_finalized": self.workflow_state.get_flag('outline_finalized'),
                                                  "compilation_requested": self.workflow_state.get_flag('report_compilation_requested'),
                                                  "level": "WARNING"})
                    # Break if it seems stuck (e.g., after a few iterations with no tasks)
                    # This needs a more robust stall detection or timeout.
                    # For now, if the queue is empty and compilation isn't the next step, we might be done or stuck.
                    # The report_generation_complete flag is the ultimate decider.
                    if iteration_count > 5 and not self.workflow_state.pending_tasks : # Arbitrary small number for test
                         logger.warning("Orchestrator: Potential stall detected (empty queue, not complete). Halting.")
                         self.workflow_state.set_flag('report_generation_complete', True) # Force stop due to stall
                         break
            else:
                # Handle the retrieved task
                self._execute_task_type(task) # This will call agent's execute_task

            iteration_count += 1

            # Log chapter completion progress
            num_total_chapters = len(self.workflow_state.parsed_outline)
            num_completed_chapters = self.workflow_state.count_completed_chapters()
            progress_log_msg = (f"Orchestrator: Workflow iteration {iteration_count} complete. "
                                f"Chapters: {num_completed_chapters}/{num_total_chapters} completed.")
            logger.debug(progress_log_msg) # More frequent, so debug level
            if iteration_count % 5 == 0 or num_total_chapters == num_completed_chapters : # Log to workflow state less frequently
                self.workflow_state.log_event(progress_log_msg)


        self.workflow_state.log_event("Orchestrator finished workflow coordination.",
                                     {"total_iterations": iteration_count,
                                      "final_status_complete": self.workflow_state.get_flag('report_generation_complete'),
                                      "chapters_completed_at_end": f"{self.workflow_state.count_completed_chapters()}/{len(self.workflow_state.parsed_outline)}"})

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
    mock_outline_generator = MockAgent("OutlineGenMock", TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER)
    # PROCESS_CHAPTER is meta, leads to RETRIEVE_FOR_CHAPTER
    mock_retriever = MockAgent("RetrieverMock", TASK_TYPE_RETRIEVE_FOR_CHAPTER, TASK_TYPE_WRITE_CHAPTER)
    mock_writer = MockAgent("WriterMock", TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER)
    mock_evaluator = MockAgent("EvaluatorMock", TASK_TYPE_EVALUATE_CHAPTER) # No next task by default, unless refinement needed
    mock_refiner = MockAgent("RefinerMock", TASK_TYPE_REFINE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER) # Refine then re-evaluate
    mock_compiler = MockAgent("CompilerMock", TASK_TYPE_COMPILE_REPORT)


    orchestrator = Orchestrator(
        workflow_state=mock_wf_state,
        topic_analyzer=mock_topic_analyzer,
        outline_generator=mock_outline_generator,
        content_retriever=mock_retriever,
        chapter_writer=mock_writer,
        evaluator=mock_evaluator,
        refiner=mock_refiner,
        report_compiler=mock_compiler,
        max_workflow_iterations=20 # Limit for test
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
