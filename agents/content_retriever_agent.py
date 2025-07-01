import logging
from typing import List, Dict, Optional, Any

from agents.base_agent import BaseAgent
from core.retrieval_service import RetrievalService, RetrievalServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_WRITE_CHAPTER, STATUS_WRITING_NEEDED # Import constants
from config import settings # Import settings for default retrieval parameters

logger = logging.getLogger(__name__)

class ContentRetrieverAgentError(Exception):
    """Custom exception for ContentRetrieverAgent errors."""
    pass

class ContentRetrieverAgent(BaseAgent):
    """
    Agent responsible for initiating content retrieval for a chapter via RetrievalService.
    It takes chapter details from WorkflowState, calls RetrievalService,
    and then updates WorkflowState with the retrieved documents and queues the next task (writing).
    """

    def __init__(self,
                 retrieval_service: RetrievalService,
                 # These parameters are passed by ReportGenerationPipeline,
                 # reflecting CLI overrides or settings defaults.
                 default_vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 default_keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 default_hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA,
                 default_final_top_n: int = settings.DEFAULT_RETRIEVAL_FINAL_TOP_N
                 ):

        super().__init__(agent_name="ContentRetrieverAgent", llm_service=None)

        if not retrieval_service:
            raise ContentRetrieverAgentError("RetrievalService is required for ContentRetrieverAgent.")
        self.retrieval_service = retrieval_service

        # Store the effective parameters passed from the pipeline
        self.vector_top_k = default_vector_top_k
        self.keyword_top_k = default_keyword_top_k
        self.hybrid_alpha = default_hybrid_alpha
        self.final_top_n = default_final_top_n

        logger.info(f"ContentRetrieverAgent initialized. Effective params: "
                    f"vector_k={self.vector_top_k}, keyword_k={self.keyword_top_k}, "
                    f"alpha={self.hybrid_alpha}, final_n={self.final_top_n}")

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Retrieves content for a specific chapter based on task_payload,
        updates WorkflowState, and adds a task for writing the chapter.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload for this task, expects:
                                 'chapter_key': Unique key/ID of the chapter.
                                 'chapter_title': Title of the chapter (for query generation).
                                 Optional retrieval params like 'vector_top_k', etc.
                                 to override agent defaults for this specific call.

        Raises:
            ContentRetrieverAgentError: If retrieval fails or essential payload info is missing.
        """
        task_id = workflow_state.current_processing_task_id
        if not task_id: # Should be set by orchestrator
            logger.error(f"{self.agent_name}: current_processing_task_id not found in workflow_state. This is unexpected.")
            # Attempt to find task_id from payload if passed by a previous version or for safety
            task_id = task_payload.get('task_id_if_passed_explicitly') # Unlikely to be there with current orchestrator

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for chapter_key: {task_payload.get('chapter_key')}, title: {task_payload.get('chapter_title')}")

        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title')

        if not chapter_key or not chapter_title:
            err_msg = "Chapter key or title not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg)

        self._log_input(chapter_key=chapter_key, chapter_title=chapter_title, overrides=task_payload)

        # Construct query - use chapter title and global keywords from topic analysis
        topic_analysis = workflow_state.topic_analysis_results or {}
        keywords_cn = topic_analysis.get('keywords_cn', [])
        # keywords_en = topic_analysis.get('keywords_en', []) # Consider if these should be added too

        query = f"{chapter_title} {' '.join(keywords_cn)}".strip()
        if not query: # Fallback if title and keywords are empty
            query = workflow_state.user_topic

        # Use the parameters stored in the agent (which were set by the pipeline from CLI/settings)
        # Task_payload could potentially override these further if we design for per-chapter retrieval variation,
        # but for now, we use the agent's configured (effective) parameters.
        current_vector_top_k = task_payload.get('vector_top_k', self.vector_top_k)
        current_keyword_top_k = task_payload.get('keyword_top_k', self.keyword_top_k)
        current_hybrid_alpha = task_payload.get('hybrid_alpha', self.hybrid_alpha)
        current_final_top_n = task_payload.get('final_top_n', self.final_top_n)

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Retrieving for chapter '{chapter_title}' (key: {chapter_key}) with query: '{query[:100]}...'. "
                    f"Params: vector_k={current_vector_top_k}, keyword_k={current_keyword_top_k}, "
                    f"alpha={current_hybrid_alpha}, final_n={current_final_top_n}")
        try:
            # RetrievalService will use its own defaults (from settings) if any of these are None,
            # but ContentRetrieverAgent now ensures they are int/float based on its init.
            retrieved_docs_for_chapter = self.retrieval_service.retrieve(
                query_text=query,
                vector_top_k=current_vector_top_k,
                keyword_top_k=current_keyword_top_k,
                hybrid_alpha=current_hybrid_alpha,
                final_top_n=current_final_top_n
            )

            # Update WorkflowState with retrieved documents for this chapter
            chapter_entry = workflow_state._get_chapter_entry(chapter_key, create_if_missing=True) # Ensure entry exists
            if chapter_entry:
                chapter_entry['retrieved_docs'] = retrieved_docs_for_chapter
                workflow_state.update_chapter_status(chapter_key, STATUS_WRITING_NEEDED) # Set next status

                # Add next task: Write Chapter
                workflow_state.add_task(
                    task_type=TASK_TYPE_WRITE_CHAPTER,
                    payload={'chapter_key': chapter_key, 'chapter_title': chapter_title}, # Pass key and title
                    priority=task_payload.get('priority', 5) + 1 # Slightly lower priority than retrieval
                )
                self._log_output({"chapter_key": chapter_key, "num_retrieved": len(retrieved_docs_for_chapter)})
                success_msg = (f"Retrieval successful for chapter '{chapter_title}'. "
                               f"{len(retrieved_docs_for_chapter)} parent contexts retrieved. Next task (Write Chapter) added.")
                logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
                if task_id: workflow_state.complete_task(task_id, success_msg, status='success')
            else: # Should not happen if _get_chapter_entry creates it
                err_msg = f"Failed to get or create chapter entry for key '{chapter_key}' in WorkflowState."
                logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
                if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
                raise ContentRetrieverAgentError(err_msg)

        except RetrievalServiceError as e:
            err_msg = f"RetrievalService failed for chapter '{chapter_title}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR") # Keep this log
            workflow_state.add_chapter_error(chapter_key, f"RetrievalService error: {e}")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg) # Re-raise
        except Exception as e:
            err_msg = f"Unexpected error during content retrieval for '{chapter_title}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="CRITICAL") # Keep this log
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error: {e}")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg) # Re-raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Mock RetrievalService (copied from its own test, simplified)
    class MockRetrievalServiceCRA: # CRA for ContentRetrieverAgent
        def retrieve(self, query_text: str, vector_top_k: int, keyword_top_k: int,
                     hybrid_alpha: float, final_top_n: Optional[int]) -> List[Dict[str, Any]]:
            logger.debug(f"MockRetrievalServiceCRA.retrieve called for '{query_text}'")
            num_results = final_top_n or 2
            return [{"document": f"Mock Parent Doc {i} for '{query_text}'", "score": 0.8,
                     "child_id": f"c{i}", "parent_id": f"p{i}", "child_text_preview":"child preview",
                     "source_document_id":"doc_mock", "retrieval_source":"mock_hybrid"} for i in range(num_results)]

    # Mock WorkflowState (copied from TopicAnalyzerAgent test, simplified)
    from core.workflow_state import WorkflowState # Ensure full class is available for test

    class MockWorkflowStateCRA(WorkflowState):
        def __init__(self, user_topic: str, topic_analysis_results: Optional[Dict] = None):
            super().__init__(user_topic)
            if topic_analysis_results: self.topic_analysis_results = topic_analysis_results
            self.added_tasks_cra = [] # Specific list for this agent's added tasks
            # Pre-populate a chapter entry for testing
            self.chapter_data['test_chap_key_1'] = {
                'title': 'Test Chapter Title 1', 'level': 1, 'status': STATUS_PENDING,
                'content': None, 'retrieved_docs': None, 'evaluations': [], 'versions': [],'errors':[]
            }


        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_cra.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority) # Call parent for full behavior
            logger.debug(f"MockWorkflowStateCRA: Task added by CRA - Type: {task_type}, Payload: {payload}")

        def update_chapter_status(self, chapter_key: str, status: str): # Override for logging
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateCRA: Chapter '{chapter_key}' status updated to {status}")


    mock_retrieval_svc_cra = MockRetrievalServiceCRA()

    # Agent default params can be set here
    retriever_agent_cra = ContentRetrieverAgent(
        retrieval_service=mock_retrieval_svc_cra,
        default_final_top_n=2
    )

    # Setup state
    mock_topic_res = {"keywords_cn": ["测试关键词"]}
    mock_state_cra = MockWorkflowStateCRA(user_topic="Test Topic CRA", topic_analysis_results=mock_topic_res)

    # Task payload for the agent's execute_task method
    task_payload_for_agent_cra = {
        'chapter_key': 'test_chap_key_1',
        'chapter_title': 'Test Chapter Title 1'
        # Can also include overrides like 'vector_top_k': 3 here
    }

    logger.info(f"\n--- Executing ContentRetrieverAgent with MockWorkflowStateCRA ---")
    try:
        retriever_agent_cra.execute_task(mock_state_cra, task_payload_for_agent_cra)

        print("\nWorkflowState after ContentRetrieverAgent execution:")
        chapter_info = mock_state_cra.get_chapter_data('test_chap_key_1')
        if chapter_info:
            print(f"  Chapter 'test_chap_key_1' Status: {chapter_info.get('status')}")
            print(f"  Retrieved Docs Count: {len(chapter_info.get('retrieved_docs', []))}")
            if chapter_info.get('retrieved_docs'):
                print(f"  First Retrieved Doc Preview: {chapter_info['retrieved_docs'][0]['document'][:50]}...")

        print(f"  Tasks added by agent: {json.dumps(mock_state_cra.added_tasks_cra, indent=2, ensure_ascii=False)}")

        assert chapter_info is not None
        assert chapter_info.get('status') == STATUS_WRITING_NEEDED
        assert len(chapter_info.get('retrieved_docs', [])) == (task_payload_for_agent_cra.get('final_top_n') or retriever_agent_cra.default_final_top_n)
        assert len(mock_state_cra.added_tasks_cra) == 1
        assert mock_state_cra.added_tasks_cra[0]['type'] == TASK_TYPE_WRITE_CHAPTER
        assert mock_state_cra.added_tasks_cra[0]['payload']['chapter_key'] == 'test_chap_key_1'

        print("\nContentRetrieverAgent test successful with MockWorkflowStateCRA.")

    except Exception as e:
        print(f"Error during ContentRetrieverAgent test: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nContentRetrieverAgent (client to RetrievalService) Example Finished.")
