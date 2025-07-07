import logging
from typing import List, Dict, Optional, Any
import json
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

        topic_analysis = workflow_state.topic_analysis_results or {}

        # Generate multiple queries for chapter content
        queries_for_chapter = self._generate_queries_for_chapter_content(
            chapter_title,
            topic_analysis,
            workflow_state.user_topic,
            max_queries=settings.DEFAULT_MAX_CHAPTER_QUERIES_CONTENT_RETRIEVAL # Use setting
        )

        if not queries_for_chapter:
            # This case should be rare given the fallback to user_topic if all else fails.
            # However, if it does happen, it means no meaningful query could be constructed.
            err_msg = f"No valid queries could be generated for chapter '{chapter_title}'. Cannot retrieve content."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            # Potentially add error to chapter_data as well
            workflow_state.add_chapter_error(chapter_key, "Query generation failed for content retrieval.")
            raise ContentRetrieverAgentError(err_msg)

        # Use the parameters stored in the agent (which were set by the pipeline from CLI/settings)
        # Task_payload can override these for this specific chapter retrieval.
        current_vector_top_k = task_payload.get('vector_top_k', self.vector_top_k)
        current_keyword_top_k = task_payload.get('keyword_top_k', self.keyword_top_k)
        current_hybrid_alpha = task_payload.get('hybrid_alpha', self.hybrid_alpha)
        current_final_top_n = task_payload.get('final_top_n', self.final_top_n)

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Retrieving for chapter '{chapter_title}' (key: {chapter_key}) "
                    f"using {len(queries_for_chapter)} queries (first: '{queries_for_chapter[0][:100]}...'). "
                    f"Params: vector_k={current_vector_top_k}, keyword_k={current_keyword_top_k}, "
                    f"alpha={current_hybrid_alpha}, final_n={current_final_top_n}")
        try:
            retrieved_docs_for_chapter = self.retrieval_service.retrieve(
                query_texts=queries_for_chapter, # Pass list of queries
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
            workflow_state.log_event(err_msg, {"error": str(e)}) # Removed level="CRITICAL"
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error: {e}")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg) # Re-raise

    def _generate_queries_for_chapter_content(self,
                                             chapter_title: str,
                                             topic_analysis: Dict,
                                             user_topic: str,
                                             max_queries: int = 4) -> List[str]:
        """
        Generates a list of diverse search queries for retrieving chapter content.
        """
        queries = []

        # Query 1: Chapter title itself (often specific enough)
        if chapter_title:
            queries.append(chapter_title)

        # Query 2: Chapter title + Chinese keywords
        keywords_cn = topic_analysis.get('keywords_cn', [])
        if chapter_title and keywords_cn:
            queries.append(f"{chapter_title} {' '.join(keywords_cn[:3])}".strip()) # Max 3 CN keywords

        # Query 3: Chapter title + English keywords (if different from CN)
        keywords_en = topic_analysis.get('keywords_en', [])
        if chapter_title and keywords_en:
            query_en_keywords = f"{chapter_title} {' '.join(keywords_en[:3])}".strip() # Max 3 EN keywords
            # Add only if it's substantially different from the CN keyword query to avoid too much similarity
            if query_en_keywords not in queries:
                 queries.append(query_en_keywords)

        # Query 4: Chapter title + Generalized Topic (Chinese)
        generalized_topic_cn = topic_analysis.get('generalized_topic_cn')
        if chapter_title and generalized_topic_cn:
            queries.append(f"{generalized_topic_cn} {chapter_title}".strip())

        # Fallback query if all above are somehow empty (should be rare)
        if not any(q.strip() for q in queries):
            if chapter_title: # If title exists but other parts were empty
                 queries.append(chapter_title)
            elif user_topic: # Absolute fallback
                queries.append(user_topic)

        # Deduplicate, filter empty strings, and limit
        final_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        return final_queries[:max_queries]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockRetrievalServiceCRA: # CRA for ContentRetrieverAgent
        def retrieve(self, query_texts: List[str], vector_top_k: int, keyword_top_k: int,
                     hybrid_alpha: float, final_top_n: Optional[int], **kwargs) -> List[Dict[str, Any]]: # Added **kwargs
            logger.debug(f"MockRetrievalServiceCRA.retrieve called with queries: {query_texts}, final_top_n={final_top_n}")
            # Simulate returning docs based on the *first* query for simplicity in mock
            # A more complex mock could try to match content from any of the queries.
            first_query_for_mock = query_texts[0] if query_texts else "empty_mock_query"
            num_results = final_top_n or 2 # Default to 2 if final_top_n is None

            # Simulate some document variety based on query terms
            docs = []
            for i in range(num_results):
                doc_content = f"Mock Parent Doc {i} for '{first_query_for_mock}'"
                if "keyword" in first_query_for_mock.lower():
                    doc_content += " (found with keyword focus)"
                elif "topic" in first_query_for_mock.lower():
                     doc_content += " (found with topic focus)"
                docs.append({
                    "document": doc_content, "score": 0.8 - (i*0.05), # Decreasing scores
                    "child_id": f"c{i}_{first_query_for_mock.replace(' ','_')[:10]}", "parent_id": f"p{i}",
                    "child_text_preview":"child preview...", "source_document_id":"doc_mock",
                    "retrieval_source":"mock_hybrid"
                })
            return docs


    # Mock WorkflowState
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
        retrieval_service=mock_retrieval_svc_cra, # Use the updated mock service
        default_final_top_n=3 # Let's try to get 3 docs for the test
    )

    # Setup state with more comprehensive topic_analysis for query generation
    mock_topic_analysis_cra = {
        "keywords_cn": ["测试关键词", "内容检索"],
        "keywords_en": ["test keyword", "content retrieval"],
        "generalized_topic_cn": "通用测试主题"
    }
    mock_state_cra = MockWorkflowStateCRA(user_topic="Global Test Topic CRA",
                                          topic_analysis_results=mock_topic_analysis_cra)
    # Simulate task_id being set by orchestrator
    mock_state_cra.current_processing_task_id = "cra_task_test_001"


    # Task payload for the agent's execute_task method
    task_payload_for_agent_cra = {
        'chapter_key': 'test_chap_key_1',
        'chapter_title': 'Test Chapter Title 1 with Keywords'
        # Optional: 'vector_top_k': 5, 'keyword_top_k': 5, 'final_top_n': 5 to override agent defaults
    }

    logger.info(f"\n--- Executing ContentRetrieverAgent with MockWorkflowStateCRA (Multi-Query) ---")
    try:
        retriever_agent_cra.execute_task(mock_state_cra, task_payload_for_agent_cra)

        print("\nWorkflowState after ContentRetrieverAgent execution:")
        chapter_info_cra = mock_state_cra.get_chapter_data('test_chap_key_1')
        if chapter_info_cra:
            print(f"  Chapter 'test_chap_key_1' Status: {chapter_info_cra.get('status')}")
            retrieved_docs_list = chapter_info_cra.get('retrieved_docs', [])
            print(f"  Retrieved Docs Count: {len(retrieved_docs_list)}")
            if retrieved_docs_list:
                for idx, d in enumerate(retrieved_docs_list):
                    print(f"    Doc {idx+1} Preview: {d.get('document','')[:60]}... (Score: {d.get('score')})")
        else:
            print("  Chapter 'test_chap_key_1' not found in workflow state after execution.")


        print(f"  Tasks added by agent: {json.dumps(mock_state_cra.added_tasks_cra, indent=2, ensure_ascii=False)}")

        assert chapter_info_cra is not None, "Chapter info should exist"
        assert chapter_info_cra.get('status') == STATUS_WRITING_NEEDED, "Chapter status should be updated"

        # The number of retrieved docs should be <= agent's default_final_top_n (or payload override)
        # And our mock service returns `final_top_n` or 2 if None. Agent default is 3 here.
        expected_doc_count = retriever_agent_cra.final_top_n
        assert len(chapter_info_cra.get('retrieved_docs', [])) == expected_doc_count, \
            f"Expected {expected_doc_count} docs, got {len(chapter_info_cra.get('retrieved_docs', []))}"

        assert len(mock_state_cra.added_tasks_cra) == 1, "One 'Write Chapter' task should be added"
        assert mock_state_cra.added_tasks_cra[0]['type'] == TASK_TYPE_WRITE_CHAPTER, "Next task should be Write Chapter"
        assert mock_state_cra.added_tasks_cra[0]['payload']['chapter_key'] == 'test_chap_key_1', "Payload should contain correct chapter key"

        print("\nContentRetrieverAgent test successful with MockWorkflowStateCRA (Multi-Query).")

    except Exception as e:
        print(f"Error during ContentRetrieverAgent test: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nContentRetrieverAgent (client to RetrievalService) Example Finished.")
