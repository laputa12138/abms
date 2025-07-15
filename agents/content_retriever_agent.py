import logging
from typing import List, Dict, Optional, Any
import json
from agents.base_agent import BaseAgent
from core.retrieval_service import RetrievalService, RetrievalServiceError
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_WRITE_CHAPTER, STATUS_WRITING_NEEDED, STATUS_PENDING # Import constants
from config import settings # Import settings for default retrieval parameters
from core.json_utils import clean_and_parse_json

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
                 llm_service: LLMService,
                 # These parameters are passed by ReportGenerationPipeline,
                 # reflecting CLI overrides or settings defaults.
                 default_vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 default_keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 default_final_top_n: int = settings.DEFAULT_RETRIEVAL_FINAL_TOP_N
                 ):

        super().__init__(agent_name="ContentRetrieverAgent", llm_service=llm_service)

        if not retrieval_service:
            raise ContentRetrieverAgentError("RetrievalService is required for ContentRetrieverAgent.")
        if not llm_service:
            raise ContentRetrieverAgentError("LLMService is required for ContentRetrieverAgent for query expansion.")
        self.retrieval_service = retrieval_service
        self.query_expansion_prompt_template = settings.CHAPTER_QUERY_EXPANSION_PROMPT

        # Store the effective parameters passed from the pipeline
        self.vector_top_k = default_vector_top_k
        self.keyword_top_k = default_keyword_top_k
        self.final_top_n = default_final_top_n

        logger.info(f"ContentRetrieverAgent initialized. Effective params: "
                    f"vector_k={self.vector_top_k}, keyword_k={self.keyword_top_k}, final_n={self.final_top_n}")

    def _execute_iterative_retrieval_for_chapter(
        self,
        initial_queries: List[str],
        chapter_title: str,
        workflow_state: WorkflowState,
        retrieval_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Executes an iterative retrieval process for a specific chapter.
        """
        max_iterations = settings.CHAPTER_RETRIEVAL_MAX_ITERATIONS
        queries_per_iteration = settings.CHAPTER_RETRIEVAL_QUERIES_PER_ITERATION

        all_queries = set(initial_queries)
        all_retrieved_docs = {}  # Use dict to store docs by a unique ID to avoid duplicates

        for i in range(max_iterations):
            logger.info(f"[{self.agent_name}] Chapter '{chapter_title}' retrieval, iteration {i+1}/{max_iterations}. Current query count: {len(all_queries)}")

            if not all_queries:
                logger.warning(f"[{self.agent_name}] No queries to process for chapter '{chapter_title}' in iteration {i+1}. Stopping.")
                break

            try:
                # 1. Retrieve documents
                retrieved_docs = self.retrieval_service.retrieve(
                    query_texts=list(all_queries),
                    **retrieval_params
                )

                # Add new documents to the collection
                new_docs_found_this_iteration = []
                for doc in retrieved_docs:
                    doc_id = doc.get('child_id') or doc.get('parent_id')
                    if doc_id and doc_id not in all_retrieved_docs:
                        all_retrieved_docs[doc_id] = doc
                        new_docs_found_this_iteration.append(doc)

                if not new_docs_found_this_iteration:
                    logger.info(f"[{self.agent_name}] No new documents found for chapter '{chapter_title}' in iteration {i+1}. Stopping expansion for this chapter.")
                    break

                # 2. Prepare for query expansion
                retrieved_content_summary = "\n".join([f"- {d.get('document', '')[:200]}..." for d in new_docs_found_this_iteration])

                expansion_prompt = self.query_expansion_prompt_template.format(
                    topic=chapter_title, # Focus is the chapter title
                    existing_queries=json.dumps(list(all_queries), ensure_ascii=False),
                    retrieved_content=retrieved_content_summary,
                    num_new_queries=queries_per_iteration
                )

                # 3. Call LLM to get new queries
                logger.info(f"[{self.agent_name}] Expanding queries for chapter '{chapter_title}' for next iteration.")
                raw_expansion_response = self.llm_service.chat(query=expansion_prompt, system_prompt="You are a helpful AI assistant specialized in research query expansion.")

                parsed_expansion = clean_and_parse_json(raw_expansion_response)

                if parsed_expansion and 'new_queries' in parsed_expansion and isinstance(parsed_expansion['new_queries'], list):
                    new_queries = set(parsed_expansion['new_queries'])
                    newly_added = new_queries - all_queries
                    if newly_added:
                        logger.info(f"[{self.agent_name}] Added {len(newly_added)} new queries for chapter '{chapter_title}': {list(newly_added)}")
                        all_queries.update(newly_added)
                    else:
                        logger.info(f"[{self.agent_name}] LLM did not generate any truly new queries for chapter '{chapter_title}'. Stopping iteration.")
                        break
                else:
                    logger.warning(f"[{self.agent_name}] Could not parse new queries for chapter '{chapter_title}' from LLM response.")

            except (RetrievalServiceError, LLMServiceError) as e:
                logger.error(f"[{self.agent_name}] A service error occurred during iterative retrieval for chapter '{chapter_title}': {e}", exc_info=True)
                break

        final_docs = list(all_retrieved_docs.values())
        logger.info(f"[{self.agent_name}] Iterative retrieval for chapter '{chapter_title}' finished. Total unique documents: {len(final_docs)}.")
        return final_docs


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
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
        # current_hybrid_alpha = task_payload.get('hybrid_alpha', self.hybrid_alpha)
        current_final_top_n = task_payload.get('final_top_n', self.final_top_n)

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Retrieving for chapter '{chapter_title}' (key: {chapter_key}) "
                    f"using {len(queries_for_chapter)} queries (first: '{queries_for_chapter[0][:100]}...'). "
                    f"Params: vector_k={current_vector_top_k}, keyword_k={current_keyword_top_k}, ")
                    # f"alpha={current_hybrid_alpha}, final_n={current_final_top_n}")
        current_vector_top_k = task_payload.get('vector_top_k', self.vector_top_k)
        current_keyword_top_k = task_payload.get('keyword_top_k', self.keyword_top_k)
        current_final_top_n = task_payload.get('final_top_n', self.final_top_n)

        retrieval_params = {
            "vector_top_k": current_vector_top_k,
            "keyword_top_k": current_keyword_top_k,
            "final_top_n": current_final_top_n
        }

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Retrieving for chapter '{chapter_title}' (key: {chapter_key}) "
                    f"using {len(queries_for_chapter)} initial queries. "
                    f"Params: {retrieval_params}")
        try:
            # Execute the new iterative retrieval method
            retrieved_docs_for_chapter = self._execute_iterative_retrieval_for_chapter(
                initial_queries=queries_for_chapter,
                chapter_title=chapter_title,
                workflow_state=workflow_state,
                retrieval_params=retrieval_params
            )

            # Update WorkflowState with the final list of retrieved documents
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
                success_msg = (f"Iterative retrieval successful for chapter '{chapter_title}'. "
                               f"{len(retrieved_docs_for_chapter)} documents retrieved. Next task (Write Chapter) added.")
                logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
                if task_id: workflow_state.complete_task(task_id, success_msg, status='success')
            else: # Should not happen if _get_chapter_entry creates it
                err_msg = f"Failed to get or create chapter entry for key '{chapter_key}' in WorkflowState."
                logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
                if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
                raise ContentRetrieverAgentError(err_msg)

        except (RetrievalServiceError, LLMServiceError) as e:
            err_msg = f"A service error occurred during retrieval for chapter '{chapter_title}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR") # Keep this log
            workflow_state.add_chapter_error(chapter_key, f"Service error during retrieval: {e}")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg) from e
        except Exception as e:
            err_msg = f"Unexpected error during content retrieval for '{chapter_title}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="CRITICAL")
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error: {e}")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ContentRetrieverAgentError(err_msg) from e

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


# The __main__ block is removed as it's no longer compatible with the new __init__ signature
# which requires an LLMService instance for query expansion. Testing would need
# to be done in an integrated test environment with proper mock service injection.
