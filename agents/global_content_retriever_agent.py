import logging
from typing import Dict, Optional, List, Any
import json
from agents.base_agent import BaseAgent
from core.workflow_state import WorkflowState, TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT
# We need access to a retrieval mechanism. This could be via ContentRetrieverAgent or a shared RetrievalService.
# For simplicity, let's assume it can instantiate or be given a ContentRetrieverAgent for its capabilities.
# Or, more ideally, it uses RetrievalService directly if ContentRetrieverAgent is too heavy or has other duties.
# Let's assume direct use of RetrievalService for now if available, or simulate if not.
from core.retrieval_service import RetrievalService # Assuming this service exists and is usable
from core.llm_service import LLMService # Optional, for query generation or summarization
from agents.content_retriever_agent import ContentRetrieverAgent # Fallback or helper
from config import settings # Added import for settings

logger = logging.getLogger(__name__)

class GlobalContentRetrieverAgentError(Exception):
    """Custom exception for GlobalContentRetrieverAgent errors."""
    pass

class GlobalContentRetrieverAgent(BaseAgent):
    """
    Agent responsible for performing a global, preliminary content retrieval
    for each chapter in the initial outline. The goal is to gather context
    that can inform the outline refinement process.
    """

    def __init__(self, retrieval_service: RetrievalService, llm_service: Optional[LLMService] = None): # LLM might be needed for query generation or summarization
        super().__init__(agent_name="GlobalContentRetrieverAgent", llm_service=llm_service)
        self.retrieval_service = retrieval_service
        if not self.retrieval_service:
            raise GlobalContentRetrieverAgentError("RetrievalService is required for GlobalContentRetrieverAgent.")
        # Alternatively, could take a ContentRetrieverAgent instance.
        # self.content_retriever = content_retriever_agent

    def _generate_queries_for_chapter(self, chapter_title: str, topic_analysis: Optional[Dict] = None, max_queries: int = 3) -> List[str]:
        """
        Generates a list of search queries for a given chapter title.
        Includes the chapter title itself, combinations with the main topic, and with keywords.
        """
        queries = []

        # Query 1: Just the chapter title
        if chapter_title:
            queries.append(chapter_title)

        if topic_analysis:
            main_topic_cn = topic_analysis.get("generalized_topic_cn", "")
            main_topic_en = topic_analysis.get("generalized_topic_en", "")

            # Query 2: Main topic + Chapter title (try both CN and EN main topics if available)
            if main_topic_cn and chapter_title:
                queries.append(f"{main_topic_cn} {chapter_title}")
            if main_topic_en and chapter_title and main_topic_en != main_topic_cn: # Avoid duplicate if they are same
                queries.append(f"{main_topic_en} {chapter_title}")

            # Query 3: Chapter title + some keywords
            keywords_cn = topic_analysis.get("keywords_cn", [])
            keywords_en = topic_analysis.get("keywords_en", [])

            all_keywords = list(dict.fromkeys(keywords_cn + keywords_en)) # Combine and deduplicate

            if chapter_title and all_keywords:
                # Take first few keywords to keep query focused
                relevant_keywords_str = " ".join(all_keywords[:2]) # e.g., first 2 keywords
                queries.append(f"{chapter_title} {relevant_keywords_str}".strip())

        # Deduplicate and filter empty/whitespace queries, then limit
        final_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        return final_queries[:max_queries]


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution.")

        parsed_outline = task_payload.get('parsed_outline')
        topic_analysis_results = task_payload.get('topic_analysis_results') # For better query generation

        if not parsed_outline:
            err_msg = "Parsed outline not found in task payload for global retrieval."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise GlobalContentRetrieverAgentError(err_msg)

        self._log_input(payload_keys=list(task_payload.keys()))

        global_docs_map: Dict[str, List[Dict[str, Any]]] = {}
        retrieved_something_overall = False

        for chapter_item in parsed_outline:
            chapter_id = chapter_item.get('id')
            chapter_title = chapter_item.get('title')

            if not chapter_id or not chapter_title:
                logger.warning(f"[{self.agent_name}] Skipping chapter item with missing ID or title: {chapter_item}")
                continue

            # Generate multiple queries for the chapter
            chapter_queries = self._generate_queries_for_chapter(
                chapter_title,
                topic_analysis_results,
                max_queries=settings.DEFAULT_MAX_CHAPTER_QUERIES_GLOBAL_RETRIEVAL # Use setting
            )

            if not chapter_queries:
                logger.warning(f"[{self.agent_name}] No valid queries generated for chapter ID '{chapter_id}' (Title: '{chapter_title}'). Skipping retrieval for this chapter.")
                global_docs_map[chapter_id] = []
                continue

            logger.info(f"[{self.agent_name}] Retrieving global context for chapter ID '{chapter_id}' (Title: '{chapter_title}') using {len(chapter_queries)} queries: {chapter_queries}")

            try:
                # MAX_GLOBAL_RESULTS_PER_CHAPTER applies to the aggregated results from multiple queries for this chapter
                MAX_GLOBAL_RESULTS_PER_CHAPTER = settings.DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER # Use setting

                retrieved_docs = self.retrieval_service.retrieve(
                    query_texts=chapter_queries, # Pass list of queries
                    final_top_n=MAX_GLOBAL_RESULTS_PER_CHAPTER
                    # Other params like vector_top_k use defaults from RetrievalService
                )

                if retrieved_docs:
                    global_docs_map[chapter_id] = retrieved_docs # Already a list
                    retrieved_something_overall = True
                    logger.debug(f"Retrieved {len(retrieved_docs)} documents for chapter '{chapter_id}'.")
                else:
                    global_docs_map[chapter_id] = [] # Store empty list if nothing found
                    logger.debug(f"No documents found for chapter '{chapter_id}'.")
            except Exception as e:
                logger.error(f"[{self.agent_name}] Error retrieving documents for chapter '{chapter_id}': {e}", exc_info=True)
                global_docs_map[chapter_id] = [] # Store empty list on error for this chapter
                workflow_state.log_event(f"Global retrieval error for chapter {chapter_id}", {"error": str(e)})


        workflow_state.set_global_retrieved_docs_map(global_docs_map)

        # Prepare payload for the next task (Suggest Outline Refinement)
        # It needs the original outline, topic analysis, etc.
        # These should have been passed in the current task's payload or be available in workflow_state

        suggest_refinement_payload = {
            "current_outline_md": task_payload.get("current_outline_md"), # Must be passed from OutlineGenerator
            "parsed_outline": parsed_outline, # The same outline we just processed
            "topic_analysis_results": topic_analysis_results,
            # Constraints also need to be passed through or fetched from config/workflow_state
            "max_chapters": task_payload.get('max_chapters_constraint', 10),
            "min_chapters": task_payload.get('min_chapters_constraint', 3)
        }
        # Ensure necessary fields are present for OutlineRefinementAgent
        if not suggest_refinement_payload.get("current_outline_md"):
             logger.warning(f"[{self.agent_name}] current_outline_md missing in task_payload. OutlineRefinementAgent might fail.")
             # Attempt to reconstruct from parsed_outline if absolutely necessary and possible
             # For now, rely on it being passed through by OutlineGeneratorAgent via Orchestrator.


        workflow_state.add_task(
            task_type=TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT,
            payload=suggest_refinement_payload,
            priority=task_payload.get('priority', 3) + 1 # Next logical step
        )

        self._log_output({"num_chapters_processed": len(parsed_outline), "num_chapters_with_retrieved_docs": sum(1 for docs in global_docs_map.values() if docs)})
        status_message = "Global content retrieval for outline completed."
        if not retrieved_something_overall:
            status_message += " No documents were retrieved for any chapter."
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - {status_message}")
        if task_id: workflow_state.complete_task(task_id, status_message, status='success')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # --- Mock dependencies ---
    class MockRetrievalServiceGCRA: # GCRA for GlobalContentRetrieverAgent
        def retrieve(self, query_texts: List[str], final_top_n: int, **kwargs) -> List[Dict[str, Any]]:
            logger.debug(f"MockRetrievalServiceGCRA.retrieve called with queries: {query_texts}, final_top_n={final_top_n}")
            docs_to_return = []
            # Simulate different results based on query content for more realistic testing
            if any("Introduction" in qt for qt in query_texts):
                docs_to_return.append({"document": "Intro doc from multi-query.", "score": 0.8, "child_id": "c_intro", "parent_id": "p_intro", "source_document_name": "intro_src.txt"})
            if any("Methods" in qt for qt in query_texts):
                docs_to_return.extend([
                    {"document": "Methods doc 1 from multi-query.", "score": 0.9, "child_id": "c_meth1", "parent_id": "p_meth1", "source_document_name": "meth_src.txt"},
                    {"document": "Methods doc 2 (lower score).", "score": 0.7, "child_id": "c_meth2", "parent_id": "p_meth2", "source_document_name": "meth_src.txt"}
                ])
            if any("Conclusion" in qt for qt in query_texts) and any("测试主题" in qt for qt in query_texts) : # Example of more specific query combination
                 docs_to_return.append({"document": "Conclusion for 测试主题.", "score": 0.85, "child_id": "c_conc", "parent_id": "p_conc", "source_document_name": "conc_src.txt"})

            docs_to_return.sort(key=lambda x: x['score'], reverse=True)
            logger.debug(f"MockRetrievalServiceGCRA returning {len(docs_to_return[:final_top_n])} docs out of {len(docs_to_return)} potential docs for queries: {query_texts}")
            return docs_to_return[:final_top_n]


    class MockWorkflowStateGCRA(WorkflowState): # GCRA for GlobalContentRetrieverAgent
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.added_tasks_gcra = []
            self.global_docs_map_set = None

        def set_global_retrieved_docs_map(self, docs_map: Dict[str, List[Dict[str, Any]]]):
            super().set_global_retrieved_docs_map(docs_map)
            self.global_docs_map_set = docs_map # For inspection

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_gcra.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)
    # --- End Mock dependencies ---

    retrieval_service_instance_gcra = MockRetrievalServiceGCRA() # Use updated mock
    agent = GlobalContentRetrieverAgent(retrieval_service=retrieval_service_instance_gcra)

    mock_parsed_outline_gcra = [
        {"id": "ch_intro", "title": "Introduction to Topic", "level": 1},
        {"id": "ch_methods", "title": "Core Methods for Topic", "level": 1},
        {"id": "ch_conclusion", "title": "Conclusion for 测试主题", "level": 1}, # Title to trigger specific mock logic
        {"id": "ch_empty", "title": "Chapter With No Mock Docs", "level":1}
    ]
    # Enriched topic_analysis for better query generation in _generate_queries_for_chapter
    mock_topic_analysis_gcra = {
        "generalized_topic_cn": "测试主题",
        "generalized_topic_en": "Test Topic",
        "keywords_cn": ["核心", "技术"],
        "keywords_en": ["core", "technology"]
    }

    mock_current_md_gcra = "# Introduction to Topic\n# Core Methods for Topic\n# Conclusion for 测试主题\n# Chapter With No Mock Docs"

    task_payload_gcra = {
        "parsed_outline": mock_parsed_outline_gcra,
        "current_outline_md": mock_current_md_gcra,
        "topic_analysis_results": mock_topic_analysis_gcra,
        "priority": 3,
        "max_chapters_constraint": 5,
        "min_chapters_constraint": 2
    }

    mock_state_gcra = MockWorkflowStateGCRA(user_topic="Test Topic Full")
    mock_state_gcra.current_processing_task_id = "gcra_task_multi_query_001"

    # Set a default for the setting used by the agent, if not already set by actual settings import
    if not hasattr(settings, 'DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER'):
        settings.DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER = 2 # Mock setting value for test

    print(f"\nExecuting GlobalContentRetrieverAgent with MockWorkflowStateGCRA (Multi-Query per Chapter)")
    try:
        agent.execute_task(mock_state_gcra, task_payload_gcra)

        print("\nWorkflowState after GlobalContentRetrieverAgent execution:")
        retrieved_map = mock_state_gcra.global_docs_map_set
        print(f"  Global Retrieved Docs Map: {json.dumps(retrieved_map, indent=2, ensure_ascii=False)}")
        print(f"  Tasks added by agent: {json.dumps(mock_state_gcra.added_tasks_gcra, indent=2, ensure_ascii=False)}")

        assert retrieved_map is not None
        # Check ch_intro (expects 1 doc from mock based on "Introduction")
        assert "ch_intro" in retrieved_map and len(retrieved_map["ch_intro"]) == 1
        assert retrieved_map["ch_intro"][0]['document'] == "Intro doc from multi-query."

        # Check ch_methods (expects 2 docs from mock based on "Methods", limited by DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER)
        assert "ch_methods" in retrieved_map
        # The number of docs for ch_methods depends on the mock logic and final_top_n for the retrieve call.
        # MockRetrievalServiceGCRA returns 2 for "Methods", and final_top_n is 2 for the test setting.
        assert len(retrieved_map["ch_methods"]) == settings.DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER
        assert retrieved_map["ch_methods"][0]['document'] == "Methods doc 1 from multi-query." # Highest score

        # Check ch_conclusion (expects 1 doc from mock based on "Conclusion" and "测试主题")
        assert "ch_conclusion" in retrieved_map and len(retrieved_map["ch_conclusion"]) == 1
        assert retrieved_map["ch_conclusion"][0]['document'] == "Conclusion for 测试主题."

        # Check ch_empty (expects 0 docs)
        assert "ch_empty" in retrieved_map and len(retrieved_map["ch_empty"]) == 0

        assert len(mock_state_gcra.added_tasks_gcra) == 1
        next_task_info_gcra = mock_state_gcra.added_tasks_gcra[0]
        assert next_task_info_gcra['type'] == TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT
        assert next_task_info_gcra['payload']['parsed_outline'] == mock_parsed_outline_gcra
        assert next_task_info_gcra['payload']['topic_analysis_results'] == mock_topic_analysis_gcra

        print("\nGlobalContentRetrieverAgent test successful (Multi-Query per Chapter).")

    except Exception as e:
        print(f"Error during GlobalContentRetrieverAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nGlobalContentRetrieverAgent example finished.")
