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

    def _generate_query_for_chapter(self, chapter_title: str, topic_analysis: Optional[Dict] = None) -> str:
        """
        Generates a search query for a given chapter title.
        Can be enhanced with topic analysis details.
        """
        # Simple query for now
        query = f"{chapter_title}"
        if topic_analysis:
            main_topic = topic_analysis.get("generalized_topic_cn", topic_analysis.get("generalized_topic_en", ""))
            if main_topic:
                query = f"{main_topic} {chapter_title}" # Combine topic with chapter title

            # Consider adding keywords, but be mindful of query length and specificity
            # keywords = topic_analysis.get("keywords_cn", []) + topic_analysis.get("keywords_en", [])
            # if keywords:
            #     query += " " + " ".join(keywords[:3]) # Add a few keywords
        return query

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

            query = self._generate_query_for_chapter(chapter_title, topic_analysis_results)
            logger.info(f"[{self.agent_name}] Retrieving global context for chapter ID '{chapter_id}' (Title: '{chapter_title}') using query: '{query}'")

            try:
                # Assuming retrieval_service.search() returns a list of document dicts
                # And ContentRetrieverAgent.MAX_RESULTS_PER_CHAPTER can be a shared constant or configured.
                # Let's use a smaller number for global retrieval to keep it light.
                MAX_GLOBAL_RESULTS_PER_CHAPTER = 3
                retrieved_docs = self.retrieval_service.search(query, k=MAX_GLOBAL_RESULTS_PER_CHAPTER)

                if retrieved_docs:
                    global_docs_map[chapter_id] = retrieved_docs
                    retrieved_something_overall = True
                    logger.debug(f"Retrieved {len(retrieved_docs)} documents for chapter '{chapter_id}'.")
                else:
                    global_docs_map[chapter_id] = [] # Store empty list if nothing found
                    logger.debug(f"No documents found for chapter '{chapter_id}'.")
            except Exception as e:
                logger.error(f"[{self.agent_name}] Error retrieving documents for chapter '{chapter_id}': {e}", exc_info=True)
                global_docs_map[chapter_id] = [] # Store empty list on error for this chapter
                workflow_state.log_event(f"Global retrieval error for chapter {chapter_id}", {"error": str(e)}, level="ERROR")


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
    class MockRetrievalService:
        def search(self, query: str, k: int) -> List[Dict[str, Any]]:
            logger.debug(f"MockRetrievalService searching for: '{query}', k={k}")
            if "Introduction" in query:
                return [{"doc_id": "doc1", "text": "This is an intro doc.", "title": "Intro Doc 1", "score":0.8}]
            if "Methods" in query:
                return [{"doc_id": "doc2", "text": "Details about methods.", "title": "Methods Doc 1", "score":0.9},
                        {"doc_id": "doc3", "text": "More methods.", "title": "Methods Doc 2", "score":0.7}]
            return []

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

    retrieval_service_instance = MockRetrievalService()
    agent = GlobalContentRetrieverAgent(retrieval_service=retrieval_service_instance)

    mock_parsed_outline = [
        {"id": "ch_intro", "title": "Introduction to Topic", "level": 1},
        {"id": "ch_methods", "title": "Core Methods", "level": 1},
        {"id": "ch_conclusion", "title": "Conclusion", "level": 1}
    ]
    mock_topic_analysis = {"generalized_topic_cn": "测试主题"}
    # This payload would be constructed by OutlineGeneratorAgent
    # Crucially, it needs 'current_outline_md' for the next step (OutlineRefinementAgent)
    # This implies OutlineGeneratorAgent must now also pass its MD outline string in the payload
    # when it creates the TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE task.
    mock_current_md = "# Introduction to Topic\n# Core Methods\n# Conclusion"

    task_payload = {
        "parsed_outline": mock_parsed_outline,
        "current_outline_md": mock_current_md, # Important for next agent
        "topic_analysis_results": mock_topic_analysis,
        "priority": 3,
        "max_chapters_constraint": 5, # Example pass-through
        "min_chapters_constraint": 2  # Example pass-through
    }

    mock_state = MockWorkflowStateGCRA(user_topic="Test Topic")
    mock_state.current_processing_task_id = "gcra_task_123" # Simulate orchestrator setting this

    print(f"\nExecuting GlobalContentRetrieverAgent with MockWorkflowState")
    try:
        agent.execute_task(mock_state, task_payload)

        print("\nWorkflowState after GlobalContentRetrieverAgent execution:")
        print(f"  Global Retrieved Docs Map: {json.dumps(mock_state.global_docs_map_set, indent=2, ensure_ascii=False)}")
        print(f"  Tasks added by agent: {json.dumps(mock_state.added_tasks_gcra, indent=2, ensure_ascii=False)}")

        assert mock_state.global_docs_map_set is not None
        assert "ch_intro" in mock_state.global_docs_map_set
        assert len(mock_state.global_docs_map_set["ch_intro"]) == 1
        assert "ch_methods" in mock_state.global_docs_map_set
        assert len(mock_state.global_docs_map_set["ch_methods"]) == 2
        assert "ch_conclusion" in mock_state.global_docs_map_set # Should be present with empty list
        assert len(mock_state.global_docs_map_set["ch_conclusion"]) == 0

        assert len(mock_state.added_tasks_gcra) == 1
        next_task_info = mock_state.added_tasks_gcra[0]
        assert next_task_info['type'] == TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT
        assert next_task_info['payload']['parsed_outline'] == mock_parsed_outline
        assert next_task_info['payload']['current_outline_md'] == mock_current_md
        assert next_task_info['payload']['topic_analysis_results'] == mock_topic_analysis


        print("\nGlobalContentRetrieverAgent test successful.")

    except Exception as e:
        print(f"Error during GlobalContentRetrieverAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nGlobalContentRetrieverAgent example finished.")
