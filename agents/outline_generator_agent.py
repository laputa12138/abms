import logging
import json
from typing import Dict, Optional, List, Any # Added Any
import uuid # For generating chapter IDs

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_PROCESS_CHAPTER # Import constants
from core.retrieval_service import RetrievalService, RetrievalServiceError # Added
# It might be useful to have a temporary ReportCompiler instance for its parsing logic,
# or replicate a simplified parser here if ReportCompilerAgent isn't available yet or for decoupling.
from agents.report_compiler_agent import ReportCompilerAgent # Assuming it's available for parsing

logger = logging.getLogger(__name__)

class OutlineGeneratorAgentError(Exception):
    """Custom exception for OutlineGeneratorAgent errors."""
    pass

class OutlineGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating a report outline based on topic analysis results
    and retrieved context.
    Updates the WorkflowState with the generated outline (both MD and parsed structure)
    and adds tasks to process each chapter.
    """

    # DEFAULT_PROMPT_TEMPLATE will be removed and sourced from settings

    def __init__(self,
                 llm_service: LLMService,
                 retrieval_service: RetrievalService, # Added
                 prompt_template: Optional[str] = None):
        super().__init__(agent_name="OutlineGeneratorAgent", llm_service=llm_service)
        from config import settings as app_settings # Import here to avoid circular dependency at module load time
        self.prompt_template = prompt_template or app_settings.DEFAULT_OUTLINE_GENERATOR_PROMPT
        if not self.llm_service:
            raise OutlineGeneratorAgentError("LLMService is required for OutlineGeneratorAgent.")
        if not retrieval_service: # Added check for retrieval_service
            raise OutlineGeneratorAgentError("RetrievalService is required for OutlineGeneratorAgent.")
        self.retrieval_service = retrieval_service # Added

        # For parsing the generated Markdown outline into a structured list with IDs
        self._outline_parser = ReportCompilerAgent() # Temporary instance for parsing

    def _parse_markdown_outline_with_ids(self, markdown_outline: str) -> List[Dict[str, Any]]:
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
                # Use a simpler, index-based ID for consistency, as UUIDs are not used for linking.
                item_id = f"outline_{len(parsed_items) + 1}"
                parsed_items.append({
                    "id": item_id,
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
        keywords_cn = analyzed_topic["keywords_cn"]
        keywords_en = analyzed_topic["keywords_en"]
        keywords_cn_str = ", ".join(keywords_cn)
        keywords_en_str = ", ".join(keywords_en)

        # --- Retrieve context using potentially multiple queries before generating outline ---
        retrieved_context_str = "无相关参考资料。" # Default if nothing found or error

        # 1. Get expanded queries from topic analysis
        expanded_queries = analyzed_topic.get('expanded_queries', [])
        if isinstance(expanded_queries, list) and all(isinstance(q, str) for q in expanded_queries):
            logger.info(f"[{self.agent_name}] Using {len(expanded_queries)} expanded queries from topic analysis.")
        else:
            logger.warning(f"[{self.agent_name}] 'expanded_queries' not found or not a list of strings in topic_details. Falling back to primary query. Value: {expanded_queries}")
            expanded_queries = []

        # 2. Construct a primary query as a fallback or addition
        primary_query_parts = [topic_cn, topic_en] + keywords_cn + keywords_en
        primary_query = " ".join(list(filter(None, dict.fromkeys(primary_query_parts)))).strip()

        # 3. Combine expanded and primary queries
        all_queries_for_retrieval = []
        if primary_query:
            all_queries_for_retrieval.append(primary_query)
        all_queries_for_retrieval.extend(q for q in expanded_queries if q.strip())

        # Deduplicate the final list of queries
        all_queries_for_retrieval = list(dict.fromkeys(all_queries_for_retrieval))


        if all_queries_for_retrieval:
            logger.info(f"[{self.agent_name}] Retrieving initial context for outline generation using {len(all_queries_for_retrieval)} queries: {all_queries_for_retrieval}")
            # TODO: Make PRE_OUTLINE_RETRIEVAL_TOP_N configurable in settings.py
            from config import settings
            PRE_OUTLINE_RETRIEVAL_TOP_N = settings.DEFAULT_OUTLINE_GENERATION_RETRIEVAL_TOP_N # Using from settings

            try:
                retrieved_docs = self.retrieval_service.retrieve(
                    query_texts=all_queries_for_retrieval, # Pass list of queries
                    final_top_n=PRE_OUTLINE_RETRIEVAL_TOP_N
                    # vector_top_k, keyword_top_k, hybrid_alpha will use defaults from RetrievalService/settings
                )
                if retrieved_docs:
                    context_parts = []
                    for i, doc in enumerate(retrieved_docs):
                        context_parts.append(f"参考资料片段 {i+1}:\n\"\"\"\n{doc.get('document', '')}\n\"\"\"")
                    retrieved_context_str = "\n\n".join(context_parts)
                    logger.info(f"[{self.agent_name}] Successfully retrieved {len(retrieved_docs)} documents for pre-outline context using {len(all_queries_for_retrieval)} queries.")
                else:
                    logger.info(f"[{self.agent_name}] No documents found during pre-outline retrieval using {len(all_queries_for_retrieval)} queries.")
            except RetrievalServiceError as r_err:
                logger.error(f"[{self.agent_name}] RetrievalService failed during pre-outline context retrieval: {r_err}")
            except Exception as e:
                logger.error(f"[{self.agent_name}] Unexpected error during pre-outline context retrieval: {e}", exc_info=True)
        else:
            logger.info(f"[{self.agent_name}] No valid queries for pre-outline retrieval. Skipping.")
        # --- End of retrieval section ---

        prompt = self.prompt_template.format(
            topic_cn=topic_cn, topic_en=topic_en,
            keywords_cn=keywords_cn_str, keywords_en=keywords_en_str,
            retrieved_context=retrieved_context_str # Added new context
        )

        try:
            logger.info(f"Sending request to LLM for outline generation. Topic: '{topic_cn}'. Context provided: {len(retrieved_context_str)} chars.")
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
            # Ensure outline_finalized is False as it will go through global retrieval then refinement
            workflow_state.set_flag('outline_finalized', False)

            # Add a task for global content retrieval for the generated outline
            global_retrieve_payload = {
                "current_outline_md": outline_markdown, # Pass along the MD string
                "parsed_outline": parsed_outline_with_ids,
                "topic_analysis_results": task_payload.get('topic_details'),
                # Pass through constraints if they were part of this task's context or from config
                "max_chapters_constraint": task_payload.get('max_chapters_constraint', workflow_state.get_flag('max_chapters_constraint', 10)),
                "min_chapters_constraint": task_payload.get('min_chapters_constraint', workflow_state.get_flag('min_chapters_constraint', 3))
            }

            from core.workflow_state import TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE # Import new task type
            workflow_state.add_task(
                task_type=TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE,
                payload=global_retrieve_payload,
                priority=task_payload.get('priority', 2) + 1 # Next step after outline generation
            )

            self._log_output({"markdown_outline": outline_markdown, "parsed_items_count": len(parsed_outline_with_ids)})
            success_msg = f"Initial outline generation successful for '{topic_cn}'. {len(parsed_outline_with_ids)} chapters. Task for global outline content retrieval added."
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
    from core.workflow_state import WorkflowState, TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE # Updated import
    from core.retrieval_service import RetrievalService # For mock
    from config import settings # For PRE_OUTLINE_RETRIEVAL_TOP_N default

    class MockRetrievalServiceForOGA: # OGA for OutlineGeneratorAgent
        def retrieve(self, query_texts: List[str], final_top_n: int, **kwargs) -> List[Dict[str, Any]]: # Added **kwargs for other params
            logger.debug(f"MockRetrievalServiceForOGA.retrieve called with queries: {query_texts}, final_top_n={final_top_n}")
            # Simulate some results if any query contains "ABMS" or "JADC2"
            # This mock is simple; a real scenario might have different docs for different queries.
            docs_to_return = []
            if any("ABMS" in qt for qt in query_texts) or any("JADC2" in qt for qt in query_texts):
                docs_to_return.extend([
                    {"document": "这是关于ABMS的第一个参考资料片段。", "score": 0.9, "child_text_preview": "ABMS片段1...", "child_id": "c1", "parent_id": "p1", "source_document_name": "docA.txt", "retrieval_source": "mock_vector_q1"},
                    {"document": "ABMS的第二个重要参考信息。", "score": 0.8, "child_text_preview": "ABMS片段2...", "child_id": "c2", "parent_id": "p2", "source_document_name": "docB.txt", "retrieval_source": "mock_vector_q2"}
                ])
            if any("具体技术" in qt for qt in query_texts):
                 docs_to_return.append(
                    {"document": "关于具体技术的文档。", "score": 0.85, "child_text_preview": "技术细节...", "child_id": "c3", "parent_id": "p3", "source_document_name": "docC.txt", "retrieval_source": "mock_vector_q3"}
                 )

            # Respect final_top_n by returning only that many, sorted by score (mock sort)
            docs_to_return.sort(key=lambda x: x['score'], reverse=True)
            return docs_to_return[:final_top_n]

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
    retrieval_service_instance = MockRetrievalServiceForOGA() # Create mock retrieval
    # Pass retrieval_service to constructor
    outline_agent = OutlineGeneratorAgent(llm_service=llm_service_instance, retrieval_service=retrieval_service_instance)

    # Include expanded_queries in mock_topic_analysis
    mock_topic_analysis_with_expanded = {
        "generalized_topic_cn": "ABMS系统", "generalized_topic_en": "ABMS",
        "keywords_cn": ["ABMS", "JADC2"], "keywords_en": ["ABMS", "JADC2"],
        "expanded_queries": [
            "ABMS系统架构",
            "JADC2与ABMS的关联",
            "ABMS 具体技术" # This query should trigger the third mock doc
        ]
    }
    # Simulate current_processing_task_id being set by orchestrator
    mock_state_oga = MockWorkflowStateOGA(user_topic="ABMS系统", topic_analysis_results=mock_topic_analysis_with_expanded)
    mock_state_oga.current_processing_task_id = "mock_outline_gen_task_id_123"

    task_payload_for_agent_oga = {'topic_details': mock_topic_analysis_with_expanded, 'priority': 2}

    # Test with no expanded queries to ensure fallback works
    mock_topic_analysis_no_expanded = {
        "generalized_topic_cn": "ABMS系统", "generalized_topic_en": "ABMS",
        "keywords_cn": ["ABMS", "JADC2"], "keywords_en": ["ABMS", "JADC2"],
        # "expanded_queries": [] # or missing key
    }
    mock_state_oga_no_expanded = MockWorkflowStateOGA(user_topic="ABMS系统", topic_analysis_results=mock_topic_analysis_no_expanded)
    mock_state_oga_no_expanded.current_processing_task_id = "mock_outline_gen_task_id_456"
    task_payload_no_expanded = {'topic_details': mock_topic_analysis_no_expanded, 'priority': 2}


    print(f"\nExecuting OutlineGeneratorAgent with MockWorkflowStateOGA (With Expanded Queries)")
    try:
        outline_agent.execute_task(mock_state_oga, task_payload_for_agent_oga)

        print("\nWorkflowState after OutlineGeneratorAgent execution:")
        print(f"  Outline Markdown: \n{mock_state_oga.current_outline_md}")
        print(f"  Parsed Outline (from state): {json.dumps(mock_state_oga.parsed_outline, indent=2, ensure_ascii=False)}")
        print(f"  Tasks added by agent: {json.dumps(mock_state_oga.added_tasks_oga, indent=2, ensure_ascii=False)}")
        print(f"  Outline Finalized Flag: {mock_state_oga.get_flag('outline_finalized')}")


        assert mock_state_oga.current_outline_md is not None
        assert mock_state_oga.parsed_outline is not None and len(mock_state_oga.parsed_outline) > 0
        assert mock_state_oga.get_flag('outline_finalized') is False

        assert len(mock_state_oga.added_tasks_oga) == 1
        added_task_details = mock_state_oga.added_tasks_oga[0]
        assert added_task_details['type'] == TASK_TYPE_GLOBAL_RETRIEVE_FOR_OUTLINE
        assert added_task_details['payload']['topic_analysis_results'] == mock_topic_analysis_with_expanded
        # More assertions can be added here based on the mock retrieval results affecting the context for LLM

        print("\nOutlineGeneratorAgent test successful with expanded queries.")

        # Test the fallback scenario (no expanded queries)
        print(f"\nExecuting OutlineGeneratorAgent with MockWorkflowStateOGA (No Expanded Queries - Fallback Test)")
        outline_agent.execute_task(mock_state_oga_no_expanded, task_payload_no_expanded)
        print("\nWorkflowState after OutlineGeneratorAgent execution (No Expanded Queries):")
        print(f"  Outline Markdown: \n{mock_state_oga_no_expanded.current_outline_md}")
        assert mock_state_oga_no_expanded.current_outline_md is not None
        # Further assertions for this case...
        print("\nOutlineGeneratorAgent fallback test successful.")


    except Exception as e:
        print(f"Error during OutlineGeneratorAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nOutlineGeneratorAgent example finished.")
