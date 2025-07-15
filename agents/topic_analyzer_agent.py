import logging
import json
import json_repair # For robust JSON parsing
from typing import Dict, Optional, List, Any
import uuid # For test section unique IDs

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.retrieval_service import RetrievalService, RetrievalServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_GENERATE_OUTLINE
from config import settings as app_settings
from core.json_utils import clean_and_parse_json

logger = logging.getLogger(__name__)

class TopicAnalyzerAgentError(Exception):
    """Custom exception for TopicAnalyzerAgent errors."""
    pass

class TopicAnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing the user's topic, generalizing it,
    extracting relevant keywords, identifying core research questions,
    potential methodologies, generating expanded search queries,
    and elaborating on a global theme. Updates the WorkflowState.
    """

    def __init__(self,
                 llm_service: LLMService,
                 retrieval_service: RetrievalService,
                 prompt_template: Optional[str] = None,
                 system_prompt: Optional[str] = None):
        super().__init__(agent_name="TopicAnalyzerAgent", llm_service=llm_service)
        self.retrieval_service = retrieval_service
        self.prompt_template = prompt_template or app_settings.DEFAULT_TOPIC_ANALYZER_PROMPT
        self.system_prompt = system_prompt or "You are a helpful AI assistant specialized in topic analysis and research planning."
        self.query_expansion_prompt_template = app_settings.QUERY_EXPANSION_PROMPT

        if not self.llm_service:
            raise TopicAnalyzerAgentError("LLMService is required for TopicAnalyzerAgent.")
        if not self.retrieval_service:
            raise TopicAnalyzerAgentError("RetrievalService is required for TopicAnalyzerAgent.")
        if not self.prompt_template:
            raise TopicAnalyzerAgentError("Prompt template (from settings.DEFAULT_TOPIC_ANALYZER_PROMPT) is required for TopicAnalyzerAgent.")

    def _execute_iterative_retrieval(self, initial_queries: List[str], user_topic: str, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
        """
        Executes an iterative retrieval process to gather a comprehensive set of global documents.
        """
        max_iterations = app_settings.GLOBAL_RETRIEVAL_MAX_ITERATIONS
        queries_per_iteration = app_settings.GLOBAL_RETRIEVAL_QUERIES_PER_ITERATION

        all_queries = set(initial_queries)
        all_retrieved_docs = {} # Use dict to store docs by a unique ID to avoid duplicates

        for i in range(max_iterations):
            logger.info(f"[{self.agent_name}] Iterative retrieval, iteration {i+1}/{max_iterations}. Current query count: {len(all_queries)}")

            if not all_queries:
                logger.warning(f"[{self.agent_name}] No queries to process in iteration {i+1}. Stopping.")
                break

            try:
                # 1. Retrieve documents with current set of queries
                retrieved_docs = self.retrieval_service.retrieve(
                    query_texts=list(all_queries),
                    final_top_n=app_settings.GLOBAL_RETRIEVAL_TOP_N_PER_ITERATION
                )

                # Add new documents to the collection, avoiding duplicates
                for doc in retrieved_docs:
                    doc_id = doc.get('child_id') or doc.get('parent_id')
                    if doc_id and doc_id not in all_retrieved_docs:
                        all_retrieved_docs[doc_id] = doc

                if not retrieved_docs:
                    logger.info(f"[{self.agent_name}] No new documents found in iteration {i+1}.")
                    # Don't expand queries if we have no new information
                    continue

                # 2. Prepare for query expansion
                retrieved_content_summary = "\n".join([f"- {d.get('document', '')[:200]}..." for d in retrieved_docs])

                expansion_prompt = self.query_expansion_prompt_template.format(
                    topic=user_topic,
                    existing_queries=json.dumps(list(all_queries), ensure_ascii=False),
                    retrieved_content=retrieved_content_summary,
                    num_new_queries=queries_per_iteration
                )

                # 3. Call LLM to get new queries
                logger.info(f"[{self.agent_name}] Expanding queries for next iteration.")
                raw_expansion_response = self.llm_service.chat(query=expansion_prompt, system_prompt="You are a helpful AI assistant specialized in research query expansion.")

                parsed_expansion = clean_and_parse_json(raw_expansion_response)

                if parsed_expansion and 'new_queries' in parsed_expansion and isinstance(parsed_expansion['new_queries'], list):
                    new_queries = set(parsed_expansion['new_queries'])
                    newly_added = new_queries - all_queries
                    if newly_added:
                        logger.info(f"[{self.agent_name}] Added {len(newly_added)} new queries: {list(newly_added)}")
                        all_queries.update(new_queries)
                    else:
                        logger.info(f"[{self.agent_name}] LLM did not generate any truly new queries. Stopping iteration.")
                        break
                else:
                    logger.warning(f"[{self.agent_name}] Could not parse new queries from LLM response. Response: {raw_expansion_response}")

            except RetrievalServiceError as e:
                logger.error(f"[{self.agent_name}] Retrieval failed during iteration {i+1}: {e}")
                # Decide if we should continue or break on retrieval failure
                break
            except LLMServiceError as e:
                logger.error(f"[{self.agent_name}] LLM query expansion failed during iteration {i+1}: {e}")
                # Decide if we should continue or break
                break

        final_docs = list(all_retrieved_docs.values())
        logger.info(f"[{self.agent_name}] Iterative retrieval finished. Total unique documents: {len(final_docs)}. Total queries generated: {len(all_queries)}")

        # Update workflow state with the final list of queries
        if workflow_state.topic_analysis_results:
            workflow_state.topic_analysis_results['expanded_queries'] = list(all_queries)

        return final_docs


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for user_topic: {task_payload.get('user_topic')}")

        user_topic = task_payload.get('user_topic')
        if not user_topic:
            err_msg = "User topic not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)

        self._log_input(user_topic=user_topic)
        print('--' * 20)
        print(f"TopicAnalyzerAgent executing for user topic: '{user_topic}'")
        print('--' * 20)

        max_queries = app_settings.DEFAULT_MAX_EXPANDED_QUERIES_TOPIC
        prompt_formatted = self.prompt_template.format(user_topic=user_topic, max_expanded_queries=max_queries)

        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for user_topic: {task_payload.get('user_topic')}")

        user_topic = task_payload.get('user_topic')
        if not user_topic:
            err_msg = "User topic not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg)

        self._log_input(user_topic=user_topic)
        print('--' * 20)
        print(f"TopicAnalyzerAgent executing for user topic: '{user_topic}'")
        print('--' * 20)

        max_queries = app_settings.DEFAULT_MAX_EXPANDED_QUERIES_TOPIC
        prompt_formatted = self.prompt_template.format(user_topic=user_topic, max_expanded_queries=max_queries)

        try:
            # Initial Topic Analysis
            logger.info(f"Sending request to LLM for initial topic analysis. User topic: '{user_topic}'.")
            raw_response = self.llm_service.chat(query=prompt_formatted, system_prompt=self.system_prompt)
            logger.debug(f"Raw LLM response for topic analysis: {raw_response}")

            parsed_response = clean_and_parse_json(raw_response)
            if not parsed_response:
                raise TopicAnalyzerAgentError(f"LLM response for initial analysis was not valid or repairable JSON: {raw_response}")

            # --- Start of Robust Key Handling and Validation Logic (remains the same) ---

            # 1. Define strictly required keys and check for their presence
            strictly_required_keys = [
                "generalized_topic_cn", "generalized_topic_en",
                "keywords_cn", "keywords_en", "core_research_questions_cn",
            ]
            missing_strict_keys = [key for key in strictly_required_keys if key not in parsed_response]
            if missing_strict_keys:
                raise TopicAnalyzerAgentError(f"LLM response missing strictly required fields: {', '.join(missing_strict_keys)}. Parsed Response: {parsed_response}")

            # 2. Define recoverable keys (these might be missing or malformed by LLM)
            recoverable_keys = ["potential_methodologies_cn", "expanded_queries"]

            # Attempt to recover these keys if they are nested within a 'class' field by mistake
            # This is a specific heuristic based on observed LLM error patterns.
            if 'class' in parsed_response and isinstance(parsed_response['class'], str):
                class_field_value = parsed_response['class']
                logger.warning(f"Found 'class' field in LLM response. Attempting to recover recoverable_keys from its content: {class_field_value[:200]}...")
                try:
                    # Check if the content of 'class' field looks like it might contain the missing JSON parts
                    if ('"potential_methodologies_cn":' in class_field_value or \
                        '"expanded_queries":' in class_field_value):

                        # Try to make the content of 'class' a valid JSON object string by wrapping with {}
                        potential_json_from_class_str = f"{{{class_field_value}}}"
                        recovered_data_from_class = json_repair.loads(potential_json_from_class_str)

                        for r_key in recoverable_keys:
                            # Only assign if the key is missing in parsed_response AND found in the recovered_data
                            if r_key not in parsed_response and r_key in recovered_data_from_class:
                                parsed_response[r_key] = recovered_data_from_class[r_key]
                                logger.info(f"Successfully recovered '{r_key}' from 'class' field's content.")
                    else:
                        logger.info("Content of 'class' field did not match expected pattern for recovery of 'potential_methodologies_cn' or 'expanded_queries'.")
                except Exception as e_recover_class:
                    # Log error but don't let it stop the process; defaults will be applied later.
                    logger.error(f"Failed to parse or recover from 'class' field. Error: {e_recover_class}. Content: {class_field_value[:500]}", exc_info=False)

            # 3. Ensure all recoverable keys exist in parsed_response, defaulting to empty lists if necessary.
            #    Also, ensure their values are lists and all items within those lists are strings.
            for key in recoverable_keys:
                if key not in parsed_response:
                    logger.warning(f"Key '{key}' is missing after all recovery attempts. Defaulting to an empty list.")
                    parsed_response[key] = []
                elif not isinstance(parsed_response[key], list):
                    logger.warning(f"Key '{key}' was found but its value is not a list (type: {type(parsed_response[key])}). Correcting to an empty list. Original value: {str(parsed_response[key])[:100]}")
                    parsed_response[key] = []
                else:
                    # If it's a list, ensure all items are strings. Non-string items are removed.
                    original_list = parsed_response[key]
                    # Convert basic types to str, filter out others (like None if json_repair puts it)
                    corrected_list = [str(item) for item in original_list if isinstance(item, (str, int, float, bool))]
                    if len(corrected_list) != len(original_list):
                         logger.warning(f"Corrected/filtered non-string items in list for key '{key}'. Original: {original_list}, Corrected: {corrected_list}")
                    parsed_response[key] = corrected_list

            # --- Type Validations for strictly_required_keys (already checked for presence) ---
            # These ensure the core fields have the correct list-of-string structure.
            for key_to_validate in ["keywords_cn", "keywords_en", "core_research_questions_cn"]:
                if not (isinstance(parsed_response[key_to_validate], list) and all(isinstance(item, str) for item in parsed_response[key_to_validate])):
                    # This error means that even a strictly required key, though present, is not a list of strings.
                    raise TopicAnalyzerAgentError(f"Field '{key_to_validate}' must be a list of strings. Found: {parsed_response[key_to_validate]}. Full Response: {parsed_response}")

            # Final type check for recoverable_keys (now they must be lists of strings)
            for key_to_validate in recoverable_keys:
                 if not (isinstance(parsed_response[key_to_validate], list) and all(isinstance(item, str) for item in parsed_response[key_to_validate])):
                    logger.error(f"Internal logic error or unrecoverable type for '{key_to_validate}'. Expected list of strings, got {type(parsed_response[key_to_validate])}. Value: {str(parsed_response[key_to_validate])[:200]}. Forcing to empty list.")
                    # Force to empty list of strings to prevent downstream errors if somehow still not a list of strings.
                    parsed_response[key_to_validate] = [str(item) for item in parsed_response.get(key_to_validate, []) if isinstance(item, str)]


            if not parsed_response.get("expanded_queries"):
                 logger.warning("'expanded_queries' list is empty after all processing. Will attempt to generate from keywords.")
                 # Synthesize queries from keywords if expanded_queries is empty
                 initial_queries = parsed_response.get('keywords_cn', []) + parsed_response.get('keywords_en', [])
                 if initial_queries:
                     parsed_response['expanded_queries'] = list(set(initial_queries))
                 else:
                     parsed_response['expanded_queries'] = [user_topic] # Absolute fallback

            # --- End of Key Handling and Validation Logic ---

            # Synthesize global theme
            theme_elaboration = parsed_response.get("report_global_theme_elaboration")
            if not theme_elaboration or not isinstance(theme_elaboration, str) or not theme_elaboration.strip():
                core_questions_str = '、'.join(parsed_response.get('core_research_questions_cn', ['未指定核心问题']))
                methodologies_str = '、'.join(parsed_response.get('potential_methodologies_cn', ['通用分析方法'])) # Now this should be a list
                theme_elaboration = (
                    f"本报告围绕主题“{parsed_response.get('generalized_topic_cn', user_topic)}”展开，"
                    f"核心研究问题包括：“{core_questions_str}”。"
                    f"分析中可能采用的方法有：“{methodologies_str}”。"
                )
                logger.info(f"Synthesized report_global_theme_elaboration. Theme: {theme_elaboration}")
                parsed_response["report_global_theme_elaboration"] = theme_elaboration

            workflow_state.update_topic_analysis(parsed_response)
            workflow_state.update_report_global_theme(parsed_response["report_global_theme_elaboration"])

            # --- Start of Iterative Global Retrieval ---
            initial_queries = parsed_response.get("expanded_queries", [])
            if not initial_queries:
                logger.warning("No initial queries found in topic analysis results. Using user topic as starting point.")
                initial_queries = [user_topic]

            global_documents = self._execute_iterative_retrieval(initial_queries, user_topic, workflow_state)
            workflow_state.set_global_retrieved_docs(global_documents)
            logger.info(f"Stored {len(global_documents)} documents in global context after iterative retrieval.")
            # --- End of Iterative Global Retrieval ---

            workflow_state.add_task(
                task_type=TASK_TYPE_GENERATE_OUTLINE,
                payload={'topic_details': parsed_response},
                priority=2
            )

            self._log_output(parsed_response)
            success_msg = f"Topic analysis and iterative global retrieval successful for '{user_topic}'. Next task (Generate Outline) added."
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except (LLMServiceError, RetrievalServiceError) as e:
            err_msg = f"A service error occurred during topic analysis or retrieval for '{user_topic}': {e}"
            workflow_state.log_event(f"Service error in TopicAnalyzerAgent for '{user_topic}'", {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg) from e
        except TopicAnalyzerAgentError as e:
            # This will catch JSON parsing errors or other specific agent errors
            err_msg = f"Topic analysis failed for '{user_topic}': {e}"
            workflow_state.log_event(err_msg, {"error": str(e)}, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise
        except Exception as e:
            err_msg = f"Unexpected error in topic analysis for '{user_topic}': {e}"
            workflow_state.log_event(f"Unexpected error in TopicAnalyzerAgent for '{user_topic}'", {"error": str(e)}, level="CRITICAL")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise TopicAnalyzerAgentError(err_msg) from e

# The __main__ block is removed as it's no longer compatible with the new __init__ signature
# which requires a RetrievalService instance. The testing of this agent would now
# need to be done in a more integrated test environment where mock services (LLM, Retrieval)
# can be properly instantiated and injected.

