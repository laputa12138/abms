import logging
import json # For parsing LLM relevance check response
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED # Import constants
from config import settings as app_settings # Import settings for prompts

logger = logging.getLogger(__name__)

class ChapterWriterAgentError(Exception):
    """Custom exception for ChapterWriterAgent errors."""
    pass

class ChapterWriterAgent(BaseAgent):
    """
    Agent responsible for writing a chapter of a report.
    It first uses an LLM to verify relevance of retrieved documents.
    Then, it uses relevant documents to generate chapter content via LLM.
    Finally, it programmatically appends citations for used documents.
    Updates WorkflowState with the written content and queues evaluation task.
    """

    def __init__(self, llm_service: LLMService,
                 writer_prompt_template: Optional[str] = None,
                 relevance_check_prompt_template: Optional[str] = None):
        super().__init__(agent_name="ChapterWriterAgent", llm_service=llm_service)
        self.writer_prompt_template = writer_prompt_template or app_settings.DEFAULT_CHAPTER_WRITER_PROMPT
        self.relevance_check_prompt_template = relevance_check_prompt_template or app_settings.DEFAULT_RELEVANCE_CHECK_PROMPT # Placeholder, will be added in settings.py
        if not self.llm_service:
            raise ChapterWriterAgentError("LLMService is required for ChapterWriterAgent.")

    def _is_document_relevant(self, chapter_title: str, document_text: str, document_id: str) -> bool:
        """
        Uses LLM to check if a document is relevant to the chapter title.
        """
        if not self.relevance_check_prompt_template:
            logger.warning(f"Relevance check prompt template is not set. Assuming all documents are relevant for doc ID: {document_id}")
            return True # Fallback if prompt is missing, though this should be configured

        prompt = self.relevance_check_prompt_template.format(
            chapter_title=chapter_title,
            document_text=document_text[:2000] # Truncate to avoid overly long prompts for relevance check
        )
        try:
            response_str = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一个内容相关性判断助手。请根据提供的章节标题和文档片段，判断文档片段是否与章节标题高度相关，并严格以JSON格式返回结果。"
            )
            logger.debug(f"Relevance check for doc ID '{document_id}', chapter '{chapter_title}'. LLM response: {response_str}")
            # Expecting a JSON response like: {"is_relevant": true/false}
            response_json = json.loads(response_str)
            is_relevant = response_json.get("is_relevant", False)
            if not isinstance(is_relevant, bool):
                logger.warning(f"LLM relevance check for doc ID '{document_id}' returned non-boolean 'is_relevant' value: {is_relevant}. Defaulting to False.")
                return False
            return is_relevant
        except LLMServiceError as e:
            logger.error(f"LLM service error during relevance check for doc ID '{document_id}': {e}. Assuming not relevant.")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM relevance check for doc ID '{document_id}': {response_str}. Error: {e}. Assuming not relevant.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during relevance check for doc ID '{document_id}': {e}. Assuming not relevant.", exc_info=True)
            return False

    def _format_content_for_generation(self, relevant_docs: List[Dict[str, any]]) -> str:
        """
        Formats the text of relevant documents for the main generation prompt.
        """
        if not relevant_docs:
            return "无相关参考资料提供。"

        formatted_str = ""
        for i, item in enumerate(relevant_docs):
            doc_text = item.get('document', '无效的参考资料片段')
            # Simple concatenation, no source info here for the LLM
            formatted_str += f"参考资料片段 {i+1}:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        return formatted_str.strip()

    def _generate_citations_string(self, used_documents: List[Dict[str, any]]) -> str:
        """
        Generates the citation string for all used documents.
        Format: [引用来源：{file_name}。原文表述：{source_text_of_parent}]
        """
        if not used_documents:
            return ""

        citations = []
        for doc in used_documents:
            file_name = doc.get('source_document_id', '未知文档')
            source_text_of_parent = doc.get('document', '无法获取原文') # 'document' holds parent_text
            citations.append(f"[引用来源：{file_name}。原文表述：{source_text_of_parent}]")

        return "\n\n" + "\n".join(citations) if citations else ""


    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        task_id = workflow_state.current_processing_task_id
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Starting execution for chapter_key: {task_payload.get('chapter_key')}, title: {task_payload.get('chapter_title')}")

        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title')

        if not chapter_key or not chapter_title:
            err_msg = "Chapter key or title not found in task payload."
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ChapterWriterAgentError(err_msg)

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data:
            err_msg = f"Chapter data for key '{chapter_key}' not found in WorkflowState."
            workflow_state.log_event(err_msg, level="ERROR")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}")
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ChapterWriterAgentError(err_msg)

        all_retrieved_docs = chapter_data.get('retrieved_docs', [])
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Chapter: '{chapter_title}'. Initial retrieved docs count: {len(all_retrieved_docs)}")
        self._log_input(chapter_title=chapter_title, initial_retrieved_docs_count=len(all_retrieved_docs))

        # 1. LLM Relevance Check for each document
        relevant_docs_for_generation = []
        if not self.relevance_check_prompt_template: # Check if template is loaded
             logger.warning("Relevance check prompt template is missing from settings. "
                            "ChapterWriterAgent will proceed assuming all documents are relevant, "
                            "but this is not the intended behavior for full traceability.")
             relevant_docs_for_generation = all_retrieved_docs # Fallback
        else:
            for doc in all_retrieved_docs:
                doc_text = doc.get('document', '')
                doc_id = doc.get('parent_id', doc.get('child_id', 'unknown_id'))
                if doc_text:
                    if self._is_document_relevant(chapter_title, doc_text, doc_id):
                        relevant_docs_for_generation.append(doc)
                        logger.debug(f"Document '{doc_id}' deemed relevant for chapter '{chapter_title}'.")
                    else:
                        logger.debug(f"Document '{doc_id}' deemed NOT relevant for chapter '{chapter_title}'.")
                else:
                    logger.warning(f"Document '{doc_id}' has no text content. Skipping relevance check.")

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Relevant docs after LLM check: {len(relevant_docs_for_generation)} (out of {len(all_retrieved_docs)})")
        workflow_state.log_event(f"Relevance check complete for chapter '{chapter_title}'. Relevant docs: {len(relevant_docs_for_generation)}/{len(all_retrieved_docs)}",
                                 {"chapter_key": chapter_key, "relevant_count": len(relevant_docs_for_generation), "total_initial_count": len(all_retrieved_docs)})


        # 2. Content Generation using relevant documents
        formatted_content_for_llm = self._format_content_for_generation(relevant_docs_for_generation)
        writer_prompt = self.writer_prompt_template.format(
            chapter_title=chapter_title,
            retrieved_content_formatted=formatted_content_for_llm # This is now just text snippets
        )

        try:
            logger.info(f"Sending request to LLM for chapter writing. Chapter: '{chapter_title}' (Key: {chapter_key}). Using {len(relevant_docs_for_generation)} relevant documents.")
            generated_chapter_text = self.llm_service.chat(
                query=writer_prompt,
                system_prompt="你是一位精通特定领域知识的专业中文报告撰写者，擅长整合信息并清晰表达。请专注于根据提供的参考资料撰写内容，不要自行添加引用标记。"
            )
            logger.debug(f"Raw LLM response for chapter '{chapter_title}' (first 200 chars): {generated_chapter_text[:200]}")

            if not generated_chapter_text or not generated_chapter_text.strip():
                logger.warning(f"LLM returned empty content for chapter: '{chapter_title}'.")
                generated_chapter_text = "" # Ensure empty string

            # 3. Programmatic Citation
            # Citations are generated based on the documents that were *actually sent* to the LLM for generation.
            # This assumes the LLM used (or had the chance to use) all `relevant_docs_for_generation`.
            citations_string = self._generate_citations_string(relevant_docs_for_generation)
            final_chapter_text_with_citations = generated_chapter_text.strip() + citations_string

            # Update WorkflowState
            # Store the list of documents that passed relevance check and were used for generation.
            workflow_state.update_chapter_content(
                chapter_key,
                final_chapter_text_with_citations,
                retrieved_docs=relevant_docs_for_generation, # Store only the relevant (used) docs
                is_new_version=False # First version
            )
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED)

            # Add next task: Evaluate Chapter
            workflow_state.add_task(
                task_type=TASK_TYPE_EVALUATE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title},
                priority=task_payload.get('priority', 6) + 1
            )

            self._log_output({"chapter_key": chapter_key, "content_length": len(final_chapter_text_with_citations), "used_sources_count": len(relevant_docs_for_generation)})
            success_msg = (f"Chapter writing successful for '{chapter_title}'. {len(relevant_docs_for_generation)} sources used. "
                           f"Next task (Evaluate Chapter) added.")
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except LLMServiceError as e:
            err_msg = f"LLM service failed for chapter '{chapter_title}': {e}"
            workflow_state.log_event(f"LLM service error writing chapter '{chapter_title}'", {"error": str(e)}, level="ERROR")
            workflow_state.add_chapter_error(chapter_key, f"LLM service error: {e}")
            logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ChapterWriterAgentError(err_msg)
        except Exception as e:
            err_msg = f"Unexpected error writing chapter '{chapter_title}': {e}"
            workflow_state.log_event(f"Unexpected error in ChapterWriterAgent for '{chapter_title}'", {"error": str(e)}, level="CRITICAL")
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error: {e}")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ChapterWriterAgentError(err_msg)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "ABMS系统概述" in query: return "这是ABMS系统概述的初稿，基于提供的父块上下文。"
            return "模拟生成的章节内容。"

    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_WRITING_NEEDED

    class MockWorkflowStateCWA(WorkflowState): # CWA for ChapterWriterAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, retrieved_docs: List):
            super().__init__(user_topic)
            # Pre-populate chapter data as if retrieval was done
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_WRITING_NEEDED,
                'content': None, 'retrieved_docs': retrieved_docs,
                'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_cwa = []

        def update_chapter_content(self, chapter_key: str, content: str,
                                   retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                                   is_new_version: bool = True):
            super().update_chapter_content(chapter_key, content, retrieved_docs, is_new_version)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' content updated.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' status updated to {status}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_cwa.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)


    llm_service_instance = MockLLMService()
    writer_agent = ChapterWriterAgent(llm_service=llm_service_instance)

    test_chapter_key = "chap_abms_overview"
    test_chapter_title = "ABMS系统概述"
    mock_retrieved_data = [
        {"document": "父块1：ABMS是一个复杂的系统...", "score": 0.9, "source": "hybrid", "child_text_preview": "子块1预览..."},
        {"document": "父块2：JADC2与ABMS的关系...", "score": 0.85, "source": "hybrid", "child_text_preview": "子块2预览..."}
    ]

    mock_state_cwa = MockWorkflowStateCWA(user_topic="ABMS",
                                          chapter_key=test_chapter_key,
                                          chapter_title=test_chapter_title,
                                          retrieved_docs=mock_retrieved_data)

    task_payload_for_agent_cwa = {'chapter_key': test_chapter_key, 'chapter_title': test_chapter_title}

    print(f"\nExecuting ChapterWriterAgent for chapter: '{test_chapter_title}' with MockWorkflowStateCWA")
    try:
        writer_agent.execute_task(mock_state_cwa, task_payload_for_agent_cwa)

        print("\nWorkflowState after ChapterWriterAgent execution:")
        chapter_info = mock_state_cwa.get_chapter_data(test_chapter_key)
        if chapter_info:
            print(f"  Chapter '{test_chapter_key}' Status: {chapter_info.get('status')}")
            print(f"  Content Preview: {chapter_info.get('content', '')[:100]}...")
            print(f"  Retrieved Docs were used: {bool(chapter_info.get('retrieved_docs'))}")

        print(f"  Tasks added by agent: {json.dumps(mock_state_cwa.added_tasks_cwa, indent=2, ensure_ascii=False)}")

        assert chapter_info is not None
        assert chapter_info.get('status') == STATUS_EVALUATION_NEEDED
        assert chapter_info.get('content') == "这是ABMS系统概述的初稿，基于提供的父块上下文。" # From mock LLM
        assert len(mock_state_cwa.added_tasks_cwa) == 1
        assert mock_state_cwa.added_tasks_cwa[0]['type'] == TASK_TYPE_EVALUATE_CHAPTER
        assert mock_state_cwa.added_tasks_cwa[0]['payload']['chapter_key'] == test_chapter_key

        print("\nChapterWriterAgent test successful with MockWorkflowStateCWA.")

    except Exception as e:
        print(f"Error during ChapterWriterAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nChapterWriterAgent example finished.")
