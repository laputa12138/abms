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
                 single_snippet_writer_prompt_template: Optional[str] = None,
                 integration_prompt_template: Optional[str] = None,
                 relevance_check_prompt_template: Optional[str] = None):
        super().__init__(agent_name="ChapterWriterAgent", llm_service=llm_service)
        self.single_snippet_writer_prompt_template = single_snippet_writer_prompt_template or app_settings.DEFAULT_SINGLE_SNIPPET_WRITER_PROMPT
        self.integration_prompt_template = integration_prompt_template or app_settings.DEFAULT_CHAPTER_INTEGRATION_PROMPT
        self.relevance_check_prompt_template = relevance_check_prompt_template or app_settings.DEFAULT_RELEVANCE_CHECK_PROMPT
        if not self.llm_service:
            raise ChapterWriterAgentError("LLMService is required for ChapterWriterAgent.")
        if not self.single_snippet_writer_prompt_template:
            raise ChapterWriterAgentError("Single snippet writer prompt template is required.")
        if not self.integration_prompt_template:
            raise ChapterWriterAgentError("Integration prompt template is required.")

    def _is_document_relevant(self, chapter_title: str, document_text: str, document_id: str) -> bool:
        """
        Uses LLM to check if a document is relevant to the chapter title.
        """
        if not self.relevance_check_prompt_template:
            logger.warning(f"Relevance check prompt template is not set. Assuming all documents are relevant for doc ID: {document_id}")
            return True # Fallback if prompt is missing, though this should be configured

        prompt = self.relevance_check_prompt_template.format(
            chapter_title=chapter_title,
            document_text=document_text[:app_settings.RELEVANCE_CHECK_MAX_TEXT_LENGTH] # Use configured length
        )
        try:
            response_str = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一个内容相关性判断助手。请根据提供的章节标题和文档片段，判断文档片段是否与章节标题高度相关，并严格以JSON格式返回结果。"
            )
            logger.debug(f"Relevance check for doc ID '{document_id}', chapter '{chapter_title}'. LLM response: {response_str}")
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

    def _format_single_snippet_for_llm(self, single_doc_text: str) -> str:
        if not single_doc_text:
            return "无有效参考资料片段。"
        return single_doc_text

    def _generate_citations_string(self, used_document: Dict[str, any]) -> str: # Takes a single doc now
        if not used_document:
            return ""
        file_name = used_document.get('source_document_name', '未知文档')
        source_text_of_parent = used_document.get('document', '无法获取原文')
        return f"[引用来源：{file_name}。原文表述：{source_text_of_parent}]"


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

        report_global_theme = workflow_state.get_report_global_theme() or "未提供全局主题"
        key_terms_definitions_dict = workflow_state.get_key_terms_definitions() or {}
        key_terms_definitions_formatted = "\n".join(
            [f"- {term}: {definition}" for term, definition in key_terms_definitions_dict.items()]
        )
        if not key_terms_definitions_formatted:
            key_terms_definitions_formatted = "未提供关键术语定义"

        all_retrieved_docs_for_chapter = chapter_data.get('retrieved_docs', [])
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Chapter: '{chapter_title}'. Initial retrieved docs count: {len(all_retrieved_docs_for_chapter)}")
        self._log_input(chapter_title=chapter_title, initial_retrieved_docs_count=len(all_retrieved_docs_for_chapter), report_global_theme=report_global_theme, key_terms_definitions=key_terms_definitions_dict)

        relevant_docs_for_generation = []
        if not self.relevance_check_prompt_template:
             logger.warning("Relevance check prompt template is missing from settings. ChapterWriterAgent will proceed assuming all documents are relevant.")
             relevant_docs_for_generation = all_retrieved_docs_for_chapter
        else:
            for doc_idx, doc in enumerate(all_retrieved_docs_for_chapter):
                doc_text = doc.get('document', '')
                doc_id = doc.get('parent_id', doc.get('child_id', f'unknown_id_idx_{doc_idx}'))
                if doc_text:
                    if self._is_document_relevant(chapter_title, doc_text, doc_id):
                        relevant_docs_for_generation.append(doc)
                        logger.debug(f"Document '{doc_id}' (Source: {doc.get('source_document_name', 'N/A')}) deemed relevant for chapter '{chapter_title}'.")
                    else:
                        logger.debug(f"Document '{doc_id}' (Source: {doc.get('source_document_name', 'N/A')}) deemed NOT relevant for chapter '{chapter_title}'.")
                else:
                    logger.warning(f"Document '{doc_id}' (Source: {doc.get('source_document_name', 'N/A')}) has no text content. Skipping relevance check.")

        num_initial_docs = len(all_retrieved_docs_for_chapter)
        num_relevant_docs = len(relevant_docs_for_generation)
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Relevant docs after LLM check: {num_relevant_docs} (out of {num_initial_docs}) for chapter '{chapter_title}'")
        workflow_state.log_event(
            f"Relevance check complete for chapter '{chapter_title}'. Relevant docs: {num_relevant_docs}/{num_initial_docs}",
            {
                "chapter_key": chapter_key,
                "relevant_count": num_relevant_docs,
                "total_initial_count": num_initial_docs,
            }
        )

        content_blocks: List[Dict[str, str]] = []
        all_snippet_citation_jsons_for_chapter: List[Dict] = []

        if not relevant_docs_for_generation:
            logger.warning(f"No relevant documents found to generate content blocks for chapter '{chapter_title}'. Integration step will handle this.")
        else:
            logger.info(f"Generating content for chapter '{chapter_title}' snippet by snippet using {len(relevant_docs_for_generation)} relevant documents.")
            for i, doc in enumerate(relevant_docs_for_generation):
                single_doc_text = doc.get('document', '')
                doc_name = doc.get('source_document_name', '未知文档')
                doc_score = doc.get('score')
                doc_chunk_id = doc.get('parent_id', doc.get('child_id', 'unknown_id'))

                if not single_doc_text.strip():
                    logger.warning(f"Skipping snippet generation for doc ID '{doc_chunk_id}' from source '{doc_name}' due to empty content.")
                    continue

                try:
                    snippet_writer_prompt = self.single_snippet_writer_prompt_template.format(
                        report_global_theme=report_global_theme,
                        key_terms_definitions_formatted=key_terms_definitions_formatted,
                        chapter_title=chapter_title,
                        single_document_snippet=single_doc_text
                    )
                except KeyError as e:
                    logger.error(f"KeyError formatting snippet_writer_prompt: {e}. Using fallback.")
                    snippet_writer_prompt = f"章节标题：\n{chapter_title}\n\n单一段落参考资料：\n\"\"\"\n{single_doc_text}\n\"\"\"\n\n请撰写内容。"

                preliminary_text_str = ""
                citation_json_for_snippet = {
                    "generated_snippet_id": f"snippet_{chapter_key}_{i}",
                    "generated_text": "",
                    "source_references": [{
                        "document_name": doc_name,
                        "original_text_snippet": single_doc_text,
                        "retrieved_score": doc_score,
                        "document_chunk_id": doc_chunk_id
                    }]
                }

                try:
                    logger.debug(f"Generating text for snippet {i+1}/{len(relevant_docs_for_generation)} of chapter '{chapter_title}'. Snippet source: {doc_name}")
                    preliminary_text_str = self.llm_service.chat(
                        query=snippet_writer_prompt,
                        system_prompt="你是一位专业的报告撰写员，请根据提供的单一段落参考资料撰写相关的描述性内容。"
                    )

                    if not preliminary_text_str or not preliminary_text_str.strip():
                        logger.warning(f"LLM returned empty content for snippet from doc: {doc_name} for chapter '{chapter_title}'. Using empty string.")
                        preliminary_text_str = ""

                except LLMServiceError as e_snippet_llm:
                    err_msg_snippet = f"LLM service failed while generating text for snippet from doc {doc_name} in chapter '{chapter_title}': {e_snippet_llm}"
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg_snippet}", exc_info=False)
                    preliminary_text_str = f"[系统提示：生成此部分内容时遇到LLM服务错误 - {e_snippet_llm}]"
                except Exception as e_snippet_unexpected:
                    err_msg_snippet_unexpected = f"Unexpected error generating content for snippet from doc {doc_name} in chapter '{chapter_title}': {e_snippet_unexpected}"
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg_snippet_unexpected}", exc_info=True)
                    preliminary_text_str = f"[系统提示：生成此部分内容时发生意外错误 - {e_snippet_unexpected}]"

                citation_json_for_snippet["generated_text"] = preliminary_text_str.strip()
                all_snippet_citation_jsons_for_chapter.append(citation_json_for_snippet)

                citation_string_for_integration = self._generate_citations_string(doc)
                content_blocks.append({
                    "generated_text": preliminary_text_str.strip(),
                    "citation_string": citation_string_for_integration
                })
                logger.debug(f"Generated content block for snippet {i+1}. Text length: {len(preliminary_text_str)}, Citation (string): {citation_string_for_integration}")

        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Preliminary content blocks generation phase complete for chapter '{chapter_title}'. {len(content_blocks)} blocks generated.")
        if not content_blocks and relevant_docs_for_generation:
             workflow_state.add_chapter_error(chapter_key, "All snippet generation attempts failed despite having relevant documents.")

        try:
            logger.info(f"Proceeding to content integration for chapter '{chapter_title}'.")
            final_integrated_content = self._integrate_chapter_content(
                chapter_title=chapter_title,
                content_blocks=content_blocks,
                report_global_theme=report_global_theme,
                key_terms_definitions_formatted=key_terms_definitions_formatted
            )

            # Check if placeholder was returned, and if so, log details
            # Use the specific placeholder defined in settings or a constant
            # This string is now fetched from settings in _integrate_chapter_content
            # We compare against that method's known return for empty content.
            # The actual string might be app_settings.DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER
            # or the fallback defined in _integrate_chapter_content.
            # For robustness, check against both if settings might not be populated.
            placeholder_from_settings = getattr(app_settings, 'DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER', "[本章节未能生成有效内容，我们正在努力改进。]")
            # Also check against the old placeholder in case it's encountered from a previous run state not yet migrated
            old_placeholder_text = "(本章节未能生成内容，因为没有相关的文本片段可供处理，或者所有片段处理均失败。)"


            if final_integrated_content == placeholder_from_settings or final_integrated_content == old_placeholder_text:
                logger.error(
                    f"ChapterWriterAgent: Chapter '{chapter_title}' (key: {chapter_key}) resulted in placeholder content. "
                    f"Initial docs retrieved: {num_initial_docs}. "
                    f"Docs deemed relevant: {num_relevant_docs}. "
                    f"Content blocks generated for integration: {len(content_blocks)}. "
                    "This indicates either no relevant documents were found, or all attempts to generate text from them failed, "
                    "or the integration step itself decided to return a placeholder."
                )
                workflow_state.add_chapter_error(chapter_key,
                    f"Content generation resulted in placeholder. Initial docs: {num_initial_docs}, Relevant: {num_relevant_docs}, Blocks: {len(content_blocks)}")

            appended_citations_text = "\n\n"
            if all_snippet_citation_jsons_for_chapter:
                appended_citations_text += "--- 参考来源列表 ---\n"
                for idx, citation_json in enumerate(all_snippet_citation_jsons_for_chapter):
                    source_ref = citation_json.get("source_references", [{}])[0]
                    doc_name = source_ref.get("document_name", "未知文档")
                    original_snippet = source_ref.get("original_text_snippet", "无原始片段")
                    appended_citations_text += (
                        f"\n[{idx+1}] 文档名: {doc_name}\n"
                        f"   原始参考依据: \"{original_snippet[:200]}...\"\n"
                    )
                appended_citations_text += "\n--- 参考来源列表结束 ---\n"
                final_content_with_appended_citations = final_integrated_content + appended_citations_text
            else:
                final_content_with_appended_citations = final_integrated_content
                if not (final_integrated_content == placeholder_from_settings or final_integrated_content == old_placeholder_text):
                    logger.info(f"No structured citations to append for chapter '{chapter_title}' but content was generated.")

            workflow_state.update_chapter_content(
                chapter_key,
                final_content_with_appended_citations,
                retrieved_docs=relevant_docs_for_generation,
                citations_structured_list=all_snippet_citation_jsons_for_chapter,
                is_new_version=False
            )
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED)

            workflow_state.add_task(
                task_type=TASK_TYPE_EVALUATE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title},
                priority=task_payload.get('priority', 6) + 1
            )

            self._log_output({
                "chapter_key": chapter_key,
                "content_length": len(final_integrated_content),
                "used_sources_count": len(relevant_docs_for_generation),
                "blocks_integrated": len(content_blocks)
            })
            success_msg = (f"Chapter '{chapter_title}' writing and integration process completed. "
                           f"{len(content_blocks)} blocks processed. Next task (Evaluate Chapter) added.")
            logger.info(f"[{self.agent_name}] Task ID: {task_id} - {success_msg}")
            if task_id: workflow_state.complete_task(task_id, success_msg, status='success')

        except Exception as e_integration_final:
            err_msg = f"Critical error during final integration or workflow update for chapter '{chapter_title}': {e_integration_final}"
            workflow_state.log_event(err_msg, {"error": str(e_integration_final)}, level="CRITICAL")
            workflow_state.add_chapter_error(chapter_key, f"Integration/Workflow critical error: {e_integration_final}")
            logger.critical(f"[{self.agent_name}] Task ID: {task_id} - {err_msg}", exc_info=True)
            if task_id: workflow_state.complete_task(task_id, err_msg, status='failed')
            raise ChapterWriterAgentError(err_msg) from e_integration_final

    def _integrate_chapter_content(self,
                                   chapter_title: str,
                                   content_blocks: List[Dict[str, str]],
                                   report_global_theme: str,
                                   key_terms_definitions_formatted: str) -> str:
        """
        Integrates multiple preliminary content blocks (each with its citation) into a single,
        coherent chapter using an LLM, incorporating global theme and key terms.
        If no content_blocks are provided, returns a standard placeholder message from settings.
        """
        if not content_blocks:
            logger.warning(
                f"No content blocks provided for integration for chapter '{chapter_title}'. "
                f"This usually means no relevant documents were found/passed relevance check, or all snippet generations failed. "
                f"Returning standard placeholder."
            )
            return getattr(app_settings, 'DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER',
                           "[本章节未能生成有效内容，我们正在努力改进。]") # New default placeholder

        formatted_blocks_for_integration = ""
        for i, block in enumerate(content_blocks):
            block_text = block.get('generated_text', "").strip()
            block_citation = block.get('citation_string', "[溯源信息缺失]").strip()
            combined_text_with_citation = f"{block_text} {block_citation}" if block_text else block_citation
            formatted_blocks_for_integration += f"文本块 {i+1}:\n{combined_text_with_citation.strip()}\n---\n\n"

        try:
            integration_prompt_str = self.integration_prompt_template.format(
                report_global_theme=report_global_theme,
                key_terms_definitions_formatted=key_terms_definitions_formatted,
                chapter_title=chapter_title,
                preliminary_content_blocks_formatted=formatted_blocks_for_integration.strip()
            )
        except KeyError as e:
            logger.error(f"KeyError formatting integration_prompt_str: {e}. Using fallback.")
            integration_prompt_str = f"章节标题：\n{chapter_title}\n\n待整合的文本块列表：\n---\n{formatted_blocks_for_integration.strip()}\n---\n请整合内容。"

        try:
            logger.info(f"Sending {len(content_blocks)} content blocks to LLM for integration for chapter '{chapter_title}'. Prompt length: {len(integration_prompt_str)}")
            integrated_chapter_text = self.llm_service.chat(
                query=integration_prompt_str,
                system_prompt="你是一位高级报告编辑。你的任务是将多个文本块整合成一个连贯的章节。每个文本块包含一段内容和其对应的引用标记。在整合时，你必须确保每个引用标记都紧跟在其对应的文本内容之后，或者以其他方式清晰地保持文本与其原始引用的直接关联。整合后的章节中，原始文本和其引用的配对关系必须清晰可见且准确无误。"
            )

            if not integrated_chapter_text or not integrated_chapter_text.strip():
                logger.warning(f"LLM returned empty content during integration for chapter: '{chapter_title}'. Falling back to simple concatenation.")
                concatenated_fallback_parts = []
                for block in content_blocks:
                    concatenated_fallback_parts.append(f"{block.get('generated_text', '').strip()}\n{block.get('citation_string', '').strip()}")
                return "\n\n".join(concatenated_fallback_parts).strip()

            logger.info(f"LLM integration successful for chapter '{chapter_title}'. Output length: {len(integrated_chapter_text)}")
            return integrated_chapter_text.strip()

        except LLMServiceError as e_llm_integrate:
            logger.error(f"LLM service error during content integration for chapter '{chapter_title}': {e_llm_integrate}", exc_info=True)
            error_intro = f"[系统提示：章节内容整合因LLM服务错误而失败 (错误详情: {e_llm_integrate})。以下为基于各片段的初步生成内容，可能未完全整合：]\n\n"
            concatenated_fallback_parts = []
            for block in content_blocks:
                 concatenated_fallback_parts.append(f"{block.get('generated_text', '').strip()}\n{block.get('citation_string', '').strip()}")
            return error_intro + "\n\n".join(concatenated_fallback_parts).strip()
        except Exception as e_integration_unexpected:
            logger.error(f"Unexpected error during content integration for chapter '{chapter_title}': {e_integration_unexpected}", exc_info=True)
            error_intro = f"[系统提示：章节内容整合因发生意外错误而失败 (错误详情: {e_integration_unexpected})。以下为基于各片段的初步生成内容，可能未完全整合：]\n\n"
            concatenated_fallback_parts = []
            for block in content_blocks:
                concatenated_fallback_parts.append(f"{block.get('generated_text', '').strip()}\n{block.get('citation_string', '').strip()}")
            return error_intro + "\n\n".join(concatenated_fallback_parts).strip()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService:
        def __init__(self):
            self.snippet_call_count = 0
            self.integration_call_count = 0
            self.last_snippet_query = None # To store the actual query for snippet generation
            self.last_integration_query = None # Store the last query received by integration

        def chat(self, query: str, system_prompt: str) -> str:
            # Updated check for snippet generation prompt
            if "专业的报告撰写员" in system_prompt and "单一段落参考资料" in query:
                self.snippet_call_count += 1
                self.last_snippet_query = query # Store the actual query
                try:
                    # This parsing is simplified and assumes the snippet is the last major block.
                    # A more robust test might need to parse based on the known prompt structure.
                    parts = query.split("单一段落参考资料：\n\"\"\"\n")
                    if len(parts) > 1:
                        snippet_content = parts[1].split("\n\"\"\"")[0]
                        return f"基于片段“{snippet_content[:20]}...”的生成内容 (片段{self.snippet_call_count})"
                    else: # Fallback if the specific split fails
                        return f"通用片段生成内容（无法提取片段） (片段{self.snippet_call_count})"
                except Exception as e: # Catch any parsing error
                    logger.error(f"MockLLMService snippet parsing error: {e} in query: {query[:200]}")
                    return f"通用片段生成内容（解析错误） (片段{self.snippet_call_count})"

            # Updated check for integration prompt
            elif "高级报告编辑" in system_prompt and "待整合的文本块列表" in query:
                self.integration_call_count += 1
                self.last_integration_query = query # Store the received query
                # The query for integration will contain all generated snippets and their citations
                # We can mock the integration by simply joining them with a header.
                # A more sophisticated mock could try to rephrase or combine.
                # For testing, we want to see if the input to this call is correctly formatted.

                # Extract chapter title for the mock response
                try:
                    chapter_title_for_mock = query.split("章节标题：\n")[1].split("\n\n待整合的文本块列表：")[0]
                except IndexError:
                    chapter_title_for_mock = "未知章节"

                # Simulate integration: prepend a title and append a footer, keeping original blocks.
                # The actual LLM is expected to do a much better job of true integration.
                # The prompt asks the LLM to preserve citations, so the mock should reflect that.
                # The content blocks in the query will look like:
                # 文本块 1:\n内容：\n[generated_text]\n溯源：[citation_string]\n---\n\n文本块 2:...

                # For a simple mock, let's just confirm it received blocks and combine them.
                if "文本块 1:" in query: # Check if there are blocks to integrate
                    # The input query now has blocks like: "文本块 1:\nGenerated text [citation]\n---\n\n"
                    # The mock should reflect that the LLM tries to integrate these, keeping text and citation together.
                    integrated_text = f"《{chapter_title_for_mock}》的整合章节内容：\n\n"
                    parts = query.split("文本块 ")[1:] # Split by "文本块 " and ignore first part

                    processed_blocks_for_mock = []
                    for part_content_with_num in parts:
                        # part_content_with_num will be like "1:\nActual text content [citation string]\n---\n\n"
                        try:
                            # Extract the content part after "N:\n" and before "\n---"
                            actual_block_content_with_citation = part_content_with_num.split(":\n", 1)[1].replace("\n---", "").strip()
                            # Simulate some light integration, e.g., by wrapping each block.
                            # A real LLM would do more, but the key is to show the citation remains with its text.
                            processed_blocks_for_mock.append(f"段落：{actual_block_content_with_citation}")
                        except IndexError:
                            processed_blocks_for_mock.append(f"无法解析的文本块（保留原始格式）：{part_content_with_num[:70]}...")

                    integrated_text += "\n\n".join(processed_blocks_for_mock)
                    return integrated_text.strip()
                else: # No blocks found in query
                    return f"《{chapter_title_for_mock}》的整合内容：(无初步文本块可供整合)"

            # Fallback for relevance check or other calls (if any)
            elif "内容相关性判断助手" in system_prompt: # Corrected system_prompt check
                 # Simulate relevance check - assume all relevant for simplicity in this test
                return json.dumps({"is_relevant": True})

            logger.warning(f"MockLLMService received unexpected query type for system_prompt: '{system_prompt}'. Query: {query[:200]}...")
            return "通用模拟LLM响应（未知请求类型）。"

    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_WRITING_NEEDED

    class MockWorkflowStateCWA(WorkflowState): # CWA for ChapterWriterAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, retrieved_docs: List,
                     global_theme: Optional[str] = None, key_terms: Optional[Dict[str, str]] = None): # Added global_theme and key_terms
            super().__init__(user_topic)
            # Pre-populate chapter data as if retrieval was done
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_WRITING_NEEDED,
                'content': None, 'retrieved_docs': retrieved_docs,
                'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_cwa = []
            # Set global theme and key terms for the mock state
            if global_theme:
                self.update_report_global_theme(global_theme)
            if key_terms:
                self.update_key_terms_definitions(key_terms)


        def update_chapter_content(self, chapter_key: str, content: str,
                                   retrieved_docs: Optional[List[Dict[str, any]]] = None,
                                   is_new_version: bool = True):
            super().update_chapter_content(chapter_key, content, retrieved_docs, is_new_version)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' content updated.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' status updated to {status}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, any]] = None, priority: int = 0):
            self.added_tasks_cwa.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)


    llm_service_instance = MockLLMService()
    # Pass the explicit prompt templates to ensure the agent uses them,
    # or ensure the mock LLM can identify calls even with default templates.
    # For this test, MockLLMService identifies calls by content keywords, so defaults in agent are fine.
    writer_agent = ChapterWriterAgent(llm_service=llm_service_instance)

    test_chapter_key = "chap_abms_overview"
    test_chapter_title = "ABMS系统概述"
    mock_retrieved_data = [
        {
            "document": "父块1：ABMS是一个复杂的系统，具有多种功能...", "score": 0.9, "retrieval_source": "hybrid",
            "child_text_preview": "子块1预览...", "child_id": "p1c1", "parent_id": "p1",
            "source_document_name": "source_doc_A.pdf"
        },
        {
            "document": "父块2：JADC2与ABMS紧密相关，是其关键组成部分...", "score": 0.85, "retrieval_source": "hybrid",
            "child_text_preview": "子块2预览...", "child_id": "p2c1", "parent_id": "p2",
            "source_document_name": "source_doc_B.txt"
        }
    ]

    # Define mock global theme and key terms for testing
    mock_global_theme = "探讨ABMS在现代军事行动中的核心作用及其对未来战争形态的影响。"
    mock_key_terms = {
        "ABMS": "Advanced Battle Management System - 先进作战管理系统，旨在连接战场上的各种传感器、平台和决策者。",
        "JADC2": "Joint All-Domain Command and Control - 联合全域指挥与控制，是ABMS旨在实现的关键军事概念。"
    }

    mock_state_cwa = MockWorkflowStateCWA(user_topic="ABMS",
                                          chapter_key=test_chapter_key,
                                          chapter_title=test_chapter_title,
                                          retrieved_docs=mock_retrieved_data,
                                          global_theme=mock_global_theme, # Pass to mock state
                                          key_terms=mock_key_terms)      # Pass to mock state

    task_payload_for_agent_cwa = {'chapter_key': test_chapter_key, 'chapter_title': test_chapter_title, 'priority': 5}

    print(f"\nExecuting ChapterWriterAgent for chapter: '{test_chapter_title}' with MockWorkflowStateCWA (New Two-Stage Logic with Global Context)")
    try:
        writer_agent.execute_task(mock_state_cwa, task_payload_for_agent_cwa)

        print("\nWorkflowState after ChapterWriterAgent execution:")
        chapter_info = mock_state_cwa.get_chapter_data(test_chapter_key)
        if chapter_info:
            print(f"  Chapter '{test_chapter_key}' Status: {chapter_info.get('status')}")
            final_content = chapter_info.get('content', '')
            print(f"  Final Content:\n{final_content}\n") # Print full content for inspection
            print(f"  Retrieved Docs were used (count): {len(chapter_info.get('retrieved_docs', []))}")

        print(f"  Tasks added by agent: {json.dumps(mock_state_cwa.added_tasks_cwa, indent=2, ensure_ascii=False)}")

        assert chapter_info is not None
        assert chapter_info.get('status') == STATUS_EVALUATION_NEEDED

        # Check if LLM mock was called for snippets and integration
        assert llm_service_instance.snippet_call_count == len(mock_retrieved_data), \
            f"Expected {len(mock_retrieved_data)} snippet calls, got {llm_service_instance.snippet_call_count}"
        assert llm_service_instance.integration_call_count == 1, \
            f"Expected 1 integration call, got {llm_service_instance.integration_call_count}"

        # Check the content based on the new MockLLMService output
        # Expected snippet 1 text: "基于片段“父块1：ABMS是一个复杂的系统...”的生成内容 (片段1)"
        # Expected citation 1: "[引用来源：source_doc_A.pdf。原文表述：父块1：ABMS是一个复杂的系统，具有多种功能...]"
        # Expected snippet 2 text: "基于片段“父块2：JADC2与ABMS紧密相...”的生成内容 (片段2)"
        # Expected citation 2: "[引用来源：source_doc_B.txt。原文表述：父块2：JADC2与ABMS紧密相关，是其关键组成部分...]"

        # New mock integration output format:
        # "《Chapter Title》的整合章节内容：\n\n段落：[snippet1_text] [citation1_text]\n\n段落：[snippet2_text] [citation2_text]"

        expected_snippet_text1 = "基于片段“父块1：ABMS是一个复杂的系统...”的生成内容 (片段1)"
        expected_citation1_str = "[引用来源：source_doc_A.pdf。原文表述：父块1：ABMS是一个复杂的系统，具有多种功能...]"
        expected_combined_block1 = f"{expected_snippet_text1} {expected_citation1_str}"

        expected_snippet_text2 = "基于片段“父块2：JADC2与ABMS紧密相...”的生成内容 (片段2)"
        expected_citation2_str = "[引用来源：source_doc_B.txt。原文表述：父块2：JADC2与ABMS紧密相关，是其关键组成部分...]"
        expected_combined_block2 = f"{expected_snippet_text2} {expected_citation2_str}"

        assert f"《{test_chapter_title}》的整合章节内容：" in final_content
        # Check for the combined blocks in the final output
        assert f"段落：{expected_combined_block1}" in final_content
        assert f"段落：{expected_combined_block2}" in final_content

        print(f"Final content structure check: Title prefix present: {'《' + test_chapter_title + '》的整合章节内容：' in final_content}")
        print(f"Final content check: Combined block 1 present as '段落：...': {f'段落：{expected_combined_block1}' in final_content}")
        print(f"Final content check: Combined block 2 present as '段落：...': {f'段落：{expected_combined_block2}' in final_content}")

        # Verify the structure of the query sent to the integration LLM call
        assert llm_service_instance.last_integration_query is not None
        # The formatting in _integrate_chapter_content is now "文本块 {i+1}:\n{block_text} {block_citation}\n---\n\n"
        # So, expected_combined_block1 and expected_combined_block2 should be directly in the query,
        # prefixed by "文本块 N:\n" and suffixed by "\n---".

        expected_integration_input_formatted_block1 = f"文本块 1:\n{expected_combined_block1}\n---"
        expected_integration_input_formatted_block2 = f"文本块 2:\n{expected_combined_block2}\n---"

        assert expected_integration_input_formatted_block1 in llm_service_instance.last_integration_query
        assert expected_integration_input_formatted_block2 in llm_service_instance.last_integration_query

        # Verify global context in integration prompt
        assert mock_global_theme in llm_service_instance.last_integration_query
        assert "ABMS: Advanced Battle Management System" in llm_service_instance.last_integration_query # Check one of the key terms

        print(f"Integration input format check: Formatted block 1 correct: {expected_integration_input_formatted_block1 in llm_service_instance.last_integration_query}")
        print(f"Integration input format check: Formatted block 2 correct: {expected_integration_input_formatted_block2 in llm_service_instance.last_integration_query}")
        print(f"Integration input check: Global theme present: {mock_global_theme in llm_service_instance.last_integration_query}")
        print(f"Integration input check: Key term present: {'ABMS: Advanced Battle Management System' in llm_service_instance.last_integration_query}")

        # Verify global context in snippet prompt (checking the last one called)
        assert llm_service_instance.last_snippet_query is not None
        assert mock_global_theme in llm_service_instance.last_snippet_query
        assert "JADC2: Joint All-Domain Command and Control" in llm_service_instance.last_snippet_query # Check the other key term
        print(f"Snippet input check: Global theme present: {mock_global_theme in llm_service_instance.last_snippet_query}")
        print(f"Snippet input check: Key term present: {'JADC2: Joint All-Domain Command and Control' in llm_service_instance.last_snippet_query}")

        assert len(mock_state_cwa.added_tasks_cwa) == 1
        assert mock_state_cwa.added_tasks_cwa[0]['type'] == TASK_TYPE_EVALUATE_CHAPTER
        assert mock_state_cwa.added_tasks_cwa[0]['payload']['chapter_key'] == test_chapter_key

        print("\nChapterWriterAgent test with new two-stage logic and global context successful.")

    except Exception as e:
        print(f"Error during ChapterWriterAgent test (New Two-Stage Logic with Global Context): {e}")
        import traceback
        traceback.print_exc()

    print("\nChapterWriterAgent example finished.")
