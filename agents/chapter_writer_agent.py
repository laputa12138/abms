import logging
import json # For parsing LLM relevance check response (though clean_and_parse_json may supersede direct json usage here)
from typing import List, Dict, Optional, Any

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED # Import constants
from config import settings as app_settings # Import settings for prompts
from core.json_utils import clean_and_parse_json # Import the new helper

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
                 use_llm_relevance_check: bool = True,
                 reranker_score_threshold: float = app_settings.RERANKER_SCORE_THRESHOLD,
                 single_snippet_writer_prompt_template: Optional[str] = None,
                 integration_prompt_template: Optional[str] = None,
                 relevance_check_prompt_template: Optional[str] = None,
                 introduction_prompt_template: Optional[str] = None): # Added new template
        super().__init__(agent_name="ChapterWriterAgent", llm_service=llm_service)
        self.use_llm_relevance_check = use_llm_relevance_check
        self.reranker_score_threshold = reranker_score_threshold
        self.single_snippet_writer_prompt_template = single_snippet_writer_prompt_template or app_settings.DEFAULT_SINGLE_SNIPPET_WRITER_PROMPT
        self.integration_prompt_template = integration_prompt_template or app_settings.DEFAULT_CHAPTER_INTEGRATION_PROMPT
        self.relevance_check_prompt_template = relevance_check_prompt_template or app_settings.DEFAULT_RELEVANCE_CHECK_PROMPT
        self.introduction_prompt_template = introduction_prompt_template or app_settings.INTRODUCTION_GENERATOR_PROMPT_TEMPLATE # Load new template
        if not self.llm_service:
            raise ChapterWriterAgentError("LLMService is required for ChapterWriterAgent.")
        if not self.single_snippet_writer_prompt_template:
            raise ChapterWriterAgentError("Single snippet writer prompt template is required.")
        if not self.integration_prompt_template:
            raise ChapterWriterAgentError("Integration prompt template is required.")
        if not self.introduction_prompt_template: # Added check
            raise ChapterWriterAgentError("Introduction generator prompt template is required.")

    def _is_introduction_chapter(self, chapter_title: str, workflow_state: WorkflowState) -> bool:
        """
        Determines if the current chapter is the introduction.
        Considers title keywords and if it's the first chapter in the outline.
        """
        # Keywords that might indicate an introduction chapter
        intro_keywords = ["引言", "介绍", "概述", "绪论", "前言"]
        normalized_title = chapter_title.lower().strip()
        for keyword in intro_keywords:
            if keyword in normalized_title:
                return True

        # Check if it's the first chapter in the parsed_outline
        if workflow_state.parsed_outline and workflow_state.parsed_outline[0].get('title') == chapter_title:
            return True

        return False

    def _format_intro_prompt_payload(self, workflow_state: WorkflowState) -> Dict[str, str]:
        """
        Prepares the payload for the introduction prompt template.
        """
        topic_analysis = workflow_state.topic_analysis_results or {}

        core_questions = topic_analysis.get('core_research_questions_cn', [])
        if core_questions:
            core_questions_formatted = "本报告将探讨以下核心研究问题：\n" + "\n".join([f"- {q}" for q in core_questions])
        else:
            core_questions_formatted = "本报告将对指定主题进行深入分析和阐述。"

        # Use keywords_cn as a proxy for main_concepts for now
        # Consider if 'generalized_topic_cn' itself can be a main concept if keywords are missing.
        main_concepts_list = topic_analysis.get('keywords_cn', [])
        if not main_concepts_list and topic_analysis.get('generalized_topic_cn'): # Fallback to generalized topic
            main_concepts_list = [topic_analysis.get('generalized_topic_cn')]

        if main_concepts_list:
            main_concepts_formatted = "报告将聚焦于以下主要概念或对象：\n" + "\n".join([f"- {c}" for c in main_concepts_list])
        else:
            main_concepts_formatted = "报告将围绕主题展开详细讨论。"

        methodologies = topic_analysis.get('potential_methodologies_cn', [])
        if methodologies:
            methodologies_formatted = "研究可能采用的研究方法或分析视角包括：\n" + "\n".join([f"- {m}" for m in methodologies])
        else:
            methodologies_formatted = "研究将采用合适的分析方法以确保结论的严谨性。"

        outline_overview_parts = []
        if workflow_state.parsed_outline:
            for item in workflow_state.parsed_outline:
                indent = "  " * (item.get('level', 1) -1)
                outline_overview_parts.append(f"{indent}- {item.get('title', '未命名章节')}")
        outline_overview = "\n".join(outline_overview_parts) if outline_overview_parts else "报告大纲尚未生成或为空。"

        return {
            "report_title": workflow_state.report_title or "未指定报告标题",
            "user_topic": workflow_state.user_topic,
            "generalized_topic_cn": topic_analysis.get('generalized_topic_cn', "未分析出泛化主题"),
            "core_research_questions_formatted": core_questions_formatted,
            "main_concepts_formatted": main_concepts_formatted,
            "potential_methodologies_formatted": methodologies_formatted,
            "outline_overview": outline_overview,
        }

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
                system_prompt="你是一个内容相关性判断助手。请根据提供的章节标题和文档片段，判断文档片段是否与章节标题高度相关，并严格以JSON格式返回结果。确保返回的是纯净的JSON，不包含Markdown标记或注释。"
            )
            logger.debug(f"Relevance check for doc ID '{document_id}', chapter '{chapter_title}'. LLM raw response: {response_str}")

            response_json = clean_and_parse_json(response_str, context=f"relevance_check_doc_{document_id}_chapter_{chapter_title}")

            if response_json is None:
                # clean_and_parse_json logs the detailed error.
                logger.error(f"Failed to parse JSON from LLM relevance check for doc ID '{document_id}' after cleaning. Raw response: {response_str[:500]}... Assuming not relevant.")
                return False

            is_relevant = response_json.get("is_relevant") # No default here, check type below
            if not isinstance(is_relevant, bool):
                logger.warning(f"LLM relevance check for doc ID '{document_id}' returned non-boolean or missing 'is_relevant' value: {is_relevant}. Type: {type(is_relevant)}. Defaulting to False.")
                return False
            return is_relevant
        except LLMServiceError as e:
            logger.error(f"LLM service error during relevance check for doc ID '{document_id}': {e}. Assuming not relevant.")
            return False
        # json.JSONDecodeError is already handled by clean_and_parse_json or its absence if json_repair is used.
        # Other exceptions might still occur if response_json is not a dict, for example.
        except Exception as e: # Catch other unexpected errors, e.g. if response_json is not a dict
            logger.error(f"Unexpected error processing relevance check for doc ID '{document_id}': {e}. LLM raw response: {response_str[:500]}... Assuming not relevant.", exc_info=True)
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

        # Prepare report_outline_summary
        outline_summary_parts = []
        current_outline = workflow_state.parsed_outline
        if current_outline:
            for item in current_outline:
                indent = "  " * (item.get('level', 1) - 1)
                outline_marker = "*" if item.get('title') == chapter_title else "" # Mark current chapter
                outline_summary_parts.append(f"{indent}- {item.get('title', '未命名章节')} {outline_marker}")
        report_outline_summary = "\n".join(outline_summary_parts) if outline_summary_parts else "报告大纲不可用或为空。"

        # Prepare previous_chapters_summary
        previous_chapters_summary_parts = []
        if current_outline:
            current_chapter_index = -1
            for i, item in enumerate(current_outline):
                if item.get('id') == chapter_key: # Assuming chapter_key is the ID in the outline
                    current_chapter_index = i
                    break

            if current_chapter_index > 0:
                for i in range(current_chapter_index):
                    prev_chap_item = current_outline[i]
                    prev_chap_key = prev_chap_item.get('id')
                    prev_chap_title = prev_chap_item.get('title', '未知标题')
                    prev_chap_data = workflow_state.get_chapter_data(prev_chap_key)
                    if prev_chap_data and prev_chap_data.get('content'):
                        # Extract a brief summary, e.g., first N chars, stripping newlines
                        content_preview = prev_chap_data['content'][:200].replace("\n", " ").strip()
                        previous_chapters_summary_parts.append(f"  - {prev_chap_title}: \"{content_preview}...\"")
                    else:
                        previous_chapters_summary_parts.append(f"  - {prev_chap_title}: (内容尚未生成或不可用)")

        previous_chapters_summary = "\n".join(previous_chapters_summary_parts) if previous_chapters_summary_parts \
                                   else "无前序章节内容可供参考，或这是报告的第一章。"


        all_retrieved_docs_for_chapter = chapter_data.get('retrieved_docs', [])
        logger.info(f"[{self.agent_name}] Task ID: {task_id} - Chapter: '{chapter_title}'. Initial retrieved docs count: {len(all_retrieved_docs_for_chapter)}")
        self._log_input(
            chapter_title=chapter_title,
            initial_retrieved_docs_count=len(all_retrieved_docs_for_chapter),
            report_global_theme=report_global_theme,
            key_terms_definitions=key_terms_definitions_dict,
            report_outline_summary_len=len(report_outline_summary),
            previous_chapters_summary_len=len(previous_chapters_summary)
        )

        relevant_docs_for_generation = []
        if self.use_llm_relevance_check:
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
        else:

            logger.info(f"Skipping LLM relevance check. Filtering documents based on reranker score threshold: {self.reranker_score_threshold}")
            print(f"Skipping LLM relevance check. Filtering documents based on reranker score threshold: {self.reranker_score_threshold}")
            for doc in all_retrieved_docs_for_chapter:
                if doc.get('score', 0.0) >= self.reranker_score_threshold:
                    relevant_docs_for_generation.append(doc)

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

        if not relevant_docs_for_generation and not self._is_introduction_chapter(chapter_title, workflow_state):
            logger.warning(f"No relevant documents found to generate content blocks for non-introduction chapter '{chapter_title}'. Integration step will handle this.")
        elif self._is_introduction_chapter(chapter_title, workflow_state):
            logger.info(f"Processing introduction chapter: '{chapter_title}'. Skipping snippet-based generation.")
            # For introduction, content_blocks and all_snippet_citation_jsons_for_chapter will remain empty.
            # The final_integrated_content will be generated directly using the introduction prompt.
            pass # Handled later by directly calling LLM with intro prompt
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
                        report_outline_summary=report_outline_summary,
                        previous_chapters_summary=previous_chapters_summary,
                        key_terms_definitions_formatted=key_terms_definitions_formatted,
                        chapter_title=chapter_title,
                        single_document_snippet=single_doc_text
                    )
                except KeyError as e:
                    logger.error(f"KeyError formatting snippet_writer_prompt: {e}. Using fallback with available context.")
                    # Fallback might be less effective but better than crashing
                    snippet_writer_prompt = (
                        f"报告全局主题：\n{report_global_theme}\n\n"
                        f"当前章节标题：\n{chapter_title}\n\n"
                        f"单一段落参考资料：\n\"\"\"\n{single_doc_text}\n\"\"\"\n\n"
                        f"请撰写内容。注意：部分上下文信息可能因错误未能加载。"
                    )

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
            final_integrated_content = ""
            if self._is_introduction_chapter(chapter_title, workflow_state):
                logger.info(f"Generating content directly for introduction chapter: '{chapter_title}'.")
                intro_payload = self._format_intro_prompt_payload(workflow_state)
                intro_prompt_formatted = self.introduction_prompt_template.format(**intro_payload)

                try:
                    final_integrated_content = self.llm_service.chat(
                        query=intro_prompt_formatted,
                        system_prompt="你是一位专业的报告引言撰写专家。" # System prompt for introduction
                    )
                    if not final_integrated_content or not final_integrated_content.strip():
                        logger.warning(f"LLM returned empty content for introduction chapter '{chapter_title}'. Using placeholder.")
                        final_integrated_content = app_settings.DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER
                except LLMServiceError as e_intro_llm:
                    err_msg_intro = f"LLM service failed while generating introduction for chapter '{chapter_title}': {e_intro_llm}"
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg_intro}", exc_info=False)
                    final_integrated_content = f"[系统提示：生成引言时遇到LLM服务错误 - {e_intro_llm}]"
                except Exception as e_intro_unexpected:
                    err_msg_intro_unexpected = f"Unexpected error generating introduction for chapter '{chapter_title}': {e_intro_unexpected}"
                    logger.error(f"[{self.agent_name}] Task ID: {task_id} - {err_msg_intro_unexpected}", exc_info=True)
                    final_integrated_content = f"[系统提示：生成引言时发生意外错误 - {e_intro_unexpected}]"

                # For introduction, citations are typically not from snippets, so all_snippet_citation_jsons_for_chapter remains empty.
                # relevant_docs_for_generation for intro might be empty or a global set, not used for snippet citations here.

            else: # For non-introduction chapters, use existing integration logic
                logger.info(f"Proceeding to content integration for chapter '{chapter_title}'.")
                final_integrated_content = self._integrate_chapter_content(
                    chapter_title=chapter_title,
                    content_blocks=content_blocks,
                    report_global_theme=report_global_theme,
                    key_terms_definitions_formatted=key_terms_definitions_formatted,
                    report_outline_summary=report_outline_summary, # Pass new context
                    previous_chapters_summary=previous_chapters_summary # Pass new context
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
                                   key_terms_definitions_formatted: str,
                                   report_outline_summary: str, # Added
                                   previous_chapters_summary: str, # Added
                                   is_recursive_call: bool = False
                                   ) -> str:
        """
        Integrates multiple preliminary content blocks (each with its citation) into a single,
        coherent chapter using an LLM, incorporating global theme and key terms.
        If no content_blocks are provided, returns a standard placeholder message from settings.
        Handles LLM context length errors by recursively splitting the content blocks.
        """
        if not content_blocks:
            logger.warning(
                f"No content blocks provided for integration for chapter '{chapter_title}'. "
                f"This usually means no relevant documents were found/passed relevance check, or all snippet generations failed. "
                f"Returning standard placeholder."
            )
            return getattr(app_settings, 'DEFAULT_CHAPTER_MISSING_CONTENT_PLACEHOLDER',
                           "[本章节未能生成有效内容，我们正在努力改进。]")

        # Context truncation logic
        truncated_outline = report_outline_summary
        if len(truncated_outline) > 2000: # Example threshold
            logger.warning(f"Report outline summary for chapter '{chapter_title}' is very long ({len(truncated_outline)} chars). Truncating.")
            truncated_outline = truncated_outline[:1000] + "\n...\n" + truncated_outline[-1000:]

        truncated_prev_summary = previous_chapters_summary
        if len(truncated_prev_summary) > 2000: # Example threshold
            logger.warning(f"Previous chapters summary for chapter '{chapter_title}' is very long ({len(truncated_prev_summary)} chars). Truncating.")
            truncated_prev_summary = truncated_prev_summary[:1000] + "\n...\n" + truncated_prev_summary[-1000:]


        formatted_blocks_for_integration = ""
        for i, block in enumerate(content_blocks):
            block_text = block.get('generated_text', "").strip()
            block_citation = block.get('citation_string', "[溯源信息缺失]").strip()
            combined_text_with_citation = f"{block_text} {block_citation}" if block_text else block_citation
            formatted_blocks_for_integration += f"文本块 {i+1}:\n{combined_text_with_citation.strip()}\n---\n\n"

        try:
            integration_prompt_str = self.integration_prompt_template.format(
                report_global_theme=report_global_theme,
                report_outline_summary=truncated_outline,
                previous_chapters_summary=truncated_prev_summary,
                key_terms_definitions_formatted=key_terms_definitions_formatted,
                chapter_title=chapter_title,
                preliminary_content_blocks_formatted=formatted_blocks_for_integration.strip()
            )
        except KeyError as e:
            logger.error(f"KeyError formatting integration_prompt_str: {e}. Using fallback with available context.")
            integration_prompt_str = (
                f"报告全局主题：\n{report_global_theme}\n\n"
                f"当前章节标题：\n{chapter_title}\n\n"
                f"待整合的文本块列表：\n---\n{formatted_blocks_for_integration.strip()}\n---\n"
                f"请整合内容。注意：部分上下文信息可能因错误未能加载。"
            )

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
            error_message = str(e_llm_integrate)
            if "is longer than the maximum model length" in error_message and not is_recursive_call:
                logger.warning(f"LLM prompt for chapter '{chapter_title}' is too long. Splitting content blocks and retrying.")

                if len(content_blocks) > 1:
                    mid_index = len(content_blocks) // 2
                    first_half_blocks = content_blocks[:mid_index]
                    second_half_blocks = content_blocks[mid_index:]

                    logger.info(f"Splitting into two chunks: {len(first_half_blocks)} and {len(second_half_blocks)} blocks.")

                    # Recursively integrate each half
                    integrated_first_half = self._integrate_chapter_content(
                        chapter_title=chapter_title, content_blocks=first_half_blocks,
                        report_global_theme=report_global_theme, key_terms_definitions_formatted=key_terms_definitions_formatted,
                        report_outline_summary=report_outline_summary, previous_chapters_summary=previous_chapters_summary,
                        is_recursive_call=True) # Mark as recursive call to prevent infinite loops

                    integrated_second_half = self._integrate_chapter_content(
                        chapter_title=chapter_title, content_blocks=second_half_blocks,
                        report_global_theme=report_global_theme, key_terms_definitions_formatted=key_terms_definitions_formatted,
                        report_outline_summary=report_outline_summary, previous_chapters_summary=previous_chapters_summary,
                        is_recursive_call=True)

                    # Now, integrate the two integrated halves
                    new_content_blocks = [
                        {'generated_text': integrated_first_half, 'citation_string': "[部分整合结果]"},
                        {'generated_text': integrated_second_half, 'citation_string': "[部分整合结果]"}
                    ]

                    logger.info(f"Re-integrating the two processed chunks for chapter '{chapter_title}'.")
                    return self._integrate_chapter_content(
                        chapter_title=chapter_title, content_blocks=new_content_blocks,
                        report_global_theme=report_global_theme, key_terms_definitions_formatted=key_terms_definitions_formatted,
                        report_outline_summary=report_outline_summary, previous_chapters_summary=previous_chapters_summary,
                        is_recursive_call=True) # Still a recursive call

                elif len(content_blocks) == 1:
                    # If a single block is too long, split its text content
                    single_block = content_blocks[0]
                    text_to_split = single_block.get('generated_text', '')
                    if len(text_to_split) > 1:
                        mid_text_index = len(text_to_split) // 2
                        first_text_half = text_to_split[:mid_text_index]
                        second_text_half = text_to_split[mid_text_index:]

                        split_blocks = [
                            {'generated_text': first_text_half, 'citation_string': single_block.get('citation_string', '')},
                            {'generated_text': second_text_half, 'citation_string': single_block.get('citation_string', '')}
                        ]
                        logger.info(f"Splitting a single oversized block's text content for chapter '{chapter_title}'.")
                        return self._integrate_chapter_content(
                            chapter_title=chapter_title, content_blocks=split_blocks,
                            report_global_theme=report_global_theme, key_terms_definitions_formatted=key_terms_definitions_formatted,
                            report_outline_summary=report_outline_summary, previous_chapters_summary=previous_chapters_summary,
                            is_recursive_call=True)

                    else:
                        logger.error(f"Single content block is too large but has no text to split for chapter '{chapter_title}'. Cannot split further.")
                        return f"[系统提示：章节内容整合失败，因为单个内容块过长且无法分割。错误详情: {e_llm_integrate}]"
                else:
                    logger.error(f"Content block is too large for chapter '{chapter_title}', but it's not a single block. This should not happen. Returning error content.")
                    return f"[系统提示：章节内容整合失败，因为内容块过长，无法处理。错误详情: {e_llm_integrate}]"

            else:
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
            self.integration_call_count = 0 # Will be used for both regular integration and intro generation
            self.last_snippet_query = None
            self.last_integration_query = None

        def chat(self, query: str, system_prompt: str) -> str:
            if "专业的报告撰写员" in system_prompt and "单一段落参考资料" in query:
                self.snippet_call_count += 1
                self.last_snippet_query = query
                parts = query.split("单一段落参考资料：\n\"\"\"\n")
                if len(parts) > 1:
                    snippet_content = parts[1].split("\n\"\"\"")[0]
                    return f"基于片段“{snippet_content[:20]}...”的生成内容 (片段{self.snippet_call_count})"
                return f"通用片段生成内容（无法提取片段） (片段{self.snippet_call_count})"
            elif "高级报告编辑" in system_prompt:
                self.integration_call_count += 1
                self.last_integration_query = query

                # Match the simplified prompt used in the test
                title_match = re.search(r"章节标题：(.*?)\n", query)
                chapter_title_for_mock = title_match.group(1) if title_match else "未知章节"

                # Handle both initial and recursive integration calls
                integrated_text = f"《{chapter_title_for_mock}》的最终整合章节内容：\n\n"
                parts = query.split("文本块 ")[1:]
                if not parts: # Handle recursive calls where "文本块" might not be present
                    parts = query.split("内容：")[1:]

                processed_blocks_for_mock = [part.split(":\n", 1)[1].replace("\n---", "").strip() for part in parts if ":\n" in part]

                # A bit of a hack to handle the recursive case where the input is already integrated
                if not processed_blocks_for_mock and "《" in query:
                     # Extract the content from the already integrated parts
                    processed_blocks_for_mock = [p.strip() for p in query.split("》的整合章节内容：") if "》的整合章节内容：" in p]


                integrated_text += "\n\n".join([f"最终段落：{p}" for p in processed_blocks_for_mock])
                return integrated_text.strip()
            elif "内容相关性判断助手" in system_prompt:
                return json.dumps({"is_relevant": True})
            elif "专业的报告引言撰写专家" in system_prompt: # Mock for introduction
                self.integration_call_count +=1
                title_in_query = "未知报告标题（引言）"
                if "报告标题：" in query:
                    try: title_in_query = query.split("报告标题：")[1].split("\n")[0]
                    except: pass
                return f"这是为报告《{title_in_query}》生成的模拟引言。它将包含背景、问题、范围和结构概述。"
            logger.warning(f"MockLLMService received unexpected query: {system_prompt} / {query[:100]}")
            return "通用模拟LLM响应（未知请求类型）。"

    import re
    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_WRITING_NEEDED

    class MockWorkflowStateCWA(WorkflowState):
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, retrieved_docs: List,
                     global_theme: Optional[str] = None, key_terms: Optional[Dict[str, str]] = None):
            super().__init__(user_topic, report_title=f"关于“{user_topic}”的报告") # Ensure report_title is set
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_WRITING_NEEDED,
                'content': None, 'retrieved_docs': retrieved_docs,
                'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_cwa = []
            if global_theme: self.update_report_global_theme(global_theme)
            if key_terms: self.update_key_terms_definitions(key_terms)

        def update_chapter_content(self, chapter_key: str, content: str, retrieved_docs: Optional[List[Dict[str, any]]] = None, is_new_version: bool = True, citations_structured_list: Optional[List[Dict[str, Any]]] = None): # Added citations_structured_list
            super().update_chapter_content(chapter_key, content, retrieved_docs, citations_structured_list, is_new_version) # Pass it on
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' content updated.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' status updated to {status}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, any]] = None, priority: int = 0):
            self.added_tasks_cwa.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)

    llm_service_instance = MockLLMService()
    writer_agent = ChapterWriterAgent(llm_service=llm_service_instance)

    def test_regular_chapter(writer_agent_instance, llm_service_instance_mock):
        test_chapter_key_regular = "chap_regular_content"
        test_chapter_title_regular = "ABMS核心技术分析"
        mock_retrieved_data = [
            {"document": "父块1：ABMS...", "score": 0.9, "source_document_name": "docA.pdf"},
            {"document": "父块2：JADC2...", "score": 0.85, "source_document_name": "docB.txt"}
        ]
        mock_global_theme = "探讨ABMS在现代军事行动中的核心作用。"
        mock_key_terms = {"ABMS": "先进作战管理系统", "JADC2": "联合全域指挥与控制"}

        mock_state_regular = MockWorkflowStateCWA(user_topic="ABMS",
                                              chapter_key=test_chapter_key_regular,
                                              chapter_title=test_chapter_title_regular,
                                              retrieved_docs=mock_retrieved_data,
                                              global_theme=mock_global_theme,
                                              key_terms=mock_key_terms)
        mock_state_regular.current_processing_task_id = "task_regular_chap"
        task_payload_regular = {'chapter_key': test_chapter_key_regular, 'chapter_title': test_chapter_title_regular, 'priority': 5}

        print(f"\nExecuting ChapterWriterAgent for REGULAR chapter: '{test_chapter_title_regular}'")
        llm_service_instance_mock.snippet_call_count = 0
        llm_service_instance_mock.integration_call_count = 0
        writer_agent_instance.execute_task(mock_state_regular, task_payload_regular)

        chapter_info = mock_state_regular.get_chapter_data(test_chapter_key_regular)
        assert chapter_info is not None and chapter_info.get('status') == STATUS_EVALUATION_NEEDED
        assert llm_service_instance_mock.snippet_call_count == len(mock_retrieved_data)
        assert llm_service_instance_mock.integration_call_count == 1
        final_content = chapter_info.get('content', '')
        assert f"《{test_chapter_title_regular}》的最终整合章节内容：" in final_content
        assert "最终段落" in final_content
        assert "--- 参考来源列表 ---" in final_content # Check for citation block
        print(f"REGULAR chapter test successful. LLM calls: Snippets={llm_service_instance_mock.snippet_call_count}, Integration={llm_service_instance_mock.integration_call_count}")

    def test_introduction_chapter(writer_agent_instance, llm_service_instance_mock):
        test_chapter_key_intro = "chap_introduction"
        test_chapter_title_intro = "引言"

        mock_state_intro = MockWorkflowStateCWA(user_topic="未来战争形态",
                                            chapter_key=test_chapter_key_intro,
                                            chapter_title=test_chapter_title_intro,
                                            retrieved_docs=[])
        mock_state_intro.report_title = "未来战争形态深度分析报告"
        mock_state_intro.current_processing_task_id = "task_intro_chap"
        mock_state_intro.update_topic_analysis({
            "generalized_topic_cn": "未来战争的演变与挑战",
            "keywords_cn": ["AI军事应用", "无人作战"],
            "core_research_questions_cn": ["AI技术将如何重塑未来战场？"],
            "potential_methodologies_cn": ["趋势分析"]
        })
        mock_state_intro.parsed_outline = [
            {'title': '引言', 'level': 1, 'id': 'chap_introduction'},
            {'title': 'AI技术', 'level': 1, 'id': 'chap_ai_military'}
        ]
        task_payload_intro = {'chapter_key': test_chapter_key_intro, 'chapter_title': test_chapter_title_intro, 'priority': 1}

        print(f"\nExecuting ChapterWriterAgent for INTRODUCTION chapter: '{test_chapter_title_intro}'")
        llm_service_instance_mock.snippet_call_count = 0
        llm_service_instance_mock.integration_call_count = 0
        writer_agent_instance.execute_task(mock_state_intro, task_payload_intro)

        chapter_info = mock_state_intro.get_chapter_data(test_chapter_key_intro)
        assert chapter_info is not None and chapter_info.get('status') == STATUS_EVALUATION_NEEDED
        assert llm_service_instance_mock.snippet_call_count == 0
        assert llm_service_instance_mock.integration_call_count == 1
        final_content = chapter_info.get('content', '')
        assert f"这是为报告《{mock_state_intro.report_title}》生成的模拟引言" in final_content
        assert "--- 参考来源列表 ---" not in final_content # Intro usually doesn't have this specific citation block
        print(f"INTRODUCTION chapter test successful. LLM calls: Snippets={llm_service_instance_mock.snippet_call_count}, IntroGen={llm_service_instance_mock.integration_call_count}")

    def test_long_chapter_recursive_split(writer_agent_instance, llm_service_instance_mock):

        class MockLLMServiceWithLengthCheck(MockLLMService):
            def __init__(self):
                super().__init__()
                self.first_long_call = True

            def chat(self, query: str, system_prompt: str) -> str:
                # Only apply length check to the integration prompt
                if "高级报告编辑" in system_prompt:
                    # A crude way to simulate prompt length exceeding a limit
                    if self.first_long_call and len(query) > 2000: # Set a mock limit, e.g. 2000
                        self.first_long_call = False
                        raise LLMServiceError("Failed to generate chat completion, detail: The decoder prompt is longer than the maximum model length of 40960.")

                # For recursive calls, the query will be smaller, so it should pass the length check and call the parent's chat method
                return super().chat(query, system_prompt)

        llm_service_with_check = MockLLMServiceWithLengthCheck()
        writer_agent_with_check = ChapterWriterAgent(llm_service=llm_service_with_check)


        test_chapter_key_long = "chap_long_content"
        test_chapter_title_long = "超长章节递归分割测试"
        # Create a large number of blocks to ensure the prompt exceeds the mock limit
        mock_retrieved_data_long = [
            {"document": f"父块{i}：一些内容...", "score": 0.9, "source_document_name": f"doc{i}.pdf"} for i in range(20)
        ]
        mock_global_theme = "测试递归分割功能"
        mock_key_terms = {"递归": "一种调用自身的编程技巧"}

        mock_state_long = MockWorkflowStateCWA(user_topic="超长内容处理",
                                              chapter_key=test_chapter_key_long,
                                              chapter_title=test_chapter_title_long,
                                              retrieved_docs=mock_retrieved_data_long,
                                              global_theme=mock_global_theme,
                                              key_terms=mock_key_terms)
        mock_state_long.current_processing_task_id = "task_long_chap"
        task_payload_long = {'chapter_key': test_chapter_key_long, 'chapter_title': test_chapter_title_long, 'priority': 5}

        print(f"\nExecuting ChapterWriterAgent for LONG chapter (expecting recursive split): '{test_chapter_title_long}'")
        writer_agent_with_check.execute_task(mock_state_long, task_payload_long)

        chapter_info = mock_state_long.get_chapter_data(test_chapter_key_long)
        assert chapter_info is not None and chapter_info.get('status') == STATUS_EVALUATION_NEEDED

        final_content = chapter_info.get('content', '')
        # The final integration will contain text from the sub-integrations.
        # In our mock, this means it will contain the titles of the sub-integrations.
        assert f"《{test_chapter_title_long}》的最终整合章节内容：" in final_content
        assert "最终段落" in final_content

        print(f"LONG chapter test successful. Recursive splitting was triggered and handled correctly.")


    try:
        print("\n--- Running ChapterWriterAgent Tests ---")
        test_regular_chapter(writer_agent, llm_service_instance)
        test_introduction_chapter(writer_agent, llm_service_instance)
        test_long_chapter_recursive_split(writer_agent, llm_service_instance)
        print("\nAll ChapterWriterAgent tests passed.")
    except AssertionError as e:
        print(f"\nChapterWriterAgent test FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nError during ChapterWriterAgent tests: {e}")
        import traceback
        traceback.print_exc()

    print("\nChapterWriterAgent example finished.")
