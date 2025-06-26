import logging
import json
from typing import List, Dict, Optional

from config import settings
from core.llm_service import LLMService, LLMServiceError
from core.embedding_service import EmbeddingService, EmbeddingServiceError
from core.reranker_service import RerankerService, RerankerServiceError
from core.document_processor import DocumentProcessor, DocumentProcessorError
from core.vector_store import VectorStore, VectorStoreError

from agents.topic_analyzer_agent import TopicAnalyzerAgent, TopicAnalyzerAgentError
from agents.outline_generator_agent import OutlineGeneratorAgent, OutlineGeneratorAgentError
from agents.content_retriever_agent import ContentRetrieverAgent, ContentRetrieverAgentError
from agents.chapter_writer_agent import ChapterWriterAgent, ChapterWriterAgentError
from agents.evaluator_agent import EvaluatorAgent, EvaluatorAgentError
from agents.refiner_agent import RefinerAgent, RefinerAgentError
from agents.report_compiler_agent import ReportCompilerAgent, ReportCompilerAgentError

logger = logging.getLogger(__name__)

class ReportGenerationPipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

class ReportGenerationPipeline:
    """
    Orchestrates the entire report generation process, from document processing
    to final report compilation, using a sequence of specialized agents.
    """

    def __init__(self,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 reranker_service: Optional[RerankerService] = None, # Reranker is optional
                 max_refinement_iterations: int = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS):
        """
        Initializes the ReportGenerationPipeline.

        Args:
            llm_service (LLMService): Instance of LLMService.
            embedding_service (EmbeddingService): Instance of EmbeddingService.
            reranker_service (Optional[RerankerService]): Instance of RerankerService.
            max_refinement_iterations (int): Maximum number of times to refine each chapter.
        """
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.max_refinement_iterations = max_refinement_iterations

        # Initialize services and agents
        self.document_processor = DocumentProcessor(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        # VectorStore dimension will be inferred upon first embedding
        self.vector_store = VectorStore(embedding_service=self.embedding_service)

        self.topic_analyzer = TopicAnalyzerAgent(llm_service=self.llm_service)
        self.outline_generator = OutlineGeneratorAgent(llm_service=self.llm_service)
        self.content_retriever = ContentRetrieverAgent(
            vector_store=self.vector_store,
            reranker_service=self.reranker_service, # Pass the reranker here
            default_top_k_retrieval=settings.DEFAULT_VECTOR_STORE_TOP_K
        )
        self.chapter_writer = ChapterWriterAgent(llm_service=self.llm_service)
        self.evaluator = EvaluatorAgent(llm_service=self.llm_service)
        self.refiner = RefinerAgent(llm_service=self.llm_service)
        self.report_compiler = ReportCompilerAgent(add_table_of_contents=True) # Default to add TOC

        logger.info("ReportGenerationPipeline initialized with all services and agents.")

    def _process_and_load_documents(self, pdf_paths: List[str]):
        """Processes PDFs, extracts text, splits into chunks, and loads into VectorStore."""
        logger.info(f"Starting document processing for {len(pdf_paths)} PDF(s).")
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                raw_text = self.document_processor.extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    logger.warning(f"No text extracted from PDF: {pdf_path}. Skipping.")
                    continue
                chunks = self.document_processor.split_text_into_chunks(raw_text)
                all_chunks.extend(chunks)
                logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}.")
            except (DocumentProcessorError, FileNotFoundError) as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}. It will be skipped.")
                # Optionally, re-raise if one PDF failure should halt the process
                # raise ReportGenerationPipelineError(f"Critical error processing PDF {pdf_path}: {e}")

        if not all_chunks:
            logger.error("No text chunks were extracted from any PDF. Cannot proceed.")
            raise ReportGenerationPipelineError("No usable content extracted from provided PDFs.")

        try:
            logger.info(f"Adding {len(all_chunks)} total chunks to VectorStore.")
            self.vector_store.add_documents(all_chunks)
            logger.info("Successfully added document chunks to VectorStore.")
        except VectorStoreError as e:
            logger.error(f"Failed to add documents to VectorStore: {e}")
            raise ReportGenerationPipelineError(f"VectorStore population failed: {e}")


    def _parse_outline_to_chapter_titles(self, markdown_outline: str) -> List[str]:
        """
        Parses a Markdown outline (potentially hierarchical) and extracts a flat list
        of chapter/section titles that require content generation.
        This is a simplified parser; ReportCompilerAgent has a more detailed one for TOC.
        """
        titles = []
        lines = markdown_outline.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Simple heuristic: lines starting with '-', '*', '+', or '#' are titles
            if line.startswith(("- ", "* ", "+ ")):
                titles.append(line[2:].strip())
            elif line.startswith("#"):
                titles.append(line.lstrip("# ").strip())
            # More complex parsing (like in ReportCompilerAgent) could be used for levels,
            # but for content generation, we just need the list of titles to write for.

        if not titles:
            logger.warning("Could not parse any chapter titles from the generated outline.")
        logger.debug(f"Parsed outline titles for content generation: {titles}")
        return titles


    def run(self, user_topic: str, pdf_paths: List[str], report_title: Optional[str] = None) -> str:
        """
        Executes the full report generation pipeline.

        Args:
            user_topic (str): The main topic for the report, provided by the user.
            pdf_paths (List[str]): A list of file paths to PDF documents to be used as context.
            report_title (Optional[str]): The desired title for the final report.
                                          If None, derived from user_topic.

        Returns:
            str: The final compiled report in Markdown format.

        Raises:
            ReportGenerationPipelineError: If any critical step in the pipeline fails.
        """
        logger.info(f"Starting report generation pipeline for topic: '{user_topic}'")
        final_report_title = report_title or f"关于“{user_topic}”的分析报告"

        # 1. Process and Load Documents
        try:
            self._process_and_load_documents(pdf_paths)
        except ReportGenerationPipelineError as e: # Catch errors from the helper
             raise # Re-raise critical errors

        # 2. Analyze Topic
        logger.info("Step 2: Analyzing topic...")
        try:
            analyzed_topic_details = self.topic_analyzer.run(user_topic)
            logger.info(f"Topic analysis complete. Generalized CN topic: {analyzed_topic_details.get('generalized_topic_cn')}")
        except TopicAnalyzerAgentError as e:
            logger.error(f"Failed to analyze topic: {e}")
            raise ReportGenerationPipelineError(f"Topic analysis failed: {e}")

        # 3. Generate Outline
        logger.info("Step 3: Generating outline...")
        try:
            markdown_outline = self.outline_generator.run(analyzed_topic_details)
            if not markdown_outline.strip():
                logger.error("Outline generation resulted in an empty outline.")
                raise ReportGenerationPipelineError("Outline generation failed: Empty outline.")
            logger.info("Outline generation complete.")
            logger.debug(f"Generated Markdown Outline:\n{markdown_outline}")
        except OutlineGeneratorAgentError as e:
            logger.error(f"Failed to generate outline: {e}")
            raise ReportGenerationPipelineError(f"Outline generation failed: {e}")

        # 4. Process each chapter/section from the outline
        logger.info("Step 4: Processing chapters...")
        # Use the simple parser for titles to iterate for content generation
        chapter_titles_for_writing = self._parse_outline_to_chapter_titles(markdown_outline)

        if not chapter_titles_for_writing:
            logger.error("No chapter titles could be parsed from the outline. Cannot proceed with chapter writing.")
            raise ReportGenerationPipelineError("Failed to parse chapter titles from outline.")

        compiled_chapter_contents: Dict[str, str] = {}

        for i, chapter_title in enumerate(chapter_titles_for_writing):
            logger.info(f"Processing chapter {i+1}/{len(chapter_titles_for_writing)}: '{chapter_title}'")

            # a. Content Retrieval
            # Combine chapter title with keywords from topic analysis for a richer query
            query_keywords_cn = analyzed_topic_details.get('keywords_cn', [])
            query_keywords_en = analyzed_topic_details.get('keywords_en', [])
            # Simple concatenation for retrieval query. Could be more sophisticated.
            retrieval_query = f"{chapter_title} {' '.join(query_keywords_cn)} {' '.join(query_keywords_en)}".strip()

            try:
                retrieved_docs = self.content_retriever.run(query=retrieval_query) # Uses default top_k, top_n
                logger.info(f"Retrieved {len(retrieved_docs)} documents for chapter '{chapter_title}'.")
                if not retrieved_docs:
                    logger.warning(f"No relevant documents found for chapter '{chapter_title}'. Chapter content might be sparse.")
            except ContentRetrieverAgentError as e:
                logger.error(f"Content retrieval failed for chapter '{chapter_title}': {e}. Skipping content for this chapter.")
                retrieved_docs = [] # Proceed with empty docs for this chapter

            # b. Chapter Writing (Initial Draft)
            try:
                current_chapter_content = self.chapter_writer.run(
                    chapter_title=chapter_title,
                    retrieved_content=retrieved_docs
                )
                logger.info(f"Initial draft written for chapter '{chapter_title}'. Length: {len(current_chapter_content)}")
            except ChapterWriterAgentError as e:
                logger.error(f"Chapter writing failed for '{chapter_title}': {e}. Using placeholder content.")
                current_chapter_content = f"Error generating content for this chapter: {e}"


            # c. Evaluation and Refinement Loop
            if current_chapter_content.strip() and not current_chapter_content.startswith("Error generating content"): # Only refine if there's actual content
                for ref_iter in range(self.max_refinement_iterations):
                    logger.info(f"Refinement iteration {ref_iter + 1}/{self.max_refinement_iterations} for chapter '{chapter_title}'")
                    try:
                        evaluation = self.evaluator.run(content_to_evaluate=current_chapter_content)
                        logger.info(f"Evaluation for '{chapter_title}': Score {evaluation.get('score')}")

                        # Basic refinement condition: if score is below a threshold (e.g., 80)
                        # This threshold could be configurable.
                        # For now, we refine regardless of score for the specified number of iterations,
                        # as the prompt asks the refiner to improve based on *any* feedback.
                        # A more sophisticated check could be: if evaluation['score'] < REFINEMENT_SCORE_THRESHOLD:

                        refined_content = self.refiner.run(
                            original_content=current_chapter_content,
                            evaluation_feedback=evaluation
                        )
                        # Log length change to see if refinement is substantial
                        logger.info(f"Refinement complete for '{chapter_title}'. Original length: {len(current_chapter_content)}, Refined length: {len(refined_content)}")
                        current_chapter_content = refined_content

                    except (EvaluatorAgentError, RefinerAgentError) as e:
                        logger.error(f"Evaluation or Refinement failed for chapter '{chapter_title}' on iteration {ref_iter + 1}: {e}. Using previous version of content.")
                        break # Break from refinement loop for this chapter on error
            else:
                logger.warning(f"Skipping refinement for chapter '{chapter_title}' due to empty or error content.")

            compiled_chapter_contents[chapter_title] = current_chapter_content

        # 5. Compile Report
        logger.info("Step 5: Compiling final report...")
        try:
            final_report_md = self.report_compiler.run(
                report_title=final_report_title,
                markdown_outline=markdown_outline, # Pass the original generated outline
                chapter_contents=compiled_chapter_contents,
                report_topic_details=analyzed_topic_details # For optional intro
            )
            logger.info("Report compilation complete.")
            logger.debug(f"Final Report Markdown (first 500 chars):\n{final_report_md[:500]}")
        except ReportCompilerAgentError as e:
            logger.error(f"Failed to compile final report: {e}")
            raise ReportGenerationPipelineError(f"Report compilation failed: {e}")

        logger.info("Report generation pipeline finished successfully.")
        return final_report_md


if __name__ == '__main__':
    # This example demonstrates the pipeline structure.
    # It requires mock services or a fully running Xinference setup with all models.
    # For simplicity, we will use Mocks for external services here.

    logging.basicConfig(level=logging.INFO, # Use INFO for pipeline overview, DEBUG for agent details
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("ReportGenerationPipeline Example Start")

    # --- Mock Services Setup ---
    # (These would be more detailed in a real test setup)
    class MockLLMServiceForPipeline(LLMService):
        def __init__(self): super().__init__(api_url="mock://llm", model_name="mock-llm")
        def chat(self, query, system_prompt, **kwargs):
            logger.debug(f"MockLLMService.chat called. Query starts with: {query[:60]}")
            if "你是一个主题分析专家" in query: # Topic Analyzer
                return json.dumps({
                    "generalized_topic_cn": "模拟主题：先进系统", "generalized_topic_en": "Mock Topic: Advanced Systems",
                    "keywords_cn": ["核心技术", "未来发展"], "keywords_en": ["core tech", "future dev"]
                })
            if "你是一个报告大纲撰写助手" in query: # Outline Generator
                return "- 章节一：介绍\n  - 1.1 背景\n- 章节二：核心分析\n- 章节三：结论"
            if "你是一位专业的报告撰写员" in query: # Chapter Writer
                title_search = "章节标题：\n"
                title_start = query.find(title_search)
                if title_start != -1:
                    title_end = query.find("\n", title_start + len(title_search))
                    title = query[title_start + len(title_search):title_end].strip()
                    return f"这是关于“{title}”的模拟章节内容。包含了一些基于模拟检索资料的分析。"
                return "模拟章节内容。"
            if "你是一位资深的报告评审员" in query: # Evaluator
                return json.dumps({
                    "score": 80, "feedback_cn": "模拟反馈：内容基本合格，可以更充实。",
                    "evaluation_criteria_met": {"relevance": "高", "fluency": "良好", "completeness": "一般", "accuracy": "待核实"}
                })
            if "你是一位报告修改专家" in query: # Refiner
                 original_content_marker = "原始内容：\n---\n"
                 original_content_start = query.find(original_content_marker)
                 if original_content_start != -1:
                    original_content_end = query.find("\n---", original_content_start + len(original_content_marker))
                    original_text = query[original_content_start + len(original_content_marker) : original_content_end]
                    return original_text + "\n（已根据模拟反馈进行修订）"
                 return "模拟修订后的内容。"
            return "Unknown LLM query for mock."
        # Need to mock get_model if Client() is called inside LLMService init
        def get_model(self, model_name): return self # Return self to allow attribute access like .chat

    class MockEmbeddingServiceForPipeline(EmbeddingService):
        def __init__(self): super().__init__(api_url="mock://emb", model_name="mock-emb")
        def create_embeddings(self, texts): return [[0.1] * 10 for _ in texts] # 10-dim dummy
        def get_model(self, model_name): return self

    class MockRerankerServiceForPipeline(RerankerService):
        def __init__(self): super().__init__(api_url="mock://rerank", model_name="mock-rerank")
        def rerank(self, query, documents, top_n=None):
            results = [{"document": doc, "relevance_score": 0.9 - (i*0.1), "original_index": i} for i, doc in enumerate(documents)]
            return results[:top_n] if top_n else results
        def get_model(self, model_name): return self

    # --- (End Mock Services) ---

    # Create dummy PDF files for testing DocumentProcessor
    dummy_pdf_path1 = "dummy_doc1.pdf"
    dummy_pdf_path2 = "dummy_doc2.pdf"

    try:
        from PyPDF2 import PdfWriter # Use PdfWriter
        # Create Dummy PDF 1
        writer1 = PdfWriter()
        writer1.add_blank_page(width=210, height=297) # A4 size in points (approx)
        # PyPDF2 PdfWriter doesn't have a direct way to add text easily to a new PDF.
        # We'll rely on DocumentProcessor's error handling if it can't extract text from these blank PDFs.
        # For a real test, actual PDFs with text are needed.
        # This setup is primarily to test the file handling part.
        # A more robust mock for DocumentProcessor itself might be needed if PyPDF2 causes issues.
        # For now, let's assume it will extract empty string from blank PDF.
        with open(dummy_pdf_path1, "wb") as f:
            writer1.write(f)

        # Create Dummy PDF 2 (similar, or with some metadata if possible)
        writer2 = PdfWriter()
        writer2.add_blank_page(width=210, height=297)
        writer2.add_metadata({"/Title": "Dummy Document 2"})
        with open(dummy_pdf_path2, "wb") as f:
            writer2.write(f)

        logger.info(f"Created dummy PDF files: {dummy_pdf_path1}, {dummy_pdf_path2}")

        # Instantiate mock services
        mock_llm = MockLLMServiceForPipeline()
        mock_embed = MockEmbeddingServiceForPipeline()
        mock_rerank = MockRerankerServiceForPipeline() # Optional

        # Initialize Pipeline with Mocks
        pipeline = ReportGenerationPipeline(
            llm_service=mock_llm,
            embedding_service=mock_embed,
            reranker_service=mock_rerank, # Can be None
            max_refinement_iterations=1 # Keep it short for example
        )

        # Override document processor for the dummy PDFs to return some text
        # As creating PDFs with text via PyPDF2 is complex.
        original_extract_text = pipeline.document_processor.extract_text_from_pdf
        def mock_extract_text(pdf_path):
            if pdf_path == dummy_pdf_path1:
                return "This is the content of the first dummy PDF document. It talks about core technologies."
            if pdf_path == dummy_pdf_path2:
                return "The second dummy PDF discusses future developments and challenges in advanced systems."
            return original_extract_text(pdf_path) # Fallback for other paths
        pipeline.document_processor.extract_text_from_pdf = mock_extract_text


        # Run the pipeline
        user_topic_example = "未来先进系统的发展趋势"
        pdf_files_example = [dummy_pdf_path1, dummy_pdf_path2]

        logger.info(f"Running pipeline with topic: '{user_topic_example}' and PDFs: {pdf_files_example}")

        final_report = pipeline.run(
            user_topic=user_topic_example,
            pdf_paths=pdf_files_example
        )

        print("\n" + "="*30 + " FINAL REPORT (Mocked) " + "="*30)
        print(final_report)
        print("="*78)

    except ReportGenerationPipelineError as e:
        logger.error(f"Pipeline execution failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline example: {e}", exc_info=True)
    finally:
        # Clean up dummy PDF files
        import os
        if os.path.exists(dummy_pdf_path1): os.remove(dummy_pdf_path1)
        if os.path.exists(dummy_pdf_path2): os.remove(dummy_pdf_path2)
        logger.info("Cleaned up dummy PDF files.")

    logger.info("ReportGenerationPipeline Example End")
