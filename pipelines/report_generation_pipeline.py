import logging
import json
import os
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi

from config import settings
from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from core.document_processor import DocumentProcessor, DocumentProcessorError
from core.vector_store import VectorStore, VectorStoreError
from core.retrieval_service import RetrievalService, RetrievalServiceError
# Import WorkflowState and task/status constants
from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT, STATUS_PENDING, STATUS_COMPLETED, STATUS_ERROR, \
    STATUS_WRITING_NEEDED, STATUS_EVALUATION_NEEDED, STATUS_REFINEMENT_NEEDED, \
    TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT, TASK_TYPE_APPLY_OUTLINE_REFINEMENT


from agents.topic_analyzer_agent import TopicAnalyzerAgent
from agents.outline_generator_agent import OutlineGeneratorAgent
from agents.content_retriever_agent import ContentRetrieverAgent # Will use RetrievalService
from agents.chapter_writer_agent import ChapterWriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.refiner_agent import RefinerAgent
from agents.report_compiler_agent import ReportCompilerAgent

logger = logging.getLogger(__name__)

class ReportGenerationPipelineError(Exception):
    pass

class ReportGenerationPipeline:
    def __init__(self,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 reranker_service: Optional[RerankerService] = None,
                 parent_chunk_size: int = settings.DEFAULT_PARENT_CHUNK_SIZE,
                 parent_chunk_overlap: int = settings.DEFAULT_PARENT_CHUNK_OVERLAP,
                 child_chunk_size: int = settings.DEFAULT_CHILD_CHUNK_SIZE,
                 child_chunk_overlap: int = settings.DEFAULT_CHILD_CHUNK_OVERLAP,
                 vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA,
                 final_top_n_retrieval: Optional[int] = None,
                 max_refinement_iterations: int = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
                 # New: Max overall iterations for the main loop to prevent infinite loops
                 max_workflow_iterations: int = 50 # Default max iterations for the main task loop
                ):

        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.max_refinement_iterations = max_refinement_iterations
        self.max_workflow_iterations = max_workflow_iterations


        self.document_processor = DocumentProcessor(
            parent_chunk_size=parent_chunk_size, parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size, child_chunk_overlap=child_chunk_overlap,
            supported_extensions=settings.SUPPORTED_DOC_EXTENSIONS
        )
        self.vector_store = VectorStore(embedding_service=self.embedding_service)

        self.bm25_index: Optional[BM25Okapi] = None
        self.all_child_chunks_for_bm25_mapping: List[Dict[str, Any]] = []

        self.retrieval_service: Optional[RetrievalService] = None
        self.content_retriever_agent: Optional[ContentRetrieverAgent] = None # Renamed from self.content_retriever

        self.retrieval_params = { # Stored for deferred initialization
            "vector_top_k": vector_top_k, "keyword_top_k": keyword_top_k,
            "hybrid_alpha": hybrid_alpha, "final_top_n": final_top_n_retrieval or vector_top_k
        }

        # Agents (most will now operate based on WorkflowState)
        self.topic_analyzer = TopicAnalyzerAgent(llm_service=self.llm_service)
        self.outline_generator = OutlineGeneratorAgent(llm_service=self.llm_service)
        self.chapter_writer = ChapterWriterAgent(llm_service=self.llm_service)
        self.evaluator = EvaluatorAgent(llm_service=self.llm_service)
        self.refiner = RefinerAgent(llm_service=self.llm_service)
        self.report_compiler = ReportCompilerAgent(add_table_of_contents=True)

        self.workflow_state: Optional[WorkflowState] = None # Will be created in run()

        logger.info("ReportGenerationPipeline initialized (WorkflowState and Retrieval components to be set up during run).")

    def _initialize_retrieval_components(self):
        """Initializes RetrievalService and ContentRetrieverAgent using data from WorkflowState (or self)."""
        if not self.retrieval_service: # Check if already initialized
            self.retrieval_service = RetrievalService(
                vector_store=self.vector_store, # Assumes vector_store is populated
                bm25_index=self.bm25_index, # Assumes bm25_index is built
                all_child_chunks_for_bm25_mapping=self.all_child_chunks_for_bm25_mapping,
                reranker_service=self.reranker_service
            )
            logger.info("RetrievalService initialized.")

        if not self.content_retriever_agent: # Check if already initialized
            self.content_retriever_agent = ContentRetrieverAgent(
                retrieval_service=self.retrieval_service, # Pass the service instance
                default_vector_top_k=self.retrieval_params["vector_top_k"],
                default_keyword_top_k=self.retrieval_params["keyword_top_k"],
                default_hybrid_alpha=self.retrieval_params["hybrid_alpha"],
                default_final_top_n=self.retrieval_params["final_top_n"]
            )
            logger.info("ContentRetrieverAgent initialized using RetrievalService.")


    def _process_and_load_data(self, data_path: str):
        # This method remains largely the same but now assumes self.workflow_state exists for logging.
        self.workflow_state.log_event(f"Starting document processing from data_path: {data_path}")
        if not os.path.isdir(data_path):
            msg = f"Invalid data_path: {data_path} is not a directory."
            self.workflow_state.log_event(msg, {"level": "ERROR"})
            raise ReportGenerationPipelineError(msg)

        all_parent_child_data: List[Dict[str, Any]] = []
        processed_file_count = 0
        # ... (file iteration and processing logic as before) ...
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            if not os.path.isfile(file_path): continue
            _, extension = os.path.splitext(filename.lower())
            if extension not in settings.SUPPORTED_DOC_EXTENSIONS: continue
            try:
                raw_text = self.document_processor.extract_text_from_file(file_path)
                if not raw_text.strip(): continue
                doc_id_base = os.path.splitext(filename)[0]
                parent_child_chunks = self.document_processor.split_text_into_parent_child_chunks(raw_text, doc_id_base)
                all_parent_child_data.extend(parent_child_chunks)
                processed_file_count +=1
            except Exception as e:
                self.workflow_state.log_event(f"Error processing file {file_path}", {"error": str(e), "level": "ERROR"})


        if not all_parent_child_data:
            raise ReportGenerationPipelineError("No usable content extracted from provided data_path.")

        self.vector_store.add_documents(all_parent_child_data)
        self.workflow_state.log_event(f"Data from {processed_file_count} files added to VectorStore.",
                                     {"child_chunks_count": self.vector_store.count_child_chunks})
        self.workflow_state.set_flag('data_loaded', True)

        # Prepare and build BM25 Index
        self.all_child_chunks_for_bm25_mapping = [{"child_id": i['child_id'], "child_text": i['child_text']} for i in self.vector_store.document_store]
        if self.all_child_chunks_for_bm25_mapping:
            # Tokenization for BM25 (simple split, TODO: improve for Chinese)
            tokenized_corpus = [item['child_text'].lower().split() for item in self.all_child_chunks_for_bm25_mapping]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.workflow_state.log_event(f"BM25 index built with {len(tokenized_corpus)} child chunks.")
        else:
            self.bm25_index = None
            self.workflow_state.log_event("No child chunks for BM25 index.", {"level": "WARNING"})

        self._initialize_retrieval_components() # Initialize services that depend on loaded data


    def _handle_task(self, task: Dict[str, Any]):
        """Handles a single task from the WorkflowState queue."""
        task_type = task['type']
        payload = task.get('payload', {})
        task_id = task['id']

        self.workflow_state.log_event(f"Handling task: {task_type}", {"task_id": task_id, "payload": payload})

        try:
            if task_type == TASK_TYPE_ANALYZE_TOPIC:
                user_topic = payload.get('user_topic', self.workflow_state.user_topic)
                analysis_results = self.topic_analyzer.run(user_topic=user_topic)
                self.workflow_state.update_topic_analysis(analysis_results)
                self.workflow_state.add_task(TASK_TYPE_GENERATE_OUTLINE, payload={'topic_details': analysis_results})
                self.workflow_state.complete_task(task_id, "Topic analysis complete.")

            elif task_type == TASK_TYPE_GENERATE_OUTLINE:
                if not self.workflow_state.topic_analysis_results:
                    raise ReportGenerationPipelineError("Topic analysis results not found for outline generation.")

                # Use ReportCompilerAgent's parser to get structured outline with IDs
                temp_compiler = ReportCompilerAgent() # Temporary instance for its parsing logic

                markdown_outline = self.outline_generator.run(self.workflow_state.topic_analysis_results)
                if not markdown_outline.strip():
                    raise ReportGenerationPipelineError("Outline generation resulted in an empty outline.")

                parsed_outline_struct = temp_compiler._parse_markdown_outline(markdown_outline) # This adds 'id'

                self.workflow_state.update_outline(markdown_outline, parsed_outline_struct)

                for item in self.workflow_state.parsed_outline:
                    chapter_key = item['id'] # Use the generated ID
                    self.workflow_state.add_task(TASK_TYPE_PROCESS_CHAPTER,
                                                 payload={'chapter_key': chapter_key, 'chapter_title': item['title']})
                self.workflow_state.complete_task(task_id, "Outline generated and chapter processing tasks added.")

            elif task_type == TASK_TYPE_PROCESS_CHAPTER: # This is now a meta-task, first step is retrieval
                chapter_key = payload['chapter_key']
                self.workflow_state.update_chapter_status(chapter_key, STATUS_RETRIEVAL_NEEDED)
                self.workflow_state.add_task(TASK_TYPE_RETRIEVE_FOR_CHAPTER, payload=payload) # Pass on payload
                self.workflow_state.complete_task(task_id, f"Chapter '{payload['chapter_title']}' processing initiated.")

            elif task_type == TASK_TYPE_RETRIEVE_FOR_CHAPTER:
                chapter_key = payload['chapter_key']
                chapter_title = payload['chapter_title']
                keywords = self.workflow_state.topic_analysis_results.get('keywords_cn', [])
                query = f"{chapter_title} {' '.join(keywords)}".strip()

                if not self.content_retriever_agent: # Ensure it's initialized
                     self._initialize_retrieval_components() # Should have been done after data load
                     if not self.content_retriever_agent:
                         raise ReportGenerationPipelineError("ContentRetrieverAgent still not available after trying to initialize.")

                retrieved_docs = self.content_retriever_agent.run(query=query) # Uses its defaults

                chapter_entry = self.workflow_state._get_chapter_entry(chapter_key, create_if_missing=True)
                chapter_entry['retrieved_docs'] = retrieved_docs
                self.workflow_state.update_chapter_status(chapter_key, STATUS_WRITING_NEEDED)
                self.workflow_state.add_task(TASK_TYPE_WRITE_CHAPTER, payload=payload)
                self.workflow_state.complete_task(task_id, f"Retrieval complete for chapter '{chapter_title}'.")

            elif task_type == TASK_TYPE_WRITE_CHAPTER:
                chapter_key = payload['chapter_key']
                chapter_title = payload['chapter_title']
                chapter_data = self.workflow_state.get_chapter_data(chapter_key)
                if not chapter_data or chapter_data.get('status') != STATUS_WRITING_NEEDED:
                    raise ReportGenerationPipelineError(f"Chapter '{chapter_title}' not ready for writing or data missing.")

                content = self.chapter_writer.run(chapter_title, chapter_data.get('retrieved_docs', []))
                self.workflow_state.update_chapter_content(chapter_key, content, chapter_data.get('retrieved_docs'))
                self.workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED)
                self.workflow_state.add_task(TASK_TYPE_EVALUATE_CHAPTER, payload=payload)
                self.workflow_state.complete_task(task_id, f"Writing complete for chapter '{chapter_title}'.")

            elif task_type == TASK_TYPE_EVALUATE_CHAPTER:
                chapter_key = payload['chapter_key']
                chapter_data = self.workflow_state.get_chapter_data(chapter_key)
                if not chapter_data or not chapter_data.get('content'):
                    raise ReportGenerationPipelineError(f"No content to evaluate for chapter '{payload['chapter_title']}'.")

                evaluation = self.evaluator.run(chapter_data['content'])
                self.workflow_state.add_chapter_evaluation(chapter_key, evaluation)

                # Logic for refinement or completion
                # For simplicity, assume one refinement iteration for now if score is below threshold
                # This logic can be made more complex (e.g. based on number of evaluations for this chapter)
                current_eval_count = len(chapter_data.get('evaluations', []))
                if evaluation.get('score', 0) < 80 and current_eval_count <= self.max_refinement_iterations : # Threshold is arbitrary
                    self.workflow_state.update_chapter_status(chapter_key, STATUS_REFINEMENT_NEEDED)
                    self.workflow_state.add_task(TASK_TYPE_REFINE_CHAPTER, payload=payload)
                    self.workflow_state.log_event(f"Chapter '{payload['chapter_title']}' queued for refinement based on score {evaluation.get('score')}.")
                else:
                    self.workflow_state.update_chapter_status(chapter_key, STATUS_COMPLETED)
                    self.workflow_state.log_event(f"Chapter '{payload['chapter_title']}' marked as completed.")
                self.workflow_state.complete_task(task_id, f"Evaluation complete for chapter '{payload['chapter_title']}'.")

            elif task_type == TASK_TYPE_REFINE_CHAPTER:
                chapter_key = payload['chapter_key']
                chapter_data = self.workflow_state.get_chapter_data(chapter_key)
                if not chapter_data or not chapter_data.get('content') or not chapter_data.get('evaluations'):
                    raise ReportGenerationPipelineError(f"Not enough data to refine chapter '{payload['chapter_title']}'.")

                # Use the latest evaluation for refinement
                latest_evaluation = chapter_data['evaluations'][-1]
                refined_content = self.refiner.run(chapter_data['content'], latest_evaluation)
                self.workflow_state.update_chapter_content(chapter_key, refined_content, is_new_version=True)
                self.workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Back to evaluation
                self.workflow_state.add_task(TASK_TYPE_EVALUATE_CHAPTER, payload=payload) # Re-evaluate
                self.workflow_state.complete_task(task_id, f"Refinement complete for chapter '{payload['chapter_title']}'. Queued for re-evaluation.")

            elif task_type == TASK_TYPE_COMPILE_REPORT:
                if not self.workflow_state.are_all_chapters_completed() or \
                   not self.workflow_state.get_flag('outline_finalized', False): # Check if outline is finalized
                    self.workflow_state.log_event("Compile report task received, but not all chapters completed or outline not final. Re-queueing.", {"level":"WARNING"})
                    self.workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=task.get('priority', 100)+1) # Re-queue with lower priority
                    self.workflow_state.complete_task(task_id, "Re-queued as prerequisites not met.", status='deferred')
                    return

                report_context = self.workflow_state.get_full_report_context_for_compilation()
                final_report = self.report_compiler.run(**report_context)
                self.workflow_state.set_flag('final_report_md', final_report) # Store report in state
                self.workflow_state.set_flag('report_generation_complete', True)
                self.workflow_state.complete_task(task_id, "Report compilation complete.")

            # TODO: Implement TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT and TASK_TYPE_APPLY_OUTLINE_REFINEMENT if agents generate these

            else:
                self.workflow_state.log_event(f"Unknown task type: {task_type}", {"task_id": task_id, "level": "ERROR"})
                self.workflow_state.complete_task(task_id, f"Unknown task type {task_type}", status='failed')

        except Exception as e:
            self.workflow_state.log_event(f"Error handling task {task_type} ({task_id})", {"error": str(e), "level": "CRITICAL"}, )
            self.workflow_state.complete_task(task_id, f"Failed with error: {str(e)}", status='failed')
            # Optionally, add specific error handling for chapter tasks to mark chapter as error
            if 'chapter_key' in payload:
                 self.workflow_state.add_chapter_error(payload['chapter_key'], str(e))


    def run(self, user_topic: str, data_path: str, report_title: Optional[str] = None) -> str:
        self.workflow_state = WorkflowState(user_topic, report_title)
        self.workflow_state.log_event("Pipeline run started.")

        try:
            self._process_and_load_data(data_path) # This also initializes retrieval components
        except Exception as e:
            self.workflow_state.log_event(f"Critical error during data processing: {e}", {"level": "CRITICAL"},)
            self.workflow_state.set_flag('report_generation_complete', True) # End workflow
            self.workflow_state.increment_error_count()
            logger.error(f"Pipeline run failed during data processing: {e}", exc_info=True)
            return f"Error: Data processing failed: {e}"


        # Add initial task to start the workflow
        self.workflow_state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={'user_topic': user_topic}, priority=0)

        iteration_count = 0
        while not self.workflow_state.get_flag('report_generation_complete', False):
            if iteration_count >= self.max_workflow_iterations:
                self.workflow_state.log_event("Max workflow iterations reached. Halting.", {"level": "ERROR"})
                self.workflow_state.set_flag('report_generation_complete', True) # Force stop
                break

            task = self.workflow_state.get_next_task()
            if not task:
                # No more tasks. Check if all chapters are done. If so, request compilation.
                # Otherwise, something might be wrong, or it's genuinely finished.
                if self.workflow_state.are_all_chapters_completed() and \
                   self.workflow_state.get_flag('outline_finalized', False) and \
                   not self.workflow_state.get_flag('report_compilation_requested', False):
                    self.workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100) # Low priority
                    self.workflow_state.set_flag('report_compilation_requested', True)
                    self.workflow_state.log_event("All chapters completed, outline final. Compilation task added.")
                    continue # Restart loop to pick up compile task

                self.workflow_state.log_event("No more pending tasks and report not yet complete. Checking conditions.",
                                             {"all_chapters_done": self.workflow_state.are_all_chapters_completed(),
                                              "outline_finalized": self.workflow_state.get_flag('outline_finalized')})
                # If stuck, break after some checks or more iterations.
                # This simple break assumes if no tasks and not complete, it's an issue or truly done.
                if iteration_count > self.max_workflow_iterations / 2 and not self.workflow_state.pending_tasks: # Heuristic
                    self.workflow_state.log_event("Stuck: No tasks and not complete. Halting.", {"level":"ERROR"})
                    break
                # If there are truly no tasks and conditions for compilation aren't met, something is amiss.
                # For now, if queue is empty and not complete, we might break or wait if tasks could be added by async process.
                # Given current synchronous model, empty queue and not complete means it's stuck or finished without compile.
                if not self.workflow_state.pending_tasks:
                    logger.warning("Task queue empty but report generation not flagged as complete. Workflow might be stuck or finished unexpectedly.")
                    break


            self._handle_task(task)
            iteration_count += 1
            self.workflow_state.log_event(f"Workflow iteration {iteration_count} complete.")

        final_report_md = self.workflow_state.get_flag('final_report_md')
        if final_report_md:
            self.workflow_state.log_event("Report generation process concluded successfully.")
            return final_report_md
        else:
            self.workflow_state.log_event("Report generation failed or did not produce a report.", {"level": "ERROR"})
            # Compile a summary of errors from workflow_state if possible
            error_summary = "Workflow finished without generating a report. Check logs. "
            if self.workflow_state.error_count > 0:
                error_summary += f"Total errors: {self.workflow_state.error_count}. "
            # You could try to compile what's available if needed:
            # partial_context = self.workflow_state.get_full_report_context_for_compilation()
            # partial_report = self.report_compiler.run(**partial_context)
            # return f"PARTIAL REPORT (Errors Occurred):\n{partial_report}"
            return error_summary + "See logs for details."


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("ReportGenerationPipeline (WorkflowState Integrated) Example Start")

    class MockLLMServiceForPipeline(LLMService): # Simplified mock
        def __init__(self): super().__init__(api_url="mock://llm", model_name="mock-llm")
        def chat(self, query, system_prompt, **kwargs):
            if TASK_TYPE_ANALYZE_TOPIC in query or "主题分析专家" in query: return json.dumps({"generalized_topic_cn": "WS主题", "keywords_cn": ["测试"]})
            if TASK_TYPE_GENERATE_OUTLINE in query or "报告大纲撰写助手" in query: return "- 章节1\n  - 1.1 子章节\n- 章节2"
            if TASK_TYPE_WRITE_CHAPTER in query or "专业的报告撰写员" in query: return "章节内容模拟。"
            if TASK_TYPE_EVALUATE_CHAPTER in query or"资深的报告评审员" in query: return json.dumps({"score": 85, "feedback_cn": "ok"}) # High score to avoid refinement loop in test
            if TASK_TYPE_REFINE_CHAPTER in query or "报告修改专家" in query: return "精炼内容模拟。"
            return "LLM Mock Response"
        def get_model(self, model_name): return self
    class MockEmbeddingServiceForPipeline(EmbeddingService):
        def __init__(self): super().__init__(api_url="mock://emb", model_name="mock-emb")
        def create_embeddings(self, texts): return [[0.1]*5 for _ in texts]
        def get_model(self, model_name): return self
    class MockRerankerServiceForPipeline(RerankerService):
        def __init__(self): super().__init__(api_url="mock://rerank", model_name="mock-rerank")
        def rerank(self, query, documents, top_n=None): return [{"document":d, "relevance_score":0.9, "original_index":i} for i,d in enumerate(documents)][:top_n if top_n else len(documents)]
        def get_model(self, model_name): return self

    dummy_data_dir = "temp_pipeline_ws_test_data"
    if not os.path.exists(dummy_data_dir): os.makedirs(dummy_data_dir)
    with open(os.path.join(dummy_data_dir, "test_doc.txt"), "w", encoding="utf-8") as f:
        f.write("Sentence one for topic. Sentence two for topic.\n\nAnother paragraph here with sentence three and sentence four about topic.")

    try:
        pipeline = ReportGenerationPipeline(
            llm_service=MockLLMServiceForPipeline(),
            embedding_service=MockEmbeddingServiceForPipeline(),
            reranker_service=MockRerankerServiceForPipeline(),
            max_refinement_iterations=0, # No refinement for this simple test
            max_workflow_iterations=20 # Limit workflow iterations for test
        )
        # The outline parser in ReportCompilerAgent needs to be called by WorkflowState.update_outline
        # or by the outline generation task handler.
        # The mock OutlineGeneratorAgent returns a string, WorkflowState.update_outline needs both md and parsed.
        # Let's refine the mock or the handler for TASK_TYPE_GENERATE_OUTLINE.
        # For the test, we'll ensure the mock for outline generator is simple.

        final_report = pipeline.run(user_topic="Test Topic WorkflowState", data_path=dummy_data_dir)
        print("\n" + "="*30 + " FINAL REPORT (Mocked - WorkflowState) " + "="*30)
        print(final_report)
        print("="*80)

        print("\nWorkflow Log Snippet:")
        for i, log_entry in enumerate(pipeline.workflow_state.workflow_log[-5:]): # Last 5 entries
            print(f"  {i}: {log_entry[0].strftime('%H:%M:%S')} - {log_entry[1]} - {log_entry[2]}")

    except Exception as e:
        logger.error(f"Pipeline example failed: {e}", exc_info=True)
    finally:
        import shutil
        if os.path.exists(dummy_data_dir): shutil.rmtree(dummy_data_dir)

    logger.info("ReportGenerationPipeline (WorkflowState Integrated) Example End")
