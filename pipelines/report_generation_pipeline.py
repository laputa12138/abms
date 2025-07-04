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
from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT # STATUS constants are used by agents & orchestrator
from core.orchestrator import Orchestrator # Import Orchestrator

from agents.topic_analyzer_agent import TopicAnalyzerAgent
from agents.outline_generator_agent import OutlineGeneratorAgent
from agents.content_retriever_agent import ContentRetrieverAgent
from agents.chapter_writer_agent import ChapterWriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.refiner_agent import RefinerAgent
from agents.report_compiler_agent import ReportCompilerAgent
from agents.global_content_retriever_agent import GlobalContentRetrieverAgent
from agents.outline_refinement_agent import OutlineRefinementAgent

logger = logging.getLogger(__name__)

class ReportGenerationPipelineError(Exception):
    pass

class ReportGenerationPipeline:
    def __init__(self,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 # Services
                 reranker_service: Optional[RerankerService] = None,
                 # Execution context parameters (from CLI or their defaults from settings)
                 vector_store_path: str = settings.DEFAULT_VECTOR_STORE_PATH,
                 index_name: Optional[str] = None,
                 force_reindex: bool = False,
                 max_workflow_iterations: int = settings.DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS,
                 # CLI-overridden hyperparameters (main.py passes these using args.*)
                 # Their names in constructor match the names in main.py's args
                 cli_overridden_parent_chunk_size: int = settings.DEFAULT_PARENT_CHUNK_SIZE,
                 cli_overridden_parent_chunk_overlap: int = settings.DEFAULT_PARENT_CHUNK_OVERLAP,
                 cli_overridden_child_chunk_size: int = settings.DEFAULT_CHILD_CHUNK_SIZE,
                 cli_overridden_child_chunk_overlap: int = settings.DEFAULT_CHILD_CHUNK_OVERLAP,
                 cli_overridden_vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 cli_overridden_keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 cli_overridden_hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA,
                 cli_overridden_final_top_n_retrieval: int = settings.DEFAULT_RETRIEVAL_FINAL_TOP_N,
                 cli_overridden_max_refinement_iterations: int = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
                 # New parameter for key terms
                 key_terms_definitions: Optional[Dict[str, str]] = None
                ):

        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.key_terms_definitions = key_terms_definitions # Store for use in run()
        self.reranker_service = reranker_service

        # Store execution context parameters
        self.vector_store_path = vector_store_path
        self.index_name = index_name
        self.force_reindex = force_reindex
        self.max_workflow_iterations = max_workflow_iterations # Passed to Orchestrator

        # Store CLI-overridden hyperparameters (or their defaults from settings if not overridden by CLI)
        # These will be used to initialize components like DocumentProcessor and ContentRetrieverAgent
        self.parent_chunk_size = cli_overridden_parent_chunk_size
        self.parent_chunk_overlap = cli_overridden_parent_chunk_overlap
        self.child_chunk_size = cli_overridden_child_chunk_size
        self.child_chunk_overlap = cli_overridden_child_chunk_overlap
        self.vector_top_k = cli_overridden_vector_top_k
        self.keyword_top_k = cli_overridden_keyword_top_k
        self.hybrid_alpha = cli_overridden_hybrid_alpha
        self.final_top_n_retrieval = cli_overridden_final_top_n_retrieval
        self.max_refinement_iterations = cli_overridden_max_refinement_iterations # Used by WorkflowState

        # Initialize DocumentProcessor with effective chunking parameters
        self.document_processor = DocumentProcessor(
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
            child_chunk_size=self.child_chunk_size,
            child_chunk_overlap=self.child_chunk_overlap
            # supported_extensions is read from settings by DocumentProcessor itself
        )
        self.vector_store = VectorStore(embedding_service=self.embedding_service)

        self.bm25_index: Optional[BM25Okapi] = None
        self.all_child_chunks_for_bm25_mapping: List[Dict[str, Any]] = []

        self.retrieval_service: Optional[RetrievalService] = None # Initialized later
        self.content_retriever_agent: Optional[ContentRetrieverAgent] = None # Initialized later

        # self.retrieval_params dictionary is no longer needed as params are passed directly or sourced from settings.

        # Initialize agents that don't depend on dynamically configured retrieval params here
        self.topic_analyzer = TopicAnalyzerAgent(llm_service=self.llm_service)
        # self.outline_generator will be initialized in _initialize_retrieval_and_orchestration_components
        # after retrieval_service is available.
        self.outline_generator: Optional[OutlineGeneratorAgent] = None
        # ContentRetrieverAgent is initialized in _initialize_retrieval_and_orchestration_components
        self.chapter_writer = ChapterWriterAgent(llm_service=self.llm_service)
        # EvaluatorAgent now gets its threshold from config.settings
        self.evaluator = EvaluatorAgent(llm_service=self.llm_service)
        self.refiner = RefinerAgent(llm_service=self.llm_service)
        self.report_compiler = ReportCompilerAgent(add_table_of_contents=True)

        self.workflow_state: Optional[WorkflowState] = None
        self.global_content_retriever= None
        outline_refiner_prompt_template = None
        self.outline_refinement_agent = None

        self.orchestrator: Optional[Orchestrator] = None

        logger.info("ReportGenerationPipeline initialized.")

    def _initialize_retrieval_and_orchestration_components(self):
        """Initializes RetrievalService, ContentRetrieverAgent, and Orchestrator."""
        if not self.workflow_state:
            raise ReportGenerationPipelineError("WorkflowState must be initialized before retrieval/orchestration components.")

        if not self.retrieval_service:
            self.retrieval_service = RetrievalService(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                all_child_chunks_for_bm25_mapping=self.all_child_chunks_for_bm25_mapping,
                reranker_service=self.reranker_service
            )
            self.workflow_state.log_event("RetrievalService initialized.")

        if not self.content_retriever_agent:
            # ContentRetrieverAgent is initialized with the effective parameters
            # (either from CLI overrides via main.py -> pipeline, or settings defaults if no CLI override)
            self.content_retriever_agent = ContentRetrieverAgent(
                retrieval_service=self.retrieval_service,
                default_vector_top_k=self.vector_top_k, # Use the value stored in self, which came from CLI/settings
                default_keyword_top_k=self.keyword_top_k,
                default_hybrid_alpha=self.hybrid_alpha,
                default_final_top_n=self.final_top_n_retrieval
            )
            self.workflow_state.log_event("ContentRetrieverAgent initialized using RetrievalService with effective parameters.")

        # Initialize OutlineGeneratorAgent now that retrieval_service is available
        if not self.outline_generator:
            self.outline_generator = OutlineGeneratorAgent(
                llm_service=self.llm_service,
                retrieval_service=self.retrieval_service
                # prompt_template can be omitted to use default from settings
            )
            self.workflow_state.log_event("OutlineGeneratorAgent initialized with RetrievalService.")

        self.global_content_retriever=GlobalContentRetrieverAgent(retrieval_service=self.retrieval_service,llm_service=self.llm_service)
        outline_refiner_prompt_template = getattr(settings, 'OUTLINE_REFINEMENT_PROMPT_TEMPLATE', None)
        self.outline_refinement_agent = OutlineRefinementAgent(llm_service=self.llm_service, prompt_template=outline_refiner_prompt_template)

        if not self.orchestrator:
            # Ensure all agents required by orchestrator are initialized
            if not self.outline_generator:
                raise ReportGenerationPipelineError("OutlineGeneratorAgent failed to initialize before Orchestrator setup.")

            self.orchestrator = Orchestrator(
                workflow_state=self.workflow_state,
                topic_analyzer=self.topic_analyzer,
                outline_generator=self.outline_generator, # Now guaranteed to be initialized
                global_content_retriever=self.global_content_retriever,
                outline_refiner=self.outline_refinement_agent,
                content_retriever=self.content_retriever_agent,
                chapter_writer=self.chapter_writer,
                evaluator=self.evaluator,
                refiner=self.refiner,
                report_compiler=self.report_compiler,
                max_workflow_iterations=self.max_workflow_iterations
            )
            self.workflow_state.log_event("Orchestrator initialized.")


    def _process_and_load_data(self, data_path: str):
        logger.debug(f"Starting _process_and_load_data: initial data_path='{data_path}', "
                     f"initial self.vector_store_path='{self.vector_store_path}', "
                     f"initial self.index_name='{self.index_name}', "
                     f"initial self.force_reindex={self.force_reindex}")
        self.workflow_state.log_event(f"Data processing: data_path='{data_path}', vs_path='{self.vector_store_path}', "
                                     f"index_name='{self.index_name}', force_reindex={self.force_reindex}")
        loaded_from_file = False

        # Determine the effective index name
        # If self.index_name is provided, use it. Otherwise, derive from data_path.
        if self.index_name:
            effective_index_name = self.index_name
        else:
            if data_path and os.path.isdir(data_path): # Ensure data_path is a directory before using its basename
                effective_index_name = os.path.basename(os.path.normpath(data_path))
            else: # Fallback if data_path is not suitable for basename
                effective_index_name = "default_rag_index"
            if not effective_index_name: # Further fallback if basename was empty (e.g. data_path was '/')
                effective_index_name = "default_rag_index"

        logger.debug(f"Determined effective_index_name: '{effective_index_name}'")
        self.workflow_state.log_event(f"Effective index name for VectorStore: '{effective_index_name}'")

        # Prepare vector store directory and paths
        vs_dir = os.path.abspath(self.vector_store_path)
        logger.debug(f"Absolute vector store directory (vs_dir): '{vs_dir}'")
        if not os.path.exists(vs_dir):
            try:
                os.makedirs(vs_dir, exist_ok=True)
                self.workflow_state.log_event(f"Created vector store directory: {vs_dir}")
            except OSError as e:
                self.workflow_state.log_event(f"Failed to create vector store directory {vs_dir}: {e}. "
                                             "Will attempt to proceed but saving/loading may fail.", {"level": "ERROR"})
                # Allow to proceed, VectorStore will handle errors if paths are unusable

        faiss_index_path = os.path.join(vs_dir, f"{effective_index_name}.faiss")
        metadata_path = os.path.join(vs_dir, f"{effective_index_name}.meta.json")
        logger.debug(f"Calculated FAISS index path: '{faiss_index_path}'")
        logger.debug(f"Calculated metadata path: '{metadata_path}'")

        if not self.force_reindex:
            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                try:
                    logger.info(f"Attempting to load existing VectorStore: index='{faiss_index_path}', meta='{metadata_path}'")
                    self.workflow_state.log_event(f"Attempting to load existing VectorStore: index='{faiss_index_path}', meta='{metadata_path}'")
                    self.vector_store.load_store(faiss_index_path, metadata_path)
                    if self.vector_store.count_child_chunks > 0:
                        loaded_from_file = True
                        self.workflow_state.log_event(f"Successfully loaded VectorStore. {self.vector_store.count_child_chunks} child chunks.")
                    else:
                        self.workflow_state.log_event("Loaded VectorStore files but store is empty. Will re-process.", {"level": "WARNING"})
                except Exception as e:
                    self.workflow_state.log_event(f"Failed to load existing VectorStore from '{effective_index_name}': {e}. Will re-process.", {"level": "WARNING"})
            else:
                self.workflow_state.log_event(f"No existing index found for '{effective_index_name}' at '{vs_dir}'. Will process documents from data_path.")

        if not loaded_from_file:
            self.workflow_state.log_event(f"Processing documents from directory: {data_path}")
            if not data_path or not os.path.isdir(data_path): # Added check for data_path being None or not a dir
                # If data_path is invalid and we couldn't load an index, we cannot proceed.
                raise ReportGenerationPipelineError(f"Invalid data_path for processing: '{data_path}' is not a directory or not provided, and no existing index could be loaded.")

            all_parent_child_data: List[Dict[str, Any]] = []
            processed_file_count = 0
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
                raise ReportGenerationPipelineError(f"No usable content extracted/chunked from data_path '{data_path}' to build a new index.")

            # Re-initialize vector_store to ensure it's clean before adding new documents
            self.vector_store = VectorStore(embedding_service=self.embedding_service)
            self.vector_store.add_documents(all_parent_child_data)
            self.workflow_state.log_event(f"Data from {processed_file_count} files processed and added to new VectorStore.",
                                         {"child_chunks_count": self.vector_store.count_child_chunks})

            # Use the same faiss_index_path and metadata_path determined earlier for saving
            try:
                self.vector_store.save_store(faiss_index_path, metadata_path)
                self.workflow_state.log_event(f"New VectorStore saved: index='{faiss_index_path}', meta='{metadata_path}'")
            except Exception as e:
                 self.workflow_state.log_event(f"Failed to save new VectorStore: {e}. Processing will continue with in-memory store.", {"level": "ERROR"})

        self.workflow_state.set_flag('data_loaded', True)

        self.all_child_chunks_for_bm25_mapping = [{"child_id": i['child_id'], "child_text": i['child_text']}
                                                  for i in self.vector_store.document_store]
        if self.all_child_chunks_for_bm25_mapping:
            # TODO: Use a better tokenizer for Chinese for BM25.
            tokenized_corpus = [item['child_text'].lower().split() for item in self.all_child_chunks_for_bm25_mapping]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.workflow_state.log_event(f"BM25 index built with {len(tokenized_corpus)} child chunks.")
        else:
            self.bm25_index = None
            self.workflow_state.log_event("No child chunks available to build BM25 index.", {"level": "WARNING"})

        self._initialize_retrieval_and_orchestration_components()


    def run(self, user_topic: str, data_path: str, report_title: Optional[str] = None,
            # Allow key_terms_definitions to be passed at runtime as well, potentially overriding __init__
            key_terms_definitions: Optional[Dict[str, str]] = None) -> str:
        self.workflow_state = WorkflowState(user_topic, report_title)
        # Pass max_refinement_iterations to workflow_state so agents (Evaluator) can access it
        self.workflow_state.set_flag('max_refinement_iterations', self.max_refinement_iterations)

        # Prioritize runtime key_terms_definitions, then __init__ provided, then None
        final_key_terms = key_terms_definitions if key_terms_definitions is not None else self.key_terms_definitions
        if final_key_terms:
            self.workflow_state.update_key_terms_definitions(final_key_terms)
            self.workflow_state.log_event("Key terms definitions populated in WorkflowState.", {"source": "pipeline_input", "count": len(final_key_terms)})

        self.workflow_state.log_event("Pipeline run initiated.")

        try:
            self._process_and_load_data(data_path)
        except Exception as e:
            self.workflow_state.log_event(f"Critical error during data processing: {e}", {"level": "CRITICAL", "details": str(e)})
            self.workflow_state.set_flag('report_generation_complete', True)
            self.workflow_state.increment_error_count()
            logger.error(f"Pipeline run failed during data processing: {e}", exc_info=True)
            return f"Error: Data processing failed: {e}. Check logs at {self.workflow_state.get_flag('log_file_path', 'log file (path not set)') if self.workflow_state else 'log file'}."


        if not self.orchestrator:
            msg = "Orchestrator not initialized. This is a critical error in pipeline setup."
            self.workflow_state.log_event(msg, {"level": "CRITICAL"})
            raise ReportGenerationPipelineError(msg)

        self.workflow_state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={'user_topic': user_topic}, priority=0)

        try:
            self.orchestrator.coordinate_workflow()
        except Exception as e:
            self.workflow_state.log_event(f"Critical error during workflow coordination: {e}", {"level": "CRITICAL", "details": str(e)})
            self.workflow_state.set_flag('report_generation_complete', True) # Ensure loop terminates
            self.workflow_state.increment_error_count()
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            # Fall through to return error summary

        final_report_md = self.workflow_state.get_flag('final_report_md')
        if final_report_md and self.workflow_state.get_flag('report_generation_complete'):
            self.workflow_state.log_event("Report generation process concluded successfully.")
            return final_report_md
        else:
            self.workflow_state.log_event("Report generation failed or did not produce a complete report.", {"level": "ERROR"})
            error_summary = "Workflow finished without generating a report. Check logs. "
            if self.workflow_state.error_count > 0: error_summary += f"Total errors: {self.workflow_state.error_count}. "
            log_path_info = self.workflow_state.get_flag('log_file_path', 'log file (path not set)') if self.workflow_state else 'log file'
            return error_summary + f"See logs (e.g., {log_path_info}) for details."


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("ReportGenerationPipeline (Orchestrator & Indexing Logic) Example Start")

    class MockLLMServiceForPipeline(LLMService):
        def __init__(self): super().__init__(api_url="mock://llm", model_name="mock-llm")
        def chat(self, query, system_prompt, **kwargs):
            # Simulate different responses based on task type indicators in query
            if "主题分析专家" in query: return json.dumps({"generalized_topic_cn": "测试主题", "keywords_cn": ["测试", "流程"]})
            if "报告大纲撰写助手" in query: return "- 章节 A\n- 章节 B"
            if "专业的报告撰写员" in query: return f"这是关于 {kwargs.get('chapter_title', '未知章节')} 的模拟内容。"
            if "资深的报告评审员" in query: return json.dumps({"score": 88, "feedback_cn": "内容良好，符合预期。"})
            if "报告修改专家" in query: return "这是精炼后的模拟内容。"
            return "Default Mock LLM Response"
        def get_model(self, model_name): return self

    class MockEmbeddingServiceForPipeline(EmbeddingService):
        def __init__(self): super().__init__(api_url="mock://emb", model_name="mock-emb")
        def create_embeddings(self, texts): return [[0.5]*5 for _ in texts] # Ensure consistent dummy embeddings
        def get_model(self, model_name): return self

    # Create dummy dirs for test
    dummy_data_dir_pipe = os.path.abspath("temp_pipeline_main_test_data")
    dummy_vs_dir_pipe = os.path.abspath("temp_pipeline_main_test_vs")
    for d_path in [dummy_data_dir_pipe, dummy_vs_dir_pipe]:
        if not os.path.exists(d_path): os.makedirs(d_path, exist_ok=True)

    # Create a dummy document
    with open(os.path.join(dummy_data_dir_pipe, "sample_doc_for_pipeline.txt"), "w", encoding="utf-8") as f:
        f.write("This is sentence one. This is sentence two about testing.\n\nThis is a new paragraph with sentence three for the test.")

    try:
        pipeline = ReportGenerationPipeline(
            llm_service=MockLLMServiceForPipeline(),
            embedding_service=MockEmbeddingServiceForPipeline(),
            reranker_service=None, # Test without reranker first
            vector_store_path=dummy_vs_dir_pipe,
            index_name="pipeline_test_index",
            force_reindex=True, # Force reindex for consistent test runs
            max_refinement_iterations=0, # No refinement for this test
            max_workflow_iterations=30, # Generous limit for test
            key_terms_definitions={"CLI_TERM": "Command Line Interface Term, provided at init."} # Test __init__ propagation
        )

        # Set log file path in workflow_state for the error message, if needed by main.py setup
        # In a real run, main.py's setup_logging would handle this.
        # For this test, we can simulate it.
        # Note: workflow_state is created inside pipeline.run(), so this log path setting might be tricky here.
        # It's better if main.py or the test runner sets up logging that WorkflowState can use.
        # For this specific test, we'll rely on the default logging.

        mock_runtime_key_terms = {
            "ADTC": "Advanced Data Transmission and Control technology.",
            "RUNTIME_TERM": "Term provided at pipeline.run() call."
        }

        final_report = pipeline.run(
            user_topic="Comprehensive Test of Pipeline with Orchestrator",
            data_path=dummy_data_dir_pipe,
            report_title="Test Report on Pipeline Orchestration",
            key_terms_definitions=mock_runtime_key_terms # Test runtime override/provision
        )
        print("\n" + "="*30 + " FINAL REPORT (Mocked - Pipeline with Orchestrator) " + "="*30)
        print(final_report)

        # Check if key terms were correctly set in workflow_state (runtime should override init)
        # This requires pipeline.workflow_state to be accessible after run()
        if pipeline.workflow_state:
            final_ws_key_terms = pipeline.workflow_state.get_key_terms_definitions()
            print(f"\nKey terms in WorkflowState after run: {final_ws_key_terms}")
            assert final_ws_key_terms is not None
            assert "ADTC" in final_ws_key_terms
            assert "RUNTIME_TERM" in final_ws_key_terms
            assert "CLI_TERM" not in final_ws_key_terms # Because runtime arg should take precedence
        else:
            print("\nWorkflowState not accessible post-run for key term check.")

        print("="*80)

        # Verify index files were created
        expected_faiss_path = os.path.join(dummy_vs_dir_pipe, "pipeline_test_index.faiss")
        expected_meta_path = os.path.join(dummy_vs_dir_pipe, "pipeline_test_index.meta.json")
        print(f"Checking for Faiss index: {expected_faiss_path} - Exists: {os.path.exists(expected_faiss_path)}")
        print(f"Checking for Meta json: {expected_meta_path} - Exists: {os.path.exists(expected_meta_path)}")
        assert os.path.exists(expected_faiss_path)
        assert os.path.exists(expected_meta_path)


    except Exception as e:
        logger.error(f"Pipeline example with Orchestrator failed: {e}", exc_info=True)
    finally:
        import shutil
        if os.path.exists(dummy_data_dir_pipe): shutil.rmtree(dummy_data_dir_pipe)
        if os.path.exists(dummy_vs_dir_pipe): shutil.rmtree(dummy_vs_dir_pipe)

    logger.info("ReportGenerationPipeline (Orchestrator & Indexing Logic) Example End")
