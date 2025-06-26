import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings # Import after path adjustment
from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from pipelines.report_generation_pipeline import ReportGenerationPipeline, ReportGenerationPipelineError

# Setup basic logging configuration
log_level_from_settings = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level_from_settings,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # Added filename and lineno
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the RAG Multi-Agent Report Generation System.
    Parses command-line arguments, initializes the pipeline, and runs it.
    """
    parser = argparse.ArgumentParser(
        description="RAG Multi-Agent Report Generation System with Parent-Child Chunking and Hybrid Search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core arguments
    parser.add_argument(
        "--topic", type=str, required=True,
        help="The main topic for the report."
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/",
        help="Path to the directory containing source documents (PDF, DOCX, TXT)."
    )
    parser.add_argument(
        "--output_path", type=str, default=f"output/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        help="File path to save the generated Markdown report."
    )
    parser.add_argument(
        "--report_title", type=str, default=None,
        help="Optional custom title for the report. If not provided, one will be generated based on the topic."
    )

    # Xinference and Model Configuration arguments
    xinference_group = parser.add_argument_group('Xinference and Model Configuration')
    xinference_group.add_argument(
        "--xinference_url", type=str, default=settings.XINFERENCE_API_URL,
        help="URL of the Xinference API server."
    )
    xinference_group.add_argument(
        "--llm_model", type=str, default=settings.DEFAULT_LLM_MODEL_NAME,
        help="Name of the LLM model to use via Xinference."
    )
    xinference_group.add_argument(
        "--embedding_model", type=str, default=settings.DEFAULT_EMBEDDING_MODEL_NAME,
        help="Name of the Embedding model to use via Xinference."
    )
    xinference_group.add_argument(
        "--reranker_model", type=str, default=settings.DEFAULT_RERANKER_MODEL_NAME,
        help="Name of the Reranker model. Set to 'None' or empty to disable."
    )

    # Document Processing (Chunking) arguments
    chunking_group = parser.add_argument_group('Document Processing - Chunking Parameters')
    chunking_group.add_argument(
        "--parent_chunk_size", type=int, default=settings.DEFAULT_PARENT_CHUNK_SIZE,
        help="Target character size for parent chunks."
    )
    chunking_group.add_argument(
        "--parent_chunk_overlap", type=int, default=settings.DEFAULT_PARENT_CHUNK_OVERLAP,
        help="Character overlap for parent chunks."
    )
    chunking_group.add_argument(
        "--child_chunk_size", type=int, default=settings.DEFAULT_CHILD_CHUNK_SIZE,
        help="Target character size for child chunks."
    )
    chunking_group.add_argument(
        "--child_chunk_overlap", type=int, default=settings.DEFAULT_CHILD_CHUNK_OVERLAP,
        help="Character overlap for child chunks."
    )

    # Retrieval arguments
    retrieval_group = parser.add_argument_group('Retrieval Parameters')
    retrieval_group.add_argument(
        "--vector_top_k", type=int, default=settings.DEFAULT_VECTOR_STORE_TOP_K,
        help="Number of top documents to retrieve from vector search."
    )
    retrieval_group.add_argument(
        "--keyword_top_k", type=int, default=settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
        help="Number of top documents to retrieve from keyword search (BM25)."
    )
    retrieval_group.add_argument(
        "--hybrid_search_alpha", type=float, default=settings.DEFAULT_HYBRID_SEARCH_ALPHA,
        help="Blending factor for hybrid search (0.0 for keyword-only, 1.0 for vector-only)."
    )
    retrieval_group.add_argument(
        "--final_top_n_retrieval", type=int, default=None, # Will default in pipeline if None
        help="Final number of documents to use for chapter generation after retrieval and reranking. Defaults to vector_top_k."
    )

    # Pipeline execution arguments
    pipeline_group = parser.add_argument_group('Pipeline Execution Parameters')
    pipeline_group.add_argument(
        "--max_refinement_iterations", type=int, default=settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
        help="Maximum number of refinement iterations for each chapter."
    )

    args = parser.parse_args()

    logger.info("Starting Report Generation System with resolved arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Validate data_path
    if not os.path.isdir(args.data_path):
        logger.error(f"The provided data_path '{args.data_path}' is not a valid directory or does not exist.")
        print(f"Error: Data path '{args.data_path}' is invalid. Please provide a valid directory path.")
        sys.exit(1)
    logger.info(f"Using data_path: {os.path.abspath(args.data_path)}")


    # Initialize services
    try:
        logger.info(f"Initializing LLMService (URL: {args.xinference_url}, Model: {args.llm_model})")
        llm_service = LLMService(api_url=args.xinference_url, model_name=args.llm_model)

        logger.info(f"Initializing EmbeddingService (URL: {args.xinference_url}, Model: {args.embedding_model})")
        embedding_service = EmbeddingService(api_url=args.xinference_url, model_name=args.embedding_model)

        reranker_service = None
        if args.reranker_model and args.reranker_model.lower() != 'none' and args.reranker_model.strip() != '':
            logger.info(f"Initializing RerankerService (URL: {args.xinference_url}, Model: {args.reranker_model})")
            try:
                reranker_service = RerankerService(api_url=args.xinference_url, model_name=args.reranker_model)
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService for model '{args.reranker_model}': {e}. Proceeding without reranker.")
        else:
            logger.info("Reranker model not specified or disabled. Proceeding without reranker.")

    except Exception as e:
        logger.error(f"Failed to initialize core AI services: {e}", exc_info=True)
        print(f"Error: Could not initialize AI services. Ensure Xinference is running and models are available at {args.xinference_url}.")
        sys.exit(1)

    # Initialize the pipeline with all relevant parameters
    try:
        pipeline = ReportGenerationPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            reranker_service=reranker_service,
            parent_chunk_size=args.parent_chunk_size,
            parent_chunk_overlap=args.parent_chunk_overlap,
            child_chunk_size=args.child_chunk_size,
            child_chunk_overlap=args.child_chunk_overlap,
            vector_top_k=args.vector_top_k,
            keyword_top_k=args.keyword_top_k,
            hybrid_alpha=args.hybrid_search_alpha,
            final_top_n_retrieval=args.final_top_n_retrieval,
            max_refinement_iterations=args.max_refinement_iterations
        )
    except Exception as e:
        logger.error(f"Failed to initialize the report generation pipeline: {e}", exc_info=True)
        print(f"Error: Could not initialize the report generation pipeline.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True) # exist_ok=True to prevent error if dir already exists
            logger.info(f"Ensured output directory exists: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            print(f"Error: Could not create output directory {output_dir}.")
            sys.exit(1)

    # Run the pipeline
    try:
        logger.info(f"Running report generation pipeline for topic: '{args.topic}'...")
        final_report_md = pipeline.run(
            user_topic=args.topic,
            data_path=args.data_path, # Pass the directory path
            report_title=args.report_title
        )

        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(final_report_md)
        logger.info(f"Successfully generated report and saved to: {os.path.abspath(args.output_path)}")
        print(f"\nReport generation complete. Output saved to: {os.path.abspath(args.output_path)}")

    except ReportGenerationPipelineError as e:
        logger.error(f"Report generation pipeline failed: {e}", exc_info=True)
        print(f"\nError: Report generation process failed. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        print(f"\nError: An unexpected error occurred. Check logs for details. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage:
    # python main.py --topic "The Future of Renewable Energy" --data_path "./sample_documents/"
    #                --output_path "reports/renewable_energy_report.md"
    #                --parent_chunk_size 1500 --child_chunk_size 300 --hybrid_search_alpha 0.6
    main()
