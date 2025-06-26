import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is in PYTHONPATH to allow imports from core, agents, etc.
# This is often needed when running scripts from a subdirectory or when modules are not installed.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings # Import after path adjustment
from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from pipelines.report_generation_pipeline import ReportGenerationPipeline, ReportGenerationPipelineError

# Setup basic logging configuration
# The level can be overridden by LOG_LEVEL from settings if more verbosity is needed.
log_level_from_settings = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level_from_settings,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the RAG Multi-Agent Report Generation System.
    Parses command-line arguments, initializes the pipeline, and runs it.
    """
    parser = argparse.ArgumentParser(
        description="RAG Multi-Agent Report Generation System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The main topic for the report."
    )
    parser.add_argument(
        "--pdfs",
        type=str,
        required=True,
        help="Comma-separated list of paths to PDF documents to use as context."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"output/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        help="File path to save the generated Markdown report."
    )
    parser.add_argument(
        "--report_title",
        type=str,
        default=None,
        help="Optional custom title for the report. If not provided, one will be generated based on the topic."
    )
    parser.add_argument(
        "--xinference_url",
        type=str,
        default=settings.XINFERENCE_API_URL,
        help="URL of the Xinference API server."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=settings.DEFAULT_LLM_MODEL_NAME,
        help="Name of the LLM model to use via Xinference."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=settings.DEFAULT_EMBEDDING_MODEL_NAME,
        help="Name of the Embedding model to use via Xinference."
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default=settings.DEFAULT_RERANKER_MODEL_NAME,
        help="Name of the Reranker model to use via Xinference. Set to 'None' or empty to disable reranker."
    )
    parser.add_argument(
        "--max_refinement_iterations",
        type=int,
        default=settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
        help="Maximum number of refinement iterations for each chapter."
    )

    args = parser.parse_args()

    logger.info("Starting Report Generation System with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Initialize services
    try:
        logger.info(f"Initializing LLMService with URL: {args.xinference_url}, Model: {args.llm_model}")
        llm_service = LLMService(api_url=args.xinference_url, model_name=args.llm_model)

        logger.info(f"Initializing EmbeddingService with URL: {args.xinference_url}, Model: {args.embedding_model}")
        embedding_service = EmbeddingService(api_url=args.xinference_url, model_name=args.embedding_model)

        reranker_service = None
        if args.reranker_model and args.reranker_model.lower() != 'none':
            logger.info(f"Initializing RerankerService with URL: {args.xinference_url}, Model: {args.reranker_model}")
            try:
                reranker_service = RerankerService(api_url=args.xinference_url, model_name=args.reranker_model)
            except Exception as e: # Catch if reranker model is specified but fails to load
                logger.warning(f"Failed to initialize RerankerService for model '{args.reranker_model}': {e}. Proceeding without reranker.")
                reranker_service = None # Ensure it's None if init fails
        else:
            logger.info("Reranker model not specified or set to 'None'. Proceeding without reranker.")

    except Exception as e: # Catch errors from service initialization (e.g., Xinference connection)
        logger.error(f"Failed to initialize core AI services: {e}", exc_info=True)
        print(f"Error: Could not initialize AI services. Please ensure Xinference is running and models are available at {args.xinference_url}.")
        sys.exit(1)

    # Initialize the pipeline
    try:
        pipeline = ReportGenerationPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            reranker_service=reranker_service,
            max_refinement_iterations=args.max_refinement_iterations
        )
    except Exception as e:
        logger.error(f"Failed to initialize the report generation pipeline: {e}", exc_info=True)
        print(f"Error: Could not initialize the report generation pipeline.")
        sys.exit(1)

    # Parse PDF paths
    pdf_paths_list = [path.strip() for path in args.pdfs.split(',') if path.strip()]
    if not pdf_paths_list:
        logger.error("No PDF files provided. Please specify at least one PDF path.")
        print("Error: No PDF files specified. Use the --pdfs argument.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            print(f"Error: Could not create output directory {output_dir}.")
            sys.exit(1)

    # Run the pipeline
    try:
        logger.info(f"Running report generation pipeline for topic: '{args.topic}'...")
        final_report_md = pipeline.run(
            user_topic=args.topic,
            pdf_paths=pdf_paths_list,
            report_title=args.report_title
        )

        # Save the report
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(final_report_md)
        logger.info(f"Successfully generated report and saved to: {args.output_path}")
        print(f"\nReport generation complete. Output saved to: {args.output_path}")

    except ReportGenerationPipelineError as e:
        logger.error(f"Report generation pipeline failed: {e}", exc_info=True)
        print(f"\nError: Report generation process failed. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        print(f"\nError: An unexpected error occurred. Check logs for details. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage (for when user runs `python main.py --help` or with args):
    # python main.py --topic "AI in Healthcare" --pdfs "path/to/doc1.pdf,path/to/doc2.pdf" --output_path "reports/ai_healthcare_report.md"
    #
    # To test without actual Xinference running, one would typically mock the services
    # or use a development/testing configuration that points to mock endpoints.
    # The current main.py is intended for use with a live Xinference server.
    #
    # If you want to run a quick test with dummy PDFs and mocked services (like in pipeline example),
    # you would need to modify this main() or create a separate test script.
    # For now, this main.py assumes real services.

    # A simple check to guide user if no args are provided (though argparse handles required ones)
    if len(sys.argv) == 1:
        # No arguments provided, print help message (argparse does this, but can be more explicit)
        # logger.info("No command-line arguments provided. Use --help for usage information.")
        # parser.print_help(sys.stderr) # Argparse handles this for missing required args
        pass # Argparse will handle missing required arguments.

    main()
