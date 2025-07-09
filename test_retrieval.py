import argparse
import logging
import json
import os # Added os import

from rank_bm25 import BM25Okapi

from core.vector_store import VectorStore, VectorStoreError
from core.retrieval_service import RetrievalService, RetrievalServiceError
from core.reranker_service import RerankerService, RerankerServiceError
from core.embedding_service import EmbeddingService, EmbeddingServiceError
from core.document_processor import DocumentProcessor, DocumentProcessorError
from config import settings

# Configure basic logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test retrieval functionality.")
    parser.add_argument("--query", type=str, required=True, help="The query to search for.")
    parser.add_argument("--index_name", type=str, default="abms-V2", help="Name of the FAISS index.")
    parser.add_argument("--vector_store_path", type=str, default="./my_vector_indexes/", help="Path to the vector store directory.")
    parser.add_argument("--data_path", type=str, default="./data/", help="Path to the raw data directory (for BM25 if needed).")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")

    # Retrieval parameters from settings.py, allowing override
    parser.add_argument("--vector_top_k", type=int, default=settings.DEFAULT_VECTOR_STORE_TOP_K, help="Top K results from vector search.")
    parser.add_argument("--keyword_top_k", type=int, default=settings.DEFAULT_KEYWORD_SEARCH_TOP_K, help="Top K results from keyword search.")
    parser.add_argument("--hybrid_alpha", type=float, default=settings.DEFAULT_HYBRID_SEARCH_ALPHA, help="Alpha for hybrid search (0=keyword, 1=vector).")
    parser.add_argument("--final_top_n", type=int, default=settings.DEFAULT_RETRIEVAL_FINAL_TOP_N, help="Final top N results to return.")
    parser.add_argument("--min_score_threshold", type=float, default=settings.DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD, help="Minimum score threshold for results.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=args.log_level.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    logger.info(f"Starting retrieval test with query: '{args.query}'")
    logger.info(f"Parameters: {args}")

    try:
        # 1. Initialize EmbeddingService
        logger.info("Initializing EmbeddingService...")
        embedding_service = EmbeddingService(
            model_name=settings.DEFAULT_EMBEDDING_MODEL_NAME,
            api_url=settings.XINFERENCE_API_URL
        )
        logger.info(f"EmbeddingService initialized with model: {settings.DEFAULT_EMBEDDING_MODEL_NAME}")

        # 2. Initialize VectorStore and load index
        logger.info("Initializing VectorStore...")
        vector_store = VectorStore(
            embedding_service=embedding_service
            # dimension can be inferred by the VectorStore from embeddings
        )

        # Construct paths for loading
        index_file_path = os.path.join(args.vector_store_path, f"{args.index_name}.faiss")
        metadata_file_path = os.path.join(args.vector_store_path, f"{args.index_name}_meta.json")

        try:
            logger.info(f"Attempting to load VectorStore from: Index='{index_file_path}', Metadata='{metadata_file_path}'")
            vector_store.load_store(index_path=index_file_path, metadata_path=metadata_file_path)
            logger.info(f"VectorStore loaded successfully. Index name (from args): '{args.index_name}'. "
                        f"FAISS index dimension: {vector_store.dimension}, "
                        f"Found {vector_store.count_child_chunks} child chunks in store.")
            if not vector_store.document_store: # Should be populated by load_store()
                 logger.warning("VectorStore's document_store is empty after load_store(). BM25 might not work as expected if it relies on this.")

        except VectorStoreError as e:
            logger.error(f"Failed to load VectorStore from '{index_file_path}' and '{metadata_file_path}': {e}. Attempting to build BM25 from data_path as a fallback for BM25 data.")
            # If vector store loading fails, we might still try to proceed if BM25 can be built
            # from raw data, but retrieval service will likely fail if vector_store is not usable.
            # For this script, we'll assume vector_store must load for vector search part.
            # If only BM25 is needed, the script logic would be different.
            raise

        # 3. Initialize RerankerService (optional)
        reranker_service = None
        if settings.DEFAULT_RERANKER_MODEL_NAME:
            logger.info("Initializing RerankerService...")
            try:
                reranker_service = RerankerService(
                    model_name=settings.DEFAULT_RERANKER_MODEL_NAME,
                    api_url=settings.XINFERENCE_API_URL
                )
                logger.info(f"RerankerService initialized with model: {settings.DEFAULT_RERANKER_MODEL_NAME}")
            except RerankerServiceError as e:
                logger.warning(f"Could not initialize RerankerService: {e}. Proceeding without reranking.")
            except Exception as e:
                logger.warning(f"Unexpected error initializing RerankerService: {e}. Proceeding without reranking.")
        else:
            logger.info("No Reranker model specified in settings. Proceeding without reranking.")


        # 4. Prepare BM25 Index and mapping data
        logger.info("Preparing BM25 Index...")
        all_child_chunks_for_bm25_mapping = [] # List of dicts: {'child_id': str, 'child_text': str, ...other_meta}
        bm25_corpus_texts = []

        # Strategy A: Try to use vector_store.document_store (which should be List[Dict[str,Any]])
        # The document_store in VectorStore is expected to hold metadata for each child chunk.
        if vector_store.document_store and isinstance(vector_store.document_store, list):
            logger.info(f"Attempting to use vector_store.document_store for BM25 data ({len(vector_store.document_store)} items).")
            # Ensure each item in document_store has 'child_id' and 'child_text'
            valid_items_count = 0
            for item_meta in vector_store.document_store:
                if isinstance(item_meta, dict) and 'child_id' in item_meta and 'child_text' in item_meta:
                    all_child_chunks_for_bm25_mapping.append(item_meta) # Store the whole dict
                    bm25_corpus_texts.append(item_meta['child_text'])
                    valid_items_count +=1
                else:
                    logger.warning(f"Item in document_store is missing 'child_id' or 'child_text': {str(item_meta)[:100]}...")
            if valid_items_count > 0:
                 logger.info(f"Successfully prepared {valid_items_count} child chunks from vector_store.document_store for BM25.")
            else:
                logger.warning("Could not extract valid child_id and child_text from vector_store.document_store. BM25 index might be empty or based on fallback.")
                all_child_chunks_for_bm25_mapping = [] # Reset if no valid items
                bm25_corpus_texts = []


        # Strategy B (Fallback if vector_store.document_store was not sufficient or empty)
        if not all_child_chunks_for_bm25_mapping:
            logger.warning("BM25 mapping data from vector_store.document_store is empty or invalid. "
                           "Falling back to DocumentProcessor to load data from --data_path for BM25.")
            try:
                doc_processor = DocumentProcessor(
                    # Embedding service not strictly needed for just getting text for BM25
                    # but some internal methods might expect it if we were to use its full processing.
                    embedding_service=None, # Or pass the existing one if full processing was intended
                    parent_chunk_size=settings.DEFAULT_PARENT_CHUNK_SIZE,
                    parent_chunk_overlap=settings.DEFAULT_PARENT_CHUNK_OVERLAP,
                    child_chunk_size=settings.DEFAULT_CHILD_CHUNK_SIZE,
                    child_chunk_overlap=settings.DEFAULT_CHILD_CHUNK_OVERLAP
                )
                # This loads and chunks all documents.
                # We only need the child chunks for BM25.
                # The `parent_child_data` is a list of parent dicts, each containing a list of child dicts.
                parent_child_data, _ = doc_processor.load_and_process_documents(
                    data_path=args.data_path,
                    supported_extensions=settings.SUPPORTED_DOC_EXTENSIONS
                )
                if not parent_child_data:
                    logger.warning(f"DocumentProcessor returned no data from {args.data_path}. BM25 index will be empty.")
                else:
                    for parent_doc in parent_child_data:
                        for child_chunk in parent_doc.get('children', []):
                            # Ensure child_chunk has 'child_id' and 'child_text'
                            if 'child_id' in child_chunk and 'child_text' in child_chunk:
                                # We need to ensure the structure matches what RetrievalService expects for mapping
                                # This typically includes parent context if possible, but RetrievalService
                                # primarily uses child_id from BM25 to look up full context from its internal map
                                # (which is built from vector_store.document_store).
                                # So, for BM25 mapping, 'child_id' and 'child_text' are crucial.
                                # Other fields like 'parent_id', 'parent_text', 'source_document_name'
                                # are also good to have if available from DocumentProcessor's output.
                                bm25_corpus_texts.append(child_chunk['child_text'])
                                all_child_chunks_for_bm25_mapping.append({
                                    'child_id': child_chunk['child_id'],
                                    'child_text': child_chunk['child_text'],
                                    'parent_id': parent_doc.get('parent_id'),
                                    'parent_text': parent_doc.get('parent_text'),
                                    'source_document_name': parent_doc.get('source_document_name', parent_doc.get('source_file_name'))
                                })
                    logger.info(f"Prepared {len(all_child_chunks_for_bm25_mapping)} child chunks for BM25 from DocumentProcessor.")

            except DocumentProcessorError as e:
                logger.error(f"DocumentProcessor failed: {e}. BM25 index will likely be empty.")
            except Exception as e:
                logger.error(f"Unexpected error during DocumentProcessor use for BM25: {e}", exc_info=True)


        bm25_index = None
        if bm25_corpus_texts:
            # Tokenize for BM25 (simple whitespace split, consider a proper tokenizer for Chinese)
            # The RetrievalService itself has a _tokenize_query method, we can emulate that or use a simpler one here for corpus.
            # For consistency, let's assume simple split for now.
            tokenized_corpus_for_bm25 = [text.lower().split() for text in bm25_corpus_texts]
            if tokenized_corpus_for_bm25:
                bm25_index = BM25Okapi(tokenized_corpus_for_bm25)
                logger.info(f"BM25Okapi index created with {len(tokenized_corpus_for_bm25)} documents.")
            else:
                logger.warning("Tokenized corpus for BM25 is empty. BM25 index will not be used.")
        else:
            logger.warning("No corpus texts available for BM25. BM25 index will not be used.")


        # 5. Initialize RetrievalService
        logger.info("Initializing RetrievalService...")
        if not vector_store or not vector_store.is_loaded: # Check if vector_store is usable
             logger.error("VectorStore is not loaded or unavailable. Cannot initialize RetrievalService.")
             return

        retrieval_service = RetrievalService(
            vector_store=vector_store,
            bm25_index=bm25_index,
            all_child_chunks_for_bm25_mapping=all_child_chunks_for_bm25_mapping,
            reranker_service=reranker_service
        )
        logger.info("RetrievalService initialized.")

        # 6. Perform retrieval
        logger.info(f"Performing retrieval for query: '{args.query}'")
        # Override settings with args for retrieval parameters
        settings.DEFAULT_VECTOR_STORE_TOP_K = args.vector_top_k
        settings.DEFAULT_KEYWORD_SEARCH_TOP_K = args.keyword_top_k
        settings.DEFAULT_HYBRID_SEARCH_ALPHA = args.hybrid_alpha
        settings.DEFAULT_RETRIEVAL_FINAL_TOP_N = args.final_top_n
        settings.DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD = args.min_score_threshold

        retrieved_results = retrieval_service.retrieve(
            query_texts=[args.query], # retrieve expects a list of queries
            vector_top_k=args.vector_top_k,
            keyword_top_k=args.keyword_top_k,
            hybrid_alpha=args.hybrid_alpha,
            final_top_n=args.final_top_n
            # min_score_threshold is applied inside retrieve based on settings.DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD
        )

        # 7. Print results
        logger.info(f"Retrieval complete. Found {len(retrieved_results)} results.")
        if retrieved_results:
            print("\n--- Retrieved Results ---")
            for i, result in enumerate(retrieved_results):
                print(f"\nResult {i+1}:")
                # Pretty print the dictionary
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("No results found for the query.")

    except EmbeddingServiceError as e:
        logger.error(f"EmbeddingService error: {e}", exc_info=True)
    except VectorStoreError as e:
        logger.error(f"VectorStore error: {e}", exc_info=True)
    except RetrievalServiceError as e:
        logger.error(f"RetrievalService error: {e}", exc_info=True)
    except RerankerServiceError as e:
        logger.error(f"RerankerService error: {e}", exc_info=True)
    except DocumentProcessorError as e:
        logger.error(f"DocumentProcessor error during BM25 preparation: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
