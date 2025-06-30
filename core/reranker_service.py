import logging
from xinference.client import Client as XinferenceClient
from config import settings # Import the settings module

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in main
logger = logging.getLogger(__name__)

class RerankerServiceError(Exception):
    """Custom exception for RerankerService errors."""
    pass

class RerankerService:
    """
    A service class to interact with a Reranker model
    deployed via Xinference.
    """
    def __init__(self, api_url: str = None, model_name: str = None):
        """
        Initializes the RerankerService.

        Args:
            api_url (str, optional): The URL of the Xinference API.
                                     Defaults to settings.XINFERENCE_API_URL.
            model_name (str, optional): The name of the Reranker model to use.
                                        Defaults to settings.DEFAULT_RERANKER_MODEL_NAME.
        """
        self.api_url = api_url or settings.XINFERENCE_API_URL
        self.model_name = model_name or settings.DEFAULT_RERANKER_MODEL_NAME

        try:
            self.client = XinferenceClient(self.api_url)
            self.model = self.client.get_model(self.model_name)
            logger.info(f"Successfully connected to Xinference API at {self.api_url} and loaded reranker model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Xinference client or load reranker model {self.model_name} from {self.api_url}: {e}")
            raise RerankerServiceError(f"Xinference client/reranker model initialization failed: {e}")

    def rerank(self,
                 query: str,
                 documents: list[str],
                 top_n: int = None,
                 batch_size: int = None, # Now defaults to setting
                 max_text_length: int = None # Now defaults to setting
                ) -> list[dict]:
        """
        Reranks a list of documents based on a query, processing in batches and truncating long texts.

        Args:
            query (str): The query string.
            documents (list[str]): A list of documents (strings) to be reranked.
            top_n (int, optional): The number of top documents to return after processing all batches.
                                   If None, returns all reranked documents.
            batch_size (int, optional): The number of documents to send to the reranker model in each batch.
                                        Defaults to settings.DEFAULT_RERANKER_BATCH_SIZE.
            max_text_length (int, optional): Maximum character length for each document sent to the reranker.
                                             Longer texts will be truncated.
                                             Defaults to settings.DEFAULT_RERANKER_MAX_TEXT_LENGTH.

        Returns:
            list[dict]: A list of reranked results. Each dict contains:
                        'document': str (original document, possibly truncated if sent to model truncated,
                                         but here we return the original full document for context)
                        'relevance_score': float
                        'original_index': int (index in the input 'documents' list)

        Raises:
            RerankerServiceError: If the rerank request fails or the response is malformed.
        """
        if not query or not documents:
            logger.warning("rerank called with empty query or documents.")
            return []

        # Use defaults from settings if parameters are None
        effective_batch_size = batch_size if batch_size is not None else settings.DEFAULT_RERANKER_BATCH_SIZE
        effective_max_length = max_text_length if max_text_length is not None else settings.DEFAULT_RERANKER_MAX_TEXT_LENGTH


        if effective_batch_size <= 0:
            logger.warning(f"rerank called with invalid batch_size {effective_batch_size}. Defaulting to 1.")
            effective_batch_size = 1

        num_documents = len(documents)
        logger.info(f"Requesting rerank for query '{query[:100]}...' with {num_documents} documents using model {self.model_name}. "
                    f"Batch_size={effective_batch_size}, Max_text_length={effective_max_length}.")

        all_batched_results = []

        for i in range(0, num_documents, effective_batch_size):
            batch_documents_original = documents[i : i + effective_batch_size]
            batch_original_indices = list(range(i, min(i + effective_batch_size, num_documents)))

            # Truncate documents in the batch if necessary
            batch_documents_for_model = []
            for doc_idx, doc_text in enumerate(batch_documents_original):
                if effective_max_length > 0 and len(doc_text) > effective_max_length:
                    truncated_text = doc_text[:effective_max_length]
                    batch_documents_for_model.append(truncated_text)
                    if i + doc_idx < 5 : # Log truncation only for first few docs to avoid spam
                         logger.debug(f"Document {i + doc_idx} (len {len(doc_text)}) truncated to {effective_max_length} chars for reranker.")
                else:
                    batch_documents_for_model.append(doc_text)


            logger.debug(f"Processing batch {i//effective_batch_size + 1}: "
                         f"documents original indices {batch_original_indices[0]} to {batch_original_indices[-1]}")

            try:
                response = self.model.rerank(
                    documents=batch_documents_for_model, # Send potentially truncated documents
                    query=query
                )

                if response and "results" in response and isinstance(response["results"], list):
                    for result_item in response["results"]:
                        if isinstance(result_item, dict) and "index" in result_item and "relevance_score" in result_item:
                            batch_internal_index = result_item["index"]
                            original_document_index = batch_original_indices[batch_internal_index]

                            all_batched_results.append({
                                "document": documents[original_document_index], # Return original full document
                                "relevance_score": result_item["relevance_score"],
                                "original_index": original_document_index
                            })
                        else:
                            logger.warning(f"Skipping malformed result item in reranker batch response: {result_item}")
                else:
                    logger.error(f"No valid 'results' in reranker model response for batch. Response: {response}")
                    raise RerankerServiceError(f"No valid 'results' in reranker model response for batch {i//effective_batch_size + 1}.")

            except Exception as e:
                logger.error(f"Error during rerank operation for batch {i//effective_batch_size + 1}: {e}")
                if "unexpected keyword argument 'corpus'" in str(e) or "unexpected keyword argument 'documents'" in str(e):
                     logger.warning("The error might indicate a mismatch in parameter names for the rerank method (e.g., 'corpus' vs 'documents'). Check Xinference SDK version.")
                raise RerankerServiceError(f"Rerank operation failed for batch: {e}")

        all_batched_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        final_results = all_batched_results
        if top_n is not None and top_n > 0:
            final_results = all_batched_results[:top_n]

        logger.info(f"Successfully reranked {num_documents} documents in batches. Returned {len(final_results)} final results.")
        return final_results

if __name__ == '__main__':
    # This is an example of how to use the RerankerService.
    # It requires a running Xinference server with the configured reranker model.
    # It requires a running Xinference server with the 'Qwen3-Reranker-0.6B' model.
    # As per instructions, this will not be run during the automated process.
    print("RerankerService Example (requires running Xinference server)")
    print("This part will not be executed by the agent but is for local testing.")

    try:
        reranker_service = RerankerService() # Uses defaults from settings.py

        # Example: Rerank documents
        # query_example = "A man is eating pasta."
        # corpus_example = [
        #     "A man is eating food.",
        #     "A man is eating a piece of bread.",
        #     "The girl is carrying a baby.",
        #     "A man is riding a horse.",
        #     "A woman is playing violin."
        # ]
        # print(f"\nReranking documents for query: '{query_example}'")
        # print(f"Original corpus: {corpus_example}")

        # reranked_docs = reranker_service.rerank(query_example, corpus_example, top_n=3)

        # if reranked_docs:
        #     print("\nReranked documents (top 3):")
        #     for i, doc_info in enumerate(reranked_docs):
        #         print(f"{i+1}. Document: '{doc_info['document']}' (Score: {doc_info['relevance_score']:.4f}, Original Index: {doc_info['original_index']})")
        # else:
        #     print("No documents were reranked or an error occurred.")

        print("\nRerankerService example finished. If no output, ensure Xinference server is running and configured.")
        print("Note: Actual API calls are commented out to prevent errors if server is not available.")

    except RerankerServiceError as e:
        print(f"Error initializing or using RerankerService: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
