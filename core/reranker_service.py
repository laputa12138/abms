import logging
from xinference.client import Client as XinferenceClient
from config.settings import XINFERENCE_API_URL, DEFAULT_RERANKER_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                                     Defaults to XINFERENCE_API_URL from settings.
            model_name (str, optional): The name of the Reranker model to use.
                                        Defaults to DEFAULT_RERANKER_MODEL_NAME from settings.
        """
        self.api_url = api_url or XINFERENCE_API_URL
        self.model_name = model_name or DEFAULT_RERANKER_MODEL_NAME

        try:
            self.client = XinferenceClient(self.api_url)
            self.model = self.client.get_model(self.model_name)
            logger.info(f"Successfully connected to Xinference API at {self.api_url} and loaded reranker model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Xinference client or load reranker model {self.model_name} from {self.api_url}: {e}")
            raise RerankerServiceError(f"Xinference client/reranker model initialization failed: {e}")

    def rerank(self, query: str, documents: list[str], top_n: int = None, batch_size: int = 4) -> list[dict]:
        """
        Reranks a list of documents based on a query, processing in batches.

        Args:
            query (str): The query string.
            documents (list[str]): A list of documents (strings) to be reranked.
            top_n (int, optional): The number of top documents to return after processing all batches.
                                   If None, returns all reranked documents.
            batch_size (int): The number of documents to send to the reranker model in each batch.

        Returns:
            list[dict]: A list of reranked results, typically containing 'index',
                        'relevance_score', and potentially 'document'. The 'document'
                        field in the response from Xinference is often None, so we
                        will augment it with the original document content.
                        The 'original_index' refers to the index in the input 'documents' list.

        Raises:
            RerankerServiceError: If the rerank request fails or the response is malformed.
        """
        if not query or not documents:
            logger.warning("rerank called with empty query or documents.")
            return []

        if batch_size <= 0:
            logger.warning(f"rerank called with invalid batch_size {batch_size}. Defaulting to 1.")
            batch_size = 1

        num_documents = len(documents)
        logger.info(f"Requesting rerank for query '{query}' with {num_documents} documents using model {self.model_name}, batch_size={batch_size}.")

        all_batched_results = []

        for i in range(0, num_documents, batch_size):
            batch_documents = documents[i:i + batch_size]
            batch_original_indices = list(range(i, min(i + batch_size, num_documents)))

            logger.debug(f"Processing batch {i//batch_size + 1}: documents {i} to {min(i + batch_size, num_documents) - 1}")

            try:
                # The Xinference rerank method expects 'documents' as the parameter name.
                # It does not have a top_n parameter at the batch level; top_n is applied globally later.
                response = self.model.rerank(
                    documents=batch_documents,
                    query=query
                    # top_n is not applied per batch, but to the final aggregated list
                )

                if response and "results" in response and isinstance(response["results"], list):
                    for result_item in response["results"]:
                        if isinstance(result_item, dict) and "index" in result_item and "relevance_score" in result_item:
                            # 'index' from Xinference is the index within the current batch_documents
                            batch_internal_index = result_item["index"]
                            # Map it back to the original index in the input 'documents' list
                            original_document_index = batch_original_indices[batch_internal_index]

                            all_batched_results.append({
                                "document": documents[original_document_index], # Get original document text
                                "relevance_score": result_item["relevance_score"],
                                "original_index": original_document_index
                            })
                        else:
                            logger.warning(f"Skipping malformed result item in reranker batch response: {result_item}")
                else:
                    logger.error(f"No valid 'results' in reranker model response for batch. Response: {response}")
                    # Depending on desired robustness, could continue or raise error.
                    # For now, let's log and continue, results might be partial.
                    # If a single batch fails, it might be better to raise RerankerServiceError.
                    # Let's assume for now that a partial failure means we can't trust the overall reranking.
                    raise RerankerServiceError(f"No valid 'results' in reranker model response for batch {i//batch_size + 1}.")

            except Exception as e:
                logger.error(f"Error during rerank operation for batch {i//batch_size + 1}: {e}")
                if "unexpected keyword argument 'corpus'" in str(e) or "unexpected keyword argument 'documents'" in str(e):
                     logger.warning("The error might indicate a mismatch in parameter names for the rerank method (e.g., 'corpus' vs 'documents'). Check Xinference SDK version.")
                raise RerankerServiceError(f"Rerank operation failed for batch: {e}")

        # Sort all collected results by relevance_score in descending order
        all_batched_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_n to the sorted, aggregated list
        final_results = all_batched_results
        if top_n is not None and top_n > 0:
            final_results = all_batched_results[:top_n]

        logger.info(f"Successfully reranked {num_documents} documents in batches. Returned {len(final_results)} final results.")
        return final_results

if __name__ == '__main__':
    # This is an example of how to use the RerankerService.
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
