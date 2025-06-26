import logging
from typing import List, Dict, Optional, Tuple

from agents.base_agent import BaseAgent
from core.vector_store import VectorStore, VectorStoreError
from core.reranker_service import RerankerService, RerankerServiceError
from config.settings import DEFAULT_VECTOR_STORE_TOP_K

logger = logging.getLogger(__name__)

class ContentRetrieverAgentError(Exception):
    """Custom exception for ContentRetrieverAgent errors."""
    pass

class ContentRetrieverAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant content chunks from the VectorStore
    based on a query (e.g., chapter title, keywords) and then reranking them.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 reranker_service: Optional[RerankerService] = None, # Reranker is optional
                 default_top_k_retrieval: int = DEFAULT_VECTOR_STORE_TOP_K,
                 default_top_n_rerank: Optional[int] = None): # How many to keep after reranking
        """
        Initializes the ContentRetrieverAgent.

        Args:
            vector_store (VectorStore): An instance of the VectorStore.
            reranker_service (Optional[RerankerService]): An instance of RerankerService.
            default_top_k_retrieval (int): Default number of documents to retrieve from vector store.
            default_top_n_rerank (Optional[int]): Default number of documents to return after reranking.
                                                  If None, returns all reranked documents.
        """
        super().__init__(agent_name="ContentRetrieverAgent",
                         vector_store=vector_store,
                         reranker_service=reranker_service)

        if not self.vector_store:
            raise ContentRetrieverAgentError("VectorStore is required for ContentRetrieverAgent.")

        self.default_top_k_retrieval = default_top_k_retrieval
        self.default_top_n_rerank = default_top_n_rerank
        logger.info(f"ContentRetrieverAgent initialized. Reranker service is {'ENABLED' if self.reranker_service else 'DISABLED'}.")


    def _combine_query_elements(self, chapter_title: str, keywords: List[str]) -> str:
        """Combines chapter title and keywords into a single query string."""
        # Simple combination, can be made more sophisticated
        keyword_str = " ".join(keywords)
        return f"{chapter_title} {keyword_str}".strip()

    def run(self,
            query: str,
            top_k_retrieval: Optional[int] = None,
            top_n_rerank: Optional[int] = None,
            metadata_filter: Optional[Dict] = None # Placeholder for future metadata filtering
           ) -> List[Dict[str, any]]: # Returns list of dicts with 'document' and 'score'
        """
        Retrieves and reranks content.

        Args:
            query (str): The query string (e.g., combined chapter title and keywords).
            top_k_retrieval (Optional[int]): Number of documents to retrieve from vector store.
                                            Defaults to agent's default.
            top_n_rerank (Optional[int]): Number of documents to return after reranking.
                                          Defaults to agent's default. If None, all reranked docs.
            metadata_filter (Optional[Dict]): Placeholder for future metadata-based filtering.

        Returns:
            List[Dict[str, any]]: A list of dictionaries, where each dictionary
                                  contains 'document' (str) and 'score' (float).
                                  Score is relevance_score from reranker if used,
                                  otherwise it's the distance from vector search.

        Raises:
            ContentRetrieverAgentError: If retrieval or reranking fails.
        """
        self._log_input(query=query, top_k_retrieval=top_k_retrieval, top_n_rerank=top_n_rerank)

        current_top_k = top_k_retrieval if top_k_retrieval is not None else self.default_top_k_retrieval
        current_top_n_rerank = top_n_rerank if top_n_rerank is not None else self.default_top_n_rerank

        try:
            # 1. Retrieve from VectorStore
            logger.info(f"Retrieving top {current_top_k} documents for query: '{query[:100]}...'")
            # VectorStore search returns List[Tuple[str, float]] -> (document, distance_score)
            retrieved_docs_with_scores = self.vector_store.search(query_text=query, k=current_top_k)

            if not retrieved_docs_with_scores:
                logger.warning(f"No documents found in VectorStore for query: '{query[:100]}...'")
                return []

            logger.info(f"Retrieved {len(retrieved_docs_with_scores)} documents from VectorStore.")

            # Prepare for reranking or direct output
            documents_to_process = [doc for doc, score in retrieved_docs_with_scores]

            # If no reranker, or no documents, return based on vector search results
            if not self.reranker_service or not documents_to_process:
                logger.info("Skipping reranking (no reranker service or no documents to rerank).")
                # Convert to the common output format
                output_results = [{"document": doc, "score": score, "source": "vector_search"}
                                  for doc, score in retrieved_docs_with_scores]
                # Sort by vector search score (distance, lower is better) if not reranking
                output_results.sort(key=lambda x: x["score"])
                final_results = output_results[:current_top_n_rerank] if current_top_n_rerank is not None else output_results
                self._log_output(final_results)
                return final_results

            # 2. Rerank if RerankerService is available
            logger.info(f"Reranking {len(documents_to_process)} documents with RerankerService. Query: '{query[:100]}...'")
            try:
                # RerankerService.rerank returns List[Dict] with 'document', 'relevance_score', 'original_index'
                reranked_results = self.reranker_service.rerank(
                    query=query,
                    documents=documents_to_process, # Pass only the document texts
                    top_n=current_top_n_rerank # Reranker can also do top_n
                )

                # Ensure the output format is consistent: list of dicts with 'document' and 'score'
                # The reranker already returns in a suitable format, just ensure 'score' key exists.
                # The reranker service already sorts by relevance_score descending.
                final_results = [{"document": res["document"], "score": res["relevance_score"], "source": "reranker"}
                                 for res in reranked_results]

                logger.info(f"Reranked documents. Returning {len(final_results)} documents.")
                self._log_output(final_results)
                return final_results

            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Falling back to VectorStore results.")
                # Fallback: use the initially retrieved documents if reranking fails
                output_results = [{"document": doc, "score": score, "source": "vector_search_fallback"}
                                  for doc, score in retrieved_docs_with_scores]
                output_results.sort(key=lambda x: x["score"]) # Sort by distance
                final_results = output_results[:current_top_n_rerank] if current_top_n_rerank is not None else output_results
                self._log_output(final_results)
                return final_results

        except VectorStoreError as e:
            logger.error(f"VectorStore error during content retrieval: {e}")
            raise ContentRetrieverAgentError(f"VectorStore failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in ContentRetrieverAgent: {e}")
            raise ContentRetrieverAgentError(f"Unexpected error in content retrieval: {e}")

if __name__ == '__main__':
    print("ContentRetrieverAgent Example")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Services ---
    class MockEmbeddingService:
        def create_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * 10 for _ in texts] # Dummy 10-dim embeddings

    class MockVectorStore(BaseAgent): # Inherit for structure, not functionality
        def __init__(self):
            super().__init__(agent_name="MockVectorStore")
            self.documents = {} # Store text and dummy score
            self.doc_list = []

        def add_documents(self, texts: List[str]):
            for i, text in enumerate(texts):
                # Simulate adding with a dummy score (distance)
                self.documents[text] = float(i + 1) * 0.1
                self.doc_list.append(text)

        def search(self, query_text: str, k: int) -> List[Tuple[str, float]]:
            logger.info(f"MockVectorStore searching for: '{query_text}', k={k}")
            # Simple mock: return first k docs, or fewer if not enough
            results = []
            # Sort by values (scores) to make it somewhat deterministic if needed
            # sorted_docs = sorted(self.documents.items(), key=lambda item: item[1])

            # Just return from list for simplicity
            for i in range(min(k, len(self.doc_list))):
                doc = self.doc_list[i]
                score = self.documents.get(doc, 0.0) # Get its dummy score
                results.append((doc, score)) # (document, score)
            return results

        @property
        def count(self):
            return len(self.doc_list)

    class MockRerankerService(BaseAgent):
        def __init__(self):
            super().__init__(agent_name="MockRerankerService")

        def rerank(self, query: str, documents: List[str], top_n: Optional[int]) -> List[Dict[str, any]]:
            logger.info(f"MockRerankerService reranking {len(documents)} docs for query: '{query}', top_n={top_n}")
            # Simple mock: reverse the order and assign new scores
            reranked = []
            for i, doc_text in enumerate(reversed(documents)):
                reranked.append({
                    "document": doc_text,
                    "relevance_score": 0.9 - (i * 0.1), # Higher score for earlier in reversed list
                    "original_index": documents.index(doc_text)
                })
            return reranked[:top_n] if top_n is not None else reranked

    # --- Example Usage ---
    try:
        mock_vs = MockVectorStore()
        sample_contents = [
            "Content about ABMS system design and architecture.",
            "Details on JADC2 integration with ABMS.",
            "Challenges in implementing advanced battle management.",
            "The role of AI in future military command systems.",
            "A unrelated document about cooking pasta."
        ]
        mock_vs.add_documents(sample_contents)

        mock_reranker = MockRerankerService()

        # Case 1: With Reranker
        print("\n--- Case 1: Retrieval with Reranker ---")
        retriever_agent_with_reranker = ContentRetrieverAgent(
            vector_store=mock_vs,
            reranker_service=mock_reranker,
            default_top_k_retrieval=5, # Retrieve 5
            default_top_n_rerank=3    # Keep top 3 after rerank
        )
        query1 = "ABMS and JADC2"
        retrieved_content1 = retriever_agent_with_reranker.run(query=query1)
        print(f"Retrieved and Reranked Content for '{query1}':")
        for item in retrieved_content1:
            print(f"  - Doc: \"{item['document'][:50]}...\", Score: {item['score']:.4f} (Source: {item.get('source')})")

        # Case 2: Without Reranker
        print("\n--- Case 2: Retrieval without Reranker ---")
        retriever_agent_no_reranker = ContentRetrieverAgent(
            vector_store=mock_vs,
            reranker_service=None, # No reranker
            default_top_k_retrieval=3 # Retrieve 3
        )
        query2 = "AI in command systems"
        retrieved_content2 = retriever_agent_no_reranker.run(query=query2)
        print(f"Retrieved Content for '{query2}' (No Reranker):")
        for item in retrieved_content2:
            print(f"  - Doc: \"{item['document'][:50]}...\", Score: {item['score']:.4f} (Source: {item.get('source')})") # Score is distance

        # Case 3: No documents found
        print("\n--- Case 3: No documents found ---")
        empty_vs = MockVectorStore() # A store with no documents
        retriever_agent_empty = ContentRetrieverAgent(vector_store=empty_vs, reranker_service=mock_reranker)
        query3 = "Non existent topic"
        retrieved_content3 = retriever_agent_empty.run(query=query3)
        print(f"Retrieved Content for '{query3}': {retrieved_content3}")


    except ContentRetrieverAgentError as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nContentRetrieverAgent example finished.")
