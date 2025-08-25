import logging
from typing import List, Dict, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi

from core.vector_store import VectorStore, VectorStoreError
from core.reranker_service import RerankerService, RerankerServiceError
from config import settings # Import settings for reranker defaults

logger = logging.getLogger(__name__)

class RetrievalServiceError(Exception):
    """Custom exception for RetrievalService errors."""
    pass

class RetrievalService:
    """
    Service responsible for performing hybrid retrieval (vector + keyword)
    from a knowledge base, optionally followed by reranking.
    It works with parent-child chunked documents stored in a VectorStore.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 bm25_index: Optional[BM25Okapi],
                 all_child_chunks_for_bm25_mapping: List[Dict[str, Any]], # For mapping BM25 results
                 reranker_service: Optional[RerankerService] = None):
        """
        Initializes the RetrievalService.

        Args:
            vector_store (VectorStore): Instance of VectorStore.
            bm25_index (Optional[BM25Okapi]): Pre-computed BM25 index over child chunks.
            all_child_chunks_for_bm25_mapping (List[Dict[str, Any]]): List of dictionaries,
                where each dict contains at least 'child_id' and 'child_text' for every
                child chunk that was used to build the bm25_index. This is crucial for
                mapping BM25's results (which are often indices or raw texts) back to
                the structured child chunk data (including parent context).
            reranker_service (Optional[RerankerService]): Instance of RerankerService.
        """
        if not vector_store:
            raise RetrievalServiceError("VectorStore is required for RetrievalService.")

        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.all_child_chunks_for_bm25_mapping = all_child_chunks_for_bm25_mapping
        self.reranker_service = reranker_service

        # Build a quick lookup map from child_id to its full context (parent, etc.)
        # This map is essential for re-associating BM25 results if they only return child_id or index.
        self.child_id_to_full_context_map: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.vector_store, 'document_store'):
            for item_meta in self.vector_store.document_store:
                self.child_id_to_full_context_map[item_meta['child_id']] = item_meta
        else:
            logger.warning("VectorStore does not have 'document_store' attribute. "
                           "BM25 result mapping might be incomplete if it relies on child_ids not found elsewhere.")


        log_msg = (f"RetrievalService initialized. "
                   f"VectorStore has {self.vector_store.count_child_chunks} child chunks. "
                   f"BM25 index is {'PRESENT' if self.bm25_index else 'ABSENT'}. "
                   f"Reranker service is {'ENABLED' if self.reranker_service else 'DISABLED'}.")
        logger.info(log_msg)

    def _tokenize_query(self, query: str) -> List[str]:
        """Simple whitespace tokenizer. For Chinese, a proper tokenizer is recommended."""
        # TODO: Integrate a proper Chinese tokenizer like jieba if docs are primarily Chinese.
        # e.g., import jieba; return list(jieba.cut_for_search(query))
        return query.lower().split()

    def _normalize_scores(self, scores: List[float], reverse: bool = False) -> List[float]:
        """Min-max normalize scores to [0, 1]. Reverse if higher score is worse (e.g., distance)."""
        if not scores: return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: # Avoid division by zero if all scores are same
            return [0.5] * len(scores) # Default to mid-point

        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        return [1.0 - s for s in normalized] if reverse else normalized

    def retrieve(self,
                 query_texts: List[str],
                 vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 # hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA, # Removed as per new logic
                 final_top_n: int = settings.DEFAULT_RETRIEVAL_FINAL_TOP_N
                ) -> List[Dict[str, Any]]:
        """
        Performs retrieval by combining results from vector search and keyword search,
        then reranks them using a RerankerService, applies a score threshold,
        and returns the top N results.

        Args:
            query_texts (List[str]): A list of user's queries. The first query is used for reranking.
            vector_top_k (int): Number of results to fetch from vector search for each query.
            keyword_top_k (int): Number of results to fetch from keyword search for each query.
            final_top_n (int): Number of final results to return after reranking and thresholding.
                               This is also passed to the reranker as its 'top_n' parameter.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries, structured for consumption.
                                  Each dictionary includes a 'score' from the reranker.
        """
        if not query_texts:
            logger.warning("RetrievalService.retrieve called with empty query_texts list. Returning empty list.")
            return []

        logger.info(f"RetrievalService called with {len(query_texts)} queries. First query for reranking: '{query_texts[0][:100]}...' "
                    f"v_k={vector_top_k}, k_k={keyword_top_k}, final_n={final_top_n}")

        # --- 1. Gather results from Vector Search and Keyword Search for all queries ---
        all_retrieved_child_chunks: Dict[str, Dict[str, Any]] = {} # child_id -> data

        for query_idx, query_text in enumerate(query_texts):
            logger.debug(f"Processing query {query_idx + 1}/{len(query_texts)}: '{query_text[:100]}...'")

            # Vector Search
            try:
                raw_vector_hits = self.vector_store.search(query_text=query_text, k=vector_top_k)
                for hit in raw_vector_hits:
                    child_id = hit['child_id']
                    if child_id not in all_retrieved_child_chunks:
                        all_retrieved_child_chunks[child_id] = {
                            **hit, # Includes child_id, parent_id, parent_text, child_text, source_document_name
                            'retrieval_source_types': {'vector'} # Store how it was found
                        }
                    else: # Already found, just add the source type
                        all_retrieved_child_chunks[child_id]['retrieval_source_types'].add('vector')
                logger.debug(f"Query '{query_text[:30]}...': Vector search added/updated {len(raw_vector_hits)} potential chunks.")
            except VectorStoreError as e:
                logger.error(f"VectorStore search failed for query '{query_text[:30]}...': {e}")
            except Exception as e:
                logger.error(f"Unexpected error during vector search for query '{query_text[:30]}...': {e}", exc_info=True)

            # Keyword Search (BM25)
            if self.bm25_index and self.all_child_chunks_for_bm25_mapping:
                try:
                    tokenized_query = self._tokenize_query(query_text)
                    bm25_doc_scores = self.bm25_index.get_scores(tokenized_query)
                    num_bm25_candidates = min(keyword_top_k, len(self.all_child_chunks_for_bm25_mapping))
                    top_bm25_indices = np.argsort(bm25_doc_scores)[::-1][:num_bm25_candidates]

                    keyword_hits_count = 0
                    for doc_idx in top_bm25_indices:
                        # We don't use BM25 scores directly for ranking anymore, just for candidate selection.
                        # A minimal score check might be useful if BM25 scores are very low, but for now, take top_k.
                        child_meta = self.all_child_chunks_for_bm25_mapping[doc_idx]
                        child_id = child_meta['child_id']
                        full_context = self.child_id_to_full_context_map.get(child_id)
                        if not full_context:
                            logger.warning(f"Query '{query_text[:30]}...': BM25 found child_id '{child_id}' but no full context mapping. Skipping.")
                            continue

                        keyword_hits_count += 1
                        if child_id not in all_retrieved_child_chunks:
                            all_retrieved_child_chunks[child_id] = {
                                **full_context, # Includes child_id, parent_id, parent_text, child_text, source_document_name
                                'retrieval_source_types': {'keyword'}
                            }
                        else:
                            all_retrieved_child_chunks[child_id]['retrieval_source_types'].add('keyword')
                    logger.debug(f"Query '{query_text[:30]}...': Keyword (BM25) added/updated {keyword_hits_count} potential chunks.")
                except Exception as e:
                    logger.error(f"Keyword search (BM25) failed for query '{query_text[:30]}...': {e}", exc_info=True)

        # Combined list of unique child chunks for reranking
        unique_child_chunks_for_reranking = list(all_retrieved_child_chunks.values())
        logger.info(f"Total {len(unique_child_chunks_for_reranking)} unique child chunks aggregated from {len(query_texts)} queries for reranking.")

        if not unique_child_chunks_for_reranking:
            logger.info("No documents found from vector or keyword search. Returning empty list.")
            return []

        # --- 2. Reranking ---
        # Use the first query_text as the representative query for reranking.
        representative_query_for_reranking = query_texts[0]
        results_after_processing: List[Dict[str, Any]] = []

        if self.reranker_service:
            # Limit the number of documents sent to the reranker
            if len(unique_child_chunks_for_reranking) > settings.DEFAULT_RERANKER_INPUT_LIMIT:
                logger.info(f"Limiting documents for reranker from {len(unique_child_chunks_for_reranking)} to {settings.DEFAULT_RERANKER_INPUT_LIMIT}.")
                # We don't have a good pre-rerank score, so we just truncate the list.
                # This is a simple strategy to prevent OOM errors.
                unique_child_chunks_for_reranking = unique_child_chunks_for_reranking[:settings.DEFAULT_RERANKER_INPUT_LIMIT]

            parents_for_reranking = [res['parent_text'] for res in unique_child_chunks_for_reranking]
            # Keep original items to map back after reranking, as reranker works on indices
            original_items_before_rerank = list(unique_child_chunks_for_reranking)

            try:
                logger.debug(f"Calling reranker service for {len(parents_for_reranking)} documents "
                             f"with representative query: '{representative_query_for_reranking[:100]}...'. "
                             f"Reranker top_n (from final_top_n): {final_top_n}, "
                             f"Reranker score threshold: {settings.RERANKER_SCORE_THRESHOLD}")

                # The reranker service itself will sort by relevance_score and apply its own top_n
                reranked_outputs = self.reranker_service.rerank(
                    query=representative_query_for_reranking,
                    documents=parents_for_reranking,
                    top_n=final_top_n, # Reranker applies its own top_n
                    batch_size=settings.DEFAULT_RERANKER_BATCH_SIZE,
                    max_text_length=settings.DEFAULT_RERANKER_MAX_TEXT_LENGTH
                )

                temp_reranked_list = []
                for reranked_item_from_service in reranked_outputs:
                    original_idx = reranked_item_from_service['original_index']
                    original_full_data = original_items_before_rerank[original_idx]

                    # Apply reranker score threshold
                    reranker_score = reranked_item_from_service['relevance_score']
                    if reranker_score >= settings.RERANKER_SCORE_THRESHOLD:
                        temp_reranked_list.append({
                            **original_full_data,
                            'final_score': reranker_score, # Use reranker's score as the final score
                            'retrieval_source': f"reranked_from_({','.join(sorted(list(original_full_data.get('retrieval_source_types', ['unknown']))))})"
                        })
                    else:
                        # logger.debug(f"Document with original index {original_idx} (child_id: {original_full_data.get('child_id')}) "
                        #              f"discarded due to rerank score {reranker_score:.4f} < threshold {settings.RERANKER_SCORE_THRESHOLD}.")
                        # print all documents full text and score, with reranker_score < threshold
                        # logger.info("Discarded documents:\n"+ '-'*50)
                        # logger.info(f"Full text: {original_full_data.get('parent_text', '')}")
                        # logger.info(f"Score: {reranker_score}")
                        continue # Explicitly continue to avoid appending to results

                # RerankerService.rerank already sorts by score and applies top_n.
                # The list temp_reranked_list is already sorted and thresholded.
                results_after_processing = temp_reranked_list
                logger.info(f"Reranking and thresholding complete. Produced {len(results_after_processing)} results.")

            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Proceeding without reranking, applying final_top_n to combined results.")
                # Fallback: if reranker fails, use the combined list, no specific order, then apply final_top_n
                # No reliable 'final_score' can be assigned here.
                for item in unique_child_chunks_for_reranking: item['final_score'] = 0.0 # Default score
                results_after_processing = unique_child_chunks_for_reranking[:final_top_n] if final_top_n is not None else unique_child_chunks_for_reranking
            except Exception as e:
                 logger.error(f"Unexpected error during reranking: {e}. Proceeding without reranking, applying final_top_n.", exc_info=True)
                 for item in unique_child_chunks_for_reranking: item['final_score'] = 0.0 # Default score
                 results_after_processing = unique_child_chunks_for_reranking[:final_top_n] if final_top_n is not None else unique_child_chunks_for_reranking

        else: # No reranker service available
            logger.warning("Reranker service is not available. Returning combined results from vector/keyword search, "
                           "without reranking or score-based thresholding. Applying final_top_n.")
            # No reliable 'final_score' can be assigned here.
            for item in unique_child_chunks_for_reranking: item['final_score'] = 0.0 # Default score
            results_after_processing = unique_child_chunks_for_reranking[:final_top_n] if final_top_n is not None else unique_child_chunks_for_reranking


        # --- 3. Format for Output ---
        output_for_chapter_writer: List[Dict[str, Any]] = []
        for res_item in results_after_processing:
            output_for_chapter_writer.append({
                "document": res_item["parent_text"],
                "score": res_item.get("final_score", 0.0), # Ensure score exists, default to 0.0
                "child_text_preview": res_item.get("child_text", "")[:150] + "...",
                "child_id": res_item["child_id"],
                "parent_id": res_item["parent_id"],
                "source_document_name": res_item["source_document_name"],
                "retrieval_source": res_item.get("retrieval_source", "unknown_initial_retrieval")
            })

        logger.info(f"Retrieval process finished. Returning {len(output_for_chapter_writer)} items.")
        return output_for_chapter_writer

if __name__ == '__main__':
    # This example requires mocks for VectorStore, BM25Okapi, RerankerService
    # It focuses on the internal logic of RetrievalService.
    logging.basicConfig(level=logging.DEBUG)
    logger.info("RetrievalService Example Start")

    # --- Mock Dependencies ---
    class MockEmbeddingServiceForRS:
        def create_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [[np.random.rand() for _ in range(5)] for _ in texts]

    class MockVectorStoreForRS:
        def __init__(self, embedding_service):
            self.embedding_service = embedding_service
            self.document_store = [] # Populated by add_documents_mock
            self.count_child_chunks = 0
            logger.debug("MockVectorStoreForRS initialized.")

        def add_documents_mock(self, data: List[Dict[str, Any]]): # Simulate adding parent_child_data
            # Simplified: just populate document_store for the mock search to use
            for p_data in data:
                for c_data in p_data['children']:
                    self.document_store.append({
                        'child_id': c_data['child_id'], 'child_text': c_data['child_text'],
                        'parent_id': p_data['parent_id'], 'parent_text': p_data['parent_text'],
                        'source_document_name': p_data['source_document_name'] # Changed key
                    })
            self.count_child_chunks = len(self.document_store)
            logger.debug(f"MockVectorStoreForRS populated with {self.count_child_chunks} items via mock add.")


        def search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
            logger.debug(f"MockVectorStoreForRS.search called for '{query_text}', k={k}")
            results = []
            # Return first k/2 items from document_store as mock results
            for i, item in enumerate(self.document_store):
                if i < k // 2 + 1:
                    results.append({**item, 'score': np.random.uniform(0.1, 0.5)}) # distance score
                else: break
            logger.debug(f"MockVectorStore.search returns {len(results)} items.")
            return results

    class MockBM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
            logger.debug(f"MockBM25Okapi initialized with {len(corpus)} documents.")
        def get_scores(self, query_tokens: List[str]) -> np.ndarray:
            logger.debug(f"MockBM25Okapi.get_scores called for query: {query_tokens}")
            # Return random scores for all docs in corpus
            return np.random.rand(len(self.corpus)) * 10

    class MockRerankerServiceForRS:
        def rerank(self, query: str, documents: List[str], top_n: Optional[int], batch_size: int, max_text_length: int) -> List[Dict[str, Any]]:
            logger.debug(f"MockRerankerService.rerank called for '{query}', {len(documents)} docs, top_n={top_n}")
            reranked = []
            for i, doc_text in enumerate(reversed(documents)):
                reranked.append({"document": doc_text, "relevance_score": 0.9 - (i * 0.05), "original_index": documents.index(doc_text)})
            return reranked[:top_n] if top_n else reranked

    # --- Setup Data for Mocks ---
    sample_p_c_data = [
        {"parent_id": "P1", "parent_text": "Parent One: Apples are red. Oranges are orange.", "source_document_name": "DocA.txt", # Changed key
         "children": [{"child_id": "P1C1", "child_text": "Apples are red."}, {"child_id": "P1C2", "child_text": "Oranges are orange."}]},
        {"parent_id": "P2", "parent_text": "Parent Two: Bananas are yellow. Grapes are purple.", "source_document_name": "DocA.txt", # Changed key
         "children": [{"child_id": "P2C1", "child_text": "Bananas are yellow."}, {"child_id": "P2C2", "child_text": "Grapes are purple."}]},
        {"parent_id": "P3", "parent_text": "Parent Three: Cars are fast. Bikes are good for exercise.", "source_document_name": "DocB.pdf", # Changed key
         "children": [{"child_id": "P3C1", "child_text": "Cars are fast and come in red or blue."}, {"child_id": "P3C2", "child_text": "Bikes are good for exercise and fun."}]}
    ]

    mock_vs = MockVectorStoreForRS(MockEmbeddingServiceForRS())
    mock_vs.add_documents_mock(sample_p_c_data) # Manually populate its store for this test

    all_child_chunks_map_data = [] # This is List[Dict{'child_id': str, 'child_text': str}]
    bm25_corpus_tokens = []
    for p_item in sample_p_c_data:
        for c_item in p_item['children']:
            all_child_chunks_map_data.append({'child_id': c_item['child_id'], 'child_text': c_item['child_text']})
            bm25_corpus_tokens.append(c_item['child_text'].lower().split())

    mock_bm25 = MockBM25Okapi(bm25_corpus_tokens) if bm25_corpus_tokens else None
    mock_reranker = MockRerankerServiceForRS()

    # --- Initialize RetrievalService with Mocks ---
    retrieval_svc = RetrievalService(
        vector_store=mock_vs,
        bm25_index=mock_bm25,
        all_child_chunks_for_bm25_mapping=all_child_chunks_map_data,
        reranker_service=mock_reranker
    )

    # --- Test Cases ---
    test_query_sets = [
        (["red apples and fast cars"], "Single Query Test"),
        (["yellow bananas", "bikes"], "Multi Query Test"),
        (["non_existent_topic"], "Single Non-Existent Topic"),
        (["apples are red", "Oranges are orange", "P1C1 P1C2"], "Multiple Relevant Queries for P1"),
        ([], "Empty Query List Test")
    ]

    for t_queries, test_name in test_query_sets:
        logger.info(f"\n--- Testing RetrievalService with: {test_name} ({len(t_queries)} queries) ---")
        if t_queries: logger.info(f"    Queries: {t_queries}")
        try:
            results = retrieval_svc.retrieve(
                query_texts=t_queries, # Pass list of queries
                vector_top_k=3,  # Increased k to see more potential overlaps
                keyword_top_k=3, # Increased k
                # hybrid_alpha=0.5, # Removed
                final_top_n=3    # Get top 3 overall
            )
            if results:
                logger.info(f"Found {len(results)} items for '{test_name}':")
                for i, item in enumerate(results):
                    print(f"  Result {i+1}:")
                    print(f"    Child ID: {item['child_id']}, Parent ID: {item['parent_id']}")
                    print(f"    Child Preview: {item['child_text_preview']}")
                    print(f"    Parent Text: '{item['document'][:60]}...'")
                    print(f"    Score: {item['score']:.4f}, Source: {item['retrieval_source']}")
                    # print(f"    Original Query if available: {item.get('original_query_text', 'N/A')}") # If added
            else:
                logger.info(f"No results found for '{test_name}'.")
        except RetrievalServiceError as e:
            logger.error(f"RetrievalServiceError for '{test_name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error for '{test_name}': {e}", exc_info=True)

    logger.info("\nRetrievalService Example End")
