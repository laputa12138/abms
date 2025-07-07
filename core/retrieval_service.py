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
                 query_texts: List[str], # Changed from query_text: str
                 vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA,
                 final_top_n: int = settings.DEFAULT_RETRIEVAL_FINAL_TOP_N
                ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval for multiple query texts, aggregates results,
        optionally reranks, and applies score thresholding.

        Args:
            query_texts (List[str]): A list of user's queries.
            vector_top_k (int): Number of results from vector search (per query).
            keyword_top_k (int): Number of results from keyword search (per query).
            hybrid_alpha (float): Weight for blending vector and keyword scores.
            final_top_n (int): Number of final results to return after all steps.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries, structured for consumption.
        """
        if not query_texts:
            logger.warning("RetrievalService.retrieve called with empty query_texts list. Returning empty list.")
            return []

        logger.info(f"RetrievalService called with {len(query_texts)} queries. First query: '{query_texts[0][:100]}...' "
                    f"v_k={vector_top_k}, k_k={keyword_top_k}, alpha={hybrid_alpha}, final_n={final_top_n}")

        all_queries_aggregated_results: Dict[str, Dict[str, Any]] = {} # child_id -> best_result_data_for_child

        for query_idx, query_text in enumerate(query_texts):
            logger.debug(f"Processing query {query_idx + 1}/{len(query_texts)}: '{query_text[:100]}...'")

            # --- 1. Vector Search (per query) ---
            current_query_vector_results: Dict[str, Dict[str, Any]] = {}
            if hybrid_alpha > 0: # Check if vector search is relevant
                try:
                    raw_vector_hits = self.vector_store.search(query_text=query_text, k=vector_top_k)
                    if raw_vector_hits: # Only normalize if there are hits
                        distances = [hit['score'] for hit in raw_vector_hits]
                        norm_similarity_scores = self._normalize_scores(distances, reverse=True)
                        for i, hit in enumerate(raw_vector_hits):
                            child_id = hit['child_id']
                            current_query_vector_results[child_id] = {
                                **hit,
                                'vector_score': norm_similarity_scores[i],
                                'keyword_score': 0.0  # Initialize keyword score
                            }
                    logger.debug(f"Query '{query_text[:30]}...': Vector search found {len(current_query_vector_results)} distinct child chunks.")
                except VectorStoreError as e:
                    logger.error(f"VectorStore search failed for query '{query_text[:30]}...': {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during vector search for query '{query_text[:30]}...': {e}", exc_info=True)

            # --- 2. Keyword Search (BM25 - per query) ---
            current_query_keyword_results: Dict[str, Dict[str, Any]] = {}
            if hybrid_alpha < 1.0 and self.bm25_index and self.all_child_chunks_for_bm25_mapping: # Check if keyword search is relevant
                try:
                    tokenized_query = self._tokenize_query(query_text)
                    bm25_doc_scores = self.bm25_index.get_scores(tokenized_query)

                    num_bm25_candidates = min(keyword_top_k, len(self.all_child_chunks_for_bm25_mapping))
                    top_bm25_indices = np.argsort(bm25_doc_scores)[::-1][:num_bm25_candidates]

                    if top_bm25_indices.size > 0: # Only normalize if there are candidates
                        top_bm25_scores_only = [bm25_doc_scores[i] for i in top_bm25_indices]
                        norm_bm25_scores = self._normalize_scores(top_bm25_scores_only, reverse=False)
                        for i, doc_idx in enumerate(top_bm25_indices):
                            if norm_bm25_scores[i] <= 1e-6: continue
                            child_meta = self.all_child_chunks_for_bm25_mapping[doc_idx]
                            child_id = child_meta['child_id']
                            full_context = self.child_id_to_full_context_map.get(child_id)
                            if not full_context:
                                logger.warning(f"Query '{query_text[:30]}...': BM25 found child_id '{child_id}' but no full context. Skipping.")
                                continue
                            current_query_keyword_results[child_id] = {
                                **full_context,
                                'vector_score': 0.0, # Initialize vector score
                                'keyword_score': norm_bm25_scores[i]
                            }
                    logger.debug(f"Query '{query_text[:30]}...': Keyword (BM25) found {len(current_query_keyword_results)} distinct child chunks.")
                except Exception as e:
                    logger.error(f"Keyword search (BM25) failed for query '{query_text[:30]}...': {e}", exc_info=True)

            # --- 3. Combine and Score Results for the Current Query ---
            current_query_combined_results: Dict[str, Dict[str, Any]] = {}
            # Merge vector results
            for child_id, data in current_query_vector_results.items():
                current_query_combined_results[child_id] = data
            # Merge keyword results, updating scores if child_id already exists
            for child_id, data in current_query_keyword_results.items():
                if child_id in current_query_combined_results:
                    current_query_combined_results[child_id]['keyword_score'] = data['keyword_score']
                else:
                    current_query_combined_results[child_id] = data

            # Calculate final_score for this query's results and add to global aggregation
            for child_id, data in current_query_combined_results.items():
                current_final_score = (hybrid_alpha * data['vector_score']) + ((1.0 - hybrid_alpha) * data['keyword_score'])

                # Determine source for this specific query
                source = "hybrid"
                if hybrid_alpha == 1.0 and data['vector_score'] > 0: source = "vector"
                elif hybrid_alpha == 0.0 and data['keyword_score'] > 0: source = "keyword"
                elif data['vector_score'] == 0 and data['keyword_score'] == 0 : source = "none"

                if current_final_score > 1e-6: # Only consider if there's a meaningful score
                    # If this child_id is already in all_queries_aggregated_results, update if current score is higher
                    if child_id not in all_queries_aggregated_results or \
                       current_final_score > all_queries_aggregated_results[child_id]['final_score']:
                        all_queries_aggregated_results[child_id] = {
                            **data, # Contains original hit details like parent_text, child_text etc.
                            'final_score': current_final_score,
                            'retrieval_source': f"query_{query_idx}_{source}", # Tag with query index and source type
                            'original_query_text': query_text # Store which query found this best version
                        }

        # Convert aggregated results dict to a list for sorting and further processing
        scored_results = list(all_queries_aggregated_results.values())
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        logger.info(f"Total {len(scored_results)} unique child chunks aggregated from {len(query_texts)} queries before thresholding/reranking.")

        # --- 3a. Apply Global Score Threshold (before reranking) ---
        # The scores at this stage are normalized [0,1] where higher is better.
        # This threshold is applied to the 'final_score' from hybrid search.
        min_score_threshold = settings.DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD
        if min_score_threshold > 0:
            results_before_thresholding_count = len(scored_results)
            results_after_thresholding = [res for res in scored_results if res['final_score'] >= min_score_threshold]
            logger.debug(f"Applied score threshold {min_score_threshold}. "
                         f"Reduced results from {results_before_thresholding_count} to {len(results_after_thresholding)}.")

            if results_before_thresholding_count > 0 and not results_after_thresholding:
                logger.warning(f"All {results_before_thresholding_count} potential documents were filtered out by the score threshold {min_score_threshold}. "
                               "This might lead to empty content for chapters if no documents met the quality criteria. "
                               "Consider reviewing threshold, query formulation, or document quality/relevance if this happens frequently.")
            scored_results = results_after_thresholding

        # --- 4. Optional Reranking (operates on parent_text of aggregated results) ---
        # Reranker needs a single representative query if it's query-dependent.
        # Using the first query_text as the representative query for reranking.
        # This is a simplification; a more complex strategy might involve reranking
        # against each query or using a combined query representation.
        representative_query_for_reranking = query_texts[0] if query_texts else ""

        results_after_processing = scored_results
        if self.reranker_service and scored_results and representative_query_for_reranking:
            parents_for_reranking = [res['parent_text'] for res in scored_results]
            # Keep original items to map back after reranking
            original_items_before_rerank = list(scored_results)

            try:
                logger.debug(f"Calling reranker service for {len(parents_for_reranking)} documents "
                             f"with representative query: '{representative_query_for_reranking[:100]}...'. "
                             f"Reranker top_n (final_top_n for retrieval): {final_top_n}")

                reranked_outputs = self.reranker_service.rerank(
                    query=representative_query_for_reranking, # Use the representative query
                    documents=parents_for_reranking,
                    top_n=final_top_n, # Reranker applies its own top_n based on this
                    batch_size=settings.DEFAULT_RERANKER_BATCH_SIZE, # from settings
                    max_text_length=settings.DEFAULT_RERANKER_MAX_TEXT_LENGTH # from settings
                )

                temp_reranked_list = []
                for reranked_item_from_service in reranked_outputs:
                    original_idx = reranked_item_from_service['original_index']
                    # Map back to original full data using the index
                    original_full_data = original_items_before_rerank[original_idx]
                    temp_reranked_list.append({
                        **original_full_data, # Spread the original data (child_id, parent_id, etc.)
                        'parent_text': reranked_item_from_service['document'], # This might be redundant if reranker doesn't change it
                        'final_score': reranked_item_from_service['relevance_score'], # Update score with reranker's score
                        'retrieval_source': original_full_data.get('retrieval_source', 'unknown') + "_reranked"
                    })
                results_after_processing = temp_reranked_list # This list is sorted by reranker and respects top_n
                logger.debug(f"Reranking complete. Produced {len(results_after_processing)} results.")

            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Using pre-reranked, thresholded results, then applying final_top_n.")
                # Fallback: apply final_top_n to the list that was thresholded but not reranked
                results_after_processing = scored_results[:final_top_n] if final_top_n is not None else scored_results
            except Exception as e:
                 logger.error(f"Unexpected error during reranking: {e}. Using pre-reranked, thresholded results, then applying final_top_n.", exc_info=True)
                 results_after_processing = scored_results[:final_top_n] if final_top_n is not None else scored_results
        elif final_top_n is not None: # No reranker, but final_top_n is set (apply to thresholded results)
            logger.debug(f"No reranker or no query for reranker. Applying final_top_n={final_top_n} to {len(scored_results)} thresholded results.")
            results_after_processing = scored_results[:final_top_n]
        else: # No reranker and no final_top_n, return all thresholded results
            logger.debug(f"No reranker and no final_top_n. Returning all {len(scored_results)} thresholded results.")
            results_after_processing = scored_results


        # --- 5. Format for Output ---
        # Ensure the output format is suitable for ChapterWriterAgent
        output_for_chapter_writer: List[Dict[str, Any]] = []
        for res_item in results_after_processing: # Corrected variable name
            output_for_chapter_writer.append({
                "document": res_item["parent_text"], # Key for ChapterWriterAgent
                "score": res_item["final_score"],
                "child_text_preview": res_item["child_text"][:150] + "...", # For context/logging
                "child_id": res_item["child_id"],
                "parent_id": res_item["parent_id"],
                "source_document_name": res_item["source_document_name"], # Changed key
                "retrieval_source": res_item["retrieval_source"] # For tracing
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
        def rerank(self, query: str, documents: List[str], top_n: Optional[int]) -> List[Dict[str, Any]]:
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
                hybrid_alpha=0.5,
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
