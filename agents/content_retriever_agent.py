import logging
from typing import List, Dict, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi # For keyword search

from agents.base_agent import BaseAgent
from core.vector_store import VectorStore, VectorStoreError
from core.reranker_service import RerankerService, RerankerServiceError
from config.settings import (
    DEFAULT_VECTOR_STORE_TOP_K,
    DEFAULT_HYBRID_SEARCH_ALPHA,
    DEFAULT_KEYWORD_SEARCH_TOP_K
)

# Configure logger
logger = logging.getLogger(__name__)

class ContentRetrieverAgentError(Exception):
    """Custom exception for ContentRetrieverAgent errors."""
    pass

class ContentRetrieverAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant content using a hybrid approach
    (vector search + keyword search) and then optionally reranking the results.
    It works with parent-child chunks from the VectorStore.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 bm25_index: Optional[BM25Okapi] = None, # BM25 index passed in
                 all_child_chunks_for_bm25: Optional[List[Dict[str, Any]]] = None, # child_id, child_text needed for mapping BM25 results
                 reranker_service: Optional[RerankerService] = None,
                 default_vector_top_k: int = DEFAULT_VECTOR_STORE_TOP_K,
                 default_keyword_top_k: int = DEFAULT_KEYWORD_SEARCH_TOP_K,
                 default_hybrid_alpha: float = DEFAULT_HYBRID_SEARCH_ALPHA,
                 default_final_top_n: Optional[int] = None): # How many to keep after all steps
        """
        Initializes the ContentRetrieverAgent.

        Args:
            vector_store (VectorStore): Instance of VectorStore.
            bm25_index (Optional[BM25Okapi]): Pre-computed BM25 index.
            all_child_chunks_for_bm25 (Optional[List[Dict[str, Any]]]): List of dicts,
                each containing at least 'child_id' and 'child_text' for all child chunks
                used to build the BM25 index. This is needed to map BM25 results back to structured data.
            reranker_service (Optional[RerankerService]): Instance of RerankerService.
            default_vector_top_k (int): Default K for vector search.
            default_keyword_top_k (int): Default K for keyword search.
            default_hybrid_alpha (float): Default alpha for hybrid search blending.
            default_final_top_n (Optional[int]): Default number of results after reranking/final selection.
        """
        super().__init__(agent_name="ContentRetrieverAgent",
                         vector_store=vector_store,
                         reranker_service=reranker_service)

        if not self.vector_store:
            raise ContentRetrieverAgentError("VectorStore is required.")

        self.bm25_index = bm25_index
        self.all_child_chunks_for_bm25 = all_child_chunks_for_bm25 if all_child_chunks_for_bm25 else []

        # Create a quick lookup map from child_text (if BM25 uses text) or child_id to full child metadata
        # This assumes child_texts given to BM25 are unique enough or we map by index.
        # Simpler: assume bm25_index was built on tokenized versions of self.all_child_chunks_for_bm25[*]['child_text']
        # and BM25.get_top_n returns documents (texts) or indices.
        # If BM25Okapi from rank_bm25 returns original docs (texts), we need to map them back.
        # Or, if it returns indices, those indices map to the corpus it was trained on.
        # Let's assume all_child_chunks_for_bm25 is the corpus (or contains it)
        self.child_id_to_parent_context_map: Dict[str, Dict[str, Any]] = {}
        if self.vector_store and hasattr(self.vector_store, 'document_store'):
             for item_meta in self.vector_store.document_store:
                 self.child_id_to_parent_context_map[item_meta['child_id']] = item_meta


        self.default_vector_top_k = default_vector_top_k
        self.default_keyword_top_k = default_keyword_top_k
        self.default_hybrid_alpha = default_hybrid_alpha
        self.default_final_top_n = default_final_top_n

        log_msg = f"ContentRetrieverAgent initialized. BM25 index is {'PRESENT' if self.bm25_index else 'ABSENT'}."
        log_msg += f" Reranker service is {'ENABLED' if self.reranker_service else 'DISABLED'}."
        logger.info(log_msg)

    def _tokenize_query(self, query: str) -> List[str]:
        """Simple whitespace tokenizer for BM25. For Chinese, a proper tokenizer (e.g., jieba) is better."""
        # In a real scenario with Chinese, use: `import jieba; return list(jieba.cut_for_search(query))`
        return query.lower().split()

    def _normalize_scores(self, scores: List[float], reverse: bool = False) -> List[float]:
        """Min-max normalize scores to [0, 1]. Reverse if higher score is worse (e.g., distance)."""
        if not scores or max(scores) == min(scores): # Avoid division by zero or if all scores are same
            return [0.5] * len(scores) if scores else [] # Default to mid-point or empty

        min_s, max_s = min(scores), max(scores)
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        return [1.0 - s for s in normalized] if reverse else normalized


    def run(self,
            query: str,
            vector_top_k: Optional[int] = None,
            keyword_top_k: Optional[int] = None,
            hybrid_alpha: Optional[float] = None,
            final_top_n: Optional[int] = None
           ) -> List[Dict[str, Any]]:
        """
        Retrieves and optionally reranks content using hybrid search.
        The returned documents are parent chunks associated with the best matching child chunks.

        Args:
            query (str): The query string.
            vector_top_k (Optional[int]): K for vector search.
            keyword_top_k (Optional[int]): K for keyword search.
            hybrid_alpha (Optional[float]): Alpha for blending (0-1). 0=keyword only, 1=vector only.
            final_top_n (Optional[int]): Number of final results to return.

        Returns:
            List[Dict[str, Any]]: List of result dictionaries, each containing:
                'parent_id', 'parent_text', 'child_id', 'child_text',
                'source_document_id', 'final_score', 'retrieval_source' (e.g., 'hybrid', 'vector', 'keyword').
                The 'document' for ChapterWriter should be 'parent_text'.
        """
        self._log_input(query=query, vector_top_k=vector_top_k, keyword_top_k=keyword_top_k,
                        hybrid_alpha=hybrid_alpha, final_top_n=final_top_n)

        # Set operational parameters from defaults or args
        vec_k = vector_top_k if vector_top_k is not None else self.default_vector_top_k
        key_k = keyword_top_k if keyword_top_k is not None else self.default_keyword_top_k
        alpha = hybrid_alpha if hybrid_alpha is not None else self.default_hybrid_alpha
        n_results = final_top_n if final_top_n is not None else self.default_final_top_n


        # --- 1. Vector Search ---
        vector_results_map: Dict[str, Dict[str, Any]] = {} # child_id -> result_dict
        if alpha > 0: # Perform vector search if alpha indicates it's needed
            try:
                logger.info(f"Performing vector search for query: '{query[:100]}...' with k={vec_k}")
                # VectorStore.search returns List[Dict] with child_id, child_text, parent_id, parent_text, score (distance)
                raw_vector_hits = self.vector_store.search(query_text=query, k=vec_k)

                # Normalize L2 distances (lower is better) to similarity scores [0,1] (higher is better)
                distances = [hit['score'] for hit in raw_vector_hits]
                norm_similarity_scores = self._normalize_scores(distances, reverse=True)

                for i, hit in enumerate(raw_vector_hits):
                    child_id = hit['child_id']
                    vector_results_map[child_id] = {
                        **hit, # contains all keys: child_id, child_text, parent_id, parent_text, source_document_id
                        'vector_score': norm_similarity_scores[i],
                        'keyword_score': 0.0 # Initialize keyword score
                    }
                logger.info(f"Vector search found {len(vector_results_map)} distinct child chunks.")
            except VectorStoreError as e:
                logger.error(f"VectorStore search failed: {e}. Proceeding without vector results.")
            except Exception as e: # Catch any other unexpected error from vector search part
                logger.error(f"Unexpected error during vector search: {e}. Proceeding without vector results.")


        # --- 2. Keyword Search (BM25) ---
        keyword_results_map: Dict[str, Dict[str, Any]] = {} # child_id -> result_dict
        if alpha < 1.0 and self.bm25_index and self.all_child_chunks_for_bm25:
            try:
                logger.info(f"Performing keyword search (BM25) for query: '{query[:100]}...' with k={key_k}")
                tokenized_query = self._tokenize_query(query)

                # bm25_scores will be scores for ALL documents in corpus used to train bm25_index
                bm25_scores = self.bm25_index.get_scores(tokenized_query)

                # Get top N indices from BM25 scores
                # These indices map to the order of documents in self.all_child_chunks_for_bm25
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:key_k] # Get indices of top k scores

                # Normalize BM25 scores (higher is better)
                # Only normalize scores of the top_k retrieved for fairer comparison if sets are small
                top_bm25_scores = [bm25_scores[i] for i in top_bm25_indices]
                norm_bm25_scores = self._normalize_scores(top_bm25_scores, reverse=False)

                for i, doc_idx in enumerate(top_bm25_indices):
                    if norm_bm25_scores[i] <= 0: # Skip if score is not positive after normalization
                        continue

                    # `doc_idx` is the index in the `self.all_child_chunks_for_bm25` list
                    # This list was used to create the BM25 index.
                    # We need to get the child_id from this list.
                    child_meta_from_bm25_corpus = self.all_child_chunks_for_bm25[doc_idx]
                    child_id = child_meta_from_bm25_corpus['child_id'] # Assuming this structure

                    # Fetch full parent/child context from our main store using child_id
                    full_context = self.child_id_to_parent_context_map.get(child_id)
                    if not full_context:
                        logger.warning(f"BM25 found child_id {child_id} but it's not in main context map. Skipping.")
                        continue

                    keyword_results_map[child_id] = {
                        **full_context, # child_id, child_text, parent_id, parent_text, source_document_id
                        'vector_score': 0.0, # Initialize vector score
                        'keyword_score': norm_bm25_scores[i]
                    }
                logger.info(f"Keyword search (BM25) found {len(keyword_results_map)} distinct child chunks with positive scores.")
            except Exception as e: # Catch any error from BM25 part
                logger.error(f"Keyword search (BM25) failed: {e}. Proceeding without keyword results.")
                # import traceback; traceback.print_exc()


        # --- 3. Combine and Rank Results ---
        combined_results: Dict[str, Dict[str, Any]] = {}
        # Merge results, prioritizing vector_results_map then adding/updating with keyword_results_map
        for child_id, data in vector_results_map.items():
            combined_results[child_id] = data

        for child_id, data in keyword_results_map.items():
            if child_id in combined_results:
                combined_results[child_id]['keyword_score'] = data['keyword_score'] # Update keyword score
            else:
                combined_results[child_id] = data # Add new entry from keyword search

        # Calculate final hybrid score
        scored_results = []
        for child_id, data in combined_results.items():
            final_score = (alpha * data['vector_score']) + ((1.0 - alpha) * data['keyword_score'])
            retrieval_source = "hybrid"
            if alpha == 1.0: retrieval_source = "vector"
            elif alpha == 0.0: retrieval_source = "keyword"

            scored_results.append({
                **data, # child_id, child_text, parent_id, parent_text, source_document_id, vector_score, keyword_score
                'final_score': final_score,
                'retrieval_source': retrieval_source
            })

        # Sort by final_score descending (higher is better)
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Limit to a preliminary top_n if hybrid search generated many candidates
        # This top_n could be larger than final_top_n if reranking is next.
        # For now, let's consider n_results applies after this stage if no reranker.
        # If reranker exists, it will do its own top_n.

        # The documents to pass to reranker are the PARENT texts.
        # Reranker expects a list of strings.
        documents_for_reranking = [res['parent_text'] for res in scored_results]
        # Keep track of the full result items to re-associate after reranking
        results_before_rerank = scored_results


        # --- 4. Optional Reranking ---
        # Reranker operates on the parent_text for better contextual understanding.
        final_hybrid_retrieved_content: List[Dict[str, Any]] = []

        if self.reranker_service and documents_for_reranking:
            logger.info(f"Reranking {len(documents_for_reranking)} combined results with RerankerService. Query: '{query[:100]}...'")
            try:
                # RerankerService.rerank returns List[Dict] with 'document', 'relevance_score', 'original_index'
                # 'document' here will be the parent_text we passed.
                # 'original_index' maps back to the `documents_for_reranking` list.
                reranked_outputs = self.reranker_service.rerank(
                    query=query,
                    documents=documents_for_reranking,
                    top_n=n_results # Reranker can handle the final top_n
                )

                for reranked_item in reranked_outputs:
                    original_idx = reranked_item['original_index']
                    # Get the original full data (parent, child, scores) using this index
                    original_full_result = results_before_rerank[original_idx]

                    final_hybrid_retrieved_content.append({
                        **original_full_result, # Keep all previous fields like child_id, parent_id etc.
                        'parent_text': reranked_item['document'], # This is the parent_text confirmed by reranker
                        'final_score': reranked_item['relevance_score'], # Update score with reranker's score
                        'retrieval_source': original_full_result['retrieval_source'] + "_reranked"
                    })
                logger.info(f"Reranking complete. Returning {len(final_hybrid_retrieved_content)} documents.")
            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Falling back to pre-reranked hybrid results.")
                # Fallback to results before reranking, respecting n_results
                final_hybrid_retrieved_content = results_before_rerank[:n_results] if n_results else results_before_rerank
            except Exception as e:
                logger.error(f"Unexpected error during reranking: {e}. Falling back to pre-reranked hybrid results.")
                final_hybrid_retrieved_content = results_before_rerank[:n_results] if n_results else results_before_rerank
        else:
            # No reranker, so use the hybrid scored results, respecting n_results
            logger.info("Skipping reranking (no reranker service or no documents to rerank).")
            final_hybrid_retrieved_content = scored_results[:n_results] if n_results else scored_results

        self._log_output([res['child_id'] for res in final_hybrid_retrieved_content]) # Log child_ids of final results

        # The output format for ChapterWriter needs 'document' to be the parent_text and 'score'
        # Let's ensure this structure for the final output of this agent.
        output_for_chapter_writer = []
        for res in final_hybrid_retrieved_content:
            output_for_chapter_writer.append({
                "document": res["parent_text"], # This is the key ChapterWriter expects
                "score": res["final_score"],
                "child_text_preview": res["child_text"][:100] + "...", # For context/logging
                "child_id": res["child_id"],
                "parent_id": res["parent_id"],
                "source": res["retrieval_source"]
            })

        return output_for_chapter_writer


if __name__ == '__main__':
    print("ContentRetrieverAgent (Hybrid Search & Parent-Child Adapted) Example")
    logging.basicConfig(level=logging.DEBUG)

    # --- Mock Services for Example ---
    class MockEmbeddingService:
        def create_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [[np.random.rand() for _ in range(5)] for _ in texts] # 5-dim

    class MockVectorStoreForContentRetriever(BaseAgent):
        def __init__(self, embedding_service):
            super().__init__(agent_name="MockVectorStore")
            self.embedding_service = embedding_service
            self.document_store: List[Dict[str, Any]] = [] # Stores child_meta with parent_text
            self._is_initialized = True # Assume initialized
            self.index = None # Not actually used in mock search, but normally present

        def add_documents(self, parent_child_data: List[Dict[str, Any]]):
            # Simplified: just populate document_store for the mock search to use
            for p_data in parent_child_data:
                for c_data in p_data['children']:
                    self.document_store.append({
                        'child_id': c_data['child_id'], 'child_text': c_data['child_text'],
                        'parent_id': p_data['parent_id'], 'parent_text': p_data['parent_text'],
                        'source_document_id': p_data['source_document_id']
                    })
            logger.info(f"MockVectorStore populated with {len(self.document_store)} child items.")
            # In real VS, FAISS index would be built here.

        def search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
            logger.info(f"MockVectorStore.search called for '{query_text}', k={k}")
            # Simulate some results; in real scenario, this involves FAISS.
            # Return a few items from document_store with dummy scores (distances).
            results = []
            for i, item in enumerate(self.document_store):
                if i < k : # Return first k items as mock results
                    results.append({**item, 'score': np.random.rand() * 0.5 }) # low distance = good
                else:
                    break
            return results

        def get_all_child_texts(self) -> List[str]: # For BM25
             return [item['child_text'] for item in self.document_store]

        def get_child_and_parent_text_by_child_id(self, child_id: str) -> Optional[Dict[str,str]]:
            for item in self.document_store:
                if item['child_id'] == child_id:
                    return {'child_text': item['child_text'], 'parent_text': item['parent_text']}
            return None


    class MockRerankerServiceForContentRetriever(BaseAgent):
        def rerank(self, query: str, documents: List[str], top_n: Optional[int]) -> List[Dict[str, Any]]:
            logger.info(f"MockRerankerService.rerank called for '{query}', {len(documents)} docs, top_n={top_n}")
            # Simulate reranking: reverse order and assign new scores
            reranked = []
            for i, doc_text in enumerate(reversed(documents)): # documents are parent_texts
                reranked.append({
                    "document": doc_text, # This is parent_text
                    "relevance_score": 0.85 - (i * 0.05), # Higher score for earlier in reversed list
                    "original_index": documents.index(doc_text)
                })
            return reranked[:top_n] if top_n is not None else reranked

    # --- Setup Example Data ---
    sample_parent_child_data = [ # From VectorStore example
        {"parent_id": "doc1-p1", "parent_text": "Parent One. It talks about apples and oranges. Also mentions bananas.", "source_document_id": "doc1",
         "children": [{"child_id": "doc1-p1-c1", "child_text": "Parent One. It talks about apples and oranges."},
                      {"child_id": "doc1-p1-c2", "child_text": "Also mentions bananas."}]},
        {"parent_id": "doc1-p2", "parent_text": "Parent Two. This one is about grapes and strawberries. And kiwi fruit.", "source_document_id": "doc1",
         "children": [{"child_id": "doc1-p2-c1", "child_text": "Parent Two. This one is about grapes and strawberries."},
                      {"child_id": "doc1-p2-c2", "child_text": "And kiwi fruit is tasty."}]},
        {"parent_id": "doc2-p1", "parent_text": "Another document. Parent Three. Discusses red cars and blue bikes. Fast vehicles.", "source_document_id": "doc2",
         "children": [{"child_id": "doc2-p1-c1", "child_text": "Another document. Parent Three. Discusses red cars and blue bikes."},
                      {"child_id": "doc2-p1-c2", "child_text": "Fast vehicles are exciting."}]}
    ]

    mock_emb_svc = MockEmbeddingService()
    mock_vs = MockVectorStoreForContentRetriever(embedding_service=mock_emb_svc)
    mock_vs.add_documents(sample_parent_child_data) # Populate the mock store

    # Prepare BM25 index (needs all child texts and their original child_ids for mapping)
    all_child_chunks_for_bm25_build = []
    corpus_for_bm25 = []
    for p_data in sample_parent_child_data:
        for c_data in p_data['children']:
            all_child_chunks_for_bm25_build.append({ # Store metadata needed to map back
                'child_id': c_data['child_id'],
                'child_text': c_data['child_text'],
                # parent_id, parent_text, source_document_id could also be stored if needed directly from here
            })
            corpus_for_bm25.append(c_data['child_text'].lower().split()) # BM25 needs tokenized text

    bm25 = None
    if corpus_for_bm25:
        bm25 = BM25Okapi(corpus_for_bm25)
        logger.info("Mock BM25 index built.")


    mock_reranker_svc = MockRerankerServiceForContentRetriever()

    # --- Instantiate and Run Agent ---
    try:
        retriever_agent = ContentRetrieverAgent(
            vector_store=mock_vs,
            bm25_index=bm25,
            all_child_chunks_for_bm25=all_child_chunks_for_bm25_build, # Pass the mapping data
            reranker_service=mock_reranker_svc,
            default_vector_top_k=3,
            default_keyword_top_k=3,
            default_hybrid_alpha=0.5, # Blend vector and keyword equally
            default_final_top_n=2     # Get top 2 final results
        )

        query_example = "information about bananas and cars"
        print(f"\n--- Running ContentRetrieverAgent for query: '{query_example}' ---")

        results = retriever_agent.run(query=query_example)

        print(f"\n--- Final Retrieved Content (Hybrid, Reranked, Top {retriever_agent.default_final_top_n}) ---")
        if results:
            for i, item in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Parent ID: {item['parent_id']}")
                print(f"  Child ID: {item['child_id']}")
                print(f"  Child Preview: {item['child_text_preview']}")
                print(f"  Parent Text: '{item['document'][:100]}...'") # 'document' is parent_text
                print(f"  Final Score: {item['score']:.4f}")
                print(f"  Source: {item['source']}")
                print("-" * 20)
        else:
            print("No results returned by the agent.")

        # Test with alpha = 1.0 (vector only)
        print(f"\n--- Running with Alpha=1.0 (Vector Search Only) ---")
        results_vec_only = retriever_agent.run(query=query_example, hybrid_alpha=1.0)
        print(f"Vector-Only Results (first {len(results_vec_only)}):")
        for item in results_vec_only: print(f"  Parent: {item['parent_id']}, Child: {item['child_id']}, Score: {item['score']:.3f}, Source: {item['source']}")

        # Test with alpha = 0.0 (keyword only)
        print(f"\n--- Running with Alpha=0.0 (Keyword Search Only) ---")
        results_key_only = retriever_agent.run(query=query_example, hybrid_alpha=0.0)
        print(f"Keyword-Only Results (first {len(results_key_only)}):")
        for item in results_key_only: print(f"  Parent: {item['parent_id']}, Child: {item['child_id']}, Score: {item['score']:.3f}, Source: {item['source']}")


    except ContentRetrieverAgentError as e:
        logger.error(f"ContentRetrieverAgent execution failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example: {e}", exc_info=True)

    print("\nContentRetrieverAgent Example Finished.")
