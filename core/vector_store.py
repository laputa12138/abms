import logging
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from core.embedding_service import EmbeddingService, EmbeddingServiceError
from config.settings import DEFAULT_VECTOR_STORE_TOP_K

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in main
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for VectorStore errors."""
    pass

class VectorStore:
    """
    A class for creating, managing, and searching a FAISS-based vector store.
    It uses an EmbeddingService to convert text to vectors.
    This version is adapted for parent-child chunking:
    - Embeddings are generated for child chunks.
    - Search retrieves child chunks and their associated parent chunk context.
    """

    def __init__(self, embedding_service: EmbeddingService, dimension: Optional[int] = None):
        """
        Initializes the VectorStore.

        Args:
            embedding_service (EmbeddingService): An instance of EmbeddingService.
            dimension (Optional[int]): The dimension of the embedding vectors.
                                       If None, it will be inferred.
        """
        self.embedding_service = embedding_service
        self.index: Optional[faiss.Index] = None

        # self.documents list will now store dictionaries for each child chunk,
        # including its text, its parent's text, and relevant IDs.
        # Format: {'child_id': str, 'child_text': str, 'parent_id': str, 'parent_text': str,
        #          'source_document_name': str} # Changed key name
        # The FAISS index will map to the index of this list.
        self.document_store: List[Dict[str, Any]] = []

        self.dimension = dimension
        self._is_initialized = False
        logger.info("VectorStore initialized for parent-child chunking.")

    def _initialize_index(self, first_embedding_vector: np.ndarray):
        """Helper to initialize FAISS index once dimension is known."""
        if not self.dimension:
            self.dimension = first_embedding_vector.shape[-1] # Get dimension from the embedding vector
            logger.info(f"Inferred embedding dimension: {self.dimension}")

        if self.dimension is None or self.dimension <= 0:
            raise VectorStoreError("Embedding dimension must be a positive integer.")

        self.index = faiss.IndexFlatL2(self.dimension)
        # For production, consider IndexIDMap to map FAISS indices to our child_ids directly.
        # self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension)) -> then use index.add_with_ids
        self._is_initialized = True
        logger.info(f"FAISS index initialized with dimension {self.dimension} using IndexFlatL2.")

    def add_documents(self, parent_child_data: List[Dict[str, Any]]):
        """
        Adds documents, structured as parent and child chunks, to the vector store.
        Embeddings are generated only for the child chunks.

        Args:
            parent_child_data (List[Dict[str, Any]]): A list of parent chunk dictionaries,
                each containing 'parent_id', 'parent_text', 'source_document_name', # Changed key name
                and a 'children' list (List[Dict{'child_id': str, 'child_text': str}]).
                This structure comes from DocumentProcessor.

        Raises:
            VectorStoreError: If embedding generation or FAISS indexing fails.
        """
        if not parent_child_data:
            logger.warning("add_documents called with empty parent_child_data.")
            return

        logger.info(f"Adding {len(parent_child_data)} parent documents (with their children) to the vector store.")

        child_texts_for_embedding: List[str] = []
        child_metadata_for_store: List[Dict[str, Any]] = []

        for parent_info in parent_child_data:
            parent_id = parent_info.get('parent_id')
            parent_text = parent_info.get('parent_text')

            if not parent_id or not parent_text:
                logger.warning(f"Skipping parent_info due to missing 'parent_id' or 'parent_text'. Data: {str(parent_info)[:200]}")
                continue

            source_doc_name = parent_info.get('source_document_name')
            if not source_doc_name:
                logger.warning(f"Missing 'source_document_name' for parent_id: {parent_id}. Using default value 'Unknown Source Document'.")
                source_doc_name = 'Unknown Source Document' # Provide a default value

            for child_info in parent_info.get('children', []):
                child_id = child_info.get('child_id')
                child_text = child_info.get('child_text') # Also make child_text access safe

                if not child_id:
                    logger.warning(f"Skipping child_info due to missing 'child_id' in parent {parent_id}. Data: {str(child_info)[:200]}")
                    continue

                if not child_text or not child_text.strip():
                    logger.debug(f"Skipping empty or missing child_text for child_id {child_id} from parent {parent_id}.")
                    continue

                child_texts_for_embedding.append(child_text)
                child_metadata_for_store.append({
                    'child_id': child_id,
                    'child_text': child_text,
                    'parent_id': parent_id,
                    'parent_text': parent_text, # Store parent text for easy retrieval
                    'source_document_name': source_doc_name # Changed key name
                })

        if not child_texts_for_embedding:
            logger.warning("No valid child texts found in parent_child_data to embed.")
            return

        logger.info(f"Extracted {len(child_texts_for_embedding)} child chunks for embedding.")

        try:
            # Generate embeddings for all child texts in batch
            child_embeddings_list = self.embedding_service.create_embeddings(child_texts_for_embedding)

            if not child_embeddings_list or len(child_embeddings_list) != len(child_texts_for_embedding):
                # This case implies some embeddings might be missing or empty.
                # Need to handle this carefully. For now, assume all embeddings are returned.
                # A more robust solution would filter out texts for which embeddings failed.
                logger.error("Mismatch between number of child texts and generated embeddings, or empty embeddings list.")
                raise VectorStoreError("Embedding service did not return expected embeddings for all child chunks.")

            child_embeddings_np = np.array(child_embeddings_list, dtype='float32')
            if child_embeddings_np.ndim == 1 and child_embeddings_np.size > 0: # Single embedding
                 child_embeddings_np = np.expand_dims(child_embeddings_np, axis=0)

            if child_embeddings_np.size == 0:
                logger.error("Embeddings array is empty after processing child texts.")
                raise VectorStoreError("No valid embeddings generated for child chunks.")

            if not self._is_initialized:
                self._initialize_index(child_embeddings_np[0]) # Initialize with the first child's embedding

            if self.index is None:
                raise VectorStoreError("FAISS index is not initialized (should have been by _initialize_index).")

            # Add embeddings to FAISS index
            self.index.add(child_embeddings_np)

            # Add corresponding metadata to our document_store
            # The order in child_metadata_for_store matches the order of embeddings
            self.document_store.extend(child_metadata_for_store)

            logger.info(f"Successfully added {len(child_texts_for_embedding)} child_chunk embeddings to FAISS. "
                        f"Total child chunks in store: {len(self.document_store)} (FAISS ntotal: {self.index.ntotal}).")

        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embeddings for child documents: {e}")
            raise VectorStoreError(f"Child chunk embedding generation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to add child documents to FAISS index: {e}")
            # import traceback; traceback.print_exc() # For more detailed debugging if needed
            raise VectorStoreError(f"FAISS index add operation for child chunks failed: {e}")

    def search(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Searches the vector store for child chunks ähnliche to the query text,
        and returns them along with their parent context.

        Args:
            query_text (str): The query text.
            k (int, optional): The number of top similar child chunks to retrieve.
                               Defaults to DEFAULT_VECTOR_STORE_TOP_K.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a retrieved item:
                {
                    'child_id': str,
                    'child_text': str,
                    'parent_id': str,
                    'parent_text': str,
                    'source_document_name': str, # Changed key name
                    'score': float (distance score from FAISS)
                }
                Lower scores (distances) mean higher similarity for L2.
        """
        if not self._is_initialized or self.index is None:
            raise VectorStoreError("Search attempted on uninitialized or empty vector store.")

        if not self.document_store: # Check our metadata store, not just FAISS index ntotal
            logger.warning("Search attempted on an empty document_store (no child/parent metadata).")
            return []

        k_to_use = k if k is not None else DEFAULT_VECTOR_STORE_TOP_K
        # Ensure k is not greater than the number of items in the FAISS index
        k_to_use = min(k_to_use, self.index.ntotal)

        if k_to_use <= 0 : # Check for k_to_use being 0 or negative.
            logger.warning(f"Search k is {k_to_use}, returning empty list.")
            return []

        logger.info(f"Searching for top {k_to_use} child chunks similar to query: '{query_text[:100]}...'")
        try:
            query_embedding_list = self.embedding_service.create_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                raise VectorStoreError("Query embedding generation returned empty result.")

            query_embedding_np = np.array(query_embedding_list, dtype='float32')
            if query_embedding_np.ndim == 1: # Ensure it's 2D for FAISS search
                query_embedding_np = np.expand_dims(query_embedding_np, axis=0)

            distances, indices = self.index.search(query_embedding_np, k_to_use)

            results = []
            if indices.size > 0:
                for i in range(indices.shape[1]): # Iterate through found items for the query
                    doc_index_in_store = indices[0, i]
                    if 0 <= doc_index_in_store < len(self.document_store):
                        retrieved_item_meta = self.document_store[doc_index_in_store]
                        result_entry = {
                            'child_id': retrieved_item_meta['child_id'],
                            'child_text': retrieved_item_meta['child_text'],
                            'parent_id': retrieved_item_meta['parent_id'],
                            'parent_text': retrieved_item_meta['parent_text'],
                            'source_document_name': retrieved_item_meta.get('source_document_name', 'Unknown Source Document'),
                            'score': float(distances[0, i]) # L2 distance
                        }
                        if 'source_document_name' not in retrieved_item_meta:
                            logger.warning(f"Missing 'source_document_name' in document_store item for child_id: {retrieved_item_meta.get('child_id', 'Unknown Child ID')}. "
                                           f"FAISS index: {doc_index_in_store}. Using default value.")
                        results.append(result_entry)
                    else:
                        logger.warning(f"Search returned invalid document index: {doc_index_in_store} "
                                       f"(document_store size: {len(self.document_store)}). Skipping.")

            logger.info(f"Search completed. Found {len(results)} child chunks with parent context.")
            return results

        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise VectorStoreError(f"Query embedding generation failed: {e}")
        except Exception as e:
            logger.error(f"FAISS search operation failed: {e}")
            # import traceback; traceback.print_exc()
            raise VectorStoreError(f"FAISS search operation failed: {e}")

    def save_store(self, index_path: str, metadata_path: str):
        """Saves the FAISS index and the document metadata store."""
        if not self._is_initialized or self.index is None:
            raise VectorStoreError("Cannot save uninitialized index.")
        logger.info(f"Saving FAISS index to {index_path} and metadata to {metadata_path}")
        faiss.write_index(self.index, index_path)

        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "document_store": self.document_store,
                    "dimension": self.dimension
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata store to {metadata_path}: {e}")
            raise VectorStoreError(f"Failed to save metadata: {e}")

    def load_store(self, index_path: str, metadata_path: str):
        """Loads the FAISS index and the document metadata store."""
        logger.info(f"Loading FAISS index from {index_path} and metadata from {metadata_path}")
        try:
            self.index = faiss.read_index(index_path)

            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.document_store = data.get("document_store", [])
                loaded_dimension = data.get("dimension")

            # --- Metadata Compatibility Check ---
            migrated_count = 0
            defaulted_count = 0
            if self.document_store:
                logger.info(f"Performing metadata compatibility check for 'source_document_name' on {len(self.document_store)} loaded items...")
                for i, item_meta in enumerate(self.document_store):
                    if 'source_document_name' not in item_meta or not item_meta['source_document_name']:
                        # Try to migrate from a known old key, e.g., 'doc_name'
                        old_key_name = 'doc_name' # Example old key
                        if old_key_name in item_meta and item_meta[old_key_name]:
                            item_meta['source_document_name'] = item_meta[old_key_name]
                            del item_meta[old_key_name] # Clean up old key
                            migrated_count += 1
                            logger.debug(f"Migrated '{old_key_name}' to 'source_document_name' for item at index {i} (child_id: {item_meta.get('child_id', 'N/A')}).")
                        else:
                            # If no old key to migrate from, set to default
                            item_meta['source_document_name'] = 'Unknown Source Document (Loaded)'
                            defaulted_count += 1
                            logger.warning(f"Missing 'source_document_name' and no suitable old key for migration in loaded item at index {i} (child_id: {item_meta.get('child_id', 'N/A')}). Set to default.")
            if migrated_count > 0:
                logger.info(f"Successfully migrated 'source_document_name' for {migrated_count} items during load.")
            if defaulted_count > 0:
                logger.warning(f"Set 'source_document_name' to default for {defaulted_count} items during load due to missing information.")
            # --- End Metadata Compatibility Check ---

            # Verify consistency
            if loaded_dimension is not None and self.index.d != loaded_dimension:
                logger.warning(f"Dimension mismatch: FAISS index dimension ({self.index.d}) "
                               f"vs loaded metadata dimension ({loaded_dimension}). Using FAISS index's.")
            self.dimension = self.index.d

            if self.index.ntotal != len(self.document_store):
                logger.warning(f"Mismatch in item count: FAISS index has {self.index.ntotal} items, "
                               f"but metadata store has {len(self.document_store)} items. "
                               "This might lead to issues if they are not perfectly aligned.")

            self._is_initialized = True
            logger.info(f"VectorStore loaded. FAISS Dimension: {self.dimension}, "
                        f"Total child vectors in FAISS: {self.index.ntotal}, "
                        f"Total items in metadata store: {len(self.document_store)}")
        except FileNotFoundError:
            logger.error(f"Could not find index file '{index_path}' or metadata file '{metadata_path}'.")
            raise VectorStoreError(f"Index or metadata file not found during load.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata: {e}")
            # import traceback; traceback.print_exc()
            raise VectorStoreError(f"Failed to load store: {e}")

    @property
    def count_child_chunks(self) -> int:
        """Returns the number of child chunks in the store (and FAISS index)."""
        return len(self.document_store) if self.index else 0

    def get_all_child_texts(self) -> List[str]:
        """Utility to get all child texts, e.g., for BM25 indexing."""
        return [item['child_text'] for item in self.document_store]

    def get_child_and_parent_text_by_child_id(self, child_id: str) -> Optional[Dict[str,str]]:
        """Retrieves a child's text and its parent's text using the child_id."""
        # This would be faster if self.document_store was a dict keyed by child_id.
        # For now, linear scan. If performance becomes an issue, optimize storage.
        for item in self.document_store:
            if item['child_id'] == child_id:
                return {'child_text': item['child_text'], 'parent_text': item['parent_text']}
        return None

    def get_child_and_parent_text_by_faiss_index(self, faiss_idx: int) -> Optional[Dict[str,str]]:
        """Retrieves a child's text and its parent's text using the FAISS internal index."""
        if 0 <= faiss_idx < len(self.document_store):
            item = self.document_store[faiss_idx]
            return {'child_text': item['child_text'], 'parent_text': item['parent_text'],
                    'child_id': item['child_id'], 'parent_id': item['parent_id']}
        return None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the VectorStore has been initialized (either by adding docs or loading)."""
        return self._is_initialized


if __name__ == '__main__':
    print("VectorStore (Parent-Child Adapted) Example")
    logging.basicConfig(level=logging.DEBUG)

    class MockEmbeddingService:
        def __init__(self, dim=10): self.dim = dim
        def create_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [list(np.random.rand(self.dim).astype('float32')) for _ in texts]

    mock_emb_service = MockEmbeddingService(dim=5) # Small dimension for example
    vector_store = VectorStore(embedding_service=mock_emb_service)

    # Example parent-child data (simulating output from DocumentProcessor)
    sample_data = [
        {
            "parent_id": "doc1-p1", "parent_text": "Parent One. It talks about apples and oranges. Also mentions bananas.",
            "source_document_name": "doc1.txt", # Changed key name and made it more file-like
            "children": [
                {"child_id": "doc1-p1-c1", "child_text": "Parent One. It talks about apples and oranges."},
                {"child_id": "doc1-p1-c2", "child_text": "Also mentions bananas."}
            ]
        },
        {
            "parent_id": "doc1-p2", "parent_text": "Parent Two. This one is about grapes and strawberries. And kiwi fruit.",
            "source_document_name": "doc1.txt", # Changed key name
            "children": [
                {"child_id": "doc1-p2-c1", "child_text": "Parent Two. This one is about grapes and strawberries."},
                {"child_id": "doc1-p2-c2", "child_text": "And kiwi fruit is tasty."}
            ]
        },
        {
            "parent_id": "doc2-p1", "parent_text": "Another document. Parent Three. Discusses red cars and blue bikes. Fast vehicles.",
            "source_document_name": "doc2.pdf", # Changed key name
            "children": [
                {"child_id": "doc2-p1-c1", "child_text": "Another document. Parent Three. Discusses red cars and blue bikes."},
                {"child_id": "doc2-p1-c2", "child_text": "Fast vehicles are exciting."}
            ]
        }
    ]

    print(f"\nAdding {len(sample_data)} parent documents to VectorStore...")
    try:
        vector_store.add_documents(sample_data)
        print(f"VectorStore now contains {vector_store.count_child_chunks} child chunks.")

        if vector_store.count_child_chunks > 0:
            # Search for similar child chunks
            query1 = "Tell me about bananas"
            print(f"\nSearching for child chunks similar to: '{query1}' (top 2)")
            results1 = vector_store.search(query_text=query1, k=2)
            if results1:
                for res_item in results1:
                    print(f"  - Child ID: {res_item['child_id']}, Score: {res_item['score']:.4f}")
                    print(f"    Child Text: '{res_item['child_text'][:60]}...'")
                    print(f"    Parent Text: '{res_item['parent_text'][:60]}...'")
            else:
                print("  No results found.")

            query2 = "Information on cars"
            print(f"\nSearching for child chunks similar to: '{query2}' (top 2)")
            results2 = vector_store.search(query_text=query2, k=2)
            if results2:
                for res_item in results2:
                    print(f"  - Child ID: {res_item['child_id']}, Score: {res_item['score']:.4f}")
                    print(f"    Child Text: '{res_item['child_text'][:60]}...'")
                    print(f"    Parent Text: '{res_item['parent_text'][:60]}...'")
            else:
                print("  No results found.")

            # Test saving and loading
            temp_index_path = "temp_vs_index.faiss"
            temp_meta_path = "temp_vs_meta.json"
            print(f"\nSaving store to '{temp_index_path}' and '{temp_meta_path}'...")
            vector_store.save_store(temp_index_path, temp_meta_path)

            print("Creating new VectorStore instance for loading...")
            loaded_vector_store = VectorStore(embedding_service=mock_emb_service)
            loaded_vector_store.load_store(temp_index_path, temp_meta_path)
            print(f"Loaded store has {loaded_vector_store.count_child_chunks} child chunks.")

            # Test search on loaded store
            results_loaded = loaded_vector_store.search(query_text=query1, k=1)
            if results_loaded:
                 print(f"Search result from loaded store for '{query1}': Child ID {results_loaded[0]['child_id']}")
            else:
                print(f"No results from loaded store for '{query1}'.")

            # Clean up temp files
            import os
            if os.path.exists(temp_index_path): os.remove(temp_index_path)
            if os.path.exists(temp_meta_path): os.remove(temp_meta_path)
            print("Cleaned up temporary store files.")

    except VectorStoreError as e:
        print(f"VectorStore error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in VectorStore example: {e}")
        import traceback
        traceback.print_exc()

    print("\nVectorStore (Parent-Child Adapted) Example finished.")
