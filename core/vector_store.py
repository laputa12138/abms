import logging
import faiss
import numpy as np
from typing import List, Tuple, Optional
from core.embedding_service import EmbeddingService, EmbeddingServiceError
from config.settings import DEFAULT_VECTOR_STORE_TOP_K

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for VectorStore errors."""
    pass

class VectorStore:
    """
    A class for creating, managing, and searching a FAISS-based vector store.
    It uses an EmbeddingService to convert text to vectors.
    """

    def __init__(self, embedding_service: EmbeddingService, dimension: Optional[int] = None):
        """
        Initializes the VectorStore.

        Args:
            embedding_service (EmbeddingService): An instance of EmbeddingService
                                                  to generate text embeddings.
            dimension (Optional[int]): The dimension of the embedding vectors.
                                       If None, it will be inferred from the first embedding.
        """
        self.embedding_service = embedding_service
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.dimension = dimension
        self._is_initialized = False

    def _initialize_index(self, first_embedding: np.ndarray):
        """Helper to initialize FAISS index once dimension is known."""
        if not self.dimension:
            self.dimension = first_embedding.shape[-1]
            logger.info(f"Inferred embedding dimension: {self.dimension}")

        if self.dimension is None or self.dimension <= 0:
            raise VectorStoreError("Embedding dimension must be a positive integer.")

        # Using IndexFlatL2 as a simple, common baseline.
        # For larger datasets, more advanced indexing like IndexIVFFlat might be better.
        self.index = faiss.IndexFlatL2(self.dimension)
        self._is_initialized = True
        logger.info(f"FAISS index initialized with dimension {self.dimension} using IndexFlatL2.")

    def add_documents(self, texts: List[str]):
        """
        Adds a list of texts to the vector store.
        The texts are first converted to embeddings using the EmbeddingService.

        Args:
            texts (List[str]): A list of text strings to add.

        Raises:
            VectorStoreError: If embedding generation fails or adding to FAISS index fails.
        """
        if not texts:
            logger.warning("add_documents called with an empty list of texts.")
            return

        logger.info(f"Adding {len(texts)} documents to the vector store.")
        try:
            embeddings_list_of_lists = self.embedding_service.create_embeddings(texts)
            if not embeddings_list_of_lists or not all(embeddings_list_of_lists): # ensure no empty lists within
                logger.warning("Embedding service returned no embeddings or empty embeddings for some texts.")
                # Filter out texts for which embeddings could not be generated
                valid_texts_embeddings = [(text, emb) for text, emb in zip(texts, embeddings_list_of_lists) if emb]
                if not valid_texts_embeddings:
                    return # No valid embeddings to add
                texts = [item[0] for item in valid_texts_embeddings]
                embeddings_list_of_lists = [item[1] for item in valid_texts_embeddings]


            embeddings_np = np.array(embeddings_list_of_lists, dtype='float32')
            if embeddings_np.ndim == 1: # Handle case of single embedding list
                 embeddings_np = np.expand_dims(embeddings_np, axis=0)

            if not self._is_initialized:
                if embeddings_np.size == 0: # Check if embeddings_np is empty
                    logger.error("Cannot initialize index with empty embeddings array from the first batch.")
                    raise VectorStoreError("Cannot initialize index: first batch of embeddings is empty.")
                self._initialize_index(embeddings_np[0])

            if self.index is None: # Should not happen if _initialize_index worked
                raise VectorStoreError("FAISS index is not initialized.")

            self.index.add(embeddings_np)
            self.documents.extend(texts)
            logger.info(f"Successfully added {len(texts)} documents. Total documents in store: {len(self.documents)}.")
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            raise VectorStoreError(f"Embedding generation failed: {e}")
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS index: {e}")
            raise VectorStoreError(f"FAISS index add operation failed: {e}")

    def search(self, query_text: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Searches the vector store for the top_k most similar documents to the query text.

        Args:
            query_text (str): The query text.
            k (int, optional): The number of top similar documents to retrieve.
                               Defaults to DEFAULT_VECTOR_STORE_TOP_K from settings.

        Returns:
            List[Tuple[str, float]]: A list of tuples, where each tuple contains
                                     a document string and its similarity score (distance).
                                     Lower scores (distances) mean higher similarity for L2.

        Raises:
            VectorStoreError: If the store is empty, not initialized, or search fails.
        """
        if not self._is_initialized or self.index is None:
            logger.error("Search attempted on uninitialized or empty vector store.")
            raise VectorStoreError("Vector store is not initialized or empty.")

        if len(self.documents) == 0:
            logger.warning("Search attempted on an empty document store (though index might be initialized).")
            return []

        k_to_use = k if k is not None else DEFAULT_VECTOR_STORE_TOP_K
        # Ensure k is not greater than the number of documents in the index
        k_to_use = min(k_to_use, self.index.ntotal)


        if k_to_use == 0 :
            logger.warning("Search k is 0, returning empty list.")
            return []


        logger.info(f"Searching for top {k_to_use} documents similar to query: '{query_text[:100]}...'")
        try:
            query_embedding_list = self.embedding_service.create_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                logger.error("Failed to generate embedding for the query text.")
                raise VectorStoreError("Query embedding generation failed.")

            query_embedding_np = np.array(query_embedding_list, dtype='float32')

            # Perform the search
            # D are distances (L2 squared), I are indices
            distances, indices = self.index.search(query_embedding_np, k_to_use)

            results = []
            if indices.size > 0:
                for i in range(indices.shape[1]): # Iterate through found items for the query
                    doc_index = indices[0, i]
                    if 0 <= doc_index < len(self.documents):
                        results.append((self.documents[doc_index], float(distances[0, i])))
                    else:
                        logger.warning(f"Search returned invalid document index: {doc_index}")

            logger.info(f"Search completed. Found {len(results)} documents.")
            return results
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise VectorStoreError(f"Query embedding generation failed: {e}")
        except Exception as e:
            logger.error(f"FAISS search operation failed: {e}")
            raise VectorStoreError(f"FAISS search operation failed: {e}")

    def save_index(self, path: str):
        """Saves the FAISS index to a file."""
        if not self._is_initialized or self.index is None:
            raise VectorStoreError("Cannot save uninitialized index.")
        logger.info(f"Saving FAISS index to {path}")
        faiss.write_index(self.index, path)
        # Note: Documents list also needs to be saved separately if persistence is required.
        # This example only saves the FAISS index itself. A more complete solution
        # would also save `self.documents` and `self.dimension`.

    def load_index(self, path: str, dimension: Optional[int] = None):
        """Loads a FAISS index from a file."""
        logger.info(f"Loading FAISS index from {path}")
        try:
            self.index = faiss.read_index(path)
            self.dimension = dimension or self.index.d # Use provided dimension or from index
            self._is_initialized = True
            logger.info(f"FAISS index loaded. Dimension: {self.dimension}, Total vectors: {self.index.ntotal}")
            # Note: Need to load `self.documents` separately.
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {path}: {e}")
            raise VectorStoreError(f"Failed to load FAISS index: {e}")

    @property
    def count(self) -> int:
        """Returns the number of documents in the store."""
        return len(self.documents)


if __name__ == '__main__':
    # This is an example of how to use the VectorStore.
    # Requires a running Xinference server and a configured EmbeddingService.
    print("VectorStore Example (requires running Xinference for EmbeddingService)")
    print("This part will not be executed by the agent but is for local testing.")

    try:
        # Mock EmbeddingService if Xinference isn't running, for basic structure test
        class MockEmbeddingService:
            def __init__(self, dim=10): # Using a small dimension for mock
                self.dim = dim
                logger.info("Using MockEmbeddingService.")

            def create_embeddings(self, texts: List[str]) -> List[List[float]]:
                # Generate dummy embeddings
                return [list(np.random.rand(self.dim).astype('float32')) for _ in texts]

        # Use actual EmbeddingService if available, otherwise Mock
        try:
            # This will try to connect to Xinference as defined in EmbeddingService
            # from config.settings import XINFERENCE_API_URL, DEFAULT_EMBEDDING_MODEL_NAME
            # embedding_service = EmbeddingService(XINFERENCE_API_URL, DEFAULT_EMBEDDING_MODEL_NAME)
            # print("Attempting to use actual EmbeddingService.")
            # # A quick check to see if the model provides dimension info (not standard)
            # # or embed a dummy text to infer dimension
            # test_emb = embedding_service.create_embeddings(["test"])
            # inferred_dim = len(test_emb[0]) if test_emb and test_emb[0] else None
            # if not inferred_dim:
            #    print("Could not infer dimension from actual EmbeddingService, falling back to mock for example.")
            #    raise EmbeddingServiceError("Dimension inference failed for example.")
            # print(f"Actual EmbeddingService seems to work. Inferred dimension: {inferred_dim}")
            # vector_store = VectorStore(embedding_service=embedding_service, dimension=inferred_dim)

            # Forcing mock for this example run to avoid external dependency for automated test
            raise EmbeddingServiceError("Forcing mock for example.")


        except (EmbeddingServiceError, Exception) as e: # Catch Xinference connection errors too
            print(f"Could not initialize actual EmbeddingService for example ({e}), using MockEmbeddingService with dim 10.")
            mock_dim = 10
            embedding_service = MockEmbeddingService(dim=mock_dim)
            vector_store = VectorStore(embedding_service=embedding_service, dimension=mock_dim)


        # Add documents
        sample_docs = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is rapidly changing the world.",
            "FAISS is a library for efficient similarity search.",
            "Paris is the capital of France.",
            "Large language models are a type of AI."
        ]
        print(f"\nAdding {len(sample_docs)} documents to VectorStore...")
        vector_store.add_documents(sample_docs)
        print(f"VectorStore now contains {vector_store.count} documents.")

        # Search for similar documents
        if vector_store.count > 0:
            query = "What is AI?"
            print(f"\nSearching for documents similar to: '{query}' (top 3)")
            results = vector_store.search(query, k=3)
            if results:
                for doc, score in results:
                    print(f"  - Document: '{doc}' (Distance Score: {score:.4f})")
            else:
                print("  No results found (or k was too small for available docs).")

            query2 = "Tell me about France."
            print(f"\nSearching for documents similar to: '{query2}' (top 3)")
            results2 = vector_store.search(query2, k=3)
            if results2:
                for doc, score in results2:
                    print(f"  - Document: '{doc}' (Distance Score: {score:.4f})")
            else:
                print("  No results found.")
        else:
            print("\nSkipping search as no documents were added (possibly due to embedding issues).")

        # Example of saving and loading (optional, requires a file path)
        # index_file_path = "my_faiss_index.idx"
        # doc_file_path = "my_documents.json" # Need to handle document persistence separately

        # if vector_store.count > 0:
        #     print(f"\nSaving index to {index_file_path}...")
        #     vector_store.save_index(index_file_path)
        #     # Here you would also save vector_store.documents, e.g., as a JSON file
        #     # import json
        #     # with open(doc_file_path, 'w') as f:
        #     #    json.dump({"documents": vector_store.documents, "dimension": vector_store.dimension}, f)


        #     print("Creating a new VectorStore instance to load the index...")
        #     new_embedding_service = MockEmbeddingService(dim=vector_store.dimension) # or actual
        #     loaded_vector_store = VectorStore(embedding_service=new_embedding_service)

        #     # Load documents first
        #     # with open(doc_file_path, 'r') as f:
        #     #    data = json.load(f)
        #     #    loaded_vector_store.documents = data["documents"]
        #     #    loaded_vector_store.dimension = data["dimension"]

        #     loaded_vector_store.load_index(index_file_path, dimension=vector_store.dimension) # Pass dimension if known

        #     # Important: For the loaded index to be useful with `search`, `loaded_vector_store.documents`
        #     # must be repopulated with the same documents in the same order as when the index was created.
        #     # The current `load_index` only loads FAISS index, not the document texts.
        #     # For this example, we'll manually assign them if we were to test search on loaded_vector_store.
        #     # loaded_vector_store.documents = vector_store.documents # This is crucial

        #     print(f"Loaded index contains {loaded_vector_store.index.ntotal if loaded_vector_store.index else 0} vectors.")
        #     print(f"Loaded VectorStore (after loading docs manually) would have {len(loaded_vector_store.documents)} documents.")

        #     # Test search on loaded store (if documents were properly restored)
        #     # if loaded_vector_store.documents and loaded_vector_store.index and loaded_vector_store.index.ntotal > 0:
        #     #     query3 = "efficient search"
        #     #     print(f"\nSearching in loaded VectorStore for: '{query3}'")
        #     #     loaded_results = loaded_vector_store.search(query3, k=2)
        #     #     if loaded_results:
        #     #         for doc, score in loaded_results:
        #     #             print(f"  - Document: '{doc}' (Distance Score: {score:.4f})")
        #     #     else:
        #     #         print("  No results found in loaded store.")
        #     # else:
        #     #      print("\nSkipping search on loaded store as documents or index are not fully set up.")


    except VectorStoreError as e:
        print(f"VectorStore error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in VectorStore example: {e}")
        import traceback
        traceback.print_exc()

    print("\nVectorStore example finished.")
