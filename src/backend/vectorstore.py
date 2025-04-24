import os
from tqdm import tqdm
import time
import uuid
from typing import List, Dict, Any, Optional
import threading
import asyncio
import chromadb
from chromadb.config import Settings

from src.backend.config.vectorstore_config import DEFAULT_DB_PATH
from src.backend.config.logger_config import setup_logging
from src.backend.preprocess.vectorstore_processing import process_documents
from src.backend.embedder import SentenceTransformerEmbeddings
from src.backend.config.phoenix_config import tracer
from src.backend.util.code_util import project_root


logger = setup_logging()

_vectorstore_instance = None
_vectorstore_lock = threading.Lock() # Global lock for thread safety - AI assisted
# Global variable to hold the singleton instance of Vectorstore
# Ensures Thread Safety - AI assisted

class Vectorstore:
    """
    A vector database wrapper for document storage and retrieval using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str,
        persistent_db_path: str = DEFAULT_DB_PATH,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        delete_existing: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the Vectorstore.

        Args:
            collection_name (str): Name of the ChromaDB collection
            persistent_db_path (str): Path to store the persistent DB
            model_name (str): Name of the SentenceTransformer model
            delete_existing (bool): Whether to delete existing collection with the same name
            device (str, optional): Device to use for embeddings (e.g., 'cuda', 'cpu')
        """
        logger.info("Initializing Vectorstore...")


        self.collection_name = collection_name
        self.model_name = model_name

        # Create DB directory if it doesn't exist
        os.makedirs(persistent_db_path, exist_ok=True)

        # Initialize embedding model
        start_time = time.time()
        logger.info(f"Initializing embedding model: {model_name}")
        self.embedding_model = SentenceTransformerEmbeddings(model_name, device=device)
        logger.info(f"Embedder loaded in {time.time() - start_time:.2f} seconds")

        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persistent_db_path,
                # For accessing telemetry data for observability
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Connected to ChromaDB at {persistent_db_path}")

        except Exception as e:
            logger.error(
                f"Failed to initialize ChromaDB client: {str(e)}", exc_info=True
            )
            raise

        # Initialize query cache
        self.cache = {}

        # Set up collection
        self.init_collection(delete_existing)

    def init_collection(self, delete_existing: bool) -> None:
        """
        Set up the vectorstore collection.

        Args:
            delete_existing (bool): Whether to delete existing collection with the same name
        """
        try:
            # Check if collection exists
            existing_collections = {c.name for c in self.client.list_collections()}

            # Check if the collection already exists and if required,
            # delete existing and create new collection
            if self.collection_name in existing_collections:
                if delete_existing:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(name=self.collection_name)
                    self.create_new_collection()
                else:
                    logger.info(f"Using existing collection: {self.collection_name}")
                    self.collection = self.client.get_collection(
                        name=self.collection_name
                    )
            else:
                self.create_new_collection()

        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}", exc_info=True)
            raise

    @tracer.chain
    def create_new_collection(self) -> None:
        """Create a new vectorstore collection with custom settings."""

        logger.info(f"Creating new collection: {self.collection_name}")

        # Create a new collection with custom settings
        # We are using HNSW index for fast approximate nearest neighbor search
        # with cosine similarity rather than IVF, as we want to optimize for speed and accuracy

        self.collection = self.client.create_collection(
            name=self.collection_name,
            # Adding metadata for HNSW index
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,  # Higher values = more accurate but slower indexing
                "hnsw:search_ef": 128,  # Higher values = more accurate but slower search
                "hnsw:M": 8,  # Number of bidirectional links
                "model_name": self.model_name,
            },
        )

    @tracer.chain(name="vectorstore_add_documents")
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = 100,
    ) -> bool:
        """
        Add documents to the vectorstore collection with batching for better performance.

        Args:
            documents (List[str]): List of documents to be added
            metadata (List[dict], optional): List of metadata dictionaries for each document
            batch_size (int): Size of batches for processing

        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []

        # Validate metadata
        # We are assuming that metadata is a list of dictionaries,
        # where each dictionary corresponds to a document
        # If metadata is None, we will create default metadata
        # If metadata is provided, ensure it matches the number of documents
        # Which could later be used for filtering or hybrid searching
        if metadata is None:
            metadata = [{"source": "CustomerSupport"} for _ in documents]

        elif len(metadata) != len(documents):
            logger.warning(
                f"Metadata length ({len(metadata)}) doesn't match documents length ({len(documents)})"
            )

            # Adjust metadata length to match documents
            if len(metadata) < len(documents):
                metadata.extend(
                    [
                        {"source": "CustomerSupport"}
                        for _ in range(len(documents) - len(metadata))
                    ]
                )
            else:
                metadata = metadata[: len(documents)]

        # Generate UUIDs for all documents
        # This is to ensure that each document has a unique ID
        # IDs are required for ChromaDB to identify documents and for retrieval
        all_ids = [str(uuid.uuid4()) for _ in documents]

        # Process in batches
        total_docs = len(documents)
        logger.info(
            f"Adding {total_docs} documents to collection {self.collection_name}"
        )

        with tqdm(total=total_docs, desc="Adding documents") as progress_bar:
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)

                batch_docs = documents[i:end_idx]
                batch_metadata = metadata[i:end_idx]
                batch_ids = all_ids[i:end_idx]

                try:
                    # Embed documents
                    batch_embeddings = self.embedding_model.embed_documents(batch_docs)

                    # Add to ChromaDB
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadata,
                        embeddings=batch_embeddings,
                        ids=batch_ids,
                    )

                except Exception as e:
                    logger.error(
                        f"Error adding batch {i}-{end_idx}: {str(e)}", exc_info=True
                    )
                    return False

                progress_bar.update(len(batch_docs))

        logger.info(
            f"Added {total_docs} documents to collection. Total count: {self.collection.count()}"
        )

        return True

    @tracer.chain(name="retriever")
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_condition: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection.

        Args:
            query (str): The query string
            top_k (int): Number of results to return
            use_cache (bool): Whether to use the query cache
            filter_condition (Dict, optional): Filter condition for ChromaDB query

        Returns:
            Dict[str, Any]: Query results
        """

        if not query:
            logger.warning("Empty query string provided")
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}

        try:
            # Embed the query
            query_embedding = self.embedding_model.embed_query(query)

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_condition,
            )

            # Parse results
            results = self.parse_results(results)

            return results

        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}", exc_info=True)
            # span.record_exception(e)
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}

    

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics
        """
        stats = {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "model_name": self.model_name,
        }
        return stats

    @tracer.chain
    @staticmethod
    def parse_results(
        results: Dict[str, Any], normalize: bool = False
    ) -> Dict[str, Any]:
        """
        Transforms the 'results' dictionary into a list of readable dictionaries.
        Each dictionary contains the document ID, text, metadata, and similarity score.
        """

        # To retrieve the first list from a nested list in the results dictionary.
        # As values are like
        # 'ids': [['8dbb3218-d',..]]

        try:

            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            results = {}

            for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):

                raw_sim = 1.0 - dist

                # As distance in cromadb is in the range of [0, 2]
                # and similarity is in the range of [0, 1]
                # We can use the formula: sim = 1 - dist
                if normalize:
                    # map raw_sim ∈ [-1,1] → norm_sim ∈ [0,1]
                    norm_sim = (raw_sim + 1.0) / 2.0
                    sim = round(norm_sim, 4)
                else:
                    sim = round(raw_sim, 4)

                results[doc_id] = {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "similarity": sim,
                    "distance": round(dist, 4)
                }

        except Exception as e:
            logger.info(f"Error while parsing cromaDB response: {str(e)}", exc_info=True)

        return results


def ingest_documents_to_vectorstore() -> bool:
    """
    Ingest documents into the vectorstore.

    This function processes raw documents, initializes the vectorstore,
    and adds the processed documents to the vectorstore.

    """
    raw_documents_path = os.path.join(
        project_root, "data", "raw", "raw_customer_support_dataset.json"
    )

    # Process documents
    processed_docs, metadata_list = process_documents(raw_documents_path)

    logger.info(f"Processed {len(processed_docs)} documents")

    if not processed_docs:
        logger.error("No documents processed. Exiting.")

    # Initialize Vectorstore
    logger.info("Initializing Vectorstore")

    vectorstore = get_vectorstore_instance(
        collection_name="customer_support",
        persistent_db_path=DEFAULT_DB_PATH,
        delete_existing=True)
    logger.info(f"Adding {len(processed_docs)} documents to Vectorstore")
    vectorstore_response = vectorstore.add_documents(
        documents=processed_docs, metadata=metadata_list
    )

    logger.info(f"Documents successfully: {vectorstore_response}")

    return vectorstore_response


def get_vectorstore_instance(
    collection_name: str = "customer_support",
    persistent_db_path: str = DEFAULT_DB_PATH,
    delete_existing: bool = False,
) -> Vectorstore:
    """_summary_
    Get a singleton instance of the Vectorstore.
    This function ensures that only one instance of the Vectorstore is created and optimizes
    loading of sentence transformer for reusability.

    Args:
        collection_name (str, optional): _description_. Defaults to "customer_support".
        persistent_db_path (str, optional): _description_. Defaults to DEFAULT_DB_PATH.
        delete_existing (bool, optional): _description_. Defaults to False.

    Returns:
        Vectorstore: _instance of Vectorstore
    """

    global _vectorstore_instance

    try: 
        if _vectorstore_instance is None:

            with _vectorstore_lock:
                if _vectorstore_instance is None:  # Double-check locking
                    _vectorstore_instance = Vectorstore(
                        collection_name=collection_name,
                        persistent_db_path=persistent_db_path,
                        delete_existing=delete_existing
                    )

    except Exception as e:
        logger.error(f"Error while creating vectorstore instance: {str(e)}", exc_info=True)    
    
    return _vectorstore_instance


if __name__ == "__main__":

    # Example usage
    # Double Comments to we don't run the ingesting process again
    ## Process all documents
    ## ingest_documents_to_vectorstore()

    # # Unit Testing
    # raw_documents_path = os.path.join(project_root, "data", "raw", "raw_customer_support_dataset.json")

    # # Process documents
    # processed_docs, metadata_list = process_documents(raw_documents_path)

    # logger.info(f"Processed {len(processed_docs)} documents")

    # if not processed_docs:
    #     logger.error("No documents processed. Exiting.")

    # # Display sample documents
    # logger.info("Sample processed documents:")
    # for i, doc in enumerate(processed_docs[:2]):
    #     logger.info(f"DOC {i+1}: \n {doc[:200]} \n")

    # # Initialize Vectorstore
    # logger.info("Initializing Vectorstore")

    # vectorstore = Vectorstore(
    #     collection_name="customer_support",
    #     persistent_db_path=DEFAULT_DB_PATH,
    #     delete_existing=True
    # )

    # # Add documents (using a smaller batch for demonstration)
    # sample_size = 100
    # logger.info(f"Adding {sample_size} sample documents to Vectorstore")

    # vectorstore_response = vectorstore.add_documents(
    #     documents=processed_docs[:sample_size],
    #     metadata=metadata_list[:sample_size]
    # )

    # logger.info(f"Documents successfully: {vectorstore_response}")

    vectorstore = get_vectorstore_instance()

    # Test querying
    test_queries = [
        "What is the status of my order?",
        # "How can I reset my password?",
        # "I need a refund for my purchase"
    ]

    logger.info("Testing queries:")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = asyncio.run(vectorstore.query(query, top_k=3))
        logger.info(f"Results: {results}")

    # Display collection stats
    stats = vectorstore.get_stats()
    logger.info(f"Collection stats: {stats}")
