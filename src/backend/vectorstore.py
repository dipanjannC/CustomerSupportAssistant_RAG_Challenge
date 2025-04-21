
import chromadb
from chromadb.config import Settings
import os
from tqdm import tqdm
import uuid
from typing import List, Dict, Any, Optional


from src.backend.config.logger_config import setup_logging
from src.backend.utilities.code_util import project_root
from src.backend.preprocess.vectorstore_processing import process_documents
from src.backend.embedder import SentenceTransformerEmbeddings

logger = setup_logging()
DEFAULT_DB_PATH = os.path.join(project_root, "data", "db")

class Vectorstore:
    """
    A vector database wrapper for document storage and retrieval using ChromaDB.
    """
    
    def __init__(self, 
                 collection_name: str,
                 persistent_db_path: str = DEFAULT_DB_PATH,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 delete_existing: bool = False,
                 device: Optional[str] = None):
        """
        Initialize the Vectorstore.
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            persistent_db_path (str): Path to store the persistent DB
            model_name (str): Name of the SentenceTransformer model
            delete_existing (bool): Whether to delete existing collection with the same name
            device (str, optional): Device to use for embeddings (e.g., 'cuda', 'cpu')
        """

        self.collection_name = collection_name
        self.model_name = model_name
        
        # Create DB directory if it doesn't exist
        os.makedirs(persistent_db_path, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {model_name}")
        self.embedding_model = SentenceTransformerEmbeddings(model_name, device=device)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persistent_db_path,
                # For accessing telemetry data for observability  
                settings=Settings(anonymized_telemetry=True)
            )
            logger.info(f"Connected to ChromaDB at {persistent_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
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
                    self.collection = self.client.get_collection(name=self.collection_name)
            else:
                self.create_new_collection()
                
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
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
                "hnsw:search_ef": 128,        # Higher values = more accurate but slower search
                "hnsw:M": 8,                 # Number of bidirectional links
                "model_name": self.model_name
            }
        )

    def add_documents(self, 
                    documents: List[str], 
                    metadata: Optional[List[dict]] = None, 
                    batch_size: int = 100) -> bool:
        
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
            logger.warning(f"Metadata length ({len(metadata)}) doesn't match documents length ({len(documents)})")
            
            # Adjust metadata length to match documents
            if len(metadata) < len(documents):
                metadata.extend([{"source": "CustomerSupport"} for _ in range(len(documents) - len(metadata))])
            else:
                metadata = metadata[:len(documents)]
        
        # Generate UUIDs for all documents
        # This is to ensure that each document has a unique ID
        # IDs are required for ChromaDB to identify documents and for retrieval
        all_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Process in batches 
        total_docs = len(documents)
        logger.info(f"Adding {total_docs} documents to collection {self.collection_name}")
        
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
                        ids=batch_ids
                    )
                    
                except Exception as e:
                    logger.error(f"Error adding batch {i}-{end_idx}: {str(e)}")
                    return False
                
                progress_bar.update(len(batch_docs))
        
        logger.info(f"Added {total_docs} documents to collection. Total count: {self.collection.count()}")
        
        return  True


    def query(self,
              query: str, 
              top_k: int = 5, 
              use_cache: bool = True, 
              filter_condition: Optional[Dict] = None) -> Dict[str, Any]:
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
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}
            
        # Create cache key
        cache_key = f"{query}_{top_k}_{str(filter_condition)}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            logger.info(f"Using cached result for query: {query}")
            return self.cache[cache_key]
        
        try:
            # Embed the query
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=top_k,
                where=filter_condition
            )
            
            # Cache the results
            if use_cache:
                self.cache[cache_key] = results
                
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}
        
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.cache = {}
        logger.info("Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        stats = {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "model_name": self.model_name
        }
        return stats

    @staticmethod
    def parse_results(results: Dict[str, Any],normalize:bool = False) -> List[Dict[str, Any]]:
        """
        Transforms the 'results' dictionary into a list of readable dictionaries.
        Each dictionary contains the document ID, text, metadata, and similarity score.
        """

        # To retrieve the first list from a nested list in the results dictionary.
        # As values are like 
        # 'ids': [['8dbb3218-d',..]]

        ids  = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        results = []

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

            results.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "similarity": sim,
                "distance": round(dist, 4)
            })

        return results


def ingest_documents_to_vectorstore() -> bool:
    """
    Ingest documents into the vectorstore.
    
    This function processes raw documents, initializes the vectorstore,
    and adds the processed documents to the vectorstore.

    """
    raw_documents_path = os.path.join(project_root, "data", "raw", "raw_customer_support_dataset.json")
    
    # Process documents
    processed_docs, metadata_list = process_documents(raw_documents_path)

    logger.info(f"Processed {len(processed_docs)} documents")

    if not processed_docs:
        logger.error("No documents processed. Exiting.")
    
    # Initialize Vectorstore
    logger.info("Initializing Vectorstore")

    vectorstore = Vectorstore(
        collection_name="customer_support",
        persistent_db_path=DEFAULT_DB_PATH,
        delete_existing=True
    )

    logger.info(f"Adding {len(processed_docs)} documents to Vectorstore")
    vectorstore_response = vectorstore.add_documents(
        documents=processed_docs,
        metadata=metadata_list
    )

    logger.info(f"Documents successfully: {vectorstore_response}")

    return vectorstore_response



if __name__ == "__main__":

    # Process all documents
    # ingest_documents_to_vectorstore()
    
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

    vectorstore = Vectorstore(
        collection_name="customer_support",
        persistent_db_path=DEFAULT_DB_PATH,
        delete_existing=False
    )

    # Test querying
    test_queries = [
        "What is the status of my order?",
        "How can I reset my password?",
        "I need a refund for my purchase"
    ]
    
    logger.info("Testing queries:")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = vectorstore.query(query, top_k=3)
        results = vectorstore.parse_results(results)
        
        logger.info(f"Results: {results}")
    
    # Display collection stats
    stats = vectorstore.get_stats()
    logger.info(f"Collection stats: {stats}")
    
   
