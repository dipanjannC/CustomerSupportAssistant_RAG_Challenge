
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List, Optional
from tqdm import tqdm

from src.backend.config.logger_config import setup_logging


logger = setup_logging()


class SentenceTransformerEmbeddings(Embeddings):
    """
    Implementation of langchain embeddings using SentenceTransformer models.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): The name of the SentenceTransformer model to use
            device (str, optional): Device to use ('cuda', 'cpu'). Defaults to None.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name 
        self.batch_size = 32  # Can be configured based on available memory

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents, showing progress and batching.
        
        Args:
            documents (List[str]): List of document texts to embed
            
        Returns:
            List[List[float]]: List of embeddings for each document
        """
        if not documents:
            return []
            
        embeddings = []
        total_docs = len(documents)
        
        # Use tqdm for progress tracking
        with tqdm(total=total_docs, desc="Embedding documents") as progress_bar:
            # Process in batches to reduce memory usage
            for i in range(0, total_docs, self.batch_size):
                end_idx = min(i + self.batch_size, total_docs)
                batch = documents[i:end_idx]
                
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                    
                    embeddings.extend(batch_embeddings.tolist())
                    progress_bar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error embedding batch {i}-{end_idx}: {str(e)}")
                    raise ValueError(f"Embedding failed for documents {i}-{end_idx}") from e
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query (str): The query text to embed
            
        Returns:
            List[float]: Embedding for the query
        """
        if not query:
            # Return zero vector if query is empty
            raise ValueError("Query is empty. Cannot generate embedding.")
            
        try:
            # Embed the query
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            # Return zero vector as fallback
            raise ValueError(f"Embedding failed for query: {query}")


