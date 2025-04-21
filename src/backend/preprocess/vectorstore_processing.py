import pandas as pd
import os

from src.backend.config.logger_config import setup_logging
logger = setup_logging()

def process_documents(raw_documents_path: str) -> tuple:
    """
    Process documents and metadata from raw data file.
    
    Args:
        raw_documents_path (str): Path to the raw documents file.
        
    Returns:
        tuple: (processed_docs, metadata_list) - List of processed documents and their metadata
    """
    processed_docs = []
    metadata_list = []
    
    if not os.path.exists(raw_documents_path):
        logger.error(f"File not found: {raw_documents_path}")
        return [], []
        
    try:
        # Load data from the file
        logger.info(f"Loading documents from {raw_documents_path}")
        df = pd.read_json(raw_documents_path, lines=True)
        
        # Check if required columns exist
        if "output" in df.columns and "input" in df.columns:
            # Process each document , one row at a time
            for idx, row in df.iterrows():
                try:
                    # Format document text
                    doc_text = f"Customer: {row['input'].strip()}\nAssistant: {row['output'].strip()}"
                    processed_docs.append(doc_text)
                    
                    # Create metadata 
                    metadata = {"source": "customer_support", "doc_id": idx}
                    if "id" in row:
                        metadata["original_id"] = row["id"]
                    
                    if "timestamp" in row:
                        metadata["timestamp"] = row["timestamp"]
                        
                    metadata_list.append(metadata)
                except Exception as e:
                    logger.warning(f"Error processing document at index {idx}: {str(e)}")
        else:
            logger.error("Required columns 'input' and 'output' not found in the data")
            
    except Exception as e:
        logger.error(f"Error while processing documents: {str(e)}")
   
    logger.info(f"Successfully processed {len(processed_docs)} documents")
    return processed_docs, metadata_list
