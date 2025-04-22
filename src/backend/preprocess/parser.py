from typing import List, Dict
from src.backend.config.logger_config import setup_logging


logger = setup_logging()

class ContextParser:
    """
    A class to parse context from vectorstore responses.
    """

    def __init__(self):
        pass

    @staticmethod
    def parse_vectorstore_response(vectorstore_response: List[Dict] ) -> str:
        """
        Parses the input results to create a context string.

        Args:
            chroma_results (list): List of dictionaries containing document data.

        Returns:
            str: A single string containing all documents joined by double newlines.
        """
        # Extract just the document content from each result
        documents = [result["document"] for result in vectorstore_response]

        # Create context by joining the documents
        return "\n\n".join(documents)



if __name__ == "__main__":

    # Example vectorstore response
    vectorstore_response = [
        {
            "id": "9eff0d50-ea0a-47b4-a7e9-1387789802e3",
            "document": "Customer: I need help with something I accidently purchased. I want a refund please.\nAssistant: Hi there. For refund info, please check the following link",
            "metadata": {"source": "customer_support", "doc_id": 181710},
            "similarity": 0.7835,
            "distance": 0.2165,
        },
    ]

    # Example usage
    context = ContextParser.parse_vectorstore_response(vectorstore_response)
    logger.info(f"Parsed context:\n{context}")
