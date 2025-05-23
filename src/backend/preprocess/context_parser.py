from typing import Dict, Any
from src.backend.config.logger_config import setup_logging

import asyncio

logger = setup_logging()


class ContextParser:
    @staticmethod
    def parse_vectorstore_response(response: Dict[str, Any]) -> str:
        """
        Parses the vectorstore response to create a context string.

        Args:
            chroma_results (Dict): List of dictionaries containing document data.

        Returns:
            str: A single string containing all documents joined by double newlines.
        """
        try:
            # Check if the response is empty
            if not response:
                logger.warning("Received an empty vectorstore response.")
                return ""

            # Check if the response is a list
            if isinstance(response, dict):
                                
                retrieved_documents = [doc_val.get("document","document key not present in vectorstore response") for doc_val in  response.values()]
                retrieved_context = "\n\n".join(retrieved_documents)
            else:
                # handle unexpected response format
                retrieved_context = ""
                logger.warning(f"Expected a dictionary.", exc_info=True)
                return ""

        except Exception as e:
            logger.error(f"Error parsing vectorstore response: {e}", exc_info=True)
            return ""

        return retrieved_context
    
    @staticmethod
    def parse_chat_history(memory = None) -> str:
        """
        Parses the chat history to create a context string.

        Args:
            response (Dict): List of dictionaries containing document data.

        Returns:
            str: A single string containing all documents joined by double newlines.
        """
        try:
            if memory and "chat_history" in memory:
                chat_history = memory["chat_history"]
                chat_history = [f"User: {item['user']}\nAssistant: {item['assistant']}" for item in chat_history]
                chat_history = "\n\n".join(chat_history)
            else:
                chat_history = ""
                logger.warning("Chat history not found in memory.")
                return ""
            
        except Exception as e:
            logger.error(f"Error parsing vectorstore response: {e}", exc_info=True)
            return ""

        return chat_history


if __name__ == "__main__":

    # Example vectorstore response
    vectorstore_response = {
        "9eff0d50-ea0a-47b4-a7e9-1387789802e":
        {
            "id": "9eff0d50-ea0a-47b4-a7e9-1387789802e3",
            "document": "Customer: I need help with something I accidently purchased. I want a refund please.\nAssistant: Hi there. For refund info, please check the following link",
            "metadata": {"source": "customer_support", "doc_id": 181710},
            "similarity": 0.7835,
            "distance": 0.2165,
        },
    }

    # Example usage
    context = asyncio.run(ContextParser.parse_vectorstore_response(vectorstore_response))
    logger.info(f"Parsed context:\n{context}")
