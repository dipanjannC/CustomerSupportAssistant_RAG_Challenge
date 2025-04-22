from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from llm import LLMChatbot
from vectorstore import Vectorstore
from prompt_manager import CustomerAssistantPrompt

from preprocess.context_parser import ContextParser
from src.backend.config.logger_config import setup_logging
from src.backend.utilities.code_util import project_root

vectorstore = Vectorstore(collection_name="customer_support")

def RAG(query_text: str) -> str:
    """
    This function implements a Retrieval-Augmented Generation (RAG) pipeline.
    It retrieves relevant documents from a vectorstore based on the input query,
    and then generates a response using a language model (LLM) with the retrieved context.
    Args:
        query_text (str): The input query for which a response is to be generated.
    Returns:        
        str: The generated response from the LLM.
    """

    retriever = vectorstore.query(
        query=query_text,
        top_k=4
    )

    context = ContextParser.parse_vectorstore_response(retriever)

    prompt = CustomerAssistantPrompt().get_chat_template()

    llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

    # # Create a chain that will pass the context and question to the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # # Run the chain with the context and a question
    response = chain.run(context=context, question=query_text)
    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response}")
    
    return response



logger = setup_logging()

if __name__ == "__main__":
    
    query_text = "I need help with something I accidently purchased. I want a refund please."

    
    retriever = Vectorstore(collection_name="customer_support").query(
        query=query_text,
        top_k=4
    )

    context = ContextParser.parse_vectorstore_response(retriever)

    prompt = CustomerAssistantPrompt().get_chat_template()

    # # Initialize the Mistral LLM
    llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

    # # Create a chain that will pass the context and question to the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # # Run the chain with the context and a question
    response = chain.run(context=context, question=query_text)

    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response}")