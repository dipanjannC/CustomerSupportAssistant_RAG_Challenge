from langchain_core.runnables import RunnablePassthrough
from openinference.instrumentation.langchain import LangChainInstrumentor
import asyncio

from src.backend.llm import LLMChatbot
from src.backend.vectorstore import get_vectorstore_instance, Vectorstore
from src.backend.prompt_manager import CustomerAssistantPrompt
from src.backend.preprocess.context_parser import ContextParser
from src.backend.config.logger_config import setup_logging
from src.backend.config.phoenix_config import tracer_provider, tracer
from openinference.semconv.trace import SpanAttributes

logger = setup_logging()



@tracer.start_as_current_span("RAG_Pipeline", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "chain"})
def RAG_base(query_text: str, retriever: Vectorstore) -> str:
    """
    This function implements a Retrieval-Augmented Generation (RAG) pipeline.
    It retrieves relevant documents from a vectorstore based on the input query,
    and then generates a response using a language model (LLM) with the retrieved context.
    Args:
        query_text (str): The input query for which a response is to be generated.
    Returns:
        str: The generated response from the LLM.
    """
    try:
        # Retriever
        retriever_results = retriever.query(query=query_text, top_k=4)

        # Augmentation
        context = ContextParser.parse_vectorstore_response(retriever_results)
        prompt = CustomerAssistantPrompt().get_chat_template()

        # # Initialize the Mistral LLM
        llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

        # Creating llm chain
        chain = (
            RunnablePassthrough()
            | (lambda input: {"question": input, "context": context})
            | prompt
            | llm
        )

        # Generation
        response = chain.invoke({"context": context, "question": query_text})
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
        raise e
    
    
    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response.content}")

    return {"status": "done", "response": response.content}
    # return response.content



if __name__ == "__main__":
    # pass
    vectorstore = get_vectorstore_instance()
    query_text = (
        "I need help with something I accidently purchased. I want a refund please."
    )
    RAG_base(query_text=query_text, retriever=vectorstore)
