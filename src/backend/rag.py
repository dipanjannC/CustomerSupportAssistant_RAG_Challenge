from langchain_core.runnables import RunnablePassthrough

from src.backend.llm import LLMChatbot
from src.backend.vectorstore import get_vectorstore_instance, Vectorstore
from src.backend.prompt_manager import CustomerAssistantPrompt
from src.backend.preprocess.context_parser import ContextParser
from src.backend.config.logger_config import setup_logging
from src.backend.config.phoenix_config import tracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

logger = setup_logging()

@tracer.start_as_current_span(
    "RAG_base_pipeline", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "chain"}
)
def RAG_base(query_text: str, retriever: Vectorstore) -> str:
    """
    This function implements a Retrieval-Augmented Generation (RAG) pipeline.
    It retrieves relevant documents from a vectorstore based on the input query,
    and then generates a response using a language model (LLM) with the retrieved context.

    Args:
        query_text (str): The input query for which a response is to be generated.

    Returns:
        str: The generated response from the LLM.

    Raises:
        Exception: If there is an error during the retrieval or generation process.
    """
    # Initialize the OpenTelemetry tracer

    # Reference:
    # https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing/custom-spans

    current_span = trace.get_current_span()

    try:
        # Retriever
        retriever_results = retriever.query(query=query_text, top_k=4)
        memory = None

        # Augmentation
        context = ContextParser.parse_vectorstore_response(retriever_results)
        prompt = CustomerAssistantPrompt().get_full_template()
        chat_history = ContextParser.parse_chat_history(memory)

        logger.info(f"Retriever Results: {retriever_results}")
        logger.info(f"Parsed Context: {context}")
        logger.info(f"Chat History: {chat_history}")

        # # Initialize the Mistral LLM
        llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

        # Creating llm chain
        # We can also use RunnableLambda from langchain
        chain = (
            RunnablePassthrough()
            | (lambda input: {"question": input, "context": context,"chat_history": chat_history})
            | prompt
            | llm
        )

        # Generation
        response = chain.invoke({"context": context, "question": query_text})

    except Exception as e:
        current_span.set_status(trace.Status(trace.StatusCode.ERROR))
        current_span.record_exception(e)
        logger.error(f"Error in RAG pipeline: {e}", exc_info=True)

        # Handle the error gracefully as it is exposed to the user
        return f"An error occurred while processing your request. Please try again later."

    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response.content}")
    current_span.set_status(trace.Status(trace.StatusCode.OK))

    return response.content


# TODO: Implement conversation memory
# TODO: Testing the conversation memory
def RAG_with_conversation_memory(query_text: str, retriever: Vectorstore) -> str:
    """
    This function implements a Retrieval-Augmented Generation (RAG) pipeline with conversation memory.
    It retrieves relevant documents from a vectorstore based on the input query,
    and then generates a response using a language model (LLM) with the retrieved context.

    Args:
        query_text (str): The input query for which a response is to be generated.

    Returns:
        str: The generated response from the LLM.
    """
    # TODO 
    # memory = ConversationBufferMemory(return_messages=True)



if __name__ == "__main__":

    # Example usage
    vectorstore = get_vectorstore_instance()
    query_text = (
        "I need help with something I accidently purchased. I want a refund please."
    )
    RAG_base(query_text=query_text, retriever=vectorstore)
