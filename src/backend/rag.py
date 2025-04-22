from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import phoenix as px
import os

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor,BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor


from dotenv import load_dotenv
load_dotenv()


tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter(
    endpoint=f"{os.getenv("PHOENIX_COLLECTOR_ENDPOINT")}/v1/traces",
    headers={
    "authorization": f"Bearer {os.getenv('PHOENIX_CLIENT_HEADERS')}"}
)

# Attach BatchSpanProcessor
processor = BatchSpanProcessor(span_exporter)
tracer_provider.add_span_processor(processor)
trace.set_tracer_provider(tracer_provider)

# Get a tracer instance
tracer = trace.get_tracer(__name__)
# Instrument LangChain
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


from llm import LLMChatbot
from vectorstore import Vectorstore
from prompt_manager import CustomerAssistantPrompt

from preprocess.context_parser import ContextParser
from src.backend.config.logger_config import setup_logging
from src.backend.utilities.code_util import project_root
logger = setup_logging()

vectorstore = Vectorstore(collection_name="customer_support")



@tracer.start_as_current_span("RAG_pipeline")
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
    response = chain.invoke({
    "context": context,     
    "question": query_text
    })
    
    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response.get("text","No response")}")
    
    return response





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

    # Run the chain with the context and a question
    response = chain.invoke({
    "context": context, 
    "question": query_text
})

    logger.info(f"Question:\n{query_text}\n")
    logger.info(f"Response:\n{response.get("text","No response")}")