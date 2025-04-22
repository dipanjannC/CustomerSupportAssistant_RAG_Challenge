# from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()





llm = ChatMistralAI(
    
    # mistral-7b updated version
    model="ministral-8b-latest",
    temperature=0,
    # top_p= 0.95,
    # top_k= 100,
    # max_new_tokens=256,
    # repitition_penalty=1.2,

    max_retries=2,
    api_key=os.getenv("MISTRAL_API_KEY"),
    
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
response = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

print(response)