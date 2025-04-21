# from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()




# Load the environment variables from .env file
access_key  = os.getenv("MISTRAL_API_KEY")



llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
    # other params...
)
